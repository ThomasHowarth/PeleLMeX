#include "PeleLMeX_Index.H"
#include "PeleLMeX.H"
#include "PeleLMeX_K.H"
#include "PeleLMeX_DeriveFunc.H"
#include "PeleLMeX_Utils.H"

#include <PelePhysics.H>
#include <mechanism.H>
#ifdef AMREX_USE_EB
#include <AMReX_EBFabFactory.H>
#include <AMReX_EBFArrayBox.H>
#endif

namespace pelelmex_derive_detail {

// Velocity gradient tensor: gradU(n, d) = d u_n / d x_d
using GradU =
  amrex::Array2D<amrex::Real, 0, AMREX_SPACEDIM - 1, 0, AMREX_SPACEDIM - 1>;

// d vel(n) / dx_d at (i,j,k). side = 0 uses the centred difference; side =
// +1 (-1) uses the second-order one-sided stencil reaching to the high
// (low) side, for cells next to covered EB cells.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
vel_deriv(
  amrex::Array4<const amrex::Real> const& vel,
  int i,
  int j,
  int k,
  int n,
  int d,
  int side,
  amrex::Real inv_dx) noexcept
{
  const int di = (d == 0) ? 1 : 0;
  const int dj = (d == 1) ? 1 : 0;
  const int dk = (d == 2) ? 1 : 0;
  constexpr amrex::Real c0 = -1.5;
  constexpr amrex::Real c1 = 2.0;
  constexpr amrex::Real c2 = -0.5;
  if (side == 0) {
    return 0.5 *
           (vel(i + di, j + dj, k + dk, n) - vel(i - di, j - dj, k - dk, n)) *
           inv_dx;
  }
  const int s = side;
  const amrex::Real onesided =
    c0 * vel(i, j, k, n) + c1 * vel(i + s * di, j + s * dj, k + s * dk, n) +
    c2 * vel(i + 2 * s * di, j + 2 * s * dj, k + 2 * s * dk, n);
  return (side > 0) ? onesided * inv_dx : -onesided * inv_dx;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE GradU
velocity_gradient(
  amrex::Array4<const amrex::Real> const& vel,
  int i,
  int j,
  int k,
  const int side[AMREX_SPACEDIM],
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxinv) noexcept
{
  GradU g;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      g(n, d) = vel_deriv(vel, i, j, k, n, d, side[d], dxinv[d]);
    }
  }
  return g;
}

// Vorticity from the velocity gradient (only the z component in 2D)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, 3>
vorticity(GradU const& g) noexcept
{
#if (AMREX_SPACEDIM == 3)
  return {g(2, 1) - g(1, 2), g(0, 2) - g(2, 0), g(1, 0) - g(0, 1)};
#else
  return {0.0, 0.0, g(1, 0) - g(0, 1)};
#endif
}

// Call f(i, j, k, gradU) for each cell of bx, with gradU the (mesh-mapping
// aware) velocity gradient of vel, switching to one-sided stencils next to
// covered EB cells. Covered cells call on_covered(i, j, k) instead.
template <typename F, typename C>
void
forEachVelocityGradient(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  const amrex::FArrayBox& statefab,
  const amrex::Geometry& geom,
  amrex::Array4<const amrex::Real> const& vel,
  F const& f,
  C const& on_covered)
{
  const auto dxinv0 = geom.InvCellSizeArray();
  // Physical derivatives under mesh mapping (identity otherwise).
  const MeshMapEvaluator mmap = a_pelelm->meshMapEvaluator();
  const amrex::GeometryData gd = geom.data();

#ifdef AMREX_USE_EB
  const auto& ebfab = static_cast<amrex::EBFArrayBox const&>(statefab);
  const amrex::EBCellFlagFab& flags = ebfab.getEBCellFlagFab();
  const auto typ = flags.getType(bx);

  if (typ == amrex::FabType::covered) {
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      on_covered(i, j, k);
    });
    return;
  }
  if (typ == amrex::FabType::singlevalued) {
    const auto& flag_fab = flags.const_array();
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const auto flag = flag_fab(i, j, k);
      if (flag.isCovered()) {
        on_covered(i, j, k);
        return;
      }
      // Covered neighbour on the high side: reach to the low side, etc.
      int side[AMREX_SPACEDIM];
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const int di = (d == 0) ? 1 : 0;
        const int dj = (d == 1) ? 1 : 0;
        const int dk = (d == 2) ? 1 : 0;
        if (!flag.isConnected(di, dj, dk)) {
          side[d] = -1;
        } else if (!flag.isConnected(-di, -dj, -dk)) {
          side[d] = 1;
        } else {
          side[d] = 0;
        }
      }
      const auto dxinv = mapped_inv_dx_array(i, j, k, dxinv0, mmap, gd);
      f(i, j, k, velocity_gradient(vel, i, j, k, side, dxinv));
    });
    return;
  }
#else
  amrex::ignore_unused(statefab);
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const int side[AMREX_SPACEDIM] = {0};
    const auto dxinv = mapped_inv_dx_array(i, j, k, dxinv0, mmap, gd);
    f(i, j, k, velocity_gradient(vel, i, j, k, side, dxinv));
  });
}

} // namespace pelelmex_derive_detail

//
// Extract temp
//
void
pelelmex_dertemp(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  amrex::ignore_unused(a_pelelm, ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  AMREX_ASSERT(!a_pelelm->m_incompressible);
  auto const in_dat = statefab.array();
  auto der = derfab.array(dcomp);
  amrex::ParallelFor(
    bx, [der, in_dat] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      der(i, j, k) = in_dat(i, j, k, TEMP);
    });
}

//
// Compute heat release
//
void
pelelmex_derheatrelease(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& reactfab,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  amrex::ignore_unused(a_pelelm, ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  AMREX_ASSERT(!a_pelelm->m_incompressible);

  amrex::FArrayBox EnthFab;
  EnthFab.resize(bx, NUM_SPECIES, amrex::The_Async_Arena());

  auto const temp = statefab.const_array(TEMP);
  auto const& Hi = EnthFab.array();
  auto HRR = derfab.array(dcomp);
  auto const react = reactfab.const_array();
  auto const* leosparm = a_pelelm->eos_parms.device_parm();
  amrex::ParallelFor(
    bx, [temp, Hi, leosparm, HRR,
         react] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      getHGivenT(i, j, k, temp, Hi, leosparm);
      HRR(i, j, k) = 0.0;
      for (int n = 0; n < NUM_SPECIES; ++n) {
        HRR(i, j, k) -= Hi(i, j, k, n) * react(i, j, k, n);
      }
    });
  amrex::Gpu::streamSynchronize();
}

//
// Extract species mass fractions Y_n
//
void
pelelmex_dermassfrac(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  amrex::ignore_unused(a_pelelm, ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  AMREX_ASSERT(statefab.nComp() >= NUM_SPECIES + 1);
  AMREX_ASSERT(ncomp == NUM_SPECIES);
  AMREX_ASSERT(!a_pelelm->m_incompressible);
  auto const in_dat = statefab.array();
  auto der = derfab.array(dcomp);
  amrex::ParallelFor(
    bx, NUM_SPECIES,
    [in_dat, der] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
      const amrex::Real rhoinv = 1.0 / in_dat(i, j, k, DENSITY);
      der(i, j, k, n) = in_dat(i, j, k, FIRSTSPEC + n) * rhoinv;
    });
}

//
// Extract species mole fractions X_n
//
void
pelelmex_dermolefrac(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  amrex::ignore_unused(a_pelelm, ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  AMREX_ASSERT(statefab.nComp() >= NUM_SPECIES + 1);
  AMREX_ASSERT(ncomp == NUM_SPECIES);
  AMREX_ASSERT(!a_pelelm->m_incompressible);
  auto const in_dat = statefab.array();
  auto der = derfab.array(dcomp);
  auto const* leosparm = a_pelelm->eos_parms.device_parm();
  amrex::ParallelFor(
    bx, [in_dat, leosparm, der] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real Yt[NUM_SPECIES] = {0.0};
      amrex::Real Xt[NUM_SPECIES] = {0.0};
      amrex::Real rhoinv = 1.0 / in_dat(i, j, k, DENSITY);
      for (int n = 0; n < NUM_SPECIES; ++n) {
        Yt[n] = in_dat(i, j, k, FIRSTSPEC + n) * rhoinv;
      }
      auto eos = pele::physics::PhysicsType::eos(leosparm);
      eos.Y2X(Yt, Xt);
      for (int n = 0; n < NUM_SPECIES; ++n) {
        der(i, j, k, n) = Xt[n];
      }
    });
}

//
// Extract element mass fractions Z_n
//
void
pelelmex_derelementfrac(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  amrex::ignore_unused(a_pelelm, ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  AMREX_ASSERT(statefab.nComp() >= NUM_SPECIES + 1);
  AMREX_ASSERT(ncomp == NUM_ELEMENTS);
  AMREX_ASSERT(!a_pelelm->m_incompressible);
  auto const in_dat = statefab.array();
  auto der = derfab.array(dcomp);
  auto const* leosparm = a_pelelm->eos_parms.device_parm();
  amrex::ParallelFor(
    bx, [in_dat, leosparm, der] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real Yt[NUM_SPECIES] = {0.0};
      amrex::Real Zt[NUM_ELEMENTS] = {0.0};
      amrex::Real rhoinv = 1.0 / in_dat(i, j, k, DENSITY);
      for (int n = 0; n < NUM_SPECIES; ++n) {
        Yt[n] = in_dat(i, j, k, FIRSTSPEC + n) * rhoinv;
      }
      auto eos = pele::physics::PhysicsType::eos(leosparm);
      eos.Y2Z(Yt, Zt);
      for (int n = 0; n < NUM_ELEMENTS; ++n) {
        der(i, j, k, n) = Zt[n];
      }
    });
}

//
// Extract rho - sum rhoY
//
void
pelelmex_derrhomrhoy(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  amrex::ignore_unused(a_pelelm, ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  AMREX_ASSERT(statefab.nComp() >= NUM_SPECIES + 1);
  AMREX_ASSERT(ncomp == 1);
  AMREX_ASSERT(!a_pelelm->m_incompressible);
  auto const in_dat = statefab.array();
  auto der = derfab.array(dcomp);
  amrex::ParallelFor(
    bx, [der, in_dat] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      der(i, j, k, 0) = in_dat(i, j, k, DENSITY);
      for (int n = 0; n < NUM_SPECIES; ++n) {
        der(i, j, k, 0) -= in_dat(i, j, k, FIRSTSPEC + n);
      }
    });
}

//
// Compute cell averaged pressure from nodes
//
void
pelelmex_deravgpress(
  PeleLM* /*a_pelelm*/,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int /*ncomp*/,
  const amrex::FArrayBox& /*statefab*/,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& pressfab,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  AMREX_ASSERT(derfab.box().contains(bx));
  auto const in_dat = pressfab.array();
  auto der = derfab.array(dcomp);
  amrex::ParallelFor(
    bx, [der, in_dat] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      constexpr amrex::Real factor = 1.0 / (AMREX_D_TERM(2.0, *2.0, *2.0));
      der(i, j, k) =
        factor * (in_dat(i + 1, j, k) + in_dat(i, j, k)
#if (AMREX_SPACEDIM >= 2)
                  + in_dat(i + 1, j + 1, k) + in_dat(i, j + 1, k)
#if (AMREX_SPACEDIM == 3)
                  + in_dat(i + 1, j, k + 1) + in_dat(i, j, k + 1) +
                  in_dat(i + 1, j + 1, k + 1) + in_dat(i, j + 1, k + 1)
#endif
#endif
                 );
    });
}

//
// Compute the velocity magnitude
//
void
pelelmex_dermgvel(
  PeleLM* /*a_pelelm*/,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int /*ncomp*/,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  auto const vel = statefab.array(VELX);
  auto der = derfab.array(dcomp);
  amrex::ParallelFor(
    bx, [der, vel] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      der(i, j, k) = std::sqrt((AMREX_D_TERM(
        vel(i, j, k, 0) * vel(i, j, k, 0), +vel(i, j, k, 1) * vel(i, j, k, 1),
        +vel(i, j, k, 2) * vel(i, j, k, 2))));
    });
}

//
// Compute vorticity magnitude
//
void
pelelmex_dermgvort(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int /*ncomp*/,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& geom,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  using namespace pelelmex_derive_detail;
  auto const& vort_arr = derfab.array(dcomp);
  forEachVelocityGradient(
    a_pelelm, bx, statefab, geom, statefab.const_array(VELX),
    [=] AMREX_GPU_DEVICE(int i, int j, int k, GradU const& g) noexcept {
      const auto w = vorticity(g);
#if (AMREX_SPACEDIM == 2)
      vort_arr(i, j, k) = std::abs(w[2]);
#else
      vort_arr(i, j, k) = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
#endif
    },
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      vort_arr(i, j, k) = 0.0;
    });
}

//
// Compute vorticity components
//
void
pelelmex_dervort(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& geom,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  using namespace pelelmex_derive_detail;
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  auto const& vort_arr = derfab.array(dcomp);
  forEachVelocityGradient(
    a_pelelm, bx, statefab, geom, statefab.const_array(VELX),
    [=] AMREX_GPU_DEVICE(int i, int j, int k, GradU const& g) noexcept {
      const auto w = vorticity(g);
#if (AMREX_SPACEDIM == 2)
      vort_arr(i, j, k) = w[2];
#else
      // Note: the 3D components are stored squared
      vort_arr(i, j, k, 0) = w[0] * w[0];
      vort_arr(i, j, k, 1) = w[1] * w[1];
      vort_arr(i, j, k, 2) = w[2] * w[2];
#endif
    },
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      for (int n = 0; n < ncomp; ++n) {
        vort_arr(i, j, k, n) = 0.0;
      }
    });
}

//
// Compute cell-centered coordinates
//
void
pelelmex_dercoord(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox&
#ifdef AMREX_USE_EB
    statefab
#else
/*unused*/
#endif
  ,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& geom,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  amrex::ignore_unused(ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);

  auto const& coord_arr = derfab.array(dcomp);
  const auto geomdata = geom.data();
  const MeshMapEvaluator mmap = a_pelelm->meshMapEvaluator(); // physical x

#ifdef AMREX_USE_EB
  AMREX_ASSERT(statefab.box().contains(bx));
  const auto& ebfab = static_cast<amrex::EBFArrayBox const&>(statefab);
  const amrex::EBCellFlagFab& flags = ebfab.getEBCellFlagFab();

  auto typ = flags.getType(bx);
  // Compute cell center coordinates even in covered boxes/cell. Only
  // modify the cell-center in cut cells
  if (typ == amrex::FabType::singlevalued) {
    const auto& flag_arr = flags.const_array();
    const auto& ccent_fab = ebfab.getCentroidData();
    const auto& ccent_arr = ccent_fab->const_array();
    amrex::ParallelFor(
      bx, [flag_arr, ccent_arr, coord_arr,
           geomdata] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const amrex::Real* prob_lo = geomdata.ProbLo();
        if (flag_arr(i, j, k).isCovered() || flag_arr(i, j, k).isRegular()) {
          AMREX_D_TERM(coord_arr(i, j, k, 0) =
                         prob_lo[0] + (i + 0.5) * geomdata.CellSize(0);
                       , coord_arr(i, j, k, 1) =
                           prob_lo[1] + (j + 0.5) * geomdata.CellSize(1);
                       , coord_arr(i, j, k, 2) =
                           prob_lo[2] + (k + 0.5) * geomdata.CellSize(2););
        } else {
          AMREX_D_TERM(coord_arr(i, j, k, 0) =
                         prob_lo[0] + (i + 0.5 + ccent_arr(i, j, k, 0)) *
                                        geomdata.CellSize(0);
                       , coord_arr(i, j, k, 1) =
                           prob_lo[1] + (j + 0.5 + ccent_arr(i, j, k, 1)) *
                                          geomdata.CellSize(1);
                       , coord_arr(i, j, k, 2) =
                           prob_lo[2] + (k + 0.5 + ccent_arr(i, j, k, 2)) *
                                          geomdata.CellSize(2););
        }
      });
  } else
#endif
  {
    amrex::ParallelFor(
      bx, [coord_arr, geomdata,
           mmap] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        AMREX_D_TERM(coord_arr(i, j, k, 0) = mmap.x_phys_cc(0, i, geomdata);
                     , coord_arr(i, j, k, 1) = mmap.x_phys_cc(1, j, geomdata);
                     , coord_arr(i, j, k, 2) = mmap.x_phys_cc(2, k, geomdata););
      });
  }
}

//
// Compute Q-criterion
//
void
pelelmex_derQcrit(
  PeleLM* a_pelelm,
  const amrex::Box&
#if AMREX_SPACEDIM == 3
    bx
#endif
  ,
  amrex::FArrayBox&
#if AMREX_SPACEDIM == 3
    derfab
#endif
  ,
  int
#if AMREX_SPACEDIM == 3
    dcomp
#endif
  ,
  int /*ncomp*/,
  const amrex::FArrayBox&
#if AMREX_SPACEDIM == 3
    statefab
#endif
  ,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry&
#if AMREX_SPACEDIM == 3
    geom
#endif
  ,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
#if AMREX_SPACEDIM == 3
  using namespace pelelmex_derive_detail;
  auto const& qcrit_arr = derfab.array(dcomp);
  forEachVelocityGradient(
    a_pelelm, bx, statefab, geom, statefab.const_array(VELX),
    [=] AMREX_GPU_DEVICE(int i, int j, int k, GradU const& gradU) noexcept {
      // Divu
      const amrex::Real divU = gradU(0, 0) + gradU(1, 1) + gradU(2, 2);

      // Directly Assemble Sym. & AntiSym. into Qcrit.
      // Remove divU (dilatation) from the Sym. tensor (due to mixing/reaction
      // most often)
      qcrit_arr(i, j, k) = 0.0;
      for (int dim1 = 0; dim1 < AMREX_SPACEDIM; ++dim1) {
        for (int dim2 = 0; dim2 < AMREX_SPACEDIM; ++dim2) {
          const amrex::Real Ohm = 0.5 * (gradU(dim1, dim2) - gradU(dim2, dim1));
          amrex::Real Sij = 0.5 * (gradU(dim1, dim2) + gradU(dim2, dim1));
          if (dim1 == dim2) {
            Sij -= divU / static_cast<amrex::Real>(AMREX_SPACEDIM);
          }
          qcrit_arr(i, j, k) += Ohm * Ohm - Sij * Sij;
        }
      }
    },
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      qcrit_arr(i, j, k) = 0.0;
    });
#else
  amrex::ignore_unused(a_pelelm);
#endif
}

//
// Compute the kinetic energy
//
void
pelelmex_derkineticenergy(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int /*ncomp*/,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  if (a_pelelm->m_incompressible != 0) {
    auto const vel = statefab.array(VELX);
    auto der = derfab.array(dcomp);
    const auto rho = a_pelelm->m_rho;
    amrex::ParallelFor(
      bx, [der, vel, rho] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        der(i, j, k) = 0.5 * rho *
                       (AMREX_D_TERM(
                         vel(i, j, k, 0) * vel(i, j, k, 0),
                         +vel(i, j, k, 1) * vel(i, j, k, 1),
                         +vel(i, j, k, 2) * vel(i, j, k, 2)));
      });
  } else {
    auto const rho = statefab.array(DENSITY);
    auto const vel = statefab.array(VELX);
    auto der = derfab.array(dcomp);
    amrex::ParallelFor(
      bx, [der, vel, rho] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        der(i, j, k) = 0.5 * rho(i, j, k) *
                       (AMREX_D_TERM(
                         vel(i, j, k, 0) * vel(i, j, k, 0),
                         +vel(i, j, k, 1) * vel(i, j, k, 1),
                         +vel(i, j, k, 2) * vel(i, j, k, 2)));
      });
  }
}

//
// Compute enstrophy
//
void
pelelmex_derenstrophy(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int /*ncomp*/,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& geom,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  using namespace pelelmex_derive_detail;
  const auto incomp = a_pelelm->m_incompressible;
  const auto rho = a_pelelm->m_rho;
  auto const& rho_arr = (incomp != 0) ? amrex::Array4<const amrex::Real>{}
                                      : statefab.const_array(DENSITY);
  auto const& ens_arr = derfab.array(dcomp);
  forEachVelocityGradient(
    a_pelelm, bx, statefab, geom, statefab.const_array(VELX),
    [=] AMREX_GPU_DEVICE(int i, int j, int k, GradU const& g) noexcept {
      const amrex::Real l_rho = (incomp != 0) ? rho : rho_arr(i, j, k);
      const auto w = vorticity(g);
#if (AMREX_SPACEDIM == 2)
      ens_arr(i, j, k) = 0.5 * l_rho * w[2] * w[2];
#else
      ens_arr(i, j, k) =
        0.5 * l_rho * (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
#endif
    },
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      ens_arr(i, j, k) = 0.0;
    });
}

//
// Compute mixture fraction
//
void
pelelmex_dermixfrac(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  amrex::ignore_unused(ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(ncomp == 1);

  if (a_pelelm->Zfu < 0.0) {
    amrex::Abort("Mixture fraction not initialized");
  }

  auto const density = statefab.array(DENSITY);
  auto const rhoY = statefab.array(FIRSTSPEC);
  auto mixt_frac = derfab.array(dcomp);

  amrex::Real Zox_lcl = a_pelelm->Zox;
  amrex::Real Zfu_lcl = a_pelelm->Zfu;
  amrex::Real denom_inv = 1.0 / (Zfu_lcl - Zox_lcl);
  amrex::GpuArray<amrex::Real, NUM_SPECIES> fact_Bilger;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    fact_Bilger[n] = a_pelelm->spec_Bilger_fact[n];
  }

  amrex::ParallelFor(
    bx, [density, mixt_frac, rhoY, Zox_lcl, denom_inv,
         fact_Bilger] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const amrex::Real rho_inv = 1.0 / density(i, j, k);
      mixt_frac(i, j, k) = 0.0;
      for (int n = 0; n < NUM_SPECIES; ++n) {
        mixt_frac(i, j, k) += (rhoY(i, j, k, n) * fact_Bilger[n]) * rho_inv;
      }
      mixt_frac(i, j, k) = (mixt_frac(i, j, k) - Zox_lcl) * denom_inv;
    });
}

//
// Compute progress variable
//
void
pelelmex_derprogvar(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)

{
  amrex::ignore_unused(ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(ncomp == 1);

  if (a_pelelm->m_C0 < 0.0) {
    amrex::Abort("Progress variable not initialized");
  }

  auto const density = statefab.array(DENSITY);
  auto const rhoY = statefab.array(FIRSTSPEC);
  auto const temp = statefab.array(TEMP);
  auto prog_var = derfab.array(dcomp);

  const amrex::Real C0_lcl = a_pelelm->m_C0;
  const amrex::Real C1_lcl = a_pelelm->m_C1;
  const amrex::Real denom_inv = 1.0 / (C1_lcl - C0_lcl);
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 1> Cweights;
  for (int n = 0; n < NUM_SPECIES + 1; ++n) {
    Cweights[n] = a_pelelm->m_Cweights[n];
  }

  const auto revert = a_pelelm->m_Crevert;
  amrex::ParallelFor(
    bx, [density, prog_var, rhoY, Cweights, denom_inv, C0_lcl, temp,
         revert] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real rho_inv = 1.0 / density(i, j, k);
      prog_var(i, j, k) = 0.0;
      for (int n = 0; n < NUM_SPECIES; ++n) {
        prog_var(i, j, k) += (rhoY(i, j, k, n) * Cweights[n]) * rho_inv;
      }
      prog_var(i, j, k) += temp(i, j, k) * Cweights[NUM_SPECIES];
      if (revert != 0) {
        prog_var(i, j, k) = 1.0 - (prog_var(i, j, k) - C0_lcl) * denom_inv;
      } else {
        prog_var(i, j, k) = (prog_var(i, j, k) - C0_lcl) * denom_inv;
      }
    });
}

//
// Extract mixture viscosity
//
void
pelelmex_dervisc(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  amrex::ignore_unused(ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);

  if (a_pelelm->m_incompressible != 0) {
    derfab.setVal<amrex::RunOn::Device>(a_pelelm->m_mu, bx, dcomp, 1);
  } else {
    auto const& rhoY = statefab.const_array(FIRSTSPEC);
    auto const& T = statefab.array(TEMP);
    auto der = derfab.array(dcomp);
    auto const* ltransparm = a_pelelm->trans_parms.device_parm();
    amrex::ParallelFor(
      bx, [rhoY, T, der,
           ltransparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        getVelViscosity(i, j, k, rhoY, T, der, ltransparm);
      });
  }
}

//
// Extract mixture averaged species diffusion coefficients
//
void
pelelmex_derdiffc(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  amrex::ignore_unused(ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  if (a_pelelm->m_use_soret != 0) {
    AMREX_ASSERT(ncomp == 2 * NUM_SPECIES);
  }
  if (a_pelelm->m_use_soret == 0) {
    AMREX_ASSERT(ncomp == NUM_SPECIES);
  }
  const bool do_fixed_Le = (a_pelelm->m_fixed_Le != 0);
  const bool do_fixed_Pr = (a_pelelm->m_fixed_Pr != 0);
  const bool do_soret = (a_pelelm->m_use_soret != 0);
  amrex::FArrayBox dummies(bx, NUM_SPECIES + 2, amrex::The_Async_Arena());
  auto const& rhoY = statefab.const_array(FIRSTSPEC);
  auto const& T = statefab.array(TEMP);
  auto rhoD = derfab.array(dcomp);
  auto lambda = dummies.array(0);
  auto mu = dummies.array(1);
  auto const* ltransparm = a_pelelm->trans_parms.device_parm();
  auto const* leosparm = a_pelelm->eos_parms.device_parm();
  auto rhotheta = do_soret ? derfab.array(dcomp + NUM_SPECIES)
                           : dummies.array(2); // dummy for no soret
  amrex::Real LeInv = a_pelelm->m_Lewis_inv;
  amrex::Real PrInv = a_pelelm->m_Prandtl_inv;
  amrex::ParallelFor(
    bx, [do_fixed_Le, do_fixed_Pr, do_soret, LeInv, PrInv, rhoY, T, rhoD,
         rhotheta, lambda, mu, ltransparm,
         leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      getTransportCoeff<pele::physics::PhysicsType::eos_type>(
        i, j, k, do_fixed_Le, do_fixed_Pr, do_soret, LeInv, PrInv, rhoY, T,
        rhoD, rhotheta, lambda, mu, ltransparm, leosparm);
    });
}

//
// Extract thermal diffusivity
//
void
pelelmex_derlambda(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  amrex::ignore_unused(ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  bool do_fixed_Le = (a_pelelm->m_fixed_Le != 0);
  bool do_fixed_Pr = (a_pelelm->m_fixed_Pr != 0);
  bool do_soret = (a_pelelm->m_use_soret != 0);
  amrex::FArrayBox dummies(bx, 2 * NUM_SPECIES + 1, amrex::The_Async_Arena());
  auto const& rhoY = statefab.const_array(FIRSTSPEC);
  auto const& T = statefab.array(TEMP);
  auto rhoD = dummies.array(1);
  auto lambda = derfab.array(dcomp);
  auto mu = dummies.array(0);
  auto rhotheta = dummies.array(NUM_SPECIES + 1);
  auto const* ltransparm = a_pelelm->trans_parms.device_parm();
  auto const* leosparm = a_pelelm->eos_parms.device_parm();
  amrex::Real LeInv = a_pelelm->m_Lewis_inv;
  amrex::Real PrInv = a_pelelm->m_Prandtl_inv;
  amrex::ParallelFor(
    bx, [do_fixed_Le, do_fixed_Pr, do_soret, LeInv, PrInv, rhoY, T, rhoD,
         rhotheta, lambda, mu, ltransparm,
         leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      getTransportCoeff<pele::physics::PhysicsType::eos_type>(
        i, j, k, do_fixed_Le, do_fixed_Pr, do_soret, LeInv, PrInv, rhoY, T,
        rhoD, rhotheta, lambda, mu, ltransparm, leosparm);
    });
}

//
// Extract Distribution Mapping
//
void
pelelmex_derdmap(
  PeleLM* /*a_pelelm*/,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int /*ncomp*/,
  const amrex::FArrayBox& /*statefab*/,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  AMREX_ASSERT(derfab.box().contains(bx));
  auto der = derfab.array(dcomp);
  const int myrank = amrex::ParallelDescriptor::MyProc();
  amrex::ParallelFor(
    bx, [der, myrank] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      der(i, j, k) = myrank;
    });
}

//
// Turbulent forcing term
//
void
pelelmex_derturbforcing(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& geom,
  amrex::Real time,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);

  // Need geom for forcing
  amrex::GeometryData const& geomdata = geom.data();
  amrex::Array4<amrex::Real> const& der = derfab.array(dcomp);

  // Set derfab to zero first
  derfab.setVal<amrex::RunOn::Device>(0.0, bx, dcomp, ncomp);
  amrex::FArrayBox DummyFab(bx, 1);

  // Declare a pointer for the density array view
  amrex::Array4<const amrex::Real> rho = (a_pelelm->m_incompressible != 0)
                                           ? DummyFab.const_array()
                                           : statefab.const_array(DENSITY);

  // call the function above to construct the forcing
  a_pelelm->turb_forcing.addTurbVelForces(
    geomdata, bx, time, der, rho, a_pelelm->m_incompressible, a_pelelm->m_rho);
}

//
// Derive manifold output quantities
//
#ifdef USE_MANIFOLD_EOS
void
pelelmex_dermaniout(
  PeleLM* a_pelelm,
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const amrex::FArrayBox& statefab,
  const amrex::FArrayBox& /*reactfab*/,
  const amrex::FArrayBox& /*pressfab*/,
  const amrex::Geometry& /*geom*/,
  amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcrec*/,
  int /*level*/)
{
  auto& h_manf_data =
    a_pelelm->eos_parms.host_only_parm().manfunc_par->host_parm();
  auto* d_manf_data =
    a_pelelm->eos_parms.host_only_parm().manfunc_par->device_parm();
  int nmanivar = h_manf_data.Nvar;

  amrex::ignore_unused(ncomp);
  AMREX_ASSERT(derfab.box().contains(bx));
  AMREX_ASSERT(statefab.box().contains(bx));
  AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
  AMREX_ASSERT(statefab.nComp() >= NUM_SPECIES + 1);
  AMREX_ASSERT(ncomp == nmanivar);
  AMREX_ASSERT(!a_pelelm->m_incompressible);

  auto const in_spec = statefab.array(FIRSTSPEC);
  auto der = derfab.array(dcomp);
  amrex::ParallelFor(
    bx, [in_spec, der, d_manf_data,
         nmanivar] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real rho, rhoinv, maniparm[NUM_SPECIES];
      pele::physics::PhysicsType::eos_type::RY2RRinvY(
        in_spec.cellData(i, j, k), rho, rhoinv, maniparm);
      pele::physics::BlackBoxFunctionFactory<
        pele::physics::eos::ManifoldFunctionType>
        manfunc{d_manf_data};

      // TODO: use get_all_values instead
      for (int n = 0; n < nmanivar; ++n) {
        manfunc.get_func()->get_value(n, maniparm, der(i, j, k, n));
      }
    });
}
#endif
