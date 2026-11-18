// How the interface meets complex geometry (the contact angle at an embedded
// boundary) is not defined yet
#ifdef AMREX_USE_EB
#error "The VOF solver does not support embedded boundaries"
#endif

#include <PeleLMeX.H>
#include <PeleLMeX_BCfill.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_VOF_SurfaceTension_K.H>
#include <hydro_plic.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <limits>
#include <string>

namespace {
// PLIC normal: height functions where they are defined, Mixed Youngs-Centered
// elsewhere (AMReX-Hydro PLIC)
const std::string vof_reconstruction{"myc_height"};
} // namespace

// Liquid volume fraction BCs.  Components are Interior, Inflow, Outflow,
// Symmetry, SlipWallAdiab, NoSlipWallAdiab, SlipWallIsoTherm,
// NoSlipWallIsoTherm.  Wall and outflow ghost cells are first extrapolated,
// then overwritten by PLIC::FillWallGhosts, which extends the interior
// interface through the face (a free contact line at a wall).
constexpr int vof_bc[] = {amrex::BCType::int_dir,  amrex::BCType::ext_dir,
                          amrex::BCType::foextrap, amrex::BCType::reflect_even,
                          amrex::BCType::foextrap, amrex::BCType::foextrap,
                          amrex::BCType::foextrap, amrex::BCType::foextrap};

void
PeleLM::VOFReadParameters()
{
  // GMRES by default for the face coefficient jump of the projections.  Set
  // before the MacProjector reads its options.
  int mac_use_gmres = 1;
  amrex::ParmParse("mac_proj").queryAdd("use_gmres", mac_use_gmres);
  amrex::ParmParse pp("peleLM.vof");
  pp.query("cfl", m_vof_cfl);
  pp.query("refine_band", m_vof_refine_band);
  pp.query("v", m_vof_verbose);

  // Required: peleLM.rho / peleLM.mu are rejected with the VOF solver
  pp.get("rho_liquid", m_vof_rho_liq);
  pp.get("rho_gas", m_vof_rho_gas);
  pp.get("mu_liquid", m_vof_mu_liq);
  pp.get("mu_gas", m_vof_mu_gas);
  std::string mu_avg = "arithmetic";
  pp.query("viscosity_average", mu_avg);
  if (mu_avg == "arithmetic") {
    m_vof_mu_harmonic = 0;
  } else if (mu_avg == "harmonic") {
    m_vof_mu_harmonic = 1;
  } else {
    amrex::Abort(
      "peleLM.vof.viscosity_average must be 'arithmetic' or 'harmonic'");
  }
  if (
    m_vof_rho_liq <= 0.0 || m_vof_rho_gas <= 0.0 || m_vof_mu_liq <= 0.0 ||
    m_vof_mu_gas <= 0.0) {
    amrex::Abort("peleLM.vof.{rho,mu}_{liquid,gas} must be positive");
  }
  pp.query("sigma", m_vof_sigma);
  if (m_vof_sigma < 0.0) {
    amrex::Abort("peleLM.vof.sigma must be non-negative");
  }
  // Outflow band, see VOFOutflowInnerBox
  pp.query("outflow_band", m_vof_outflow_band);
  if (m_vof_outflow_band < 0) {
    amrex::Abort("peleLM.vof.outflow_band must be non-negative");
  }
  // The sharp face density balances gravity exactly at any interface
  // orientation, but on the faces where the surface tension acts it can be
  // close to rho_g, which amplifies the curvature errors by up to
  // rho_l / rho_g: the arithmetic average is the default with surface
  // tension
  std::string face_density = (m_vof_sigma > 0.0) ? "arithmetic" : "sharp";
  pp.query("face_density", face_density);
  if (face_density == "sharp") {
    m_vof_sharp_face_density = 1;
  } else if (face_density == "arithmetic") {
    m_vof_sharp_face_density = 0;
  } else {
    amrex::Abort("peleLM.vof.face_density must be 'sharp' or 'arithmetic'");
  }
  if (m_vof_verbose > 0) {
    amrex::Print() << " VOF: surface tension " << m_vof_sigma
                   << ", projection face density " << face_density
                   << ", outflow band " << m_vof_outflow_band << " cells\n";
  }

  if (m_vof_refine_band > m_nGrowState) {
    amrex::Abort(
      "peleLM.vof.refine_band cannot exceed the number of state ghost cells (" +
      std::to_string(m_nGrowState) + ")");
  }
  if (m_vof_cfl <= 0.0 || m_vof_cfl >= 1.0) {
    amrex::Abort("peleLM.vof.cfl must be in (0, 1): the advection is explicit");
  }
}

void
PeleLM::VOFCheckSetup()
{
  if (m_incompressible == 0) {
    amrex::Abort(
      "The VOF solver currently requires peleLM.incompressible = 1. "
      "Low-Mach two-phase coupling is not implemented yet.");
  }
  if (m_mesh_mapping) {
    amrex::Abort(
      "The VOF solver does not support geometry.mesh_mapping: the PLIC "
      "reconstruction assumes uniform cubic cells.");
  }
  if (PLIC::nGrowF(vof_reconstruction) > m_nGrowVOF) {
    amrex::Abort(
      "PeleLM::m_nGrowVOF is smaller than the ghost cells the PLIC "
      "reconstruction reads");
  }
}

void
PeleLM::setVOFBoundaryConditions()
{
  const int* lo_bc = m_phys_bc.lo();
  const int* hi_bc = m_phys_bc.hi();
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    m_bcrec_vof.setLo(idim, vof_bc[lo_bc[idim]]);
    m_bcrec_vof.setHi(idim, vof_bc[hi_bc[idim]]);
  }
}

void
PeleLM::initLevelVOF(const int lev)
{
  auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
  const auto geomdata = geom[lev].data();
  ProbParm const* lprobparm = prob_parm_d;

  auto const& vof_ma = ldata_p->vof.arrays();
  amrex::ParallelFor(
    ldata_p->vof, [vof_ma, geomdata, lprobparm] AMREX_GPU_DEVICE(
                    int box_no, int i, int j, int k) noexcept {
      call_initvof<ProblemSpecificFunctions>(
        i, j, k, vof_ma[box_no], geomdata, *lprobparm);
      vof_ma[box_no](i, j, k) = amrex::Clamp(
        vof_ma[box_no](i, j, k), amrex::Real(0.0), amrex::Real(1.0));
    });
  amrex::Gpu::streamSynchronize();

  // The tagging of this level reads the ghost cells
  fillpatch_vof(lev, m_t_new[lev], m_leveldata_new[lev]->vof, m_nGrowVOF);
}

amrex::Real
PeleLM::VOFLiquidVolume(const TimeStamp a_time)
{
  amrex::Vector<const amrex::MultiFab*> vof(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    vof[lev] = &getLevelDataPtr(lev, a_time)->vof;
  }
  return MFSum(vof, 0);
}

// Liquid volume V, centroid x_c = <F x> / V, second central moments
// <F (x_d - x_c,d)^2> / V and the bounding box of the cells with F > 1/2,
// over the uncovered cells of all levels, and the interface area
// sum |grad F| dV (central differences).  With surface tension, also the
// mean curvature of the interface cells and the difference between the mean
// pressures of the pure liquid and pure gas cells.  Used by the droplet
// verification cases (centroid trajectory -> drag, moments -> oscillation
// frequency, extent and area -> deformation and breakup, pressure jump ->
// Laplace).
void
PeleLM::VOFLiquidStats(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::VOFLiquidStats()");
  // F, F x_d, F x_d^2, then interface cells and their curvature, pure
  // liquid cells and their pressure, pure gas cells and their pressure, and
  // |grad F|
  constexpr int nmom = 1 + 2 * AMREX_SPACEDIM;
  constexpr int ntot = nmom + 7;
  const bool has_st = m_vof_sigma > 0.0;
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> mom(finest_level + 1);
  amrex::Real lo[AMREX_SPACEDIM];
  amrex::Real hi[AMREX_SPACEDIM];
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    lo[d] = std::numeric_limits<amrex::Real>::max();
    hi[d] = std::numeric_limits<amrex::Real>::lowest();
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    // The level data hold F with all its ghost cells
    const auto& vof = getLevelDataPtr(lev, a_time)->vof;
    const auto geomdata = geom[lev].data();
    mom[lev] = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], ntot, 0, amrex::MFInfo(), Factory(lev));
    amrex::MultiFab kappa(
      grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
    kappa.setVal(0.0);
    if (has_st) {
      VOFCurvature(lev, vof, kappa);
    }
    auto const& vof_ma = vof.const_arrays();
    auto const& kappa_ma = kappa.const_arrays();
    auto const& p_ma =
      getLevelDataPtr(lev, AmrNewTime)->vof_press.const_arrays();
    auto const& mom_ma = mom[lev]->arrays();
    amrex::ParallelFor(
      *mom[lev],
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
          geomdata.ProbLo(0) + (i + 0.5) * geomdata.CellSize(0),
          geomdata.ProbLo(1) + (j + 0.5) * geomdata.CellSize(1),
          geomdata.ProbLo(2) + (k + 0.5) * geomdata.CellSize(2))};
        const auto& Fa = vof_ma[box_no];
        const amrex::Real f = Fa(i, j, k);
        mom_ma[box_no](i, j, k, 0) = f;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          mom_ma[box_no](i, j, k, 1 + d) = f * x[d];
          mom_ma[box_no](i, j, k, 1 + AMREX_SPACEDIM + d) = f * x[d] * x[d];
        }
        const amrex::Real mixed = PLIC::is_mixed(f) ? 1.0 : 0.0;
        const amrex::Real liq = (f >= 1.0 - PLIC::pure_tol) ? 1.0 : 0.0;
        const amrex::Real gas = (f <= PLIC::pure_tol) ? 1.0 : 0.0;
        const amrex::Real p = p_ma[box_no](i, j, k);
        mom_ma[box_no](i, j, k, nmom) = mixed;
        mom_ma[box_no](i, j, k, nmom + 1) = mixed * kappa_ma[box_no](i, j, k);
        mom_ma[box_no](i, j, k, nmom + 2) = liq;
        mom_ma[box_no](i, j, k, nmom + 3) = liq * p;
        mom_ma[box_no](i, j, k, nmom + 4) = gas;
        mom_ma[box_no](i, j, k, nmom + 5) = gas * p;
        amrex::Real g2 = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          const amrex::IntVect e = amrex::IntVect::TheDimensionVector(d);
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const amrex::Real g =
            0.5 * (Fa(iv + e) - Fa(iv - e)) / geomdata.CellSize(d);
          g2 += g * g;
        }
        mom_ma[box_no](i, j, k, nmom + 6) = std::sqrt(g2);
      });
    // Bounding box of the liquid cells not covered by a finer level
    const int masked = (lev < finest_level) ? 1 : 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      const amrex::Real dx = geom[lev].CellSize(d);
      const amrex::Real plo = geom[lev].ProbLo(d);
      amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax> reduce_op;
      amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
      using ReduceTuple = typename decltype(reduce_data)::Type;
      for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& F = vof.const_array(mfi);
        auto const& mask = masked != 0 ? m_coveredMask[lev]->const_array(mfi)
                                       : amrex::Array4<int const>{};
        reduce_op.eval(
          bx, reduce_data,
          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> ReduceTuple {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            const bool use =
              F(i, j, k) > 0.5 && (masked == 0 || mask(i, j, k) != 0);
            if (!use) {
              return {
                std::numeric_limits<amrex::Real>::max(),
                std::numeric_limits<amrex::Real>::lowest()};
            }
            return {plo + iv[d] * dx, plo + (iv[d] + 1) * dx};
          });
      }
      const auto r = reduce_data.value();
      lo[d] = amrex::min(lo[d], amrex::get<0>(r));
      hi[d] = amrex::max(hi[d], amrex::get<1>(r));
    }
  }
  amrex::ParallelDescriptor::ReduceRealMin(lo, AMREX_SPACEDIM);
  amrex::ParallelDescriptor::ReduceRealMax(hi, AMREX_SPACEDIM);

  amrex::Real sums[ntot];
  for (int n = 0; n < ntot; ++n) {
    sums[n] = MFSum(GetVecOfConstPtrs(mom), n);
  }
  const amrex::Real vol = amrex::max(sums[0], 1.0e-300);
  amrex::Print() << "   VOF: time " << getTime(0, a_time) << " centroid";
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    amrex::Print() << " " << sums[1 + d] / vol;
  }
  amrex::Print() << " moments";
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const amrex::Real xc = sums[1 + d] / vol;
    amrex::Print() << " " << sums[1 + AMREX_SPACEDIM + d] / vol - xc * xc;
  }
  amrex::Print() << " extent";
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    amrex::Print() << " " << lo[d] << " " << hi[d];
  }
  amrex::Print() << " area " << sums[nmom + 6];
  if (has_st) {
    const auto mean = [&](const int n) {
      return sums[n + 1] / amrex::max(sums[n], 1.0e-300);
    };
    amrex::Print() << " curvature " << mean(nmom) << " pressure_jump "
                   << mean(nmom + 2) - mean(nmom + 4);
  }
  amrex::Print() << "\n";
}

//-----------------------------------------------------------------------------
// Mixture properties.  The volume fraction is fillpatched at a_time (time
// interpolated between t^n and t^{n+1}, e.g. at the half time) and mapped
// in place to the property.
//-----------------------------------------------------------------------------
void
PeleLM::fillVOFMixture(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_mf,
  const int nGhost,
  const amrex::Real a_liq,
  const amrex::Real a_gas,
  const int a_harmonic)
{
  fillpatch_vof(lev, a_time, a_mf, nGhost);
  auto const& ma = a_mf.arrays();
  amrex::ParallelFor(
    a_mf, amrex::IntVect(nGhost),
    [ma, a_liq, a_gas,
     a_harmonic] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      ma[box_no](i, j, k) =
        PLIC::mixture(ma[box_no](i, j, k), a_liq, a_gas, a_harmonic != 0);
    });
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::fillVOFDensity(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_rho,
  const int nGhost)
{
  fillVOFMixture(lev, a_time, a_rho, nGhost, m_vof_rho_liq, m_vof_rho_gas, 0);
}

void
PeleLM::fillVOFViscosity(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_mu,
  const int nGhost)
{
  fillVOFMixture(
    lev, a_time, a_mu, nGhost, m_vof_mu_liq, m_vof_mu_gas, m_vof_mu_harmonic);
}

// State for the derived quantities: the incompressible state holds only the
// velocity, so the mixture density is appended in the DENSITY component, in
// all the ghost cells of a_state
std::unique_ptr<amrex::MultiFab>
PeleLM::VOFDeriveState(
  const int lev, const amrex::Real a_time, const amrex::MultiFab& a_state)
{
  static_assert(DENSITY == AMREX_SPACEDIM, "DENSITY follows the velocity");
  const int nGhost = a_state.nGrow();
  AMREX_ALWAYS_ASSERT(nGhost <= m_nGrowVOF);
  auto mf = std::make_unique<amrex::MultiFab>(
    grids[lev], dmap[lev], DENSITY + 1, nGhost, amrex::MFInfo(), Factory(lev));
  amrex::MultiFab::Copy(*mf, a_state, VELX, VELX, AMREX_SPACEDIM, nGhost);
  amrex::MultiFab rho(*mf, amrex::make_alias, DENSITY, 1);
  fillVOFDensity(lev, a_time, rho, nGhost);
  return mf;
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFFaceDensity(const int lev, const amrex::Real a_time)
{
  amrex::MultiFab rho(
    grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
  fillVOFDensity(lev, a_time, rho, 1);

  constexpr int doZeroVisc = 0;
  return getDiffusivity(lev, 0, 1, doZeroVisc, {m_bcrec_vof}, rho);
}

//-----------------------------------------------------------------------------
// The cells of the level outside the outflow band, the m_vof_outflow_band
// cells next to each outflow boundary.  A prescribed outflow pressure cannot
// carry an interface across the boundary: the liquid reaching the band is
// removed (absorbing outflow, VOFOutflowSink) and the surface tension is
// off there.
//-----------------------------------------------------------------------------
amrex::Box
PeleLM::VOFOutflowInnerBox(const int lev) const
{
  amrex::Box inner = geom[lev].Domain();
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (geom[lev].isPeriodic(d)) {
      continue;
    }
    if (m_phys_bc.lo(d) == amrex::PhysBCType::outflow) {
      inner.growLo(d, -m_vof_outflow_band);
    }
    if (m_phys_bc.hi(d) == amrex::PhysBCType::outflow) {
      inner.growHi(d, -m_vof_outflow_band);
    }
  }
  return inner;
}

//-----------------------------------------------------------------------------
// Absorbing outflow: remove the liquid in the outflow band of each level
// (VOFOutflowInnerBox) of the new F, before its average down and fill, and
// return the volume removed (peleLM.vof.v > 0, zero otherwise).
//-----------------------------------------------------------------------------
amrex::Real
PeleLM::VOFOutflowSink()
{
  bool has_band = false;
  for (int lev = 0; lev <= finest_level; ++lev) {
    has_band = has_band || (VOFOutflowInnerBox(lev) != geom[lev].Domain());
  }
  if (!has_band) {
    return 0.0;
  }
  const bool verbose = m_vof_verbose > 0;
  const amrex::Real vol_before = verbose ? VOFLiquidVolume(AmrNewTime) : 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box inner = VOFOutflowInnerBox(lev);
    auto const& vof_ma = m_leveldata_new[lev]->vof.arrays();
    amrex::ParallelFor(
      m_leveldata_new[lev]->vof,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        if (!inner.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
          vof_ma[box_no](i, j, k) = 0.0;
        }
      });
  }
  amrex::Gpu::streamSynchronize();
  return verbose ? vol_before - VOFLiquidVolume(AmrNewTime) : 0.0;
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFProjFaceDensity(const int lev, const TimeStamp a_time)
{
  if (m_vof_sharp_face_density != 0) {
    return getVOFSharpFaceDensity(lev, a_time);
  }
  // The volume fraction is linear in time between t^n and t^{n+1}
  const amrex::Real time = (a_time == AmrHalfTime)
                             ? 0.5 * (m_t_old[lev] + m_t_new[lev])
                             : getTime(lev, a_time);
  return getVOFFaceDensity(lev, time);
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFSharpFaceDensity(const int lev, const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::getVOFSharpFaceDensity()");
  // The level data hold F at t^n and t^{n+1} with all their ghost cells
  amrex::Vector<const amrex::MultiFab*> F;
  if (a_time == AmrHalfTime) {
    F = {&m_leveldata_old[lev]->vof, &m_leveldata_new[lev]->vof};
  } else {
    F = {&getLevelDataPtr(lev, a_time)->vof};
  }

  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rho_fc;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    rho_fc[idim].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
      dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
  }
  PLIC::SharpFaceDensity(
    GetArrOfPtrs(rho_fc), F, geom[lev], m_vof_rho_liq, m_vof_rho_gas,
    vof_reconstruction);
  return rho_fc;
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFFaceBeta(
  const int lev, const TimeStamp a_time, const amrex::Real a_numerator)
{
  auto beta = getVOFProjFaceDensity(lev, a_time);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    beta[idim].invert(a_numerator, 0);
  }
  return beta;
}

//-----------------------------------------------------------------------------
// Momentum advection consistent with the VOF mass transport (Rudman 1998):
// the mass and momentum that the flux volumes of the VOF step carry, from its
// cuts (advanceVOF) completed by PLIC::FarFieldMomentumFluxes, so that
//   rho' = rho - dt div(m_f),  (rho u)' = (rho u) - dt div(p_f)
// gives the mixture density of the new F, and a uniform velocity stays
// uniform.  Near the interface each piece of a flux volume carries the
// velocity at t^n of the cell it lies in; away from it u_f is the Godunov face
// state.  Coarse faces under fine faces take the average of the fine fluxes,
// so the composite update is conservative.
// The result is returned as the advection term
// AofS = (u** - u^n) / dt (updateVelocity adds dt AofS).
//-----------------------------------------------------------------------------
void
PeleLM::VOFMomentumAdvTerm(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_faces)
{
  BL_PROFILE("PeleLMeX::VOFMomentumAdvTerm()");

  const amrex::Real rho_l = m_vof_rho_liq;
  const amrex::Real rho_g = m_vof_rho_gas;
  const amrex::Real dt = m_dt;

  // The mass fluxes, and the momentum fluxes near the interface, come from the
  // cuts of the VOF step.  The other faces carry the Godunov face velocity.
  AMREX_ALWAYS_ASSERT(
    static_cast<int>(advData->vof_mflux.size()) == finest_level + 1);
  auto& mflux = advData->vof_mflux;
  auto& pflux = advData->vof_pflux;
  for (int lev = 0; lev <= finest_level; ++lev) {
    // F at t^n, as given to the VOF step
    const auto& vof = m_leveldata_old[lev]->vof;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
      PLIC::FarFieldMomentumFluxes(
        mfi.validbox(),
        AMREX_D_DECL(
          pflux[lev][0].array(mfi), pflux[lev][1].array(mfi),
          pflux[lev][2].array(mfi)),
        AMREX_D_DECL(
          mflux[lev][0].const_array(mfi), mflux[lev][1].const_array(mfi),
          mflux[lev][2].const_array(mfi)),
        vof.const_array(mfi),
        AMREX_D_DECL(
          a_faces[lev][0].const_array(mfi), a_faces[lev][1].const_array(mfi),
          a_faces[lev][2].const_array(mfi)));
    }
  }
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(mflux[lev]), GetArrOfPtrs(mflux[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
    amrex::average_down_faces(
      GetArrOfConstPtrs(pflux[lev]), GetArrOfPtrs(pflux[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
  }

  // u** = ((rho u)^n - dt div(p_f)) / (rho^n - dt div(m_f))
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto dxinv = geom[lev].InvCellSizeArray();
    auto const& vof_ma = m_leveldata_old[lev]->vof.const_arrays();
    auto const& velo_ma = m_leveldata_old[lev]->state.const_arrays();
    auto const& aofs_ma = advData->AofS[lev].arrays();
    AMREX_D_TERM(
      auto const& mx_ma = mflux[lev][0].const_arrays();
      , auto const& my_ma = mflux[lev][1].const_arrays();
      , auto const& mz_ma = mflux[lev][2].const_arrays(););
    AMREX_D_TERM(
      auto const& px_ma = pflux[lev][0].const_arrays();
      , auto const& py_ma = pflux[lev][1].const_arrays();
      , auto const& pz_ma = pflux[lev][2].const_arrays(););
    amrex::ParallelFor(
      advData->AofS[lev],
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::Real rho_n =
          PLIC::mixture(vof_ma[box_no](i, j, k), rho_l, rho_g);
        const amrex::Real rho_new =
          rho_n -
          dt *
            (AMREX_D_TERM(
              (mx_ma[box_no](i + 1, j, k) - mx_ma[box_no](i, j, k)) * dxinv[0],
              +(my_ma[box_no](i, j + 1, k) - my_ma[box_no](i, j, k)) * dxinv[1],
              +(mz_ma[box_no](i, j, k + 1) - mz_ma[box_no](i, j, k)) *
                dxinv[2]));
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
          const amrex::Real u_n = velo_ma[box_no](i, j, k, VELX + n);
          const amrex::Real mom_new =
            rho_n * u_n -
            dt *
              (AMREX_D_TERM(
                (px_ma[box_no](i + 1, j, k, n) - px_ma[box_no](i, j, k, n)) *
                  dxinv[0],
                +(py_ma[box_no](i, j + 1, k, n) - py_ma[box_no](i, j, k, n)) *
                  dxinv[1],
                +(pz_ma[box_no](i, j, k + 1, n) - pz_ma[box_no](i, j, k, n)) *
                  dxinv[2]));
          aofs_ma[box_no](i, j, k, VELX + n) = (mom_new / rho_new - u_n) / dt;
        }
      });
  }
  amrex::Gpu::streamSynchronize();
}

//-----------------------------------------------------------------------------
// Face projection umac -= beta grad phi with div(beta grad phi) = div(umac), by
// the MacProjector with its solver options (mac_proj.*).  GMRES preconditioned
// by MLMG is the VOF default, for the large face-coefficient jumps of a sharp,
// high density ratio interface.  With a_phi, the solution phi (the pressure
// when beta = dt / rho_f) is returned in it: the sum of the corrections of all
// the solves, each made from the divergence the previous ones left.
//-----------------------------------------------------------------------------
void
PeleLM::VOFFaceProjection(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_umac,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_beta,
  const amrex::Vector<amrex::MultiFab*>& a_phi)
{
  BL_PROFILE("PeleLMeX::VOFFaceProjection()");

  setupMacProjector(a_beta);
  macproj->setUMAC(a_umac);
  // Coarse faces under fine faces take the fine-face average, so that the
  // composite divergence is consistent across coarse-fine interfaces
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(a_umac[lev]), a_umac[lev - 1], refRatio(lev - 1),
      geom[lev - 1]);
  }
  // The solvers stop on their own estimate of the residual.  Convergence is
  // judged here on the divergence actually left in umac, including its mean,
  // solving again for what remains, with a warning if the target is not
  // reached: a wrong operator (a face with two coefficients) gives a converged
  // solve and a divergent umac.
  const auto max_divergence = [&]() {
    amrex::Real div = 0.0;
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab divu(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      amrex::computeDivergence(divu, GetArrOfConstPtrs(a_umac[lev]), geom[lev]);
      div = amrex::max(div, divu.norm0());
    }
    return div;
  };
  // The divergence of a face velocity cannot be resolved below the round-off
  // of its 2*dim face terms, ulp(|u|) / dx on the finest faces: below that the
  // target is met only by chance cancellations
  amrex::Real roundoff = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      roundoff = amrex::max(
        roundoff,
        a_umac[lev][idim]->norm0(0, 0, true) * geom[lev].InvCellSize(idim));
    }
  }
  amrex::ParallelDescriptor::ReduceRealMax(roundoff);
  roundoff *= 2 * AMREX_SPACEDIM * std::numeric_limits<amrex::Real>::epsilon();

  constexpr int max_solves = 10;
  amrex::Real div_left = max_divergence();
  const amrex::Real div_target =
    amrex::max(m_mac_mg_rtol * div_left, m_mac_mg_atol, roundoff);
  int nsolve = 0;
  for (auto* phi : a_phi) {
    phi->setVal(0.0);
  }
  while (div_left > div_target && nsolve < max_solves) {
    solveMacProjection();
    for (int lev = 0; lev < static_cast<int>(a_phi.size()); ++lev) {
      amrex::MultiFab::Add(*a_phi[lev], macproj->getPhi()[lev], 0, 0, 1, 0);
    }
    if (m_macproj_verbose > 0) {
      amrex::Print() << "   VOF face projection: " << macproj->getNumIters()
                     << " iterations\n";
    }
    div_left = max_divergence();
    ++nsolve;
  }
  if (div_left > div_target) {
    amrex::Print() << "   VOF: WARNING face projection left a divergence of "
                   << div_left << " after " << nsolve << " solves, target "
                   << div_target << "\n";
  } else if (m_macproj_verbose > 0 && nsolve > 1) {
    amrex::Print() << "   VOF face projection: " << nsolve
                   << " solves, divergence left " << div_left << "\n";
  }
}

//-----------------------------------------------------------------------------
// Balanced-force velocity projection (Francois et al., JCP 213, 2006).
// Gravity, surface tension and the pressure gradient act on the faces:
//   u_f = <u*>_f + dt g + dt f_f / rho_f,   div(dt / rho_f grad p) = div(u_f),
//   a_f = g + (f_f - grad_f p) / rho_f,
// with the face density rho_f of getVOFProjFaceDensity and the surface
// tension f_f = sigma kappa_f grad_f F of getVOFSurfaceTension, so that a
// fluid at rest with a planar interface normal to g (sharp face density), or
// a drop of uniform curvature, gives a_f = 0 to solver tolerance.  The
// cell-centered velocity is updated with the average of the face
// accelerations of each cell, and gp holds g - <a_f>, i.e. (grad p - f) / rho,
// used as the lagged pressure-gradient forcing of the next step.  On entry the
// new velocity contains the lagged forcing dt (g - gp^{n-1/2}), which is
// removed first.  The pressure is kept in vof_press for the plotfiles.
//-----------------------------------------------------------------------------
void
PeleLM::VOFVelocityProjection(const int a_initPress, const amrex::Real a_dt)
{
  BL_PROFILE("PeleLMeX::VOFVelocityProjection()");

  const auto grav = m_gravity;

  // Remove the lagged pressure-gradient / gravity acceleration
  if (a_initPress == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
      auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
      auto const& vel_ma = ldataNew_p->state.arrays();
      auto const& gp_ma = ldataOld_p->gp.const_arrays();
      amrex::ParallelFor(
        ldataNew_p->state, amrex::IntVect(0), AMREX_SPACEDIM,
        [=, dt = a_dt] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          vel_ma[box_no](i, j, k, VELX + n) -=
            dt * (grav[n] - gp_ma[box_no](i, j, k, n));
        });
    }
    amrex::Gpu::streamSynchronize();
    averageDownVelocity(AmrNewTime);
    fillPatchState(AmrNewTime);
  }

  // Face velocities before (ub) and after (umac) adding gravity
  const auto bcRecVel = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> ub(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> umac(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> beta(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box& domain = geom[lev].Domain();
    auto const& vel_ma = getLevelDataPtr(lev, AmrNewTime)->state.const_arrays();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const amrex::BoxArray fba =
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim));
      ub[lev][idim].define(fba, dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      umac[lev][idim].define(
        fba, dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      if (a_initPress != 0) {
        ub[lev][idim].setVal(0.0);
        continue;
      }
      // Dirichlet faces take the boundary velocity held in the ghost cell
      const bool periodic = geom[lev].isPeriodic(idim);
      const bool ext_lo =
        !periodic && bcRecVel[idim].lo(idim) == amrex::BCType::ext_dir;
      const bool ext_hi =
        !periodic && bcRecVel[idim].hi(idim) == amrex::BCType::ext_dir;
      const int dlo = domain.smallEnd(idim);
      const int dhi = domain.bigEnd(idim) + 1;
      auto const& ub_ma = ub[lev][idim].arrays();
      amrex::ParallelFor(
        ub[lev][idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const bool on_lo = ext_lo && iv[idim] == dlo;
          const bool on_hi = ext_hi && iv[idim] == dhi;
          cen2edg_cpp(
            i, j, k, idim, 1, false, on_lo, on_hi,
            amrex::Array4<amrex::Real const>(vel_ma[box_no], VELX + idim),
            ub_ma[box_no]);
        });
    }
  }
  amrex::Gpu::streamSynchronize();
  // Coarse faces under / at the boundary of fine grids see the fine faces,
  // as the projected umac does
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(ub[lev]), GetArrOfPtrs(ub[lev - 1]), refRatio(lev - 1),
      geom[lev - 1]);
  }

  const TimeStamp rhoTime = (a_initPress != 0) ? AmrNewTime : AmrHalfTime;
  const bool has_st = m_vof_sigma > 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box& domain = geom[lev].Domain();
    beta[lev] = getVOFFaceBeta(lev, rhoTime, a_dt);
    // Surface tension from the volume fraction after the advection
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> st;
    if (has_st) {
      st = getVOFSurfaceTension(lev, getTime(lev, AmrNewTime));
    }
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      // Gravity acts on all faces except those where the velocity is
      // prescribed (walls, inflow)
      const bool periodic = geom[lev].isPeriodic(idim);
      const bool fixed_lo =
        !periodic && m_phys_bc.lo(idim) != amrex::PhysBCType::outflow;
      const bool fixed_hi =
        !periodic && m_phys_bc.hi(idim) != amrex::PhysBCType::outflow;
      const int dlo = domain.smallEnd(idim);
      const int dhi = domain.bigEnd(idim) + 1;
      const amrex::Real dtg = a_dt * grav[idim];
      auto const& ub_ma = ub[lev][idim].const_arrays();
      auto const& um_ma = umac[lev][idim].arrays();
      auto const& beta_ma = beta[lev][idim].const_arrays();
      // beta = dt / rho_f: the surface tension increment is beta f_f
      auto const& st_ma = has_st ? st[idim].const_arrays() : beta_ma;
      amrex::ParallelFor(
        umac[lev][idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const bool fixed =
            (fixed_lo && iv[idim] == dlo) || (fixed_hi && iv[idim] == dhi);
          amrex::Real du = dtg;
          if (has_st) {
            du += beta_ma[box_no](i, j, k) * st_ma[box_no](i, j, k);
          }
          um_ma[box_no](i, j, k) = ub_ma[box_no](i, j, k) + (fixed ? 0.0 : du);
        });
    }
  }
  amrex::Gpu::streamSynchronize();

  amrex::Vector<amrex::MultiFab*> press(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    press[lev] = &(getLevelDataPtr(lev, AmrNewTime)->vof_press);
  }
  VOFFaceProjection(GetVecOfArrOfPtrs(umac), beta, press);

  // Cell acceleration = average of the face accelerations
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      amrex::MultiFab::Subtract(umac[lev][idim], ub[lev][idim], 0, 0, 1, 0);
    }
    amrex::MultiFab dvel(
      grids[lev], dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
    amrex::average_face_to_cellcenter(dvel, 0, GetArrOfConstPtrs(umac[lev]));
    if (a_initPress == 0) {
      amrex::MultiFab::Add(ldataNew_p->state, dvel, 0, VELX, AMREX_SPACEDIM, 0);
    }
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      ldataNew_p->gp.setVal(grav[n], n, 1, 0);
    }
    amrex::MultiFab::Saxpy(
      ldataNew_p->gp, -1.0 / a_dt, dvel, 0, 0, AMREX_SPACEDIM, 0);
  }

  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down(
      getLevelDataPtr(lev, AmrNewTime)->gp,
      getLevelDataPtr(lev - 1, AmrNewTime)->gp, 0, AMREX_SPACEDIM,
      refRatio(lev - 1));
  }
  if (a_initPress == 0) {
    averageDownVelocity(AmrNewTime);
  }
}

//-----------------------------------------------------------------------------
// Interface curvature in the cells within one cell of the interface
// (PLIC::interface_band), in the valid and first ghost cells of a_kappa.
// a_vof must hold m_nGrowVOF filled ghost cells.  In order of preference:
//  1. the height-function curvature (cell_height_curvature), second-order
//     and also defined in the pure cells next to the interface, where the
//     columns hold the same heights up to a constant;
//  2. the average of the height-function curvatures of the 3^dim neighbors
//     that have one (Popinet, JCP 228, 2009), then of the 5^dim neighbors:
//     in 3D the 3x3x7 columns of the cells near the grid diagonals often
//     extend out of the 7-cell window, and a local fit there is accurate
//     but follows small interface perturbations, which drives spurious
//     currents unstable;
//  3. a least-squares parabola / paraboloid through the height and PLIC
//     interface points of the 3^dim stencil (parabolic_curvature), in the
//     valid cells;
//  4. -div(n) from the volume fraction gradient (thin, under-resolved
//     structures, e.g. ligaments about to pinch off).
// The ghost cells shared with other boxes of the level are then overwritten
// by the valid values, so that every face sees the same curvature from
// either side.  Zero away from the interface.
//-----------------------------------------------------------------------------
void
PeleLM::VOFCurvature(
  const int lev, const amrex::MultiFab& a_vof, amrex::MultiFab& a_kappa)
{
  BL_PROFILE("PeleLMeX::VOFCurvature()");
  // Height columns of the first ghost layer, and parabolic fits of the valid
  // cells, read PLIC::hf_half + 1 cells away
  static_assert(m_nGrowVOF >= PLIC::hf_half + 1);
  AMREX_ALWAYS_ASSERT(a_vof.nGrow() >= m_nGrowVOF && a_kappa.nGrow() >= 1);
  const amrex::Real dxinv = geom[lev].InvCellSize(0);

  // Height-function curvature (component 0) and whether it is defined (1),
  // in the valid and first ghost cells, exchanged out to two ghost cells
  // for the neighbor averages
  amrex::MultiFab khf(a_kappa.boxArray(), a_kappa.DistributionMap(), 2, 2);
  khf.setVal(0.0);
  auto const& F_ma = a_vof.const_arrays();
  auto const& khf_ma = khf.arrays();
  amrex::ParallelFor(
    khf, amrex::IntVect(1),
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      const auto& F = F_ma[box_no];
      amrex::Real kap = 0.0;
      amrex::Real valid = 0.0;
      if (
        PLIC::interface_band(i, j, k, F) &&
        pelelmex::vof::cell_height_curvature(i, j, k, F, dxinv, kap)) {
        valid = 1.0;
      }
      khf_ma[box_no](i, j, k, 0) = valid > 0.5 ? kap : 0.0;
      khf_ma[box_no](i, j, k, 1) = valid;
    });
  amrex::Gpu::streamSynchronize();
  khf.FillBoundary(geom[lev].periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_kappa, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& gbx = mfi.growntilebox(1);
    const amrex::Box& vbx = mfi.validbox();
    auto const& F = a_vof.const_array(mfi);
    auto const& K = khf.const_array(mfi);
    auto const& kappa = a_kappa.array(mfi);
    amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real kap = 0.0;
      if (!PLIC::interface_band(i, j, k, F)) {
        kappa(i, j, k) = kap;
        return;
      }
      if (K(i, j, k, 1) > 0.5) {
        kappa(i, j, k) = K(i, j, k, 0);
        return;
      }
      // Average of the height-function curvatures within r cells
      const auto neighbor_average = [&](const int r, amrex::Real& avg) {
        amrex::Real sum = 0.0;
        int n = 0;
#if AMREX_SPACEDIM == 3
        for (int kk = -r; kk <= r; ++kk) {
#else
        {
          constexpr int kk = 0;
#endif
          for (int jj = -r; jj <= r; ++jj) {
            for (int ii = -r; ii <= r; ++ii) {
              if (
                K.contains(i + ii, j + jj, k + kk) &&
                K(i + ii, j + jj, k + kk, 1) > 0.5) {
                sum += K(i + ii, j + jj, k + kk, 0);
                ++n;
              }
            }
          }
        }
        if (n == 0) {
          return false;
        }
        avg = sum / static_cast<amrex::Real>(n);
        return true;
      };
      if (
        !neighbor_average(1, kap) && !neighbor_average(2, kap) &&
        (!vbx.contains(i, j, k) ||
         !pelelmex::vof::parabolic_curvature(i, j, k, F, dxinv, kap))) {
        // The fit reads planes in the 3^dim stencil: valid cells only
        kap = pelelmex::vof::normal_divergence_curvature(i, j, k, F, dxinv);
      }
      kappa(i, j, k) = kap;
    });
  }
  amrex::Gpu::streamSynchronize();
  a_kappa.FillBoundary(0, 1, amrex::IntVect(1), geom[lev].periodicity());
}

//-----------------------------------------------------------------------------
// Continuum surface force (Brackbill et al., JCP 100, 1992) in the
// balanced-force form of Francois et al. (JCP 213, 2006): on each face,
//   f_f = sigma kappa_f (F_i - F_{i-1}) / dx,
// the same two-point gradient as the face pressure gradient of the
// projection, so that the pressure jump sigma kappa balances it exactly
// for a uniform curvature.  kappa_f is taken from the interface (mixed)
// cells on either side: their average when both are mixed, the mixed one's
// otherwise (Popinet, JCP 228, 2009).  The pure cells next to an oblique
// interface often have no height function (in 3D, most of them) and only a
// fallback curvature, used when neither cell is mixed (interface on the
// face).  Zero on the faces of the cells within m_vof_outflow_band cells of
// an outflow boundary.
//-----------------------------------------------------------------------------
amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFSurfaceTension(const int lev, const amrex::Real a_time)
{
  BL_PROFILE("PeleLMeX::getVOFSurfaceTension()");
  amrex::MultiFab F(
    grids[lev], dmap[lev], 1, m_nGrowVOF, amrex::MFInfo(), Factory(lev));
  fillpatch_vof(lev, a_time, F, m_nGrowVOF);
  amrex::MultiFab kappa(
    grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
  VOFCurvature(lev, F, kappa);

  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> st;
  auto const& F_ma = F.const_arrays();
  auto const& kappa_ma = kappa.const_arrays();
  // Cells with surface tension
  const amrex::Box inner = VOFOutflowInnerBox(lev);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    st[idim].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
      dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    const amrex::Real sdxinv = m_vof_sigma * geom[lev].InvCellSize(idim);
    auto const& st_ma = st[idim].arrays();
    amrex::ParallelFor(
      st[idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        const amrex::IntVect ivm =
          iv - amrex::IntVect::TheDimensionVector(idim);
        // The two cells of the face
        const amrex::Box fcells(ivm, iv);
        if (!inner.contains(fcells)) {
          st_ma[box_no](i, j, k) = 0.0;
          return;
        }
        const auto& Fa = F_ma[box_no];
        const auto& K = kappa_ma[box_no];
        const bool mm = PLIC::is_mixed(Fa(ivm));
        const bool mp = PLIC::is_mixed(Fa(iv));
        const amrex::Real kf = (mm == mp) ? 0.5 * (K(ivm) + K(iv))
                               : mm       ? K(ivm)
                                          : K(iv);
        st_ma[box_no](i, j, k) = sdxinv * kf * (Fa(iv) - Fa(ivm));
      });
  }
  amrex::Gpu::streamSynchronize();
  return st;
}

//-----------------------------------------------------------------------------
// Advection: geometric unsplit fluxes (Owkes & Desjardins, JCP 270, 2014) with
// PLIC reconstruction.  All levels advance with the same dt; fine face fluxes
// are averaged onto the coarse faces before the update, so the composite update
// conserves volume across coarse/fine boundaries.
//-----------------------------------------------------------------------------
void
PeleLM::advanceVOF(std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::advanceVOF()");

  const amrex::Real vol_old =
    (m_vof_verbose > 0) ? VOFLiquidVolume(AmrNewTime) : 0.0;

  // Working state is the new VOF, which holds t^n and its ghost cells at this
  // point

  // Largest face velocity / dx.  The step is estimated from the
  // cell-centered velocity at t^n, while the advection uses the face velocity
  // at t^{n+1/2}, which can be larger
  const auto umac = GetVecOfArrOfConstPtrs(advData->umac);
  amrex::Real umac_rate = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      umac_rate = amrex::max(
        umac_rate,
        umac[lev][idim]->norm0(0, 0, true) * geom[lev].InvCellSize(idim));
    }
  }
  amrex::ParallelDescriptor::ReduceRealMax(umac_rate);

  // The advection is explicit
  if (umac_rate * m_dt >= 1.0) {
    amrex::Abort(
      "VOF: the advection requires a face Courant number below 1: reduce the "
      "time step.");
  }

  // Liquid crossing each face per unit area and time, F^{n+1} = F^n - dt
  // div(flux).  Faces shared by boxes get the same flux to the bit.  F is not
  // clipped or limited: it conserves the liquid volume, and departs from [0,1]
  // by the errors of the fluxes.  The same cuts give the mass and momentum
  // fluxes of the momentum advection (VOFMomentumAdvTerm).
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> flux(
    finest_level + 1);
  advData->vof_mflux.clear();
  advData->vof_pflux.clear();
  advData->vof_mflux.resize(finest_level + 1);
  advData->vof_pflux.resize(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto& vof = m_leveldata_new[lev]->vof;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> edge;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::BoxArray fba =
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(dir));
      flux[lev][dir].define(fba, dmap[lev], 1, 0);
      edge[dir].define(fba, dmap[lev], 1, 0);
      advData->vof_mflux[lev][dir].define(fba, dmap[lev], 1, 0);
      advData->vof_pflux[lev][dir].define(fba, dmap[lev], AMREX_SPACEDIM, 0);
    }
    // The velocity at t^n, for the momentum carried by the cut flux volumes
    const auto& state = m_leveldata_old[lev]->state;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
      auto& mf = advData->vof_mflux[lev];
      auto& pf = advData->vof_pflux[lev];
      PLIC::ComputeEdgeStateAndFluxes(
        mfi.validbox(),
        AMREX_D_DECL(
          flux[lev][0].array(mfi), flux[lev][1].array(mfi),
          flux[lev][2].array(mfi)),
        AMREX_D_DECL(
          edge[0].array(mfi), edge[1].array(mfi), edge[2].array(mfi)),
        AMREX_D_DECL(mf[0].array(mfi), mf[1].array(mfi), mf[2].array(mfi)),
        AMREX_D_DECL(pf[0].array(mfi), pf[1].array(mfi), pf[2].array(mfi)),
        vof.const_array(mfi),
        amrex::Array4<amrex::Real const>(state.const_array(mfi), VELX),
        AMREX_D_DECL(
          umac[lev][0]->const_array(mfi), umac[lev][1]->const_array(mfi),
          umac[lev][2]->const_array(mfi)),
        geom[lev], m_dt, m_vof_rho_liq, m_vof_rho_gas, vof_reconstruction);
    }
  }
  // The flux volumes do not tile across coarse/fine boundaries, which the
  // tagging keeps in pure cells
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(flux[lev]), GetArrOfPtrs(flux[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& F_ma = m_leveldata_new[lev]->vof.arrays();
    const auto dxinv = geom[lev].InvCellSizeArray();
    const amrex::Real dt = m_dt;
    AMREX_D_TERM(
      auto const& fx_ma = flux[lev][0].const_arrays();
      , auto const& fy_ma = flux[lev][1].const_arrays();
      , auto const& fz_ma = flux[lev][2].const_arrays(););
    amrex::ParallelFor(
      m_leveldata_new[lev]->vof,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        F_ma[box_no](i, j, k) -=
          dt *
          (AMREX_D_TERM(
            (fx_ma[box_no](i + 1, j, k) - fx_ma[box_no](i, j, k)) * dxinv[0],
            +(fy_ma[box_no](i, j + 1, k) - fy_ma[box_no](i, j, k)) * dxinv[1],
            +(fz_ma[box_no](i, j, k + 1) - fz_ma[box_no](i, j, k)) * dxinv[2]));
      });
  }
  amrex::Gpu::streamSynchronize();

  const amrex::Real vol_sink = VOFOutflowSink();

  averageDownVOF(AmrNewTime);
  fillPatchVOF(AmrNewTime);

  if (m_vof_verbose > 0) {
    // The relative change measures the conservation of the advection: it
    // leaves out the volume removed at the outflow
    const amrex::Real vol_new = VOFLiquidVolume(AmrNewTime);
    amrex::Print() << "   VOF: liquid volume " << vol_new
                   << ", relative change "
                   << (vol_new + vol_sink - vol_old) /
                        amrex::max(vol_old, 1.0e-300);
    if (vol_sink != 0.0) {
      amrex::Print() << ", removed at the outflow " << vol_sink;
    }
    amrex::Print() << "\n";
    // The discrete divergence of the face velocities, as a change of F in a
    // pure liquid cell over this step, and the departure of F from [0,1]
    amrex::Real div_max = 0.0;
    amrex::Real f_min = 1.0e200;
    amrex::Real f_max = -1.0e200;
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab divu(grids[lev], dmap[lev], 1, 0);
      amrex::computeDivergence(divu, umac[lev], geom[lev]);
      div_max = amrex::max(div_max, divu.norm0());
      f_min = amrex::min(f_min, m_leveldata_new[lev]->vof.min(0));
      f_max = amrex::max(f_max, m_leveldata_new[lev]->vof.max(0));
    }
    amrex::Print() << "   VOF: dt*max|div umac| " << div_max * m_dt
                   << ", F below 0 by " << amrex::max(0.0, -f_min)
                   << ", above 1 by " << amrex::max(0.0, f_max - 1.0) << "\n";
    VOFLiquidStats(AmrNewTime);
  }
}

amrex::Real
PeleLM::estVOFDt(const TimeStamp a_time)
{
  amrex::Real estdt = 1.0e200;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const auto dx = geom[lev].CellSizeArray();
    const amrex::Vector<amrex::Real> u_max =
      ldata_p->state.norm0({AMREX_D_DECL(VELX, VELY, VELZ)}, 0, true, true);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      if (u_max[idim] > 0.0) {
        estdt = amrex::min(estdt, m_vof_cfl * dx[idim] / u_max[idim]);
      }
    }
    // Capillary waves (Brackbill et al., JCP 100, 1992): explicit surface
    // tension is stable for dt < sqrt(rho_avg dx^3 / (2 pi sigma))
    if (m_vof_sigma > 0.0) {
      const amrex::Real rho_avg = 0.5 * (m_vof_rho_liq + m_vof_rho_gas);
      estdt = amrex::min(
        estdt, std::sqrt(
                 rho_avg * dx[0] * dx[0] * dx[0] /
                 (2.0 * amrex::Math::pi<amrex::Real>() * m_vof_sigma)));
    }
  }
  amrex::ParallelDescriptor::ReduceRealMin(estdt);
  return estdt;
}

void
PeleLM::VOFTagInterface(const int lev, amrex::TagBoxArray& tags)
{
  // The interface stays on the finest level: the flux volumes do not tile
  // across coarse/fine boundaries, which must stay in pure cells
  if (lev >= max_level) {
    return;
  }

  const int nband = m_vof_refine_band;
  amrex::MultiFab vof(grids[lev], dmap[lev], 1, nband);
  fillpatch_vof(lev, getTime(lev, AmrNewTime), vof, nband);

  auto const& tag_ma = tags.arrays();
  auto const& vof_ma = vof.const_arrays();
  amrex::ParallelFor(
    tags, [tag_ma, vof_ma,
           nband] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      if (PLIC::interface_band(i, j, k, vof_ma[box_no], nband)) {
        tag_ma[box_no](i, j, k) = amrex::TagBox::SET;
      }
    });
  amrex::Gpu::streamSynchronize();
}

//-----------------------------------------------------------------------------
// Fill patch / average down
//-----------------------------------------------------------------------------
namespace {

auto
makeVOFBndryFunc(
  const amrex::Geometry& a_geom,
  const amrex::Vector<amrex::BCRec>& a_bcrec,
  ProbParm const* a_prob_parm)
{
  using Fill = PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>;
  return amrex::PhysBCFunct<amrex::GpuBndryFuncFab<Fill>>(
    a_geom, a_bcrec, Fill{a_prob_parm});
}

} // namespace

void
PeleLM::fillPatchVOF(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchVOF()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::Real time = getTime(lev, a_time);
    fillpatch_vof(lev, time, ldata_p->vof, m_nGrowVOF);
  }
}

void
PeleLM::fillpatch_vof(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_vof,
  const int nGhost)
{
  BL_PROFILE("PeleLMeX::fillpatch_vof()");
  // The level data hold all their ghost cells at t_old and t_new, and this fill
  // is called many times in a step on the same F
  const auto* ldata = (a_time == m_t_new[lev])   ? m_leveldata_new[lev].get()
                      : (a_time == m_t_old[lev]) ? m_leveldata_old[lev].get()
                                                 : nullptr;
  if (
    ldata != nullptr && &a_vof != &m_leveldata_new[lev]->vof &&
    &a_vof != &m_leveldata_old[lev]->vof &&
    a_vof.boxArray() == ldata->vof.boxArray() &&
    a_vof.DistributionMap() == ldata->vof.DistributionMap()) {
    amrex::MultiFab::Copy(a_vof, ldata->vof, 0, 0, 1, nGhost);
    return;
  }
  // The wall ghost values read further than nGhost layers (VOFWallGhosts), so a
  // smaller fill would see zeros there and give different values for the same
  // ghost cell.  Fill all the layers and keep those requested.
  if (nGhost > 0 && nGhost < m_nGrowVOF) {
    amrex::MultiFab full(
      a_vof.boxArray(), a_vof.DistributionMap(), 1, m_nGrowVOF, amrex::MFInfo(),
      a_vof.Factory());
    fillpatch_vof(lev, a_time, full, m_nGrowVOF);
    amrex::MultiFab::Copy(a_vof, full, 0, 0, 1, nGhost);
    return;
  }
  ProbParm const* lprobparm = prob_parm_d;
  amrex::Vector<amrex::BCRec> bcrec{m_bcrec_vof};

  if (lev == 0) {
    auto bndry_func = makeVOFBndryFunc(geom[lev], bcrec, lprobparm);
    amrex::FillPatchSingleLevel(
      a_vof, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->vof), &(m_leveldata_new[lev]->vof)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev], bndry_func, 0);
  } else {
    // Limited conservative linear interpolation keeps F within [0,1]
    auto crse_bndry_func = makeVOFBndryFunc(geom[lev - 1], bcrec, lprobparm);
    auto fine_bndry_func = makeVOFBndryFunc(geom[lev], bcrec, lprobparm);
    amrex::FillPatchTwoLevels(
      a_vof, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->vof), &(m_leveldata_new[lev - 1]->vof)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->vof), &(m_leveldata_new[lev]->vof)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
      &amrex::cell_cons_interp, bcrec, 0);
  }

  a_vof.EnforcePeriodicity(geom[lev].periodicity());
  VOFWallGhosts(lev, a_vof, nGhost);
}

//-----------------------------------------------------------------------------
// Wall and outflow ghost cells from the PLIC planes of the interior interface,
// extended through the face: the interface meets a wall at whatever angle it
// has, as it should without surface tension.  Extrapolated ghost values impose
// 90 degrees instead, and the inconsistent near-wall normals unbalance gravity
// at a tilted interface.  The result does not depend on the box layout.
//-----------------------------------------------------------------------------
void
PeleLM::VOFWallGhosts(const int lev, amrex::MultiFab& a_vof, const int nGhost)
{
  if (nGhost == 0) {
    return;
  }
  // The extension reads all the ghost layers of a_vof
  AMREX_ALWAYS_ASSERT(a_vof.nGrow() == nGhost);
  PLIC::FillWallGhosts(a_vof, geom[lev], m_bcrec_vof, vof_reconstruction);
}

void
PeleLM::fillcoarsepatch_vof(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_vof,
  const int nGhost)
{
  AMREX_ASSERT(lev > 0);
  AMREX_ASSERT(nGhost == m_nGrowVOF);
  ProbParm const* lprobparm = prob_parm_d;
  amrex::Vector<amrex::BCRec> bcrec{m_bcrec_vof};

  auto crse_bndry_func = makeVOFBndryFunc(geom[lev - 1], bcrec, lprobparm);
  auto fine_bndry_func = makeVOFBndryFunc(geom[lev], bcrec, lprobparm);
  amrex::InterpFromCoarseLevel(
    a_vof, amrex::IntVect(nGhost), a_time, m_leveldata_new[lev - 1]->vof, 0, 0,
    1, geom[lev - 1], geom[lev], crse_bndry_func, 0, fine_bndry_func, 0,
    refRatio(lev - 1), &amrex::cell_cons_interp, bcrec, 0);
  VOFWallGhosts(lev, a_vof, nGhost);
}

void
PeleLM::averageDownVOF(const TimeStamp a_time)
{
  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataFine_p = getLevelDataPtr(lev, a_time);
    auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
    amrex::average_down(
      ldataFine_p->vof, ldataCrse_p->vof, 0, 1, refRatio(lev - 1));
  }
}
