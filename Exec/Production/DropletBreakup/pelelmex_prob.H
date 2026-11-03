#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PeleLMeX_VOF_K.H>
#include <PMFData.H>
#include <PelePhysics.H>

// Aerodynamic breakup of a drop by an impulsively started gas stream
// (two-phase VOF with surface tension, incompressible, no gravity).  A drop
// of diameter `diameter` centered at `center`, initially at rest, meets a
// uniform gas stream of velocity `velocity` along x, set as the Inflow value
// at x-lo.  The regime is set by the gas Weber number
// We = rho_g U^2 D / sigma and the Ohnesorge number
// Oh = mu_l / sqrt(rho_l sigma D); see README.md.

struct MyProbParm : public ProbParmDefault
{
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center = {
    AMREX_D_DECL(4.0, 0.0, 0.0)};
  amrex::Real diameter = 1.0;
  amrex::Real velocity = 1.0;
};

namespace breakup {

// > 0 inside the drop
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
phi(
  const amrex::Real x,
  const amrex::Real y,
  const amrex::Real z,
  MyProbParm const& pp)
{
  amrex::ignore_unused(z);
  const amrex::Real dx = x - pp.center[0];
  const amrex::Real dy = y - pp.center[1];
#if AMREX_SPACEDIM == 3
  const amrex::Real dz = z - pp.center[2];
#else
  constexpr amrex::Real dz = 0.0;
#endif
  return 0.5 * pp.diameter - std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace breakup

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  // Gas stream outside the drop; the initial projection makes it divergence
  // free (potential flow around the drop)
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata(
    int i,
    int j,
    int k,
    int /*is_incompressible*/,
    amrex::Array4<amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const auto pp = prob_parm;
    auto phi = [=] AMREX_GPU_DEVICE(
                 amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real {
      return breakup::phi(x, y, z, pp);
    };
    const amrex::Real F =
      pelelmex::vof::cell_fraction(i, j, k, geomdata, phi, 8);
    AMREX_D_TERM(state(i, j, k, VELX) = pp.velocity * (1.0 - F);
                 , state(i, j, k, VELY) = 0.0;, state(i, j, k, VELZ) = 0.0;)
  }

  // Inflow: gas stream along x; walls: zero normal velocity
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real* /*x*/,
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int idir,
    const int sgn,
    const amrex::Real /*time*/,
    amrex::GeometryData const& /*geomdata*/,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    AMREX_D_TERM(s_ext[VELX] = 0.0;, s_ext[VELY] = 0.0;, s_ext[VELZ] = 0.0;)
    if (idir == 0 && sgn == 1) {
      s_ext[VELX] = prob_parm.velocity;
    }
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initvof(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real> const& vof,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm)
  {
    const auto pp = prob_parm;
    auto phi = [=] AMREX_GPU_DEVICE(
                 amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real {
      return breakup::phi(x, y, z, pp);
    };
    // Sampling noise in F would be amplified by 1/dx in the curvature
    vof(i, j, k) = pelelmex::vof::smooth_cell_fraction(
      i, j, k, geomdata, phi, AMREX_SPACEDIM == 2 ? 8 : 4);
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;

#endif
