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

// Incompressible two-phase drops with surface tension (peleLM.vof.sigma),
// no gravity.  The drop of radius `radius` is centered at `center`:
//  - static:    drop at rest, the exact solution is u = 0 with the Laplace
//               pressure jump sigma kappa (kappa = 1/R in 2D, 2/R in 3D)
//  - oscillate: drop at rest with the mode-2 perturbation
//               r = R (1 + amplitude cos 2 theta) (2D) or
//               r = R (1 + amplitude P2(cos theta)) (3D), theta measured
//               from x; it oscillates at the Rayleigh / Lamb frequency
//  - breakup:   drop at rest in a uniform gas stream of velocity
//               `velocity` along x, set as the Inflow value at x-lo
//               (impulsive acceleration, Weber number
//               rho_g U^2 2R / sigma)

enum class DropletType { Static, Oscillate, Breakup };

struct MyProbParm : public ProbParmDefault
{
  DropletType type = DropletType::Static;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center = {
    AMREX_D_DECL(0.5, 0.5, 0.5)};
  amrex::Real radius = 0.25;
  amrex::Real amplitude = 0.0;
  amrex::Real velocity = 0.0;
};

namespace droplet {

// Signed distance-like function, > 0 inside the drop
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
  const amrex::Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
  amrex::Real R = pp.radius;
  if (pp.type == DropletType::Oscillate && r > 0.0) {
    const amrex::Real c = dx / r;
#if AMREX_SPACEDIM == 2
    // cos 2 theta
    R *= 1.0 + pp.amplitude * (2.0 * c * c - 1.0);
#else
    // P2(cos theta)
    R *= 1.0 + pp.amplitude * 0.5 * (3.0 * c * c - 1.0);
#endif
  }
  return R - r;
}

} // namespace droplet

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  // Gas stream outside the drop for the breakup case (the initial
  // projection makes it divergence free), rest otherwise
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
    AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) = 0.0;
                 , state(i, j, k, VELZ) = 0.0;)
    if (prob_parm.type == DropletType::Breakup) {
      const auto pp = prob_parm;
      auto phi = [=] AMREX_GPU_DEVICE(
                   amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real {
        return droplet::phi(x, y, z, pp);
      };
      const amrex::Real F =
        pelelmex::vof::cell_fraction(i, j, k, geomdata, phi, 8);
      state(i, j, k, VELX) = pp.velocity * (1.0 - F);
    }
  }

  // Inflow: gas stream along x; walls: zero velocity
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
      return droplet::phi(x, y, z, pp);
    };
    // Sampling noise in F would be amplified by 1/dx in the curvature
    vof(i, j, k) = pelelmex::vof::smooth_cell_fraction(
      i, j, k, geomdata, phi, AMREX_SPACEDIM == 2 ? 8 : 4);
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;

#endif
