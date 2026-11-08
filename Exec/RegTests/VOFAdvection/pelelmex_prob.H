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

// VOF advection verification cases on the unit square / cube:
//  - zalesak:   slotted disk (cylinder in 3D) in solid-body rotation about
//               the domain center, back to its initial position at
//               t = period (peleLM.vof.advect_only = 1)
//  - vortex:    circle in the time-reversed single vortex (2D) or sphere in
//               the LeVeque deformation field (3D), back to its initial
//               shape at t = period (peleLM.vof.advect_only = 1)
//  - translate: circle / sphere carried by a uniform flow through a periodic
//               domain; runs through the Navier-Stokes path
//               (peleLM.vof.advect_only = 0)

enum class VOFTestType { Zalesak, Vortex, Translate };

struct MyProbParm : public ProbParmDefault
{
  VOFTestType type = VOFTestType::Zalesak;
  amrex::Real radius = 0.15;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center = {
    AMREX_D_DECL(0.5, 0.75, 0.5)};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel = {
    AMREX_D_DECL(1.0, 1.0, 1.0)};
  amrex::Real slot_width = 0.05;
  amrex::Real slot_length = 0.25;
  amrex::Real period = 1.0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  // Vector potential A of the prescribed velocity u = curl(A); A[2] is the
  // stream function in 2D.  Also gives the initial velocity.
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void vof_vector_potential(
    const amrex::Real x[AMREX_SPACEDIM],
    const amrex::Real time,
    amrex::Real A[3],
    MyProbParm const& prob_parm)
  {
    constexpr amrex::Real pi = amrex::Math::pi<amrex::Real>();
    A[0] = A[1] = A[2] = 0.0;
    if (prob_parm.type == VOFTestType::Zalesak) {
      const amrex::Real omega = 2.0 * pi / prob_parm.period;
      A[2] = -0.5 * omega *
             ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5));
    } else if (prob_parm.type == VOFTestType::Vortex) {
      const amrex::Real ct = std::cos(pi * time / prob_parm.period);
      const amrex::Real sx2 = std::sin(pi * x[0]) * std::sin(pi * x[0]);
      const amrex::Real sy2 = std::sin(pi * x[1]) * std::sin(pi * x[1]);
#if AMREX_SPACEDIM == 2
      A[2] = sx2 * sy2 / pi * ct;
#else
      const amrex::Real sz2 = std::sin(pi * x[2]) * std::sin(pi * x[2]);
      A[1] = -sx2 * sz2 / pi * std::sin(2.0 * pi * x[1]) * ct;
      A[2] = sx2 * sy2 / pi * std::sin(2.0 * pi * x[2]) * ct;
#endif
    } else {
      auto const& U = prob_parm.vel;
#if AMREX_SPACEDIM == 2
      A[2] = U[0] * x[1] - U[1] * x[0];
#else
      // A = (U x X) / 2
      A[0] = 0.5 * (U[1] * x[2] - U[2] * x[1]);
      A[1] = 0.5 * (U[2] * x[0] - U[0] * x[2]);
      A[2] = 0.5 * (U[0] * x[1] - U[1] * x[0]);
#endif
    }
  }

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
    constexpr amrex::Real pi = amrex::Math::pi<amrex::Real>();
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* dx = geomdata.CellSize();
    AMREX_D_TERM(
      const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
      , const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
      , const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];)
    amrex::Real u[3] = {0.0, 0.0, 0.0};
    if (prob_parm.type == VOFTestType::Zalesak) {
      const amrex::Real omega = 2.0 * pi / prob_parm.period;
      u[0] = -omega * (y - 0.5);
      u[1] = omega * (x - 0.5);
    } else if (prob_parm.type == VOFTestType::Vortex) {
      const amrex::Real sx = std::sin(pi * x);
      const amrex::Real sy = std::sin(pi * y);
#if AMREX_SPACEDIM == 2
      u[0] = sx * sx * std::sin(2.0 * pi * y);
      u[1] = -std::sin(2.0 * pi * x) * sy * sy;
#else
      const amrex::Real sz = std::sin(pi * z);
      u[0] = 2.0 * sx * sx * std::sin(2.0 * pi * y) * std::sin(2.0 * pi * z);
      u[1] = -std::sin(2.0 * pi * x) * sy * sy * std::sin(2.0 * pi * z);
      u[2] = -std::sin(2.0 * pi * x) * std::sin(2.0 * pi * y) * sz * sz;
#endif
    } else {
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        u[d] = prob_parm.vel[d];
      }
    }
    AMREX_D_TERM(state(i, j, k, VELX) = u[0];, state(i, j, k, VELY) = u[1];
                 , state(i, j, k, VELZ) = u[2];)
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
    auto const c = prob_parm.center;
    const amrex::Real R = prob_parm.radius;
    const bool slotted = (prob_parm.type == VOFTestType::Zalesak);
    const amrex::Real hw = 0.5 * prob_parm.slot_width;
    const amrex::Real slot_top = c[1] - R + prob_parm.slot_length;
    // phi > 0 inside the liquid.  The Zalesak disk is extruded along z.
    auto phi = [=] AMREX_GPU_DEVICE(
                 amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real {
      amrex::Real r2 = (x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]);
#if AMREX_SPACEDIM == 3
      if (!slotted) {
        r2 += (z - c[2]) * (z - c[2]);
      }
#else
      amrex::ignore_unused(z);
#endif
      if (r2 > R * R) {
        return -1.0;
      }
      if (slotted && std::abs(x - c[0]) < hw && y < slot_top) {
        return -1.0;
      }
      return 1.0;
    };
    vof(i, j, k) = pelelmex::vof::cell_fraction(i, j, k, geomdata, phi, 16);
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;

#endif
