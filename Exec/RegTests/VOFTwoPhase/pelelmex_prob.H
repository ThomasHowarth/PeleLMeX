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

// Incompressible two-phase cases with the density and viscosity derived
// from the liquid volume fraction (gravity along -y):
//  - pool:     flat liquid layer at rest, below the plane n.(x - c) = 0
//              through the domain center c with n = (sin tilt, cos tilt)
//              (tilt in degrees), shifted so that the depth at the center
//              is `depth`; set peleLM.gravity = -g n for equilibrium
//  - rt:       Rayleigh-Taylor instability, heavy fluid (the "liquid",
//              F = 1) above y = depth + amplitude * cos(2 pi x / wavelength)
//  - dambreak: liquid column of width `width` and height `height` in the
//              lower-left corner, released from rest

enum class TwoPhaseType { Pool, RT, DamBreak };

struct MyProbParm : public ProbParmDefault
{
  TwoPhaseType type = TwoPhaseType::Pool;
  amrex::Real depth = 0.5;
  amrex::Real tilt = 0.0;
  amrex::Real amplitude = 0.0;
  amrex::Real wavelength = 1.0;
  amrex::Real width = 0.146;
  amrex::Real height = 0.292;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata(
    int i,
    int j,
    int k,
    int /*is_incompressible*/,
    amrex::Array4<amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& /*geomdata*/,
    MyProbParm const& /*prob_parm*/,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) = 0.0;
                 , state(i, j, k, VELZ) = 0.0;)
  }

  // Walls only: zero velocity
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real* /*x*/,
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int /*idir*/,
    const int /*sgn*/,
    const amrex::Real /*time*/,
    amrex::GeometryData const& /*geomdata*/,
    MyProbParm const& /*prob_parm*/,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    AMREX_D_TERM(s_ext[VELX] = 0.0;, s_ext[VELY] = 0.0;, s_ext[VELZ] = 0.0;)
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
    constexpr amrex::Real pi = amrex::Math::pi<amrex::Real>();
    const auto pp = prob_parm;
    // phi > 0 inside the liquid
    auto phi = [=] AMREX_GPU_DEVICE(
                 amrex::Real x, amrex::Real y, amrex::Real z) -> amrex::Real {
      amrex::ignore_unused(z);
      if (pp.type == TwoPhaseType::Pool) {
        const amrex::Real th = pp.tilt * pi / 180.0;
        const amrex::Real xc = 0.5 * (geomdata.ProbLo(0) + geomdata.ProbHi(0));
        return pp.depth - (std::sin(th) * (x - xc) + std::cos(th) * y);
      }
      if (pp.type == TwoPhaseType::RT) {
        return y - pp.depth -
               pp.amplitude * std::cos(2.0 * pi * x / pp.wavelength);
      }
      return amrex::min(pp.width - x, pp.height - y);
    };
    vof(i, j, k) = pelelmex::vof::cell_fraction(i, j, k, geomdata, phi, 16);
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;

#endif
