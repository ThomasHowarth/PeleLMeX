#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_Index.H>
#include <pelelmex_prob_parm.H>
#include <PMFData.H>
#include <PelePhysics.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pelelmex_initdata(
  int i,
  int j,
  int k,
  int is_incompressible,
  amrex::Array4<amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& /*aux*/,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{

  //
  // Fill Velocity
  //
  state(i, j, k, VELX) = 0;
  state(i, j, k, VELY) = 0;
  state(i, j, k, VELZ) = 0;

  if (is_incompressible != 0) {
      return;
  }

  state(i, j, k, TEMP) = prob_parm.T_mean;
  
  amrex::Real Yt[NUM_SPECIES];
  Yt[O2_ID] = 0.233;
  Yt[N2_ID] = 0.767;
  
  amrex::Real rho_cgs, P_cgs;
  P_cgs = prob_parm.P_mean * 10.0;
  
  auto eos = pele::physics::PhysicsType::eos();
  eos.PYT2R(P_cgs, Yt, state(i, j, k, TEMP), rho_cgs);
  state(i, j, k, DENSITY) = rho_cgs * 1.0e3; // CGS -> MKS conversion
  
  eos.TY2H(state(i, j, k, TEMP), Yt, state(i, j, k, RHOH));
  state(i, j, k, RHOH) *=
      1.0e-4 * state(i, j, k, DENSITY); // CGS -> MKS conversion
  
  for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, FIRSTSPEC + n) = Yt[n] * state(i, j, k, DENSITY);
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real* /*x[AMREX_SPACEDIM]*/,
  const int /*m_nAux*/,
  amrex::Real* /*s_ext[NVAR]*/,
  const int /*idir*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParm const& /*prob_parm*/,
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
zero_visc(
  int /*i*/,
  int /*j*/,
  int /*k*/,
  amrex::Array4<amrex::Real> const& /*beta*/,
  amrex::GeometryData const& /*geomdata*/,
  amrex::Box const& /*domainBox*/,
  const int /*dir*/,
  const int /*beta_comp*/,
  const int /*nComp*/)
{
  // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
  // otherwise this routine could be called for other face diffusivity (Temp,
  // velocity, ...)
}
#endif
