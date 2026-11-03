#ifndef PELELM_FILLPATCH_H
#define PELELM_FILLPATCH_H

// Definitions of the PeleLM fillpatch helper templates declared in
// PeleLMeX.H. Include this (rather than PeleLMeX_BCfill.H directly) in the
// translation units that fillpatch.

#include <PeleLMeX.H>
#include <PeleLMeX_BCfill.H>
#include <AMReX_FillPatchUtil.H>

template <template <typename> class BCFill>
auto
PeleLM::makeBCFill()
{
  return BCFill<ProblemSpecificFunctions>{
    prob_parm_d, pmf_data.device_parm(), m_nAux, m_map_eval};
}

inline auto
PeleLM::makeStateBCFill()
{
  return PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
    prob_parm_d,
    pmf_data.device_parm(),
    m_nAux,
    static_cast<int>(turb_inflow.is_initialized()),
    m_use_inlet_from_plane,
    m_inlet_plane_dir,
    static_cast<int>(m_recycling_mode == RecyclingMode::Full),
    m_map_eval};
}

template <typename BCFill, typename SrcFn>
void
PeleLM::fillpatchLevel(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_dst,
  const int nGhost,
  SrcFn&& a_src,
  const int scomp,
  const int dcomp,
  const int ncomp,
  amrex::Vector<amrex::BCRec> const& a_bcrec,
  BCFill const& a_bcfill,
  amrex::InterpBase* a_mapper)
{
  using PhysBC = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCFill>>;
  if (lev == 0) {
    PhysBC bndry_func(geom[lev], a_bcrec, a_bcfill);
    const FillPatchSource fine = a_src(lev);
    amrex::FillPatchSingleLevel(
      a_dst, amrex::IntVect(nGhost), a_time, fine.mf, fine.time, scomp, dcomp,
      ncomp, geom[lev], bndry_func, 0);
  } else {
    PhysBC crse_bndry_func(geom[lev - 1], a_bcrec, a_bcfill);
    PhysBC fine_bndry_func(geom[lev], a_bcrec, a_bcfill);
    const FillPatchSource crse = a_src(lev - 1);
    const FillPatchSource fine = a_src(lev);
    amrex::FillPatchTwoLevels(
      a_dst, amrex::IntVect(nGhost), a_time, crse.mf, crse.time, fine.mf,
      fine.time, scomp, dcomp, ncomp, geom[lev - 1], geom[lev], crse_bndry_func,
      0, fine_bndry_func, 0, refRatio(lev - 1), a_mapper, a_bcrec, 0);
  }
}

template <typename BCFill>
void
PeleLM::fillpatchAllLevels(
  const amrex::Real a_time,
  amrex::Vector<amrex::MultiFab*> const& a_mf,
  const int nGhost,
  const int ncomp,
  amrex::Vector<amrex::BCRec> const& a_bcrec,
  BCFill const& a_bcfill,
  amrex::InterpBase* a_mapper)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    fillpatchLevel(
      lev, a_time, *a_mf[lev], nGhost,
      [&](int l) { return FillPatchSource{{a_mf[l]}, {a_time}}; }, 0, 0, ncomp,
      a_bcrec, a_bcfill, a_mapper);
  }
}

template <typename BCFill>
void
PeleLM::fillcoarsepatchLevel(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_dst,
  const int nGhost,
  amrex::MultiFab& a_crse,
  const int ncomp,
  amrex::Vector<amrex::BCRec> const& a_bcrec,
  BCFill const& a_bcfill,
  amrex::InterpBase* a_mapper)
{
  AMREX_ASSERT(lev > 0);
  using PhysBC = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCFill>>;
  PhysBC crse_bndry_func(geom[lev - 1], a_bcrec, a_bcfill);
  PhysBC fine_bndry_func(geom[lev], a_bcrec, a_bcfill);
  amrex::InterpFromCoarseLevel(
    a_dst, amrex::IntVect(nGhost), a_time, a_crse, 0, 0, ncomp, geom[lev - 1],
    geom[lev], crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
    a_mapper, a_bcrec, 0);
}

#endif
