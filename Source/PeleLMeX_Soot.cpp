#ifdef PELE_USE_SOOT
#include <PeleLMeX.H>
#include <PeleLMeX_Derive.H>
#include "SootModel.H"

void
PeleLM::setSootIndx()
{
  const int ndim = AMREX_SPACEDIM;
  SootComps sootComps;
  // U/Q state variables indices
  sootComps.qRhoIndx = DENSITY - ndim;
  sootComps.qSpecIndx = FIRSTSPEC - ndim;
  sootComps.qTempIndx = TEMP - ndim;
  sootComps.qSootIndx = FIRSTSOOT - ndim;
  // Source variables indices
  sootComps.rhoIndx = DENSITY - ndim;
  sootComps.specIndx = FIRSTSPEC - ndim;
  sootComps.engIndx = RHOH - ndim;
  sootComps.sootIndx = FIRSTSOOT - ndim;
  soot_model->setIndices(sootComps);
}

void
PeleLM::cleanupSootModel()
{
  soot_model->cleanup();
  delete soot_model;
}

void
PeleLM::computeSootSource(const TimeStamp a_timestamp, const amrex::Real a_dt)
{
  constexpr bool pres_term =
    false; // Do not include change in pressure in energy
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_timestamp);
    const amrex::Real time = getTime(lev, a_timestamp);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*(m_extSource[lev]), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      amrex::Box const& gbx = mfi.growntilebox();
      auto const& state_arr = ldata_p->state.const_array(mfi, DENSITY);
      auto const& mu = ldata_p->visc_cc.const_array(mfi, 0);
      auto const& source_arr = m_extSource[lev]->array(mfi, DENSITY);
      soot_model->computeSootSourceTerm(
        gbx, state_arr, mu, source_arr, time, a_dt, pres_term);
    }
  }
}

void
PeleLM::computeSootSourceSDC(
  const std::unique_ptr<AdvanceDiffData>& diffData, const bool a_withForcing)
{
  BL_PROFILE("PeleLMeX::computeSootSourceSDC()");
  constexpr bool pres_term =
    false; // Do not include change in pressure in energy
  const amrex::Real dtinv = 1.0 / m_dt;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
    auto& sootSrc = diffData->sootSrc[lev];
    const auto ng = sootSrc.nGrowVect();

    // Forcing: everything that moved the t^{n+1,k} iterate except the soot
    // source it used, F = (Phi^{n+1,k} - Phi^n) / dt - S_soot
    amrex::MultiFab forcing;
    if (a_withForcing) {
      forcing.define(
        grids[lev], dmap[lev], NVAR, ng, amrex::MFInfo(), *m_factory[lev]);
      auto const& f_ma = forcing.arrays();
      auto const& so_ma = ldataOld_p->state.const_arrays();
      auto const& sn_ma = ldataNew_p->state.const_arrays();
      auto const& src_ma = sootSrc.const_arrays();
      amrex::ParallelFor(
        forcing, ng, NVAR - DENSITY,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          const int c = DENSITY + n;
          f_ma[box_no](i, j, k, c) =
            (sn_ma[box_no](i, j, k, c) - so_ma[box_no](i, j, k, c)) * dtinv -
            src_ma[box_no](i, j, k, c);
        });
      amrex::Gpu::streamSynchronize();
    }

    sootSrc.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(sootSrc, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      amrex::Box const& gbx = mfi.growntilebox();
      auto const& state_arr = ldataOld_p->state.const_array(mfi, DENSITY);
      auto const& mu = ldataOld_p->visc_cc.const_array(mfi, 0);
      auto const& source_arr = sootSrc.array(mfi, DENSITY);
      SootModel::SootForcing sf;
      if (a_withForcing) {
        sf.forcing = forcing.const_array(mfi, DENSITY);
        sf.Qstate_end = ldataNew_p->state.const_array(mfi, DENSITY);
        sf.coeff_mu_end = ldataNew_p->visc_cc.const_array(mfi, 0);
      }
      soot_model->computeSootSourceTerm(
        gbx, state_arr, mu, source_arr, getTime(lev, AmrOldTime), m_dt,
        pres_term, sf);
    }
  }
}

void
PeleLM::computeSootRate(
  const TimeStamp a_timestamp, amrex::Vector<amrex::MultiFab>& a_dst)
{
  BL_PROFILE("PeleLMeX::computeSootRate()");
  constexpr bool pres_term =
    false; // Do not include change in pressure in energy
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_timestamp);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(a_dst[lev], amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      amrex::Box const& gbx = mfi.growntilebox();
      auto const& state_arr = ldata_p->state.const_array(mfi, DENSITY);
      auto const& mu = ldata_p->visc_cc.const_array(mfi, 0);
      auto const& source_arr = a_dst[lev].array(mfi, DENSITY);
      soot_model->computeSootSourceRate(
        gbx, state_arr, mu, source_arr, pres_term);
    }
  }
}

void
PeleLM::clipSootMoments()
{
  SootData* sd = soot_model->getSootData_d();
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    auto const& state_ma = ldata_p->state.arrays();
    amrex::ParallelFor(
      ldata_p->state, [state_ma, sd] AMREX_GPU_DEVICE(
                        int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real> state_arr(state_ma[box_no], FIRSTSOOT);
        amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS + 1> moments;
        for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
          moments[mom] = state_arr(i, j, k, mom);
        }
        sd->momConvClipConv(moments.data());
        for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
          state_arr(i, j, k, mom) = moments[mom];
        }
      });
  }
  amrex::Gpu::streamSynchronize();
}

// TODO: This isn't working yet
/*
void
PeleLM::addSootDerivePlotVars(PeleLMDeriveList& derive_lst)
{
  // Add in soot variables
  amrex::Vector<std::string> sootNames = {"rho_soot", "sum_rho_soot"};
  derive_lst.add(
    "soot_vars", IndexType::TheCellType(), sootNames.size(), sootNames,
    soot_genvars, PeleLMDeriveRec::TheSameBox);

  // Variables associated with the second mode (large particles)
  amrex::Vector<std::string> large_part_names = {"NL", "soot_V_L", "soot_S_L"};
  derive_lst.add(
    "soot_large_particles", IndexType::TheCellType(), large_part_names.size(),
    large_part_names, soot_largeparticledata, PeleLMDeriveRec::TheSameBox);
}
*/
#endif
