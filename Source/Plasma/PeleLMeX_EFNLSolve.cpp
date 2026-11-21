#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_EF_K.H>
#include <MLGMRES.H>
#include <PeleLMeX_DiffusionOp.H>
#include <PeleLMeX_Utils.H>
#include <hydro_utils.H>

namespace {
// Coefficient of the phiV Laplacian in the Poisson residual
constexpr amrex::Real scalLap = eps0 * epsr / elemCharge;

// A vanishing scale is replaced by 1
amrex::Real
nonZeroScale(const amrex::Real a_scale)
{
  return (a_scale > 1.0e-12) ? a_scale : 1.0;
}
} // namespace

PrecondOp*
PeleLM::getPrecondOp()
{
  if (!m_precond_op)
    m_precond_op.reset(new PrecondOp(this));
  return m_precond_op.get();
}

void
PeleLM::implicitNonLinearSolve(
  const int sdcIter,
  const amrex::Real a_dt,
  const std::unique_ptr<AdvanceDiffData>& diffData,
  const std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE_VAR("PeleLMeX::implicitNonLinearSolve()", implicitNonLinearSolve);

  const amrex::Real strt_time = amrex::ParallelDescriptor::second();

  //------------------------------------------------------------------------
  // Pre-solve

  // Get cell-centered charged species transport coefficients
  calcEFTransport(AmrNewTime);

  // Time step of the non-linear solve, used by the GMRES callbacks
  dtsub = a_dt;

  // Pass t^{n} nE/PhiV from leveldata to leveldatanlsolve
  // t^{n} have been fillpatched already
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get t^{n} data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrOldTime);
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    amrex::MultiFab::Copy(
      ldataNLs_p->nlState, ldata_p->state, NE, 0, 1, m_nGrowState);
    amrex::MultiFab::Copy(
      ldataNLs_p->nlState, ldata_p->state, PHIV, 1, 1, m_nGrowState);
  }

  // Gradient of PhiV at t^{n}
  constexpr int do_avgDown = 0; // TODO or should I ?
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);
  getDiffusionOp()->computeGradient(
    getNLgradPhiVVect(), {}, // don't need the laplacian out
    GetVecOfConstPtrs(getPhiVVect(AmrOldTime)), {}, bcRecPhiV[0], do_avgDown);

  // Stash away a copy of umac
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      amrex::MultiFab::Copy(
        ldataNLs_p->umac[idim], advData->umac[lev][idim], 0, 0, 1, 0);
    }
  }

  // Setup MLGMRES
  MLGMRESSolver gmres;
  int GMRES_tot_count = 0;
  gmres.define(this, 2, 1);
  MLJtimesVFunc jtv = &PeleLM::jTimesV;
  gmres.setJtimesV(jtv);
  MLNormFunc normF = &PeleLM::nlSolveNorm;
  gmres.setNorm(normF);
  MLPrecondFunc prec = &PeleLM::applyPrecond;
  gmres.setPrecond(prec);

  //------------------------------------------------------------------------
  const amrex::Real newTime = getTime(0, AmrOldTime) + dtsub;

  // -----------------
  // Pre-Newton
  // Set up the NL state scaling
  getNLStateScaling(nE_scale, phiV_scale);
  nE_scale = nonZeroScale(nE_scale);
  phiV_scale = nonZeroScale(phiV_scale);
  if (ef_verbose) {
    amrex::Print() << " ne scaling: " << nE_scale << "\n";
    amrex::Print() << " phiV scaling: " << phiV_scale << "\n";
  }
  scaleNLState();

  // Compute the background charge distribution at the new time
  computeBGcharge(newTime, diffData, advData);

  // Newton initial guess
  // TODO newton initial guess
  nlSolveNorm(getNLstateVect(), nl_stateNorm);

  // Initial NL residual: update residual scaling and preconditioner
  int update_scaling = 1;
  int update_precond = 1;
  nonLinearResidual(
    dtsub, getNLstateVect(), getNLresidVect(), update_scaling, update_precond);
  nlSolveNorm(getNLresidVect(), nl_residNorm);

  // Check for convergence
  amrex::Real scaledResnE, scaledResphiV;
  getNLResidScaling(scaledResnE, scaledResphiV);
  amrex::Real max_nlres = amrex::max<amrex::Real>(scaledResnE, scaledResphiV);
  // Skip the Newton iterations if the initial state already satisfies the
  // tolerance, but still do the post-Newton forcing and state update
  int exit_newton = (max_nlres <= m_ef_newtonTol) ? 1 : 0;
  if ((exit_newton != 0) && (ef_verbose != 0)) {
    amrex::Print() << " No Newton iteration needed. \n";
  }

  // -----------------
  // Newton iteration
  int NK_ite = 0;
  while (exit_newton == 0) {
    NK_ite += 1;

    // Verbose
    if (ef_verbose) {
      amrex::Print() << " Newton it: " << NK_ite
                     << " L2**2 residual: " << 0.5 * nl_residNorm * nl_residNorm
                     << ". Linf residual: " << max_nlres << "\n";
    }

    // Solve for Newton direction
    amrex::Vector<amrex::MultiFab> newtonDir;
    newtonDir.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      newtonDir.emplace_back(
        grids[lev], dmap[lev], 2, 1, amrex::MFInfo(), Factory(lev));
      newtonDir[lev].setVal(0.0, 0, 2, 1);
    }
    const amrex::Real S_tol = m_ef_GMRES_reltol;
    const amrex::Real S_tol_abs = m_ef_GMRES_abstol;
    GMRES_tot_count +=
      gmres.solve(GetVecOfPtrs(newtonDir), getNLresidVect(), S_tol_abs, S_tol);
    amrex::Real newtonDir_Norm;
    nlSolveNorm(GetVecOfPtrs(newtonDir), newtonDir_Norm);

    // Linesearch & update state TODO
    updateNLState(GetVecOfPtrs(newtonDir));
    nlSolveNorm(getNLstateVect(), nl_stateNorm);
    update_scaling = 0;
    update_precond = 1;
    nonLinearResidual(
      dtsub, getNLstateVect(), getNLresidVect(), update_scaling,
      update_precond);
    nlSolveNorm(getNLresidVect(), nl_residNorm);
    getNLResidScaling(scaledResnE, scaledResphiV);
    max_nlres = amrex::max<amrex::Real>(scaledResnE, scaledResphiV);

    // Exit condition
    exit_newton = testExitNewton(NK_ite, max_nlres, newtonDir_Norm);
  }

  // -----------------
  // Post-Newton
  // Increment the forcing term
  incrementElectronForcing(advData);
  // Unscale nl_state
  unscaleNLState(0);

  // Update the state
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get t^{n} data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    int nGrowNL = 0;
    amrex::MultiFab::Copy(
      ldata_p->state, ldataNLs_p->nlState, 0, NE, 1, nGrowNL);
    amrex::MultiFab::Copy(
      ldata_p->state, ldataNLs_p->nlState, 1, PHIV, 1, nGrowNL);
  }

  if (ef_verbose) {
    amrex::Real run_time = amrex::ParallelDescriptor::second() - strt_time;
    amrex::ParallelDescriptor::ReduceRealMax(
      run_time, amrex::ParallelDescriptor::IOProcessorNumber());
    if (NK_ite > 0) {
      amrex::Real avgGMRES = (float)GMRES_tot_count / (float)NK_ite;
      amrex::Print() << "  [" << sdcIter << "] dt: " << a_dt
                     << " - Avg GMRES/Newton: " << avgGMRES << "\n";
    }
    amrex::Print() << "  >> PeleLMeX::implicitNLSolve() " << run_time << "\n";
  }
}

int
PeleLM::testExitNewton(
  const int newtonIter,
  const amrex::Real max_res,
  const amrex::Real norm_NewtonDir)
{
  int exit = 0;
  if (max_res <= m_ef_newtonTol || norm_NewtonDir <= 1e-11) {
    exit = 1;
    if (ef_verbose) {
      amrex::Print() << " Newton iterations converged: \n";
      amrex::Print() << " Final Newton L2**2 res norm : "
                     << 0.5 * nl_residNorm * nl_residNorm << "\n";
      amrex::Print() << " Final Newton Linf res norm : " << max_res << "\n";
    }
  }

  if (newtonIter >= m_ef_maxNewtonIter && exit == 0) {
    exit = 1;
    amrex::Print()
      << " WARNING: Max Newton iteration reached without convergence !!! \n";
    amrex::Print() << " Final Newton L2**2 res norm : "
                   << 0.5 * nl_residNorm * nl_residNorm << "\n";
    amrex::Print() << " Final Newton Linf res norm : " << max_res << "\n";
  }
  return exit;
}

void
PeleLM::updateNLState(const amrex::Vector<amrex::MultiFab*>& a_update)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev); // NL data
    ldataNLs_p->nlState.plus(*a_update[lev], 0, 2, 0);
  }

  // AverageDown the new state
  averageDown(getNLstateVect(), 0, 2);

  // Need to fillpatch the NL state
  // First unscale NLState
  unscaleNLState(m_nGrowState);

  // FillPatch
  amrex::Vector<amrex::MultiFab> nEState;
  amrex::Vector<amrex::MultiFab> phiVState;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev); // NL data
    nEState.emplace_back(ldataNLs_p->nlState, amrex::make_alias, 0, 1);
    phiVState.emplace_back(ldataNLs_p->nlState, amrex::make_alias, 1, 1);
  }
  fillPatchNLnE(m_cur_time, GetVecOfPtrs(nEState), m_nGrowState);
  fillPatchNLphiV(m_cur_time, GetVecOfPtrs(phiVState), m_nGrowState);

  // Rescale
  scaleNLState();
}

void
PeleLM::incrementElectronForcing(const std::unique_ptr<AdvanceAdvData>& advData)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldata_p = getLevelDataPtr(lev, AmrOldTime); // Old time electron
    auto ldataR_p = getLevelDataReactPtr(lev);       // Reaction
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);   // NL data

    auto const& state_o_ma = ldata_p->state.const_arrays();
    auto const& nE_n_ma = ldataNLs_p->nlState.const_arrays();
    auto const& I_R_ma = ldataR_p->I_R.const_arrays();
    auto const& F_ma = advData->Forcing[lev].arrays();
    const amrex::Real scaling = nE_scale;
    const amrex::Real dtinv = 1.0 / dtsub;
    amrex::ParallelFor(
      ldata_p->state,
      [state_o_ma, nE_n_ma, I_R_ma, F_ma, scaling,
       dtinv] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real const> nE_o(state_o_ma[box_no], NE);
        amrex::Array4<amrex::Real const> I_R_nE(I_R_ma[box_no], NUM_SPECIES);
        amrex::Array4<amrex::Real> FnE(F_ma[box_no], NUM_SPECIES + 1);
        FnE(i, j, k) =
          (nE_n_ma[box_no](i, j, k) * scaling - nE_o(i, j, k)) * dtinv -
          I_R_nE(i, j, k);
      });
    amrex::Gpu::streamSynchronize();
  }
}

void
PeleLM::computeBGcharge(
  const amrex::Real a_time,
  const std::unique_ptr<AdvanceDiffData>& diffData,
  const std::unique_ptr<AdvanceAdvData>& advData)
{
  // Get integration dt
  const amrex::Real dt_int = a_time - getTime(0, AmrOldTime);

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get data pointers
    auto ldata_p = getLevelDataPtr(lev, AmrOldTime); // Old time species
    auto ldataR_p = getLevelDataReactPtr(lev);       // Reaction
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);   // NL data

    auto const& state_old_ma = ldata_p->state.const_arrays();
    auto const& adv_ma = advData->AofS[lev].const_arrays();
    auto const& dn_ma = diffData->Dn[lev].const_arrays();
    auto const& dnp1_ma = diffData->Dnp1[lev].const_arrays();
    auto const& dhat_ma = diffData->Dhat[lev].const_arrays();
    auto const& rhoYdot_ma = ldataR_p->I_R.const_arrays();

    auto const& charge_ma = ldataNLs_p->backgroundCharge.arrays();
    constexpr amrex::Real factor = 1.0 / elemCharge;

    // Sum over the species in a fixed order in each cell, so the result is
    // deterministic on GPU (it was an atomic add per species)
    amrex::ParallelFor(
      ldataNLs_p->backgroundCharge,
      [state_old_ma, adv_ma, dn_ma, dnp1_ma, dhat_ma, rhoYdot_ma, charge_ma,
       dt_int,
       zk = zk] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real const> rhoYold(
          state_old_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real const> adv(adv_ma[box_no], FIRSTSPEC);
        amrex::Real charge = 0.0;
        for (int n = 0; n < NUM_SPECIES; ++n) {
          amrex::Real rhoYprov =
            rhoYold(i, j, k, n) +
            dt_int *
              (adv(i, j, k, n) +
               0.5 * (dn_ma[box_no](i, j, k, n) - dnp1_ma[box_no](i, j, k, n)) +
               dhat_ma[box_no](i, j, k, n) + rhoYdot_ma[box_no](i, j, k, n));
          rhoYprov = amrex::max<amrex::Real>(rhoYprov, 0.0);
          // Separate statement: no fused multiply-add with the sum
          const amrex::Real val = zk[n] * rhoYprov;
          charge += val;
        }
        charge_ma[box_no](i, j, k) = charge * factor;
      });
    amrex::Gpu::streamSynchronize();
  }
}

void
PeleLM::nonLinearResidual(
  const amrex::Real a_dt,
  const amrex::Vector<amrex::MultiFab*>& a_nlstate,
  const amrex::Vector<amrex::MultiFab*>& a_nlresid,
  const int updateScaling,
  const int updatePrecond)
{
  // Get unscaled copy of the NL state
  amrex::Vector<amrex::MultiFab> nE;
  nE.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> phiV;
  phiV.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    nE.emplace_back(
      grids[lev], dmap[lev], 1, m_nGrowState, amrex::MFInfo(), Factory(lev));
    phiV.emplace_back(
      grids[lev], dmap[lev], 1, m_nGrowState, amrex::MFInfo(), Factory(lev));
    amrex::MultiFab::Copy(nE[lev], *a_nlstate[lev], 0, 0, 1, m_nGrowState);
    nE[lev].mult(nE_scale, 0, 1, m_nGrowState);
    amrex::MultiFab::Copy(phiV[lev], *a_nlstate[lev], 1, 0, 1, m_nGrowState);
    phiV[lev].mult(phiV_scale, 0, 1, m_nGrowState);
  }

  // FillPatch nE state here
  fillPatchNLnE(m_cur_time, GetVecOfPtrs(nE), m_nGrowState);

  // Get L(phiV) and Grad(phiV)
  amrex::Vector<amrex::MultiFab> laplacian;
  laplacian.reserve(finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> gradPhiVCur(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    laplacian.emplace_back(
      grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const auto& fba =
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim));
      gradPhiVCur[lev][idim].define(
        fba, dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    }
  }
  constexpr int do_avgDown = 0; // TODO or should I ?
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);
  getDiffusionOp()->computeGradient(
    GetVecOfArrOfPtrs(gradPhiVCur), GetVecOfPtrs(laplacian),
    GetVecOfConstPtrs(phiV), {}, bcRecPhiV[0], do_avgDown);

  // Get nE diffusion term
  amrex::Vector<amrex::MultiFab> diffnE;
  diffnE.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    diffnE.emplace_back(
      grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
  }
  auto bcRecnE = fetchBCRecArray(NE, 1);
  getDiffusionOp()->computeDiffLap(
    GetVecOfPtrs(diffnE), 0, GetVecOfConstPtrs(nE), 0,
    GetVecOfConstPtrs(getnEDiffusivityVect(AmrNewTime)), 0, bcRecnE, 1);

  // Get nE advection term
  amrex::Vector<amrex::MultiFab> advnE;
  advnE.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    advnE.emplace_back(
      grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
  }
  getAdvectionTerm(
    GetVecOfConstPtrs(nE), GetVecOfPtrs(advnE),
    GetVecOfArrOfConstPtrs(gradPhiVCur));

  // Assemble non-linear residual
  // res(ne(:)) = dt * ( diff(:) + conv(:) + I_R(:) ) - ( ne(:) - ne_old(:) )
  // res(phiv(:)) = \Sum z_k * \tilde Y_k / q_e - ne + Lapl_PhiV
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldataOld_p = getLevelDataPtr(lev, AmrOldTime); // Old time
    auto ldataR_p = getLevelDataReactPtr(lev);          // Reaction

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);

    // Init the ghostcells too
    a_nlresid[lev]->setVal(0.0);

    auto const& I_R_ma = ldataR_p->I_R.const_arrays();
    auto const& lapPhiV_ma = laplacian[lev].const_arrays();
    auto const& ne_diff_ma = diffnE[lev].const_arrays();
    auto const& ne_adv_ma = advnE[lev].const_arrays();
    auto const& ne_curr_ma = nE[lev].const_arrays();
    auto const& state_old_ma = ldataOld_p->state.const_arrays();
    auto const& charge_ma = ldataNLs_p->backgroundCharge.const_arrays();
    auto const& res_ma = a_nlresid[lev]->arrays();
    amrex::ParallelFor(
      ldataNLs_p->nlResid,
      [I_R_ma, lapPhiV_ma, ne_diff_ma, ne_adv_ma, ne_curr_ma, state_old_ma,
       charge_ma, res_ma,
       a_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real> res_nE(res_ma[box_no], 0);
        amrex::Array4<amrex::Real> res_phiV(res_ma[box_no], 1);
        amrex::Array4<amrex::Real const> I_R_nE(I_R_ma[box_no], NUM_SPECIES);
        amrex::Array4<amrex::Real const> ne_old(state_old_ma[box_no], NE);
        res_nE(i, j, k) = ne_old(i, j, k) - ne_curr_ma[box_no](i, j, k) +
                          a_dt * (ne_diff_ma[box_no](i, j, k) +
                                  ne_adv_ma[box_no](i, j, k) + I_R_nE(i, j, k));
        res_phiV(i, j, k) = lapPhiV_ma[box_no](i, j, k) * scalLap -
                            ne_curr_ma[box_no](i, j, k) +
                            charge_ma[box_no](i, j, k);
        res_nE(i, j, k) *= -1.0;   // NLresidual is -RHS
        res_phiV(i, j, k) *= -1.0; // NLresidual is -RHS
      });
    // Shift outside?
    amrex::Gpu::streamSynchronize();
  }

  // AverageDown the NLresiduals
  averageDown(a_nlresid, 0, 2);

  // Update the residual scale if necessary
  if (updateScaling) {
    getNLResidScaling(FnE_scale, FphiV_scale);
    FnE_scale = nonZeroScale(FnE_scale);
    FphiV_scale = nonZeroScale(FphiV_scale);
    if (ef_verbose) {
      amrex::Print() << "    F(ne) scaling: " << FnE_scale << "\n";
      amrex::Print() << "    F(PhiV) scaling: " << FphiV_scale << "\n";
    }
  }

  // Apply scaling
  scaleNLResid(a_nlresid);

  // Update the preconditioning LinOps
  if (updatePrecond) {
    setUpPrecond(a_dt, GetVecOfConstPtrs(nE));
  }
}

void
PeleLM::getAdvectionTerm(
  const amrex::Vector<const amrex::MultiFab*>& a_nE,
  const amrex::Vector<amrex::MultiFab*>& a_advTerm,
  const amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_gPhiVCur)
{
  // Get advection fluxes on all levels
  int nGrow = 0;
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        1, nGrow, amrex::MFInfo(), factory);
    }
  }

  // nE BCRec
  auto bcRecnE = fetchBCRecArray(NE, 1);

  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n} data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);

    // Get the face centered electron mobility
    constexpr int doZeroVisc = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> mobE_ec =
      getDiffusivity(lev, 0, 1, doZeroVisc, bcRecnE, ldata_p->mobE_cc);

    // Get the electron effective velocity
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto const& ueff_ma = ldataNLs_p->uEffnE[idim].arrays();
      auto const& umac_ma = ldataNLs_p->umac[idim].const_arrays();
      auto const& gphi_c_ma = a_gPhiVCur[lev][idim]->const_arrays();
      auto const& gphi_o_ma = ldataNLs_p->gPhiVOld[idim].const_arrays();
      auto const& kappa_e_ma = mobE_ec[idim].const_arrays();
      amrex::ParallelFor(
        ldataNLs_p->uEffnE[idim],
        [ueff_ma, umac_ma, gphi_c_ma, gphi_o_ma, kappa_e_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k) noexcept {
          ueff_ma[box_no](i, j, k) =
            umac_ma[box_no](i, j, k) -
            kappa_e_ma[box_no](i, j, k) * -1.0 * 0.5 *
              (gphi_c_ma[box_no](i, j, k) + gphi_o_ma[box_no](i, j, k));
        });
      // Shift outside?
      amrex::Gpu::streamSynchronize();
    }
  }

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    // Compute advective fluxes
    getAdvectionFluxes(
      lev, GetArrOfPtrs(fluxes[lev]), *a_nE[lev],
      GetArrOfConstPtrs(ldataNLs_p->uEffnE), bcRecnE[0]);
  }

  // Average down the fluxes
  averageDownFaces(fluxes);

  // Compute divergence
  constexpr int intensiveFluxes = 1;
  fluxDivergence(
    a_advTerm, 0, GetVecOfArrOfPtrs(fluxes), 0, 1, intensiveFluxes, -1.0);
}

void
PeleLM::getAdvectionFluxes(
  const int lev,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const amrex::MultiFab& a_nE,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_ueff,
  amrex::BCRec bcrec)
{
  const amrex::Box& domain = geom[lev].Domain();

  const int order = m_nEAdvOrder;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_nE, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    amrex::Box const& bx = mfi.tilebox();
    auto const& ne_arr = a_nE.const_array(mfi);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      // Predict the edge state and multiply it by the effective velocity
      const amrex::Box ebx = amrex::surroundingNodes(bx, idim);
      const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
      const auto bc_lo = bcrec.lo(idim);
      const auto bc_hi = bcrec.hi(idim);
      const bool extdir_or_ho_lo =
        (bc_lo == amrex::BCType::ext_dir) || (bc_lo == amrex::BCType::hoextrap);
      const bool extdir_or_ho_hi =
        (bc_hi == amrex::BCType::ext_dir) || (bc_hi == amrex::BCType::hoextrap);
      const int domlo = domain.smallEnd(idim);
      const int domhi = domain.bigEnd(idim);
      auto const& ueff = a_ueff[idim]->const_array(mfi);
      auto const& flux = a_fluxes[idim]->array(mfi);
      amrex::ParallelFor(
        ebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const int idx[3] = {i, j, k};
          const bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) &&
             (idx[idim] <= edomain.smallEnd(idim)));
          const bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) &&
             (idx[idim] >= edomain.bigEnd(idim)));
          if (order == 1) {
            flux(i, j, k) =
              ueff(i, j, k) *
              ef_edge_state_extdir(i, j, k, idim, on_lo, on_hi, ne_arr, ueff);
          } else {
            flux(i, j, k) = ueff(i, j, k) * ef_edge_state_2ndO_extdir(
                                              i, j, k, idim, on_lo, on_hi,
                                              extdir_or_ho_lo, extdir_or_ho_hi,
                                              domlo, domhi, ne_arr, ueff, ebx);
          }
        });
    }
  }
}

void
PeleLM::setUpPrecond(
  const amrex::Real a_dt, const amrex::Vector<const amrex::MultiFab*>& a_nE)
{
  BL_PROFILE("PeleLMeX::setUpPrecond()");

  // Update LinearOps defs if needed -> done internally by the getPrecondOp()
  // func

  // nE BCRec
  auto bcRecnE = fetchBCRecArray(NE, 1);
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);

  //--------------------------------------------------------------------------
  // Set diff/drift operator
  getPrecondOp()->setDiffOpScalars(
    -nE_scale / FnE_scale, -a_dt * nE_scale / FnE_scale,
    a_dt * nE_scale / FnE_scale);
  amrex::Real omega = m_ABecCecOmega;
  getPrecondOp()->setDiffOpRelaxation(omega);
  getPrecondOp()->setDiffOpBCs(bcRecnE[0]);

  // Get and set coefficients on each level
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get level data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);

    constexpr int doZeroVisc = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> diffE_ec =
      getDiffusivity(lev, 0, 1, doZeroVisc, bcRecnE, ldata_p->diffE_cc);

    getPrecondOp()->setDiffOpACoeff(lev, 1.0);
    getPrecondOp()->setDiffOpBCoeff(lev, GetArrOfConstPtrs(diffE_ec));
    getPrecondOp()->setDiffOpCCoeff(lev, GetArrOfConstPtrs(ldataNLs_p->uEffnE));
  }

  // Stilda approx first-order
  amrex::Vector<amrex::MultiFab> diagDiffOp;
  diagDiffOp.reserve(finest_level + 1);
  if (m_ef_PC_approx == 2) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      diagDiffOp.emplace_back(grids[lev], dmap[lev], 1, 1);
      getPrecondOp()->getDiffOpDiagonal(lev, diagDiffOp[lev]);
      diagDiffOp[lev].mult(FnE_scale / nE_scale);
    }
    fillpatch_forces(
      m_cur_time, GetVecOfPtrs(diagDiffOp), 1, getInterpolator());
  }

  //--------------------------------------------------------------------------
  // Set Stilda and drift operators

  // Set scalars
  getPrecondOp()->setDriftOpScalars(0.0, 0.5 * phiV_scale / FnE_scale * a_dt);
  if (m_ef_PC_approx == 1 || m_ef_PC_approx == 2) {
    getPrecondOp()->setStildaOpScalars(0.0, -1.0);
  } else if (m_ef_PC_approx == 3 || m_ef_PC_approx == 4) {
    getPrecondOp()->setStildaOpScalars(0.0, 1.0);
  }

  // Set BCs
  getPrecondOp()->setDriftOpBCs(bcRecPhiV[0]);
  getPrecondOp()->setStildaOpBCs(bcRecPhiV[0]);

  // Get and set coefficients on each level
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);

    // CC neKe values
    amrex::MultiFab nEKe(grids[lev], dmap[lev], 1, 1);

    // CC Schur neKe values
    amrex::MultiFab Schur_nEKe;
    if (m_ef_PC_approx == 2) {
      Schur_nEKe.define(grids[lev], dmap[lev], 1, 1);
    }

    auto const& neke_ma = nEKe.arrays();
    auto const& kappaE_ma = ldata_p->mobE_cc.arrays();
    auto const& ne_arr_ma = a_nE[lev]->const_arrays();
    auto const& Schur_ma =
      (m_ef_PC_approx == 2) ? Schur_nEKe.arrays() : nEKe.arrays();
    auto const& diffOp_diag_ma =
      (m_ef_PC_approx == 2) ? diagDiffOp[lev].arrays() : nEKe.arrays();
    const int do_Schur = (m_ef_PC_approx == 2) ? 1 : 0;

    amrex::ParallelFor(
      nEKe, nEKe.nGrowVect(),
      [neke_ma, kappaE_ma, ne_arr_ma, do_Schur, Schur_ma, a_dt,
       diffOp_diag_ma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::Real neke =
          kappaE_ma[box_no](i, j, k) * ne_arr_ma[box_no](i, j, k);
        neke_ma[box_no](i, j, k) = neke;
        if (do_Schur == 1) {
          Schur_ma[box_no](i, j, k) =
            -a_dt * 0.5 * neke / diffOp_diag_ma[box_no](i, j, k);
        }
      });
    amrex::Gpu::streamSynchronize();
    // Upwinded edge neKe values
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> neKe_ec = getUpwindedEdge(
      lev, 0, 1, bcRecnE, nEKe, GetArrOfConstPtrs(ldataNLs_p->uEffnE));

    // Set drift Op coefficients
    getPrecondOp()->setDriftOpBCoeff(lev, GetArrOfConstPtrs(neKe_ec));

    // Set Stilda Op coefficients
    if (m_ef_PC_approx == 1) { // Assuming identity of the inverse of DiffOp
      // Add Stilda pieces
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        neKe_ec[idim].mult(0.5 * a_dt, 0, 1);
        neKe_ec[idim].plus(scalLap, 0, 1);
      }
      getPrecondOp()->setStildaOpBCoeff(lev, GetArrOfConstPtrs(neKe_ec));
    } else if (m_ef_PC_approx == 2) { // Assuming inverse of the diag of DiffOp
      // Upwinded Schur edge neKe values
      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> Schur_neKe_ec =
        getUpwindedEdge(
          lev, 0, 1, bcRecnE, Schur_nEKe,
          GetArrOfConstPtrs(ldataNLs_p->uEffnE));
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Schur_neKe_ec[idim].plus(scalLap, 0, 1);
      }
      getPrecondOp()->setStildaOpBCoeff(lev, GetArrOfConstPtrs(Schur_neKe_ec));
    } else {
      amrex::Abort("Preconditioner option /= 1 or 2 not available yet");
    }
  }
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getUpwindedEdge(
  const int lev,
  const int edge_comp,
  const int ncomp,
  const amrex::Vector<amrex::BCRec>& bcrec,
  const amrex::MultiFab& ccMF,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& ecVel)
{
  AMREX_ASSERT(bcrec.size() >= ncomp);
  AMREX_ASSERT(ccMF.nComp() >= edge_comp + ncomp);
  const auto& ba = ccMF.boxArray();
  const auto& dm = ccMF.DistributionMap();
  const auto& factory = ccMF.Factory();
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> ecMFs{AMREX_D_DECL(
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(0)), dm, ncomp, 0,
      amrex::MFInfo(), factory),
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(1)), dm, ncomp, 0,
      amrex::MFInfo(), factory),
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(2)), dm, ncomp, 0,
      amrex::MFInfo(), factory))};

  const amrex::Box& domain = geom[lev].Domain();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(ccMF, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const amrex::Box ebx = mfi.nodaltilebox(idim);
      const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
      const auto& ccVal = ccMF.const_array(mfi, edge_comp);
      const auto& ecUeff = ecVel[idim]->const_array(mfi);
      const auto& ecVal = ecMFs[idim].array(mfi);
      const auto bc_lo = bcrec[0].lo(idim);
      const auto bc_hi = bcrec[0].hi(idim);
      amrex::ParallelFor(
        ebx, [bc_lo, bc_hi, idim, edomain, ecUeff, ccVal,
              ecVal] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const int idx[3] = {i, j, k};
          const bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) &&
             (idx[idim] <= edomain.smallEnd(idim)));
          const bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) &&
             (idx[idim] >= edomain.bigEnd(idim)));
          cen2edg_upwind(i, j, k, idim, 1, on_lo, on_hi, ecUeff, ccVal, ecVal);
        });
    }
  }

  return ecMFs;
}

void
PeleLM::jTimesV(
  const amrex::Vector<amrex::MultiFab*>& a_v,
  const amrex::Vector<amrex::MultiFab*>& a_Jv)
{
  amrex::Real vNorm;
  nlSolveNorm(a_v, vNorm);

  // v is zero, Jv is zero and return
  if (vNorm == 0.0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      a_Jv[lev]->setVal(0.0);
    }
    return;
  }

  // TODO: only one-sided difference for now
  amrex::Real delta_pert =
    m_ef_lambda_jfnk * (m_ef_lambda_jfnk + nl_stateNorm / vNorm);

  // Residual of the state perturbed by a_eps * v
  auto perturbedResid = [&](const amrex::Real a_eps) {
    amrex::Vector<amrex::MultiFab> statePert;
    statePert.reserve(finest_level + 1);
    amrex::Vector<amrex::MultiFab> residPert;
    residPert.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
      statePert.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      residPert.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      amrex::MultiFab::Copy(
        statePert[lev], ldataNLs_p->nlState, 0, 0, 2, m_nGrowState);
      amrex::MultiFab::Saxpy(statePert[lev], a_eps, *a_v[lev], 0, 0, 2, 0);
    }
    constexpr int update_scaling = 0;
    constexpr int update_precond = 0;
    nonLinearResidual(
      dtsub, GetVecOfPtrs(statePert), GetVecOfPtrs(residPert), update_scaling,
      update_precond);
    return residPert;
  };

  if (m_ef_diffT_jfnk == 1) {
    const auto residPert = perturbedResid(delta_pert);
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
      amrex::MultiFab::LinComb(
        *a_Jv[lev], 1.0, residPert[lev], 0, -1.0, ldataNLs_p->nlResid, 0, 0, 2,
        0);
      a_Jv[lev]->mult(-1.0 / delta_pert);
    }
  } else if (m_ef_diffT_jfnk == 2) {
    const auto residPertPls = perturbedResid(delta_pert);
    const auto residPertMns = perturbedResid(-delta_pert);
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::LinComb(
        *a_Jv[lev], 1.0, residPertPls[lev], 0, -1.0, residPertMns[lev], 0, 0, 2,
        0);
      a_Jv[lev]->mult(-0.5 / delta_pert);
    }
  }
}

void
PeleLM::applyPrecond(
  const amrex::Vector<amrex::MultiFab*>& a_v,
  const amrex::Vector<amrex::MultiFab*>& a_Pv)
{
  BL_PROFILE("PeleLMeX::applyPrecond()");

  // Setup aliases and temps
  amrex::Vector<amrex::MultiFab> nE_al;
  nE_al.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> phiV_al;
  phiV_al.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> PnE_al;
  PnE_al.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> PphiV_al;
  PphiV_al.reserve(finest_level + 1);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> temp(finest_level + 1);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> temp2(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Aliases
    nE_al.emplace_back(*a_v[lev], amrex::make_alias, 0, 1);
    phiV_al.emplace_back(*a_v[lev], amrex::make_alias, 1, 1);
    PnE_al.emplace_back(*a_Pv[lev], amrex::make_alias, 0, 1);
    PphiV_al.emplace_back(*a_Pv[lev], amrex::make_alias, 1, 1);
    PnE_al[lev].setVal(0.0, 0, 1, 1);
    PphiV_al[lev].setVal(0.0, 0, 1, 1);

    // Temporary data holder
    temp[lev].reset(new amrex::MultiFab(
      grids[lev], dmap[lev], 1, nE_al[lev].nGrow(), amrex::MFInfo(),
      Factory(lev)));
    temp[lev]->setVal(0.0);
    temp2[lev].reset(new amrex::MultiFab(
      grids[lev], dmap[lev], 1, nE_al[lev].nGrow(), amrex::MFInfo(),
      Factory(lev)));
    temp2[lev]->setVal(0.0);
  }

  // Set LinOps Level BCs
  for (int lev = 0; lev <= finest_level; ++lev) {
    getPrecondOp()->setDiffOpLevelBC(lev, &PnE_al[lev]);
    getPrecondOp()->setDriftOpLevelBC(lev, &PphiV_al[lev]);
    getPrecondOp()->setStildaOpLevelBC(lev, &PphiV_al[lev]);
  }

  // Most inner matrix
  amrex::Real S_tol = m_ef_PC_MG_Tol;
  amrex::Real S_tol_abs = MLNorm0(GetVecOfConstPtrs(nE_al)) * m_ef_PC_MG_Tol;
  getPrecondOp()->diffOpSolve(
    GetVecOfPtrs(PnE_al), GetVecOfConstPtrs(nE_al), S_tol, S_tol_abs);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(PphiV_al[lev], phiV_al[lev], 0, 0, 1, 0);
  }

  // Pivot second matrix
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Saxpy(
      PphiV_al[lev], nE_scale / FphiV_scale, PnE_al[lev], 0, 0, 1, 0);
  }

  // PhiV estimate matrix
  // First scale PphiV
  for (int lev = 0; lev <= finest_level; ++lev) {
    PphiV_al[lev].mult(FphiV_scale / phiV_scale);
  }
  S_tol_abs = MLNorm0(GetVecOfConstPtrs(PphiV_al)) * m_ef_PC_MG_Tol;
  getPrecondOp()->StildaOpSolve(
    GetVecOfPtrs(temp), GetVecOfConstPtrs(PphiV_al), S_tol, S_tol_abs);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(PphiV_al[lev], *temp[lev], 0, 0, 1, 0);
  }

  // Final matrix
  getPrecondOp()->driftOpApply(GetVecOfPtrs(temp), GetVecOfPtrs(PphiV_al));
  S_tol_abs = MLNorm0(GetVecOfConstPtrs(temp)) * m_ef_PC_MG_Tol;
  getPrecondOp()->diffOpSolve(
    GetVecOfPtrs(temp2), GetVecOfConstPtrs(temp), S_tol, S_tol_abs);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Saxpy(PnE_al[lev], -1.0, *temp2[lev], 0, 0, 1, 0);
  }
}

void
PeleLM::nlSolveNorm(const amrex::Vector<amrex::MultiFab*>& a_MF, amrex::Real& r)
{
  r = 0.0;
  const int nComp = a_MF[0]->nComp();
  for (int comp = 0; comp < nComp; ++comp) {
    amrex::Real norm = 0.0;
    for (int lev = 0; lev < a_MF.size(); ++lev) {
      // TODO : norm not weighted by cell size, should it ?
      if (lev != a_MF.size() - 1) {
        norm += amrex::MultiFab::Dot(
          *m_coveredMask[lev], *a_MF[lev], comp, *a_MF[lev], comp, 1, 0);
      } else {
        norm += amrex::MultiFab::Dot(*a_MF[lev], comp, *a_MF[lev], comp, 1, 0);
      }
    }
    r += norm;
  }
  r = std::sqrt(r);
}
