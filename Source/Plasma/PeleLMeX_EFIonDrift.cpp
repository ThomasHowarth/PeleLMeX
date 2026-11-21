#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_EF_Constants.H>
#include <AMReX_FillPatchUtil.H>
#include <PeleLMeX_BCfill.H>

void
PeleLM::ionDriftVelocity(const std::unique_ptr<AdvanceAdvData>& advData)
{
  //----------------------------------------------------------------
  // set udrift boundaries to zero
  if (advData->uDrift[0][0].nGrow() > 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      AMREX_D_TERM(advData->uDrift[lev][0].setBndry(0.0);
                   , advData->uDrift[lev][1].setBndry(0.0);
                   , advData->uDrift[lev][2].setBndry(0.0););
    }
  }

  //----------------------------------------------------------------
  // Get the gradient of Old and New phiV
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> gphiVOld(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> gphiVNew(
    finest_level + 1);
  int nGrow = 0; // No need for ghost face on gphiV
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      gphiVOld[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        1, nGrow, amrex::MFInfo(), factory);
      gphiVNew[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        1, nGrow, amrex::MFInfo(), factory);
    }
  }

  constexpr int do_avgDown = 0; // TODO or should I ?
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);
  getDiffusionOp()->computeGradient(
    GetVecOfArrOfPtrs(gphiVOld), {}, // don't need the laplacian out
    GetVecOfConstPtrs(getPhiVVect(AmrOldTime)), {}, bcRecPhiV[0], do_avgDown);
  getDiffusionOp()->computeGradient(
    GetVecOfArrOfPtrs(gphiVNew), {}, // don't need the laplacian out
    GetVecOfConstPtrs(getPhiVVect(AmrNewTime)), {}, bcRecPhiV[0], do_avgDown);

  //----------------------------------------------------------------
  // TODO : this assumes that all the ions are grouped together at th end ...
  auto bcRecIons =
    fetchBCRecArray(FIRSTSPEC + NUM_SPECIES - NUM_IONS, NUM_IONS);

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get CC ions t^{n+1/2} mobilities
    // TODO In the old version, there's a switch to only use an instant. value
    auto ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

    amrex::MultiFab mobH_cc(grids[lev], dmap[lev], NUM_IONS, 1);

    auto const& mob_o_ma = ldataOld_p->mob_cc.const_arrays();
    auto const& mob_n_ma = ldataNew_p->mob_cc.const_arrays();
    auto const& mob_h_ma = mobH_cc.arrays();

    amrex::ParallelFor(
      mobH_cc, mobH_cc.nGrowVect(), NUM_IONS,
      [mob_o_ma, mob_n_ma, mob_h_ma] AMREX_GPU_DEVICE(
        int box_no, int i, int j, int k, int n) noexcept {
        mob_h_ma[box_no](i, j, k, n) =
          0.5 * (mob_o_ma[box_no](i, j, k, n) + mob_n_ma[box_no](i, j, k, n));
      });
    amrex::Gpu::streamSynchronize();
    // Get the face centered ions mobility
    constexpr int doZeroVisc = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> mobH_ec =
      getDiffusivity(lev, 0, NUM_IONS, doZeroVisc, bcRecIons, mobH_cc);

    // Assemble the ions drift velocity
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto const& mob_ec_ma = mobH_ec[idim].const_arrays();
      auto const& gp_o_ma = gphiVOld[lev][idim].const_arrays();
      auto const& gp_n_ma = gphiVNew[lev][idim].const_arrays();
      auto const& Ud_Sp_ma = advData->uDrift[lev][idim].arrays();
      amrex::ParallelFor(
        mobH_ec[idim], amrex::IntVect(0), NUM_IONS,
        [mob_ec_ma, gp_o_ma, gp_n_ma, Ud_Sp_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          Ud_Sp_ma[box_no](i, j, k, n) =
            mob_ec_ma[box_no](i, j, k, n) * -0.5 *
            (gp_o_ma[box_no](i, j, k) + gp_n_ma[box_no](i, j, k));
        });
      // Shift outside (idim/lev)?
      amrex::Gpu::streamSynchronize();
    }
  }

  //----------------------------------------------------------------
  // Average down faces
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
    EB_average_down_faces(
      GetArrOfConstPtrs(advData->uDrift[lev]),
      GetArrOfPtrs(advData->uDrift[lev - 1]), refRatio(lev - 1), geom[lev - 1]);
#else
    average_down_faces(
      GetArrOfConstPtrs(advData->uDrift[lev]),
      GetArrOfPtrs(advData->uDrift[lev - 1]), refRatio(lev - 1), geom[lev - 1]);
#endif
  }

  // FillPatch Udrift on levels > 0
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (lev > 0) {
      fillpatchFaces(
        lev, GetArrOfPtrs(advData->uDrift[lev - 1]),
        GetArrOfPtrs(advData->uDrift[lev]), NUM_IONS, 1,
        &amrex::face_linear_interp);
    } else {
      AMREX_D_TERM(
        advData->uDrift[lev][0].FillBoundary(geom[lev].periodicity());
        , advData->uDrift[lev][1].FillBoundary(geom[lev].periodicity());
        , advData->uDrift[lev][2].FillBoundary(geom[lev].periodicity()));
    }
  }
}

void
PeleLM::ionDriftAddUmac(
  const int lev, const std::unique_ptr<AdvanceAdvData>& advData)
{
  // Add umac to the ions drift velocity to get the effective velocity
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

    auto const& umac_ma = advData->umac[lev][idim].const_arrays();
    auto const& Ud_Sp_ma = advData->uDrift[lev][idim].arrays();

    amrex::ParallelFor(
      advData->umac[lev][idim], advData->umac[lev][idim].nGrowVect(), NUM_IONS,
      [umac_ma, Ud_Sp_ma] AMREX_GPU_DEVICE(
        int box_no, int i, int j, int k, int n) noexcept {
        Ud_Sp_ma[box_no](i, j, k, n) += umac_ma[box_no](i, j, k);
      });
    // Shift outside?
    amrex::Gpu::streamSynchronize();
    advData->uDrift[lev][idim].FillBoundary(geom[lev].periodicity());
  }
}
