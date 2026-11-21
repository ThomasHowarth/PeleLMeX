#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_Utils.H>
#include <hydro_utils.H>

void
PeleLM::computeVelocityAdvTerm(const std::unique_ptr<AdvanceAdvData>& advData)
{
  //----------------------------------------------------------------
  // Create temporary containers
  constexpr int nGrow_force = 1;
  amrex::Vector<amrex::MultiFab> divtau;
  divtau.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> velForces;
  velForces.reserve(finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> faces(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    divtau.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
    velForces.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow_force, amrex::MFInfo(),
      Factory(lev));
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
      faces[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
    }
  }

  //----------------------------------------------------------------
  // Get viscous forces
  constexpr int use_density = 0;
  computeDivTau(AmrOldTime, GetVecOfPtrs(divtau), use_density);

  // Add compensating pressure gradient for periodic channel flow
  if (m_do_periodic_channel != 0) {
    m_background_gp[m_periodic_channel_dir] =
      MFSum(GetVecOfConstPtrs(divtau), m_periodic_channel_dir) / m_uncoveredVol;
    if (m_verbose > 2) {
      amrex::Print() << "   ... Adding background pressure gradient in dir "
                     << m_periodic_channel_dir << ": "
                     << m_background_gp[m_periodic_channel_dir] << "\n";
    }
  }

  //----------------------------------------------------------------
  // Gather all the velocity forces
  // F = [ (gravity+...) - gradP + divTau ] / rho
  constexpr int add_gradP = 1;
  getVelForces(
    AmrOldTime, GetVecOfPtrs(divtau), GetVecOfPtrs(velForces), nGrow_force,
    add_gradP);

  auto bcRecVel = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  auto bcRecVel_d = convertToDeviceVector(bcRecVel);
  auto AdvTypeVel = fetchAdvTypeArray(VELX, AMREX_SPACEDIM);
  auto AdvTypeVel_d = convertToDeviceVector(AdvTypeVel);
  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    //----------------------------------------------------------------
    // Get divU
    constexpr bool divuTimesJ = false;
    amrex::MultiFab divu = getAdvectionDivu(lev, advData, divuTimesJ);

    // Under mesh mapping the extrapolation is done in Xi with the
    // Xi-velocity u~ = umac / J_fc and the physical divU; the fluxes
    // u~.q are rescaled by J_fc below (see getExtrapUmac).
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> umac_xi;
    const auto umacE = getExtrapUmac(lev, advData, umac_xi);
    if (m_debug_divu_mismatch != 0) {
      checkMACDivuMismatch(
        lev, "velocity prediction, mac_divu vs umac", divu, umacE);
    }

    //----------------------------------------------------------------
    // Compute the velocity fluxes
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      constexpr bool knownEdgeState = false;
      constexpr bool is_velocity = true;
      predictAdvFluxesOnBox(
        lev, mfi, umacE, 0, ldata_p->state, VELX, AMREX_SPACEDIM, fluxes[lev],
        0, faces[lev], 0, knownEdgeState, is_velocity, divu, velForces[lev], 0,
        bcRecVel, bcRecVel_d.dataPtr(), AdvTypeVel_d.dataPtr(), VELX);
    }
    scaleMappedAdvFluxes(lev, fluxes[lev], 0, AMREX_SPACEDIM);
#ifdef AMREX_USE_EB
    EB_set_covered_faces(GetArrOfPtrs(fluxes[lev]), 0.);
    EB_set_covered_faces(GetArrOfPtrs(faces[lev]), 0.);
#endif
  }

  //----------------------------------------------------------------
  // Average down fluxes to ensure C/F consistency
  averageDownFaces(fluxes);
  averageDownFaces(faces);

  //----------------------------------------------------------------
  // Fluxes divergence to get the velocity advection term. Convective form:
  // u.grad(u) = div(umac u) - u div(umac), with the divergence umac satisfies
  for (int lev = 0; lev <= finest_level; ++lev) {
    constexpr bool divuTimesJ = true;
    amrex::MultiFab divu = getAdvectionDivu(lev, advData, divuTimesJ);
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    std::unique_ptr<amrex::MultiFab> ebVel;
#ifdef AMREX_USE_EB
    if (m_useEBinflow != 0) {
      ebVel = getEBState(lev, VELX, AMREX_SPACEDIM, AmrOldTime);
    }
#endif
    constexpr bool setCovered = true;
    advDivergenceToAofS(
      lev, divu, fluxes[lev], faces[lev], AMREX_SPACEDIM,
      AdvTypeVel_d.dataPtr(), advData->AofS[lev], VELX, ldata_p->state, VELX,
      bcRecVel_d.dataPtr(), setCovered, ebVel.get(), ebVel.get());
  }
}

void
PeleLM::updateVelocity(const std::unique_ptr<AdvanceAdvData>& advData)
{
  //----------------------------------------------------------------
  // Compute t^n divTau
  amrex::Vector<amrex::MultiFab> divtau;
  divtau.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    divtau.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
  }
  constexpr int use_density = 0;
  constexpr amrex::Real CrankNicholsonFactor = 0.5;
  computeDivTau(
    AmrOldTime, GetVecOfPtrs(divtau), use_density, CrankNicholsonFactor);

  //----------------------------------------------------------------
  // Get velocity forcing at half time including lagged grad P term
  constexpr int nGrow_force = 1;
  amrex::Vector<amrex::MultiFab> velForces;
  velForces.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    velForces.emplace_back(grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow_force);
  }
  constexpr int add_gradP = 1;
  getVelForces(
    AmrHalfTime, GetVecOfPtrs(divtau), GetVecOfPtrs(velForces), nGrow_force,
    add_gradP);

  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get both old and new level_data
    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

    //----------------------------------------------------------------
    // Compute provisional new velocity
    // velForce holds: 1/\rho^{n+1/2} [(gravity+...)^{n+1/2} - \nabla pi^{n} +
    // 0.5 * divTau^{n}]
    auto const& state_old_ma = ldataOld_p->state.const_arrays();
    auto const& adv_aofs_ma = advData->AofS[lev].const_arrays();
    auto const& force_ma = velForces[lev].const_arrays();
    auto const& state_new_ma = ldataNew_p->state.arrays();
    amrex::ParallelFor(
      ldataOld_p->state, amrex::IntVect(0), AMREX_SPACEDIM,
      [state_old_ma, adv_aofs_ma, force_ma, state_new_ma,
       dt_loc =
         m_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real> vel_new(state_new_ma[box_no], VELX);
        amrex::Array4<amrex::Real const> vel_old(state_old_ma[box_no], VELX);
        amrex::Array4<amrex::Real const> adv_aofs(adv_aofs_ma[box_no], VELX);
        vel_new(i, j, k, n) =
          vel_old(i, j, k, n) +
          dt_loc * (adv_aofs(i, j, k, n) + force_ma[box_no](i, j, k, n));
      });
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::getScalarAdvForce(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{

  const auto aux_diffuse_v = convertToDeviceVector(m_DiffTypeAux);
  const auto* aux_diffuse_d = aux_diffuse_v.dataPtr();
  auto const* leosparm = eos_parms.device_parm();

  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n} data pointer
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataR_p = getLevelDataReactPtr(lev);

    auto const& state_ma = ldata_p->state.const_arrays();
    auto const& dn_ma = diffData->Dn[lev].const_arrays();
    auto const& adv_ma = advData->Forcing[lev].arrays();
    // No reaction data is allocated without reactions; the kernel skips it.
    auto const& r_ma = (m_do_react != 0) ? ldataR_p->I_R.const_arrays() : dn_ma;
    auto const& ext_ma = m_extSource[lev]->arrays();

    auto const& dn_aux_ma =
      (m_nAux > 0) ? diffData->Dn_aux[lev].const_arrays() : dn_ma;
    auto const& adv_aux_ma =
      (m_nAux > 0) ? advData->Forcing_aux[lev].arrays() : adv_ma;

    amrex::ParallelFor(
      advData->Forcing[lev],
      [state_ma, dn_ma, dn_aux_ma, r_ma, ext_ma, adv_ma, adv_aux_ma,
       aux_diffuse_d, leosparm, nAux = m_nAux, dp0dt = m_dp0dt,
       is_closed_ch = m_closed_chamber,
       do_react =
         m_do_react] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<const amrex::Real> rho(state_ma[box_no], DENSITY);
        amrex::Array4<const amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
        amrex::Array4<const amrex::Real> T(state_ma[box_no], TEMP);
        amrex::Array4<const amrex::Real> dn(dn_ma[box_no], 0);
        amrex::Array4<const amrex::Real> ddn(dn_ma[box_no], NUM_SPECIES + 1);
        amrex::Array4<const amrex::Real> dn_aux(dn_aux_ma[box_no], 0);
        amrex::Array4<const amrex::Real> r(r_ma[box_no], 0);
        amrex::Array4<amrex::Real> extRhoY(ext_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real> extRhoH(ext_ma[box_no], RHOH);
        amrex::Array4<amrex::Real> fY(adv_ma[box_no], 0);
        amrex::Array4<amrex::Real> fT(adv_ma[box_no], NUM_SPECIES);
        amrex::Array4<amrex::Real> fAux(adv_aux_ma[box_no], 0);
        buildAdvectionForcing(
          i, j, k, rho, rhoY, T, dn, ddn, r, extRhoY, extRhoH, dp0dt,
          is_closed_ch, do_react, fY, fT, fAux, dn_aux, aux_diffuse_d, nAux,
          leosparm);
      });
  }
  amrex::Gpu::streamSynchronize();

  // Fill forcing ghost cells
  if (advData->Forcing[0].nGrow() > 0) {
    fillpatch_forces(
      m_cur_time, GetVecOfPtrs(advData->Forcing), advData->Forcing[0].nGrow());
  }
  // Fill auxiliary forcing ghost cells
  if (m_nAux > 0 && advData->Forcing_aux[0].nGrow() > 0) {
    fillpatch_forces(
      m_cur_time, GetVecOfPtrs(advData->Forcing_aux),
      advData->Forcing_aux[0].nGrow());
  }
}

void
PeleLM::computeScalarAdvTerms(const std::unique_ptr<AdvanceAdvData>& advData)
{

  //----------------------------------------------------------------
  // Get the BCRecs and AdvectionTypes
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);
  auto bcRecSpec_d = convertToDeviceVector(bcRecSpec);
  auto AdvTypeSpec = fetchAdvTypeArray(FIRSTSPEC, NUM_SPECIES);
  auto AdvTypeSpec_d = convertToDeviceVector(AdvTypeSpec);
  auto bcRecTemp = fetchBCRecArray(TEMP, 1);
  auto bcRecTemp_d = convertToDeviceVector(bcRecTemp);
  auto AdvTypeTemp = fetchAdvTypeArray(TEMP, 1);
  auto AdvTypeTemp_d = convertToDeviceVector(AdvTypeTemp);
  auto bcRecRhoH = fetchBCRecArray(RHOH, 1);
  auto bcRecRhoH_d = convertToDeviceVector(bcRecRhoH);
  // Species and the adjacent RhoH, for the divergence step
  auto bcRecSpecRhoH = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES + 1);
  auto bcRecSpecRhoH_d = convertToDeviceVector(bcRecSpecRhoH);
  auto AdvTypeRhoH = fetchAdvTypeArray(RHOH, 1);
  auto AdvTypeRhoH_d = convertToDeviceVector(AdvTypeRhoH);
  auto bcRecAux = fetchBCRecAuxArray(0, m_nAux);
  auto bcRecAux_d = convertToDeviceVector(bcRecAux);
  auto AdvTypeAux = fetchAdvTypeAuxArray(0, m_nAux);
  auto AdvTypeAux_d = convertToDeviceVector(AdvTypeAux);
  //----------------------------------------------------------------
  // Create temporary containers
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes_aux(
    finest_level + 1);
  // Auxiliary face states, for the convective form of non-conservative ones
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> edgeState_aux(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], NUM_SPECIES + 1, 0, amrex::MFInfo(),
        Factory(lev)); // Species + RhoH
      if (m_nAux > 0) {
        fluxes_aux[lev][idim].define(
          amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
          dmap[lev], m_nAux, 0, amrex::MFInfo(),
          Factory(lev)); // auxiliary fields
        edgeState_aux[lev][idim].define(
          amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
          dmap[lev], m_nAux, 0, amrex::MFInfo(), Factory(lev));
      }
    }
  }

  //----------------------------------------------------------------
  // Loop over levels and get the fluxes
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get level data ptr Old
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    // Define edge state: Density + Species + RhoH + Temp
    constexpr int nGrow = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> edgeState;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      edgeState[idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], NUM_SPECIES + 3, nGrow, amrex::MFInfo(), Factory(lev));
    }

#ifdef PELE_USE_PLASMA
    //----------------------------------------------------------------
    // Assemble drift and mac velocities
    ionDriftAddUmac(lev, advData);
#endif

    //----------------------------------------------------------------
    // Get divU
    constexpr bool divuTimesJ = false;
    amrex::MultiFab divu = getAdvectionDivu(lev, advData, divuTimesJ);

    // Under mesh mapping the extrapolation is done in Xi with the
    // Xi-velocity u~ = umac / J_fc and the physical divU; the fluxes
    // u~.q are rescaled by J_fc below (see getExtrapUmac).
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> umac_xi;
    const auto umacE = getExtrapUmac(lev, advData, umac_xi);

    //----------------------------------------------------------------
#ifdef AMREX_USE_EB
    // Get EBFact & areafrac
    const auto& ebfact = EBFactory(lev);
    amrex::Array<const amrex::MultiCutFab*, AMREX_SPACEDIM> areafrac =
      ebfact.getAreaFrac();
#endif

    // Get the species edge state and advection term
    constexpr bool knownEdgeState = false;
    constexpr bool is_velocity = false;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
#ifdef PELE_USE_PLASMA
      // Uncharged species all at once
      predictAdvFluxesOnBox(
        lev, mfi, umacE, 0, ldata_p->state, FIRSTSPEC, NUM_SPECIES - NUM_IONS,
        fluxes[lev], 0, edgeState, 1, knownEdgeState, is_velocity, divu,
        advData->Forcing[lev], 0, bcRecSpec, bcRecSpec_d.dataPtr(),
        AdvTypeSpec_d.dataPtr(), FIRSTSPEC);

      // Ions one by one, with the drift velocity
      const auto uDrift = GetArrOfConstPtrs(advData->uDrift[lev]);
      for (int n = 0; n < NUM_IONS; ++n) {
        const int ion_idx = NUM_SPECIES - NUM_IONS + n;
        auto bcRecIons = fetchBCRecArray(FIRSTSPEC + ion_idx, 1);
        auto bcRecIons_d = convertToDeviceVector(bcRecIons);
        auto AdvTypeIons = fetchAdvTypeArray(FIRSTSPEC + ion_idx, 1);
        auto AdvTypeIons_d = convertToDeviceVector(AdvTypeIons);
        predictAdvFluxesOnBox(
          lev, mfi, uDrift, n, ldata_p->state, FIRSTSPEC + ion_idx, 1,
          fluxes[lev], ion_idx, edgeState, 1 + ion_idx, knownEdgeState,
          is_velocity, divu, advData->Forcing[lev], ion_idx, bcRecIons,
          bcRecIons_d.dataPtr(), AdvTypeIons_d.dataPtr(), FIRSTSPEC + ion_idx);
      }
#else
      predictAdvFluxesOnBox(
        lev, mfi, umacE, 0, ldata_p->state, FIRSTSPEC, NUM_SPECIES, fluxes[lev],
        0, edgeState, 1, knownEdgeState, is_velocity, divu,
        advData->Forcing[lev], 0, bcRecSpec, bcRecSpec_d.dataPtr(),
        AdvTypeSpec_d.dataPtr(), FIRSTSPEC);
#endif
    }
    // get the edge state for auxiliary components
    if (m_nAux > 0) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(ldata_p->auxiliaries, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        predictAdvFluxesOnBox(
          lev, mfi, umacE, 0, ldata_p->auxiliaries, 0, m_nAux, fluxes_aux[lev],
          0, edgeState_aux[lev], 0, knownEdgeState, is_velocity, divu,
          advData->Forcing_aux[lev], 0, bcRecAux, bcRecAux_d.dataPtr(),
          AdvTypeAux_d.dataPtr());
      }
      // Zero out fluxes, and the face states of the convective correction,
      // for non-advected auxiliaries
      for (int n = 0; n < m_nAux; ++n) {
        if (m_aux_advect[n] == 0) {
          for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            fluxes_aux[lev][idim].setVal(0.0, n, 1);
            edgeState_aux[lev][idim].setVal(0.0, n, 1);
          }
        }
      }
    }

    // Get the edge temperature, with its fluxes in place of RhoH's (they are
    // overwritten below)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      predictAdvFluxesOnBox(
        lev, mfi, umacE, 0, ldata_p->state, TEMP, 1, fluxes[lev], NUM_SPECIES,
        edgeState, NUM_SPECIES + 2, knownEdgeState, is_velocity, divu,
        advData->Forcing[lev], NUM_SPECIES, bcRecTemp, bcRecTemp_d.dataPtr(),
        AdvTypeTemp_d.dataPtr(), TEMP);
    }

    // Get the edge density, summing over the species, and the edge RhoH
    auto const* leosparm = eos_parms.device_parm();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.tilebox();

#ifdef AMREX_USE_EB
      auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
#endif

      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const amrex::Box& ebx = amrex::surroundingNodes(bx, idim);
        auto const& rho = edgeState[idim].array(mfi, 0);
        auto const& rhoY = edgeState[idim].const_array(mfi, 1);
        auto const& T = edgeState[idim].const_array(mfi, NUM_SPECIES + 2);
        auto const& rhoHm = edgeState[idim].array(mfi, NUM_SPECIES + 1);
#ifdef AMREX_USE_EB
        if (flagfab.getType(ebx) == amrex::FabType::covered) { // Covered boxes
          amrex::ParallelFor(
            ebx, [rho, rhoHm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              rho(i, j, k) = 0.0;
              rhoHm(i, j, k) = 0.0;
            });
        } else if (
          flagfab.getType(ebx) != amrex::FabType::regular) { // EB containing
                                                             // boxes
          const auto& afrac = areafrac[idim]->array(mfi);
          amrex::ParallelFor(
            ebx, [rhoHm, afrac, rhoY, T, rho,
                  leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              rho(i, j, k) = 0.0;
              if (afrac(i, j, k) <= 0.0) { // Covered faces
                rhoHm(i, j, k) = 0.0;
              } else {
                pele::physics::PhysicsType::eos_type::RY2R(
                  rhoY.cellData(i, j, k), rho(i, j, k));
                getRHmixGivenTY(i, j, k, rho, rhoY, T, rhoHm, leosparm);
              }
            });
        } else // Regular boxes
#endif
        {
          amrex::ParallelFor(
            ebx, [rho, rhoY, T, rhoHm,
                  leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              pele::physics::PhysicsType::eos_type::RY2R(
                rhoY.cellData(i, j, k), rho(i, j, k));
              getRHmixGivenTY(i, j, k, rho, rhoY, T, rhoHm, leosparm);
            });
        }
      }
    }

    // Finally get the RhoH fluxes from the known edge RhoH. Pass the Temp
    // forces again here, but they aren't used.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      constexpr bool knownEdgeRhoH = true;
      predictAdvFluxesOnBox(
        lev, mfi, umacE, 0, ldata_p->state, RHOH, 1, fluxes[lev], NUM_SPECIES,
        edgeState, NUM_SPECIES + 1, knownEdgeRhoH, is_velocity, divu,
        advData->Forcing[lev], NUM_SPECIES, bcRecRhoH, bcRecRhoH_d.dataPtr(),
        AdvTypeRhoH_d.dataPtr(), RHOH);
    }
    scaleMappedAdvFluxes(lev, fluxes[lev], 0, NUM_SPECIES + 1);
    if (m_nAux > 0) {
      scaleMappedAdvFluxes(lev, fluxes_aux[lev], 0, m_nAux);
    }
#ifdef AMREX_USE_EB
    EB_set_covered_faces(GetArrOfPtrs(fluxes[lev]), 0.);
#endif
  }

  //----------------------------------------------------------------
  // Average down fluxes to ensure C/F consistency
  averageDownFaces(fluxes);
  if (m_nAux > 0) {
    averageDownFaces(fluxes_aux);
    averageDownFaces(edgeState_aux);
  }

  //----------------------------------------------------------------
  // If balances are required, compute face domain integrals
  // using level 0 since we've averaged down the fluxes already
  if (m_sdcIter == m_nSDCmax) {
    if (m_do_massBalance != 0) {
      addMassFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }
    if (m_do_energyBalance != 0) {
      addRhoHFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }
    if (m_do_speciesBalance != 0) {
      addRhoYFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }

    if (m_do_patch_mfr != 0) {
      addRhoYFluxesPatch(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }
  }

#ifdef PELE_USE_PLASMA
  if (m_do_extraEFdiags) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int n = 0; n < NUM_IONS; ++n) {
        const int spec_idx = NUM_SPECIES - NUM_IONS + n;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> ionFlux;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          ionFlux[idim].reset(new amrex::MultiFab(
            fluxes[lev][idim], amrex::make_alias, spec_idx, 1));
        }
        average_face_to_cellcenter(
          *m_ionsFluxes[lev], n * AMREX_SPACEDIM, GetArrOfConstPtrs(ionFlux));
      }
    }
  }
#endif

  //----------------------------------------------------------------
  // Fluxes divergence to get the scalars advection term
  auto AdvTypeAll =
    fetchAdvTypeArray(FIRSTSPEC, NUM_SPECIES + 1); // Species+RhoH
  auto AdvTypeAll_d = convertToDeviceVector(AdvTypeAll);
  for (int lev = 0; lev <= finest_level; ++lev) {
    constexpr bool divuTimesJ = true;
    amrex::MultiFab divu = getAdvectionDivu(lev, advData, divuTimesJ);
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    constexpr bool setCovered = true;

    // Species and RhoH, all conservative: the face states are not used
    std::unique_ptr<amrex::MultiFab> ebVel, ebVal;
#ifdef AMREX_USE_EB
    if (m_useEBinflow != 0) {
      ebVel = getEBState(lev, VELX, AMREX_SPACEDIM, AmrOldTime);
      ebVal = getEBState(lev, FIRSTSPEC, NUM_SPECIES + 1, AmrOldTime);
    }
#endif
    advDivergenceToAofS(
      lev, divu, fluxes[lev], fluxes[lev], NUM_SPECIES + 1,
      AdvTypeAll_d.dataPtr(), advData->AofS[lev], FIRSTSPEC, ldata_p->state,
      FIRSTSPEC, bcRecSpecRhoH_d.dataPtr(), setCovered, ebVel.get(),
      ebVal.get());

    // Auxiliaries
    if (m_nAux > 0) {
      advDivergenceToAofS(
        lev, divu, fluxes_aux[lev], edgeState_aux[lev], m_nAux,
        AdvTypeAux_d.dataPtr(), advData->AofS_aux[lev], 0, ldata_p->auxiliaries,
        0, bcRecAux_d.dataPtr(), setCovered);
    }
  }

  //----------------------------------------------------------------
  // Sum over the species AofS to get the density advection term
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& aofsma = advData->AofS[lev].arrays();
    amrex::ParallelFor(
      advData->AofS[lev],
      [aofsma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        pele::physics::PhysicsType::eos_type::RY2R(
          aofsma[box_no].cellData(i, j, k), aofsma[box_no](i, j, k, DENSITY),
          FIRSTSPEC);
      });
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::updateDensity(const std::unique_ptr<AdvanceAdvData>& advData)
{
  // The caller fills the new density afterwards: no average down here
  constexpr bool avgDown = false;
  updateScalarComp(advData, DENSITY, 1, avgDown);
}

void
PeleLM::computePassiveAdvTerms(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const int state_comp,
  const int ncomp)
{
  //----------------------------------------------------------------
  // Get the BCRecs and AdvectionTypes
  auto bcRecPass = fetchBCRecArray(state_comp, ncomp);
  auto bcRecPass_d = convertToDeviceVector(bcRecPass);
  auto AdvTypePass = fetchAdvTypeArray(state_comp, ncomp);
  auto AdvTypePass_d = convertToDeviceVector(AdvTypePass);

  //----------------------------------------------------------------
  // Create temporary containers
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> edgeState(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], ncomp, 0, amrex::MFInfo(), Factory(lev));
      edgeState[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], ncomp, 0, amrex::MFInfo(), Factory(lev));
    }
  }

  //----------------------------------------------------------------
  // Loop over levels and get the fluxes
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get level data ptr Old
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    //----------------------------------------------------------------
    // Get divU
    constexpr bool divuTimesJ = false;
    amrex::MultiFab divu = getAdvectionDivu(lev, advData, divuTimesJ);

    // Under mesh mapping the extrapolation is done in Xi with the
    // Xi-velocity u~ = umac / J_fc and the physical divU; the fluxes
    // u~.q are rescaled by J_fc below (see getExtrapUmac).
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> umac_xi;
    const auto umacE = getExtrapUmac(lev, advData, umac_xi);

    // Get the passive variables edge state and advection term
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      // TODO: Find way to include diffusive forces for passive scalars that
      // diffuse
      constexpr bool knownEdgeState = false;
      constexpr bool is_velocity = false;
      predictAdvFluxesOnBox(
        lev, mfi, umacE, 0, ldata_p->state, state_comp, ncomp, fluxes[lev], 0,
        edgeState[lev], 0, knownEdgeState, is_velocity, divu, *m_extSource[lev],
        state_comp, bcRecPass, bcRecPass_d.dataPtr(), AdvTypePass_d.dataPtr());
    }
    scaleMappedAdvFluxes(lev, fluxes[lev], 0, ncomp);
#ifdef AMREX_USE_EB
    EB_set_covered_faces(GetArrOfPtrs(fluxes[lev]), 0.);
#endif
  }
  //----------------------------------------------------------------
  // Average down fluxes and edge state to ensure C/F consistency
  averageDownFaces(fluxes);
  averageDownFaces(edgeState);
  //----------------------------------------------------------------
  // Fluxes divergence to get the scalars advection term
  for (int lev = 0; lev <= finest_level; ++lev) {
    constexpr bool divuTimesJ = true;
    amrex::MultiFab divu = getAdvectionDivu(lev, advData, divuTimesJ);
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    constexpr bool setCovered = false;
    advDivergenceToAofS(
      lev, divu, fluxes[lev], edgeState[lev], ncomp, AdvTypePass_d.dataPtr(),
      advData->AofS[lev], state_comp, ldata_p->state, state_comp,
      bcRecPass_d.dataPtr(), setCovered);
  }
  // TODO: This assumes passive variables have no diffusive fluxes
  updateScalarComp(advData, state_comp, ncomp);
}

void
PeleLM::updateScalarComp(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const int state_comp,
  const int ncomp,
  const bool a_avgDown)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get level data ptr
    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

    auto const& state_old_ma = ldataOld_p->state.const_arrays();
    auto const& adv_aofs_ma = advData->AofS[lev].const_arrays();
    auto const& ext_ma = m_extSource[lev]->const_arrays();
    auto const& state_new_ma = ldataNew_p->state.arrays();

    amrex::ParallelFor(
      ldataOld_p->state, amrex::IntVect(0), ncomp,
      [state_old_ma, adv_aofs_ma, ext_ma, state_new_ma, state_comp,
       dt =
         m_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real> state_new_arr(
          state_new_ma[box_no], state_comp);
        amrex::Array4<amrex::Real const> state_old_arr(
          state_old_ma[box_no], state_comp);
        amrex::Array4<amrex::Real const> adv_aofs_arr(
          adv_aofs_ma[box_no], state_comp);
        amrex::Array4<amrex::Real const> ext_arr(ext_ma[box_no], state_comp);
        state_new_arr(i, j, k, n) =
          state_old_arr(i, j, k, n) +
          dt * (adv_aofs_arr(i, j, k, n) + ext_arr(i, j, k, n));
      });
  }
  amrex::Gpu::streamSynchronize();
  if (a_avgDown) {
    averageDown(AmrNewTime, state_comp, ncomp);
  }
}

amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>
PeleLM::getExtrapUmac(
  int a_lev,
  const std::unique_ptr<AdvanceAdvData>& advData,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_scratch)
{
  amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> r;
  if (!m_mesh_mapping) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      r[idim] = &advData->umac[a_lev][idim];
    }
    return r;
  }

  // umac = J.u~ after the MAC projection (macProject); the Godunov
  // extrapolation advances the state in Xi and needs the Xi-velocity
  // u~ = u_d / fac_d = umac / J_fc on each face.  The metric MultiFabs
  // carry at least as many ghost faces as umac.
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    const amrex::MultiFab& um = advData->umac[a_lev][idim];
    a_scratch[idim].define(
      um.boxArray(), um.DistributionMap(), 1, um.nGrowVect(), amrex::MFInfo(),
      um.Factory());
    amrex::MultiFab::Copy(a_scratch[idim], um, 0, 0, 1, um.nGrowVect());
    amrex::MultiFab::Divide(
      a_scratch[idim], m_mesh_map->detJ_fc(a_lev, idim), 0, 0, 1,
      um.nGrowVect());
    r[idim] = &a_scratch[idim];
  }
  return r;
}

void
PeleLM::scaleMappedAdvFluxes(
  int a_lev,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_fluxes,
  int a_comp,
  int a_ncomp)
{
  if (!m_mesh_mapping) {
    return;
  }
  // Fluxes were formed with u~: F = u~.q.  Multiply by J_fc so that the Xi
  // divergence of F is J.(physical divergence), the scaling advFluxDivergence
  // and the J.divU correction assume; AofS is divided by J afterwards.
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    auto const& f_ma = a_fluxes[idim].arrays();
    auto const& J_ma = m_mesh_map->detJ_fc(a_lev, idim).const_arrays();
    amrex::ParallelFor(
      a_fluxes[idim], amrex::IntVect(0), a_ncomp,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        f_ma[box_no](i, j, k, a_comp + n) *= J_ma[box_no](i, j, k);
      });
  }
  amrex::Gpu::streamSynchronize();
}

amrex::MultiFab
PeleLM::getAdvectionDivu(
  int a_lev, const std::unique_ptr<AdvanceAdvData>& advData, bool a_timesJ)
{
  amrex::MultiFab divu(
    grids[a_lev], dmap[a_lev], 1, m_nGrowdivu, amrex::MFInfo(), Factory(a_lev));
  // The face velocity satisfies div(umac) = mac_divu (S at t^{n+1/2}, less
  // theta dp0/dt in a closed chamber): the prediction and the convective-form
  // correction both need that divergence, not the old-time S.
  if (m_incompressible != 0) {
    divu.setVal(0.0);
  } else {
    amrex::MultiFab::Copy(divu, advData->mac_divu[a_lev], 0, 0, 1, m_nGrowdivu);
  }
  // Mapped fluxes are J.u~.q: the convective-form correction needs J.divU
  if (a_timesJ && m_mesh_mapping) {
    m_mesh_map->multiply_by_detJ_cc(a_lev, divu, 0, 1, m_nGrowdivu);
  }
  return divu;
}

void
PeleLM::predictAdvFluxesOnBox(
  int a_lev,
  amrex::MFIter& a_mfi,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_umac,
  int a_umac_comp,
  const amrex::MultiFab& a_state,
  int a_state_comp,
  int a_ncomp,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_fluxes,
  int a_flux_comp,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_edges,
  int a_edge_comp,
  bool a_knownEdgeState,
  bool a_isVelocity,
  const amrex::MultiFab& a_divu,
  const amrex::MultiFab& a_force,
  int a_force_comp,
  const amrex::Vector<amrex::BCRec>& a_bcRec,
  const amrex::BCRec* a_bcRec_d,
  const int* a_advType_d,
  int a_ebInflowComp)
{
  const amrex::Box& bx = a_mfi.tilebox();
  AMREX_D_TERM(
    auto const& umac = a_umac[0]->const_array(a_mfi, a_umac_comp);
    , auto const& vmac = a_umac[1]->const_array(a_mfi, a_umac_comp);
    , auto const& wmac = a_umac[2]->const_array(a_mfi, a_umac_comp);)
  AMREX_D_TERM(
    auto const& fx = a_fluxes[0].array(a_mfi, a_flux_comp);
    , auto const& fy = a_fluxes[1].array(a_mfi, a_flux_comp);
    , auto const& fz = a_fluxes[2].array(a_mfi, a_flux_comp);)
  AMREX_D_TERM(
    auto const& edgex = a_edges[0].array(a_mfi, a_edge_comp);
    , auto const& edgey = a_edges[1].array(a_mfi, a_edge_comp);
    , auto const& edgez = a_edges[2].array(a_mfi, a_edge_comp);)
  constexpr bool fluxes_are_area_weighted = false;
#ifdef AMREX_USE_EB
  // State on EB inflow faces, when there is EB inflow and the caller asks
  amrex::FArrayBox ebInflow;
  amrex::Array4<amrex::Real const> ebInflow_arr{};
  if (m_useEBinflow != 0 && a_ebInflowComp >= 0) {
    ebInflow = getEBState(a_mfi, a_lev, a_ebInflowComp, a_ncomp, AmrOldTime);
    ebInflow_arr = ebInflow.const_array();
  }
#else
  amrex::ignore_unused(a_ebInflowComp);
#endif
  HydroUtils::ComputeFluxesOnBoxFromState(
    bx, a_ncomp, a_mfi, a_state.const_array(a_mfi, a_state_comp),
    AMREX_D_DECL(fx, fy, fz), AMREX_D_DECL(edgex, edgey, edgez),
    a_knownEdgeState, AMREX_D_DECL(umac, vmac, wmac), a_divu.const_array(a_mfi),
    a_force.const_array(a_mfi, a_force_comp), geom[a_lev], m_dt, a_bcRec,
    a_bcRec_d, a_advType_d,
#ifdef AMREX_USE_EB
    EBFactory(a_lev), ebInflow_arr,
#endif
    m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, a_isVelocity,
    fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
}

void
PeleLM::advDivergenceToAofS(
  int a_lev,
  amrex::MultiFab& a_divu,
  const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_fluxes,
  const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_faceState,
  int a_ncomp,
  const int* a_advType_d,
  amrex::MultiFab& a_aofs,
  int a_aofs_comp,
  const amrex::MultiFab& a_state,
  int a_state_comp,
  const amrex::BCRec* a_bcRec_d,
  bool a_setCovered,
  const amrex::MultiFab* a_ebVel,
  const amrex::MultiFab* a_ebVal)
{
  constexpr bool fluxes_are_area_weighted = false;
#ifdef AMREX_USE_EB
  // Divergence in a temporary with ghost cells, then redistributed into AofS
  constexpr int nGrow_divTmp = 3;
  amrex::MultiFab divTmp(
    grids[a_lev], dmap[a_lev], a_ncomp, nGrow_divTmp, amrex::MFInfo(),
    EBFactory(a_lev));
  divTmp.setVal(0.0);
  advFluxDivergence(
    a_lev, divTmp, 0, a_divu, GetArrOfConstPtrs(a_fluxes), 0,
    GetArrOfConstPtrs(a_faceState), 0, a_ncomp, a_advType_d, -1.0,
    fluxes_are_area_weighted, a_ebVel, a_ebVal);
  divTmp.FillBoundary(geom[a_lev].periodicity());
  redistribute(
    a_lev, m_dt, divTmp, 0, a_aofs, a_aofs_comp, a_state, a_state_comp, a_ncomp,
    a_bcRec_d, geom[a_lev], m_adv_redist_type);
  if (a_setCovered) {
    EB_set_covered(a_aofs, 0.0);
  }
#else
  amrex::ignore_unused(a_state, a_state_comp, a_bcRec_d, a_setCovered);
  advFluxDivergence(
    a_lev, a_aofs, a_aofs_comp, a_divu, GetArrOfConstPtrs(a_fluxes), 0,
    GetArrOfConstPtrs(a_faceState), 0, a_ncomp, a_advType_d, -1.0,
    fluxes_are_area_weighted, a_ebVel, a_ebVal);
#endif
  // Under mesh mapping the Xi-space divergence is J.(physical divergence):
  // restore the physical advection term
  if (m_mesh_mapping) {
    m_mesh_map->divide_by_detJ_cc(a_lev, a_aofs, a_aofs_comp, a_ncomp);
  }
}
