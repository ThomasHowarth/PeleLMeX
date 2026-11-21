#include <PeleLMeX.H>
#include <memory>

#ifdef PELE_USE_SPRAY
#include "SprayParticles.H"
#endif

pele::physics::PeleParams<pele::physics::transport::TransParm<
  pele::physics::PhysicsType::eos_type,
  pele::physics::PhysicsType::transport_type>>
  PeleLM::trans_parms;

pele::physics::PeleParams<
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>>
  PeleLM::eos_parms;

PeleLM::PeleLM() = default;

PeleLM::~PeleLM()
{
  if (m_incompressible == 0) {
    trans_parms.deallocate();
    eos_parms.deallocate();
    if (m_reactor) {
      m_reactor->close();
    }
  }

  closeTempFile();
  typical_values.clear();
  freeProbParm();
  delete prob_parm;
  amrex::The_Arena()->free(prob_parm_d);
  m_initial_ba.clear();
  m_regrid_ba.clear();
#ifdef PELE_USE_SPRAY
  SprayParticleContainer::SprayCleanUp();
  GhostPC.reset();
  VirtPC.reset();
  SprayPC.reset();
#endif
#ifdef PELE_USE_SOOT
  cleanupSootModel();
#endif
}

PeleLM::LevelData*
PeleLM::getLevelDataPtr(
  const int lev, const TimeStamp a_time, const int /*useUMac*/)
{
  AMREX_ASSERT(
    a_time == AmrOldTime || a_time == AmrNewTime || a_time == AmrHalfTime);
  if (a_time == AmrOldTime) {
    return m_leveldata_old[lev].get();
  }
  if (a_time == AmrNewTime) {
    return m_leveldata_new[lev].get();
  }
  m_leveldata_floating = makeLevelData(grids[lev], dmap[lev], *m_factory[lev]);
  const amrex::Real time = getTime(lev, a_time);
  fillpatch_state(lev, time, m_leveldata_floating->state, m_nGrowState);
  if (m_nAux > 0) {
    fillpatch_aux(lev, time, m_leveldata_floating->auxiliaries, m_nGrowState);
  }
  return m_leveldata_floating.get();
}

PeleLM::LevelDataReact*
PeleLM::getLevelDataReactPtr(const int lev)
{
  if (m_do_react != 0) {
    return m_leveldatareact[lev].get();
  }
  return nullptr;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getAliasVect(
  const TimeStamp a_time,
  amrex::MultiFab LevelData::* a_field,
  const int a_scomp,
  const int a_ncomp)
{
  auto& ldata = (a_time == AmrOldTime) ? m_leveldata_old : m_leveldata_new;
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(
      std::make_unique<amrex::MultiFab>(
        (*ldata[lev]).*a_field, amrex::make_alias, a_scomp, a_ncomp));
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getFieldVect(
  const TimeStamp a_time, amrex::MultiFab LevelData::* a_field)
{
  return levelFieldVect(
    (a_time == AmrOldTime) ? m_leveldata_old : m_leveldata_new, a_field);
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getStateVect(const TimeStamp a_time)
{
  return getAliasVect(a_time, &LevelData::state, 0, nCompState());
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getVelocityVect(const TimeStamp a_time)
{
  return getAliasVect(a_time, &LevelData::state, VELX, AMREX_SPACEDIM);
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getSpeciesVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  return getAliasVect(a_time, &LevelData::state, FIRSTSPEC, NUM_SPECIES);
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getDensityVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  if (a_time == AmrOldTime || a_time == AmrNewTime) {
    return getAliasVect(a_time, &LevelData::state, DENSITY, 1);
  }
  // Intermediate time: fillpatch a temporary
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Real time = getTime(lev, a_time);
    r.push_back(
      std::make_unique<amrex::MultiFab>(
        grids[lev], dmap[lev], 1, m_nGrowState));
    fillpatch_density(lev, time, *(r[lev]), 0, m_nGrowState);
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getTempVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  return getAliasVect(a_time, &LevelData::state, TEMP, 1);
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getRhoHVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  return getAliasVect(a_time, &LevelData::state, RHOH, 1);
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getDivUVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  return getFieldVect(a_time, &LevelData::divu);
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getDiffusivityVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  return getFieldVect(a_time, &LevelData::diff_cc);
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getViscosityVect(const TimeStamp a_time)
{
  return getFieldVect(a_time, &LevelData::visc_cc);
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getIRVect()
{
  return levelFieldVect(m_leveldatareact, &LevelDataReact::I_R);
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getAuxVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_nAux > 0);
  return getAliasVect(a_time, &LevelData::auxiliaries, 0, m_nAux);
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getAuxDiffusivityVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_nAux > 0);
  return getFieldVect(a_time, &LevelData::diff_aux_cc);
}

void
PeleLM::averageDownState(const TimeStamp a_time)
{
  // Velocity components need the Xi-space-conservative restriction;
  // delegate to averageDownVelocity (which is mapping-aware).
  averageDownVelocity(a_time);

  // Remaining components AFTER velocity, if any (compressible runs).
  const int nc_rest = nCompState() - AMREX_SPACEDIM;
  if (nc_rest > 0) {
    if (m_mesh_mapping) {
      // detJ-weighted (volume-conservative) restriction of [DENSITY..RHOH];
      // linear per component, so rho = sum(rhoY) is preserved. Components
      // after RHOH (TEMP, RHORT, aux) use arithmetic and the derived ones
      // are overwritten by the EOS recompute below.
      averageDownMetricWeighted(a_time, DENSITY, RHOH - DENSITY + 1, false);

      const int nc_after = NVAR - (RHOH + 1);
      if (nc_after > 0) {
        averageDown(a_time, RHOH + 1, nc_after);
      }

      // Recompute TEMP/RHORT from the restricted densities via the EOS.
      // Done only on the mesh-mapping path here; applying this
      // (more thermodynamically consistent) treatment to the non-mapped
      // path too is a deferred enhancement -- see the tracking GitHub
      // issue -- since it would shift existing results and add cost.
      setTemperature(a_time);
      setThermoPress(a_time);
    } else {
      averageDown(a_time, AMREX_SPACEDIM, nc_rest);
    }
  }
}

void
PeleLM::averageDownScalars(const TimeStamp a_time)
{
#ifdef PELE_USE_PLASMA
  constexpr int nScal = NUM_SPECIES + 5;
#else
  constexpr int nScal = NUM_SPECIES + 3;
#endif

  if (m_mesh_mapping) {
    // detJ-weighted restriction of [DENSITY..RHOH]; the rest of the
    // nScal block (TEMP, RHORT/plasma extras) uses arithmetic, with the
    // derived state restored by the EOS recompute below.
    averageDownMetricWeighted(a_time, DENSITY, RHOH - DENSITY + 1, false);

    const int nc_dens = RHOH - DENSITY + 1; // rho + NUM_SPECIES rhoY + rhoH
    const int nc_after = nScal - nc_dens;
    if (nc_after > 0) {
      averageDown(a_time, DENSITY + nc_dens, nc_after);
    }

    // Recompute TEMP/RHORT from the restricted densities via the EOS
    // (mesh-mapping path only; see averageDownState for the note on the
    // deferred generalization to the non-mapped path).
    setTemperature(a_time);
    setThermoPress(a_time);
    return;
  }

  averageDown(a_time, DENSITY, nScal);
}

void
PeleLM::averageDownAux(const TimeStamp a_time)
{
  averageDown(a_time, 0, m_nAux, &LevelData::auxiliaries);
}

void
PeleLM::averageDown(
  const TimeStamp a_time,
  const int scomp,
  const int ncomp,
  amrex::MultiFab LevelData::* a_field)
{
  averageDown(getFieldVect(a_time, a_field), scomp, ncomp);
}

void
PeleLM::averageDown(
  amrex::Vector<amrex::MultiFab*> const& a_mf, const int scomp, const int ncomp)
{
  for (int lev = finest_level; lev > 0; --lev) {
    eb_average_down(
      *a_mf[lev], *a_mf[lev - 1], scomp, ncomp, refRatio(lev - 1));
  }
}

void
PeleLM::averageDownFaces(
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_faces)
{
  for (int lev = finest_level; lev > 0; --lev) {
    eb_average_down_faces(
      GetArrOfConstPtrs(a_faces[lev]), GetArrOfPtrs(a_faces[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
  }
}

void
PeleLM::rebuildMappedInterps()
{
  // Long-lived interp objects (one per C/F pair) so the address stays
  // stable: AMReX's TheFPinfo cache holds a raw pointer to the interp
  // across all fillpatch calls between regrids.
  if (!m_mesh_mapping || (m_mesh_map == nullptr)) {
    m_mapped_interps.clear();
    return;
  }
  // Size from the MeshMap's defined metric levels, NOT finest_level:
  // during regrid AmrCore updates finest_level only after the
  // MakeNewLevel*/RemakeLevel calls, so finest_level is stale here and a
  // pair would be missed for a newly added top level.
  const int n_pairs = amrex::max<int>(0, m_mesh_map->num_levels() - 1);
  m_mapped_interps.clear();
  m_mapped_interps.resize(n_pairs);
  // Component layout for the interpolator's weights (see its class doc).
  // Incompressible state has only velocity, so leave the density range empty.
  const int dens_start = (m_incompressible != 0) ? -1 : DENSITY;
  const int dens_end = (m_incompressible != 0) ? -2 : RHOH;
  for (int c = 0; c < n_pairs; ++c) {
    // Only build a pair whose two levels both have defined metrics;
    // clear_level() empties a level's MultiFabs without shrinking the
    // vector, so num_levels() can include cleared (coarsened-away) levels.
    if (m_mesh_map->detJ_cc(c).empty() || m_mesh_map->detJ_cc(c + 1).empty()) {
      continue; // leave entry null -> getInterpolator falls back to legacy
    }
    m_mapped_interps[c] = std::make_unique<MeshMappedCellConsInterp>(
      &m_mesh_map->detJ_cc(c), &m_mesh_map->fac_cc(c),
      &m_mesh_map->detJ_cc(c + 1), &m_mesh_map->fac_cc(c + 1), VELX,
      AMREX_SPACEDIM, dens_start, dens_end);
  }
}

void
PeleLM::averageDownVelocity(const TimeStamp a_time)
{
  if (m_mesh_mapping) {
    // Mass-conservative restriction under mesh mapping: weight by
    // detJ / fac_i (u_xi_i = v_i * detJ / fac_i). Required to preserve
    // discrete div-free across the C/F boundary.
    averageDownMetricWeighted(a_time, VELX, AMREX_SPACEDIM, true);
  } else {
    averageDown(a_time, VELX, AMREX_SPACEDIM);
  }
}

amrex::MultiFab
PeleLM::metricWeightedToXi(
  amrex::MultiFab const& a_state,
  const int a_lev,
  const int scomp,
  const int ncomp,
  const bool divide_by_fac)
{
  // Copy of a_state components [scomp, scomp + ncomp) multiplied by detJ,
  // and divided by fac_n when divide_by_fac (fac_cc is only read then).
  amrex::MultiFab xi(
    a_state.boxArray(), a_state.DistributionMap(), ncomp, 0, amrex::MFInfo(),
    a_state.Factory());
  amrex::MultiFab::Copy(xi, a_state, scomp, 0, ncomp, 0);
  amrex::MultiArray4<amrex::Real const> fac_ma{};
  if (divide_by_fac) {
    fac_ma = m_mesh_map->fac_cc(a_lev).const_arrays();
  }
  auto const& detJ_ma = m_mesh_map->detJ_cc(a_lev).const_arrays();
  auto const& xi_ma = xi.arrays();
  amrex::ParallelFor(
    xi, amrex::IntVect(0), ncomp,
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
      if (divide_by_fac) {
        xi_ma[box_no](i, j, k, n) *=
          detJ_ma[box_no](i, j, k) / fac_ma[box_no](i, j, k, n);
      } else {
        xi_ma[box_no](i, j, k, n) *= detJ_ma[box_no](i, j, k);
      }
    });
  return xi;
}

void
PeleLM::averageDownMetricWeighted(
  const TimeStamp a_time,
  const int scomp,
  const int ncomp,
  const bool divide_by_fac)
{
  // Stage state components [scomp, scomp + ncomp) in Xi-space (see
  // metricWeightedToXi), arithmetic-average (which IS volume-conservative on
  // the uniform Xi mesh), and convert back to physical space on the coarse
  // level. The coarse level is staged too, so average_down overwrites only
  // the covered region and the rest round-trips. No divide guard needed
  // (unlike MeshMappedCellConsInterp, which operates on seeded temporaries):
  // detJ/fac are the MeshMap metrics, strictly positive by construction and
  // defined on every valid cell.
  AMREX_ASSERT(m_mesh_mapping && (m_mesh_map != nullptr));
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::MultiFab& stateCrse = getLevelDataPtr(lev - 1, a_time)->state;
    amrex::MultiFab xi_fine = metricWeightedToXi(
      getLevelDataPtr(lev, a_time)->state, lev, scomp, ncomp, divide_by_fac);
    amrex::MultiFab xi_crse =
      metricWeightedToXi(stateCrse, lev - 1, scomp, ncomp, divide_by_fac);

    eb_average_down(xi_fine, xi_crse, 0, ncomp, refRatio(lev - 1));

    // fac_cc is only read (and fetched) when divide_by_fac
    amrex::MultiArray4<amrex::Real const> fac_ma{};
    if (divide_by_fac) {
      fac_ma = m_mesh_map->fac_cc(lev - 1).const_arrays();
    }
    auto const& detJ_ma = m_mesh_map->detJ_cc(lev - 1).const_arrays();
    auto const& xi_ma = xi_crse.const_arrays();
    auto const& state_ma = stateCrse.arrays();
    amrex::ParallelFor(
      xi_crse, amrex::IntVect(0), ncomp,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        if (divide_by_fac) {
          state_ma[box_no](i, j, k, scomp + n) = xi_ma[box_no](i, j, k, n) *
                                                 fac_ma[box_no](i, j, k, n) /
                                                 detJ_ma[box_no](i, j, k);
        } else {
          state_ma[box_no](i, j, k, scomp + n) =
            xi_ma[box_no](i, j, k, n) / detJ_ma[box_no](i, j, k);
        }
      });
    // Before the Xi-space temporaries go out of scope.
    amrex::Gpu::streamSynchronize();
  }
}

void
PeleLM::averageDownReaction()
{
  // I_R is only allocated when reactions are on
  if (m_do_react == 0) {
    return;
  }
  averageDown(getIRVect(), 0, nCompIR());
}

#ifdef PELE_USE_PLASMA
amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getPhiVVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  return getAliasVect(a_time, &LevelData::state, PHIV, 1);
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getnEDiffusivityVect(const TimeStamp a_time)
{
  return getFieldVect(a_time, &LevelData::diffE_cc);
}
#endif
