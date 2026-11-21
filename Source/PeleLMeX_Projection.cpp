#include <PeleLMeX.H>
#include <memory>

void
PeleLM::initialProjection()
{
  BL_PROFILE("PeleLMeX::initialProjection()");

  if (m_verbose != 0) {
    amrex::Vector<amrex::Real> velMax(AMREX_SPACEDIM);
    velMax = MLNorm0(
      GetVecOfConstPtrs(getVelocityVect(AmrNewTime)), 0, AMREX_SPACEDIM);
    amrex::Print() << " Initial velocity projection: ";
    amrex::Print() << AMREX_D_TERM(
      "  U: " << velMax[0] <<, "  V: " << velMax[1] <<,
      "  W: " << velMax[2] <<) "\n";
  }

  constexpr int incremental = 0;
  constexpr int nGhost = 0;
  constexpr amrex::Real dummy_dt = 1.0;

  // Get sigma (1/rho; empty when incompressible without mesh mapping)
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> sigma(finest_level + 1);
  if ((m_incompressible == 0) || m_mesh_mapping) {
    makeProjectionSigma(
      sigma, levelFieldVect(m_leveldata_new, &LevelData::state), DENSITY,
      dummy_dt, nGhost, false);
  }

  // Get velocity
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> vel(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    vel[lev] = std::make_unique<amrex::MultiFab>(
      m_leveldata_new[lev]->state, amrex::make_alias, VELX, AMREX_SPACEDIM);
    vel[lev]->setBndry(0.0);
    setInflowBoundaryVel(*vel[lev], lev, AmrNewTime);
#if AMREX_SPACEDIM == 2
    if (geom[lev].IsRZ()) {
      scaleProj_RZ(lev, *vel[lev]);
    };
#endif
  }

  // Get RHS cc: - divU (- \int{divU})
  amrex::Real Sbar = 0.0;
  amrex::Vector<amrex::MultiFab> rhs_cc;
  if ((m_incompressible == 0) && (m_has_divu != 0)) {
    // Ensure integral of RHS is zero for closed chamber
    if (m_closed_chamber != 0) {
      Sbar = MFSum(GetVecOfConstPtrs(getDivUVect(AmrNewTime)), 0);
      Sbar /= m_uncoveredVol; // Transform in Mean.
    }
    rhs_cc.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      rhs_cc.emplace_back(
        grids[lev], dmap[lev], 1, m_leveldata_new[lev]->divu.nGrow());
      amrex::MultiFab::Copy(
        rhs_cc[lev], m_leveldata_new[lev]->divu, 0, 0, 1,
        m_leveldata_new[lev]->divu.nGrow());
      if (m_closed_chamber != 0) {
        rhs_cc[lev].plus(-Sbar, 0, 1);
      }
#if AMREX_SPACEDIM == 2
      if (geom[lev].IsRZ()) {
        scaleProj_RZ(lev, rhs_cc[lev]);
      }
#endif
      rhs_cc[lev].mult(-1.0, 0, 1, rhs_cc[lev].nGrow());
    }
  }

  // Mesh mapping: convert velocity and rhs_cc to uniform (Xi) space
  if (m_mesh_mapping) {
    scaleMappedProjVel(GetVecOfPtrs(vel), true);
    if (!rhs_cc.empty()) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        m_mesh_map->multiply_by_detJ_cc(
          lev, rhs_cc[lev], 0, 1, rhs_cc[lev].nGrow());
      }
    }
  }

  checkNodalCompatibility("initial projection", GetVecOfConstPtrs(rhs_cc));

  doNodalProject(
    GetVecOfPtrs(vel), GetVecOfPtrs(sigma), GetVecOfPtrs(rhs_cc), {},
    incremental, dummy_dt);

  correctMappedNodalVel(GetVecOfPtrs(vel), GetVecOfPtrs(sigma));

  // Mesh mapping: convert projected velocity back to physical space
  scaleMappedProjVel(GetVecOfPtrs(vel), false);

  // Set back press and gpress to zero and restore divu
  // and rescale velocity if 2D-RZ
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    ldata_p->press.setVal(0.0);
    ldata_p->gp.setVal(0.0);
    if ((m_incompressible == 0) && (m_has_divu != 0)) {
      m_leveldata_new[lev]->divu.mult(-1.0, 0, 1, rhs_cc[lev].nGrow());
      // Restore divU integral
      if (m_closed_chamber != 0) {
        m_leveldata_new[lev]->divu.plus(Sbar, 0, 1);
      }
    }
#if AMREX_SPACEDIM == 2
    if (geom[lev].IsRZ()) {
      unscaleProj_RZ(lev, *vel[lev]);
    }
#endif
  }

  // R-Z (r*vel) and mesh mapping both leave the coarse velocity stale
  // under fine grids after the per-level unscale; sync it down from the
  // fine level to keep div(U) consistent across the C/F boundary.
  if (Geom(0).IsRZ() || m_mesh_mapping) {
    averageDownVelocity(AmrNewTime);
  }

  if (m_verbose != 0) {
    amrex::Vector<amrex::Real> velMax = MLNorm0(
      GetVecOfConstPtrs(getVelocityVect(AmrNewTime)), 0, AMREX_SPACEDIM);
    amrex::Print() << " >> After initial velocity projection: ";
    amrex::Print() << AMREX_D_TERM(
      "  U: " << velMax[0] <<, "  V: " << velMax[1] <<,
      "  W: " << velMax[2] <<) "\n";
  }
}

void
PeleLM::initialPressProjection()
{
  BL_PROFILE("PeleLMeX::initialPressProjection()");

  if (m_verbose != 0) {
    amrex::Print() << " Initial pressure projection \n";
  }

  constexpr amrex::Real dummy_dt = 1.0;
  constexpr int incremental = 0;
  constexpr int nGhost = 1;

  // Get sigma (1/rho; empty when incompressible without mesh mapping)
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> sigma;
  if ((m_incompressible == 0) || m_mesh_mapping) {
    sigma.resize(finest_level + 1);
    makeProjectionSigma(
      sigma, levelFieldVect(m_leveldata_new, &LevelData::state), DENSITY,
      dummy_dt, nGhost, false);
  }

  // Set the velocity to the gravity field
  amrex::Vector<amrex::MultiFab> vel;
  vel.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    vel.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, nGhost, amrex::MFInfo(),
      *m_factory[lev]);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      vel[lev].setVal(m_gravity[idim], idim, 1, 1);
    }
    vel[lev].setBndry(0.0);
    setInflowBoundaryVel(vel[lev], lev, AmrNewTime);
#if AMREX_SPACEDIM == 2
    if (geom[lev].IsRZ()) {
      scaleProj_RZ(lev, vel[lev]);
    }
#endif
  }

  // Mesh mapping: convert velocity to uniform (Xi) space
  scaleMappedProjVel(GetVecOfPtrs(vel), true);

  // Done without divU in IAMR
  doNodalProject(
    GetVecOfPtrs(vel), GetVecOfPtrs(sigma), {}, {}, incremental, dummy_dt);

  // Velocity is local to this function; no post-project unscaling needed.
}

void
PeleLM::velocityProjection(
  const int is_initIter, const TimeStamp a_rhoTime, const amrex::Real a_dt)
{
  BL_PROFILE("PeleLMeX::velocityProjection()");

  constexpr int nGhost = 0;
  const int incremental = (is_initIter) != 0 ? 1 : 0;

  // Density at a_rhoTime (low-Mach only), shared by the sigma build and the
  // gp correction below; the projection only changes velocity.
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhoHalf;
  if (m_incompressible == 0) {
    rhoHalf = getDensityVect(a_rhoTime);
  }

  // Get sigma: dt/rho^{a_rhoTime} (dt/m_rho incompressible); empty when
  // incompressible without mesh mapping (constant sigma in doNodalProject)
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> sigma(finest_level + 1);
  if ((m_incompressible == 0) || m_mesh_mapping) {
    makeProjectionSigma(sigma, GetVecOfPtrs(rhoHalf), 0, a_dt, nGhost, true);
  }

  // Under mesh mapping the pressure-gradient correction carries a 1/fac_i
  // factor per component because grad_phys p = (T^T / J) grad_Xi p, and
  // only the diagonal 1/fac_i part survives the face-aligned mapping.
  if (incremental == 0) {
    const bool lowMach = (m_incompressible == 0);
    const bool mapped = m_mesh_mapping;
    // Incompressible: constant dt/rho (using m_dt, as before)
    const amrex::Real soverrho_const = lowMach ? 0.0 : m_dt / m_rho;
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
      auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
      auto const& state_new_ma = ldataNew_p->state.arrays();
      auto const& gp_old_ma = ldataOld_p->gp.const_arrays();
      amrex::MultiArray4<amrex::Real const> rho_ma{};
      if (lowMach) {
        rho_ma = rhoHalf[lev]->const_arrays();
      }
      amrex::MultiArray4<amrex::Real const> fac_ma{};
      if (mapped) {
        fac_ma = m_mesh_map->fac_cc(lev).const_arrays();
      }
      amrex::ParallelFor(
        ldataNew_p->state, amrex::IntVect(0), AMREX_SPACEDIM,
        [=, dt = a_dt] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real> vel(state_new_ma[box_no], VELX);
          const amrex::Real soverrho =
            lowMach ? dt / rho_ma[box_no](i, j, k) : soverrho_const;
          if (mapped) {
            vel(i, j, k, n) += gp_old_ma[box_no](i, j, k, n) * soverrho /
                               fac_ma[box_no](i, j, k, n);
          } else {
            vel(i, j, k, n) += gp_old_ma[box_no](i, j, k, n) * soverrho;
          }
        });
    }
    amrex::Gpu::streamSynchronize();
  }

  // If incremental
  // define "vel" to be U^{np1*} - U^{n} rather than U^{np1*}
  if (incremental != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
      auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
      amrex::MultiFab::Subtract(
        ldataNew_p->state, ldataOld_p->state, VELX, VELX, AMREX_SPACEDIM, 0);
    }
  }

  // Get velocity
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> vel(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    vel[lev] = std::make_unique<amrex::MultiFab>(
      m_leveldata_new[lev]->state, amrex::make_alias, VELX, AMREX_SPACEDIM);
    eb_set_covered(*vel[lev], 0.0);
    vel[lev]->setBndry(0.0);
    if (incremental == 0) {
      setInflowBoundaryVel(*vel[lev], lev, AmrNewTime);
    }
#if AMREX_SPACEDIM == 2
    if (geom[lev].IsRZ()) {
      scaleProj_RZ(lev, *vel[lev]);
    }
#endif
  }

  // To ensure integral of RHS is zero for closed chamber, get mean divU
  amrex::Real SbarOld = 0.0;
  amrex::Real SbarNew = 0.0;
  if ((m_closed_chamber != 0) && (m_incompressible == 0)) {
    SbarNew = MFSum(GetVecOfConstPtrs(getDivUVect(AmrNewTime)), 0) /
              m_uncoveredVol; // Transform in Mean.
    if (incremental != 0) {
      SbarOld = MFSum(GetVecOfConstPtrs(getDivUVect(AmrOldTime)), 0) /
                m_uncoveredVol; // Transform in Mean.
    }
  }

  // Get RHS cc
  amrex::Vector<amrex::MultiFab> rhs_cc;
  if ((m_incompressible == 0) && (m_has_divu != 0)) {
    rhs_cc.reserve(finest_level + 1);
    if (incremental == 0) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
        rhs_cc.emplace_back(
          grids[lev], dmap[lev], 1, ldata_p->divu.nGrow(), amrex::MFInfo(),
          *m_factory[lev]);
        amrex::MultiFab::Copy(
          rhs_cc[lev], ldata_p->divu, 0, 0, 1, ldata_p->divu.nGrow());
        if (m_closed_chamber != 0) {
          rhs_cc[lev].plus(-SbarNew, 0, 1);
        }
        rhs_cc[lev].mult(-1.0, 0, 1, ldata_p->divu.nGrow());
        eb_set_covered(rhs_cc[lev], 0.0);
#if AMREX_SPACEDIM == 2
        if (geom[lev].IsRZ()) {
          scaleProj_RZ(lev, rhs_cc[lev]);
        }
#endif
      }
    } else {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
        auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
        rhs_cc.emplace_back(
          grids[lev], dmap[lev], 1, ldataOld_p->divu.nGrow(), amrex::MFInfo(),
          *m_factory[lev]);

        auto const& divu_o_ma = ldataOld_p->divu.const_arrays();
        auto const& divu_n_ma = ldataNew_p->divu.const_arrays();
        auto const& rhs_ma = rhs_cc[lev].arrays();

        amrex::ParallelFor(
          rhs_cc[lev], rhs_cc[lev].nGrowVect(),
          [divu_o_ma, divu_n_ma,
           rhs_ma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            rhs_ma[box_no](i, j, k) =
              -(divu_n_ma[box_no](i, j, k) - divu_o_ma[box_no](i, j, k));
          });
        if (m_closed_chamber != 0) {
          rhs_cc[lev].plus(SbarNew - SbarOld, 0, 1, ldataOld_p->divu.nGrow());
        }
#ifdef AMREX_USE_EB
        EB_set_covered(rhs_cc[lev], 0.0);
#endif
#if AMREX_SPACEDIM == 2
        if (geom[lev].IsRZ()) {
          scaleProj_RZ(lev, rhs_cc[lev]);
        }
#endif
      }
      amrex::Gpu::streamSynchronize();
    }
  }

  // Mesh mapping: scale U^* to uniform space (u_i *= J/fac_i), and
  // scale rhs_cc by J.  These are reversed after the nodal project
  // returns -- vel is aliased into state, so we must restore it, but
  // rhs_cc is function-local and doesn't need a restore.
  if (m_mesh_mapping) {
    scaleMappedProjVel(GetVecOfPtrs(vel), true);

    if (!rhs_cc.empty()) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        m_mesh_map->multiply_by_detJ_cc(
          lev, rhs_cc[lev], 0, 1, rhs_cc[lev].nGrow());
      }
    }
  }

  checkNodalCompatibility("velocity projection", GetVecOfConstPtrs(rhs_cc));

  doNodalProject(
    GetVecOfPtrs(vel), GetVecOfPtrs(sigma), GetVecOfPtrs(rhs_cc), {},
    incremental, a_dt);

  correctMappedNodalVel(GetVecOfPtrs(vel), GetVecOfPtrs(sigma));

  // Mesh mapping: scale U^{n+1} back to physical space (u_i *= fac_i/J).
  scaleMappedProjVel(GetVecOfPtrs(vel), false);

#if AMREX_SPACEDIM == 2
  // Unscaling New vel before adding back old one
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (geom[lev].IsRZ()) {
      unscaleProj_RZ(lev, *vel[lev]);
    }
  }
#endif

  // If incremental
  // define back to be U^{np1} by adding U^{n}
  if (incremental != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
      auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
      amrex::MultiFab::Add(
        ldataNew_p->state, ldataOld_p->state, VELX, VELX, AMREX_SPACEDIM, 0);
    }
  }

  // R-Z (r*vel) and mesh mapping both leave the coarse velocity stale
  // under fine grids after the per-level unscale; sync it down from the
  // fine level for cross-level consistency on the next step.
#if AMREX_SPACEDIM == 2
  if (Geom(0).IsRZ() || m_mesh_mapping) {
    averageDownVelocity(AmrNewTime);
  }
#else
  if (m_mesh_mapping) {
    averageDownVelocity(AmrNewTime);
  }
#endif
}

// Projection sigma, allocated here with a_nGhost ghost cells: a_dt/rho, or
// with mesh mapping a_dt/rho . J/fac_i^2 in component i. rho is component
// a_rhoComp of a_rho, or m_rho when incompressible (a_rho unused).
void
PeleLM::makeProjectionSigma(
  amrex::Vector<std::unique_ptr<amrex::MultiFab>>& a_sigma,
  const amrex::Vector<amrex::MultiFab*>& a_rho,
  const int a_rhoComp,
  const amrex::Real a_dt,
  const int a_nGhost,
  const bool a_setCovered)
{
  const int sigma_ncomp = m_mesh_mapping ? AMREX_SPACEDIM : 1;
  for (int lev = 0; lev <= finest_level; ++lev) {
    a_sigma[lev] = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], sigma_ncomp, a_nGhost, amrex::MFInfo(),
      *m_factory[lev]);
    auto const& sigma_ma = a_sigma[lev]->arrays();
    if (!m_mesh_mapping) {
      auto const& rho_ma = a_rho[lev]->const_arrays();
      amrex::ParallelFor(
        *a_rho[lev],
        [rho_ma, sigma_ma, rc = a_rhoComp,
         dt = a_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          sigma_ma[box_no](i, j, k) = dt / rho_ma[box_no](i, j, k, rc);
        });
    } else {
      auto const& fac_ma = m_mesh_map->fac_cc(lev).const_arrays();
      auto const& detJ_ma = m_mesh_map->detJ_cc(lev).const_arrays();
      if (m_incompressible == 0) {
        auto const& rho_ma = a_rho[lev]->const_arrays();
        amrex::ParallelFor(
          *a_sigma[lev], amrex::IntVect(0), AMREX_SPACEDIM,
          [=, rc = a_rhoComp, dt = a_dt] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k, int n) noexcept {
            const amrex::Real f = fac_ma[box_no](i, j, k, n);
            const amrex::Real dJ = detJ_ma[box_no](i, j, k);
            sigma_ma[box_no](i, j, k, n) =
              dt * dJ / (f * f * rho_ma[box_no](i, j, k, rc));
          });
      } else {
        const amrex::Real rho_const = m_rho;
        amrex::ParallelFor(
          *a_sigma[lev], amrex::IntVect(0), AMREX_SPACEDIM,
          [=, dt = a_dt] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k, int n) noexcept {
            const amrex::Real f = fac_ma[box_no](i, j, k, n);
            const amrex::Real dJ = detJ_ma[box_no](i, j, k);
            sigma_ma[box_no](i, j, k, n) = dt * dJ / (f * f * rho_const);
          });
      }
    }
#ifdef AMREX_USE_EB
    if (a_setCovered) {
      EB_set_covered(*a_sigma[lev], 0.0);
    }
#else
    amrex::ignore_unused(a_setCovered);
#endif
#if AMREX_SPACEDIM == 2
    if (geom[lev].IsRZ()) {
      scaleProj_RZ(lev, *a_sigma[lev]);
    }
#endif
  }
  amrex::Gpu::streamSynchronize();
}

// Mesh mapping: projection velocity to Xi space (a_toXi, u_i *= J/fac_i) or
// back (u_i *= fac_i/J). No-op without mapping. Ghost cells included:
// setInflowBoundaryVel() writes the inflow Dirichlet values there in
// physical units, and the nodal projector reads them as the boundary
// velocity. Leaving them unscaled makes the projection enforce
// u_phys = u_in * fac_n / J at the inlet instead of u_in.
void
PeleLM::scaleMappedProjVel(
  const amrex::Vector<amrex::MultiFab*>& a_vel, const bool a_toXi)
{
  if (!m_mesh_mapping) {
    return;
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& fac_ma = m_mesh_map->fac_cc(lev).const_arrays();
    auto const& detJ_ma = m_mesh_map->detJ_cc(lev).const_arrays();
    auto const& vel_ma = a_vel[lev]->arrays();
    amrex::ParallelFor(
      *a_vel[lev], a_vel[lev]->nGrowVect(), AMREX_SPACEDIM,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        const amrex::Real f = fac_ma[box_no](i, j, k, n);
        const amrex::Real dJ = detJ_ma[box_no](i, j, k);
        vel_ma[box_no](i, j, k, n) *= a_toXi ? dJ / f : f / dJ;
      });
  }
  amrex::Gpu::streamSynchronize();
}

// Correct the AmrWind-inherited limitation in AMReX's nodal projector:
// MLNodeLaplacian::mknewu collapses the anisotropic sigma to sigma_x
// (the first component of the stored 3-component sigma) when applying
// the velocity update to all three components.  That means doNodalProject
// left us with
//     u_i = u_entry_i - sigma_x . (grad phi)_i
// instead of the correct
//     u_i = u_entry_i - sigma_i . (grad phi)_i
// After doNodalProject, ldata_p->gp holds grad phi (cell-centered,
// AMREX_SPACEDIM components).  Apply the per-component correction here
// so the mesh-mapping path produces the right answer even without an
// AMReX-level fix for mknewu.  Under identity mapping all sigma_i are
// equal and the correction is exactly zero, preserving byte-identity.
//
// Gated on AMREX_MLNODELAP_HAS_MKNEWU_HA: when the AMReX fix is present
// (updateVelocity/getFluxes route through the anisotropic mknewu_ha
// kernel automatically), this post-hoc correction is unnecessary and
// is compiled out.  This keeps PeleLMeX working against both pre- and
// post-fix AMReX.
void
PeleLM::correctMappedNodalVel(
  const amrex::Vector<amrex::MultiFab*>& a_vel,
  const amrex::Vector<amrex::MultiFab*>& a_sigma)
{
#if !defined(AMREX_MLNODELAP_HAS_MKNEWU_HA)
  if (!m_mesh_mapping) {
    return;
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    auto const& sigma_ma = a_sigma[lev]->const_arrays();
    auto const& gp_ma = ldata_p->gp.const_arrays();
    auto const& vel_ma = a_vel[lev]->arrays();
    amrex::ParallelFor(
      *a_vel[lev], amrex::IntVect(0), AMREX_SPACEDIM,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        const amrex::Real sig_x = sigma_ma[box_no](i, j, k, 0);
        const amrex::Real sig_n = sigma_ma[box_no](i, j, k, n);
        vel_ma[box_no](i, j, k, n) +=
          (sig_x - sig_n) * gp_ma[box_no](i, j, k, n);
      });
  }
  amrex::Gpu::streamSynchronize();
#else
  amrex::ignore_unused(a_vel, a_sigma);
#endif
}

void
PeleLM::checkNodalCompatibility(
  const std::string& a_who, const amrex::Vector<const amrex::MultiFab*>& a_rhs)
{
  // Sealed box only: the NodalProjector forms div(U) + rhs_cc, and with no
  // inflow/outflow face the boundary term is zero so sum(rhs_cc) must
  // vanish.  With open faces the nodal boundary term is not reconstructed
  // here and the check is skipped.
  if (compatibilityCheckEnabled() && !a_rhs.empty() && !domainHasOpenFace()) {
    checkProjectionCompatibility(a_who, a_rhs, 0.0, 1.0, geom[0].IsRZ());
  }
}

void
PeleLM::doNodalProject(
  const amrex::Vector<amrex::MultiFab*>& a_vel,
  const amrex::Vector<amrex::MultiFab*>& a_sigma,
  const amrex::Vector<amrex::MultiFab*>& rhs_cc,
  const amrex::Vector<const amrex::MultiFab*>& rhs_nd,
  const int incremental,
  const amrex::Real scaling_factor)
{
  // Asserts
  AMREX_ASSERT(a_vel.size() == a_sigma.size());
  AMREX_ASSERT(rhs_cc.empty() || (a_vel.size() == rhs_cc.size()));
  AMREX_ASSERT(rhs_nd.empty() || (a_vel.size() == rhs_nd.size()));
  AMREX_ASSERT(a_vel[0]->nComp() == AMREX_SPACEDIM);

  amrex::LPInfo info;
  info.setMaxCoarseningLevel(m_nodal_mg_max_coarsening_level);

  // BCs
  std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc;
  std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (Geom(0).isPeriodic(idim)) {
      lobc[idim] = hibc[idim] = amrex::LinOpBCType::Periodic;
    } else {
      if (m_phys_bc.lo(idim) == amrex::PhysBCType::outflow) {
        lobc[idim] = amrex::LinOpBCType::Dirichlet;
      } else if (m_phys_bc.lo(idim) == amrex::PhysBCType::inflow) {
        lobc[idim] = amrex::LinOpBCType::inflow;
      } else {
        lobc[idim] = amrex::LinOpBCType::Neumann;
      }
      if (m_phys_bc.hi(idim) == amrex::PhysBCType::outflow) {
        hibc[idim] = amrex::LinOpBCType::Dirichlet;
      } else if (m_phys_bc.hi(idim) == amrex::PhysBCType::inflow) {
        hibc[idim] = amrex::LinOpBCType::inflow;
      } else {
        hibc[idim] = amrex::LinOpBCType::Neumann;
      }
    }
  }

  // Setup NodalProjector
  std::unique_ptr<Hydro::NodalProjector> nodal_projector;

  if ((m_incompressible != 0) && !m_mesh_mapping) {
    const amrex::Real constant_sigma = scaling_factor / m_rho;
    nodal_projector = std::make_unique<Hydro::NodalProjector>(
      a_vel, constant_sigma, Geom(0, finest_level), info);
  } else {
    // Mesh mapping (or low-Mach): sigma is a MultiFab.  Under mesh
    // mapping it carries AMREX_SPACEDIM components; velocityProjection
    // builds it that way.
    if (!rhs_cc.empty()) {
      nodal_projector = std::make_unique<Hydro::NodalProjector>(
        a_vel, GetVecOfConstPtrs(a_sigma), Geom(0, finest_level), info, rhs_cc,
        rhs_nd);
    } else {
      nodal_projector = std::make_unique<Hydro::NodalProjector>(
        a_vel, GetVecOfConstPtrs(a_sigma), Geom(0, finest_level), info);
    }
  }

  nodal_projector->setDomainBC(lobc, hibc);

#ifdef AMREX_USE_EB
  if (m_useEBinflow != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      nodal_projector->getLinOp().setEBInflowVelocity(
        lev, *getEBState(lev, VELX, AMREX_SPACEDIM, AmrNewTime));
    }
  }
#endif

#ifdef AMREX_USE_HYPRE
  if (
    nodal_projector->getMLMG().getBottomSolver() ==
    amrex::MLMG::BottomSolver::hypre) {
    nodal_projector->getMLMG().setHypreOptionsNamespace(
      m_hypre_namespace_nodal);
  }
#endif

  // Solve
  if (!m_mlmg_fail_plt_residuals) {
    nodal_projector->project(m_nodal_mg_rtol, m_nodal_mg_atol);
  } else {
    nodal_projector->setThrowException(true);
    nodal_projector->getMLMG().setConvergenceNormType(
      amrex::MLMGNormType::bnorm);
    try {
      nodal_projector->project(m_nodal_mg_rtol, m_nodal_mg_atol);
    } catch (const std::exception& e) {
      amrex::Print() << "\n";
      amrex::Print() << "  *** Nodal projection MLMG solve failed! ***\n";
      amrex::Print() << "  Error: " << e.what() << "\n";
      amrex::Print()
        << "  Dumping nodal projection residuals for debugging...\n";

      auto phi = nodal_projector->getPhi();
      auto rhs = nodal_projector->getRHSConst();

      WriteMLMGResidual(
        nodal_projector->getMLMG(), phi, rhs, "nodal_projection", m_nstep);

      amrex::Abort("MLMG solve for nodal_projection failed");
    }
  }

  auto phi = nodal_projector->getPhi();
  auto gphi = nodal_projector->getGradPhi();

  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->gp, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      amrex::Box const& tbx = mfi.tilebox();
      amrex::Box const& nbx = mfi.nodaltilebox();
      auto const& p_lev_arr = ldata_p->press.array(mfi);
      auto const& gp_lev_arr = ldata_p->gp.array(mfi);
      auto const& p_proj_arr = phi[lev]->const_array(mfi);
      auto const& gp_proj_arr = gphi[lev]->const_array(mfi);
      if (incremental != 0) {
        amrex::ParallelFor(
          tbx, AMREX_SPACEDIM,
          [gp_lev_arr,
           gp_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            gp_lev_arr(i, j, k, n) += gp_proj_arr(i, j, k, n);
          });
        amrex::ParallelFor(
          nbx, [p_lev_arr,
                p_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            p_lev_arr(i, j, k) += p_proj_arr(i, j, k);
          });
      } else {
        amrex::ParallelFor(
          tbx, AMREX_SPACEDIM,
          [gp_lev_arr,
           gp_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            gp_lev_arr(i, j, k, n) = gp_proj_arr(i, j, k, n);
          });
        amrex::ParallelFor(
          nbx, [p_lev_arr,
                p_proj_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            p_lev_arr(i, j, k) = p_proj_arr(i, j, k);
          });
      }
    }
  }

  // Average down grad P from fine to coarse.  Under mesh mapping gp is
  // the Xi-space gradient (dphi/dxi), so plain average_down on the uniform
  // Xi mesh is already mass-conservative; an extra Xi-space transform here
  // would double-weight gp and worsen the drift.
  averageDown(AmrNewTime, 0, AMREX_SPACEDIM, &LevelData::gp);
}

#if AMREX_SPACEDIM == 2
void
PeleLM::scaleProj_RZ(const int a_lev, amrex::MultiFab& a_mf)
{
  // Scale nodal projection cell-centered mfs by radius
  amrex::Box domain = geom[a_lev].Domain();
  auto BCRecVel = fetchBCRecArray(VELX, 1);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (BCRecVel[0].lo(idim) == amrex::BCType::ext_dir) {
      domain.growLo(idim, 1);
    }
    if (BCRecVel[0].hi(idim) == amrex::BCType::ext_dir) {
      domain.growHi(idim, 1);
    }
  }
  const amrex::Real dr = geom[a_lev].CellSize()[0];
  auto const& mf_ma = a_mf.arrays();
  amrex::ParallelFor(
    a_mf, a_mf.nGrowVect(), a_mf.nComp(),
    [mf_ma, dr,
     domain] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
      auto mf = mf_ma[box_no];
      if (domain.contains(i, j, k)) {
        mf(i, j, k, n) *= (static_cast<amrex::Real>(i) + 0.5) * dr;
      } else {
        mf(i, j, k, n) = 0.0;
      }
    });
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::unscaleProj_RZ(const int a_lev, amrex::MultiFab& a_mf)
{
  // Unscale nodal projection cell-centered mfs by radius
  const amrex::Box& domain = geom[a_lev].Domain();
  const amrex::Real dr = geom[a_lev].CellSize()[0];
  auto const& mf_ma = a_mf.arrays();
  amrex::ParallelFor(
    a_mf, a_mf.nGrowVect(), a_mf.nComp(),
    [mf_ma, dr,
     domain] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
      auto mf = mf_ma[box_no];
      if (domain.contains(i, j, k)) {
        mf(i, j, k, n) /= (static_cast<amrex::Real>(i) + 0.5) * dr;
      } else {
        mf(i, j, k, n) = 0.0;
      }
    });
  amrex::Gpu::streamSynchronize();
}
#endif
