#include <PeleLMeX.H>
#include <PeleLMeX_BCfill.H>
#include <PeleLMeX_VOF_K.H>
#include <PeleLMeX_VOF_Unsplit_K.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>

static_assert(
  PeleLM::m_nGrowVOF == pelelmex::vof::nghost,
  "PeleLM::m_nGrowVOF must match the VOF reconstruction stencil");

// Liquid volume fraction BCs.  Components are Interior, Inflow, Outflow,
// Symmetry, SlipWallAdiab, NoSlipWallAdiab, SlipWallIsoTherm,
// NoSlipWallIsoTherm.  Wall ghost cells are first extrapolated, then
// overwritten by VOFWallGhosts (free contact line).
constexpr int vof_bc[] = {amrex::BCType::int_dir,  amrex::BCType::ext_dir,
                          amrex::BCType::foextrap, amrex::BCType::reflect_even,
                          amrex::BCType::foextrap, amrex::BCType::foextrap,
                          amrex::BCType::foextrap, amrex::BCType::foextrap};

void
PeleLM::VOFReadParameters()
{
  // GMRES by default for the face coefficient jump of the projections.  Set
  // before the MacProjector reads its options.
  int mac_use_gmres = 1;
  amrex::ParmParse("mac_proj").queryAdd("use_gmres", mac_use_gmres);
  amrex::ParmParse pp("peleLM.vof");
  pp.query("cfl", m_vof_cfl);
  pp.query("refine_max_level", m_vof_refine_maxlev);
  pp.query("refine_band", m_vof_refine_band);
  pp.query("v", m_vof_verbose);
  pp.query("advect_only", m_vof_advect_only);

  // Required: peleLM.rho / peleLM.mu are rejected with the VOF solver
  pp.get("rho_liquid", m_vof_rho_liq);
  pp.get("rho_gas", m_vof_rho_gas);
  pp.get("mu_liquid", m_vof_mu_liq);
  pp.get("mu_gas", m_vof_mu_gas);
  std::string mu_avg = "arithmetic";
  pp.query("viscosity_average", mu_avg);
  if (mu_avg == "arithmetic") {
    m_vof_mu_harmonic = 0;
  } else if (mu_avg == "harmonic") {
    m_vof_mu_harmonic = 1;
  } else {
    amrex::Abort(
      "peleLM.vof.viscosity_average must be 'arithmetic' or 'harmonic'");
  }
  if (
    m_vof_rho_liq <= 0.0 || m_vof_rho_gas <= 0.0 || m_vof_mu_liq <= 0.0 ||
    m_vof_mu_gas <= 0.0) {
    amrex::Abort("peleLM.vof.{rho,mu}_{liquid,gas} must be positive");
  }

  if (
    m_vof_advect_only != 0 &&
    !has_vof_vector_potential<ProblemSpecificFunctions>::value) {
    amrex::Abort(
      "peleLM.vof.advect_only = 1 requires "
      "ProblemSpecificFunctions::vof_vector_potential in pelelmex_prob.H");
  }

  if (m_vof_refine_band > m_nGrowState) {
    amrex::Abort(
      "peleLM.vof.refine_band cannot exceed the number of state ghost cells (" +
      std::to_string(m_nGrowState) + ")");
  }
  if (m_vof_cfl <= 0.0 || m_vof_cfl >= 1.0) {
    amrex::Abort("peleLM.vof.cfl must be in (0, 1): the advection is explicit");
  }
}

void
PeleLM::VOFCheckSetup()
{
  if (m_incompressible == 0) {
    amrex::Abort("The VOF solver currently requires peleLM.incompressible = 1. "
                 "Low-Mach two-phase coupling is not implemented yet.");
  }
  if (m_mesh_mapping) {
    amrex::Abort(
      "The VOF solver does not support geometry.mesh_mapping: the PLIC "
      "reconstruction assumes uniform cubic cells.");
  }
  // The flux volumes do not tile across coarse/fine boundaries, which must
  // stay in pure cells
  if (m_vof_refine_maxlev >= 0 && m_vof_refine_maxlev < max_level) {
    amrex::Abort(
      "The VOF advection requires the interface on the finest level "
      "(peleLM.vof.refine_max_level).");
  }
#ifdef AMREX_USE_EB
  amrex::Abort("The VOF solver does not support embedded boundaries yet.");
#endif
}

void
PeleLM::setVOFBoundaryConditions()
{
  const int* lo_bc = m_phys_bc.lo();
  const int* hi_bc = m_phys_bc.hi();
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    m_bcrec_vof.setLo(idim, vof_bc[lo_bc[idim]]);
    m_bcrec_vof.setHi(idim, vof_bc[hi_bc[idim]]);
  }
}

void
PeleLM::initLevelVOF(const int lev)
{
  auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
  const auto geomdata = geom[lev].data();
  ProbParm const* lprobparm = prob_parm_d;

  auto const& vof_ma = ldata_p->vof.arrays();
  amrex::ParallelFor(
    ldata_p->vof, [vof_ma, geomdata, lprobparm] AMREX_GPU_DEVICE(
                    int box_no, int i, int j, int k) noexcept {
      call_initvof<ProblemSpecificFunctions>(
        i, j, k, vof_ma[box_no], geomdata, *lprobparm);
      vof_ma[box_no](i, j, k) = amrex::Clamp(
        vof_ma[box_no](i, j, k), amrex::Real(0.0), amrex::Real(1.0));
    });
  amrex::Gpu::streamSynchronize();

  // The tagging of this level reads the ghost cells
  fillpatch_vof(lev, m_t_new[lev], m_leveldata_new[lev]->vof, m_nGrowVOF);
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getVOFVect(const TimeStamp a_time)
{
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto& vof = (a_time == AmrOldTime) ? m_leveldata_old[lev]->vof
                                       : m_leveldata_new[lev]->vof;
    r.push_back(
      std::make_unique<amrex::MultiFab>(vof, amrex::make_alias, 0, 1));
  }
  return r;
}

amrex::Real
PeleLM::VOFLiquidVolume(const TimeStamp a_time)
{
  return MFSum(GetVecOfConstPtrs(getVOFVect(a_time)), 0);
}

//-----------------------------------------------------------------------------
// Mixture properties.  The volume fraction is fillpatched at a_time (time
// interpolated between t^n and t^{n+1}, e.g. at the half time) and mapped
// in place to the property.
//-----------------------------------------------------------------------------
void
PeleLM::fillVOFDensity(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_rho,
  const int nGhost)
{
  fillpatch_vof(lev, a_time, a_rho, nGhost);
  auto const& rho_ma = a_rho.arrays();
  amrex::ParallelFor(
    a_rho, amrex::IntVect(nGhost),
    [rho_ma, rho_l = m_vof_rho_liq, rho_g = m_vof_rho_gas] AMREX_GPU_DEVICE(
      int box_no, int i, int j, int k) noexcept {
      rho_ma[box_no](i, j, k) =
        pelelmex::vof::mixture_density(rho_ma[box_no](i, j, k), rho_l, rho_g);
    });
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::fillVOFViscosity(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_mu,
  const int nGhost)
{
  fillpatch_vof(lev, a_time, a_mu, nGhost);
  auto const& mu_ma = a_mu.arrays();
  amrex::ParallelFor(
    a_mu, amrex::IntVect(nGhost),
    [mu_ma, mu_l = m_vof_mu_liq, mu_g = m_vof_mu_gas,
     harmonic =
       m_vof_mu_harmonic] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      mu_ma[box_no](i, j, k) = pelelmex::vof::mixture_viscosity(
        mu_ma[box_no](i, j, k), mu_l, mu_g, harmonic);
    });
  amrex::Gpu::streamSynchronize();
}

// State for the derived quantities: the incompressible state holds only the
// velocity, so the mixture density is appended in the DENSITY component, in
// all the ghost cells of a_state
std::unique_ptr<amrex::MultiFab>
PeleLM::VOFDeriveState(
  const int lev, const amrex::Real a_time, const amrex::MultiFab& a_state)
{
  static_assert(DENSITY == AMREX_SPACEDIM, "DENSITY follows the velocity");
  const int nGhost = a_state.nGrow();
  AMREX_ALWAYS_ASSERT(nGhost <= m_nGrowVOF);
  auto mf = std::make_unique<amrex::MultiFab>(
    grids[lev], dmap[lev], DENSITY + 1, nGhost, amrex::MFInfo(), Factory(lev));
  amrex::MultiFab::Copy(*mf, a_state, VELX, VELX, AMREX_SPACEDIM, nGhost);
  amrex::MultiFab rho(
    grids[lev], dmap[lev], 1, nGhost, amrex::MFInfo(), Factory(lev));
  fillVOFDensity(lev, a_time, rho, nGhost);
  amrex::MultiFab::Copy(*mf, rho, 0, DENSITY, 1, nGhost);
  return mf;
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFFaceDensity(const int lev, const amrex::Real a_time)
{
  amrex::MultiFab rho(
    grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
  fillVOFDensity(lev, a_time, rho, 1);

  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rho_fc;
  const amrex::Box& domain = geom[lev].Domain();
  auto const& rho_ma = rho.const_arrays();
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    rho_fc[idim].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
      dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    // Dirichlet faces take the boundary value held in the ghost cell
    const bool ext_lo = m_bcrec_vof.lo(idim) == amrex::BCType::ext_dir;
    const bool ext_hi = m_bcrec_vof.hi(idim) == amrex::BCType::ext_dir;
    const int dlo = domain.smallEnd(idim);
    const int dhi = domain.bigEnd(idim) + 1;
    auto const& fc_ma = rho_fc[idim].arrays();
    amrex::ParallelFor(
      rho_fc[idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        const amrex::IntVect ivm =
          iv - amrex::IntVect::TheDimensionVector(idim);
        const auto& r = rho_ma[box_no];
        if (ext_lo && iv[idim] == dlo) {
          fc_ma[box_no](i, j, k) = r(ivm);
        } else if (ext_hi && iv[idim] == dhi) {
          fc_ma[box_no](i, j, k) = r(iv);
        } else {
          fc_ma[box_no](i, j, k) = 0.5 * (r(ivm) + r(iv));
        }
      });
  }
  amrex::Gpu::streamSynchronize();
  return rho_fc;
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFSharpFaceDensity(const int lev, const TimeStamp a_time)
{
  // Planes are fitted in the cells on either side of each face
  constexpr int nGhost = m_nGrowVOF;
  amrex::Vector<amrex::Real> times;
  if (a_time == AmrHalfTime) {
    times = {m_t_old[lev], m_t_new[lev]};
  } else {
    times = {getTime(lev, a_time)};
  }
  const amrex::Real weight = 1.0 / static_cast<amrex::Real>(times.size());

  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rho_fc;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    rho_fc[idim].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
      dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    rho_fc[idim].setVal(0.0);
  }

  // Segment liquid fraction, averaged over the requested times
  amrex::MultiFab F(
    grids[lev], dmap[lev], 1, nGhost, amrex::MFInfo(), Factory(lev));
  for (const amrex::Real time : times) {
    fillpatch_vof(lev, time, F, nGhost);
    auto const& F_ma = F.const_arrays();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto const& fc_ma = rho_fc[idim].arrays();
      amrex::ParallelFor(
        rho_fc[idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
          fc_ma[box_no](i, j, k) +=
            weight *
            pelelmex::vof::face_segment_fraction(i, j, k, idim, F_ma[box_no]);
        });
    }
    amrex::Gpu::streamSynchronize();
  }

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    auto const& fc_ma = rho_fc[idim].arrays();
    amrex::ParallelFor(
      rho_fc[idim],
      [fc_ma, rho_l = m_vof_rho_liq, rho_g = m_vof_rho_gas] AMREX_GPU_DEVICE(
        int box_no, int i, int j, int k) {
        fc_ma[box_no](i, j, k) =
          rho_g + (rho_l - rho_g) * fc_ma[box_no](i, j, k);
      });
  }
  amrex::Gpu::streamSynchronize();
  // A face must have one coefficient, or the projection operator is neither
  // conservative nor symmetric.  Keep one copy against round-off.
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    rho_fc[idim].OverrideSync(geom[lev].periodicity());
  }
  return rho_fc;
}

//-----------------------------------------------------------------------------
// Momentum advection consistent with the VOF mass transport (Rudman 1998), with
// the fluxes of the VOF steps.  Each sub-step carries the mass that the flux
// volumes carry,
//   m_f = rho_g C_f + (rho_l - rho_g) L_f   (per cell volume)
// with C_f the face Courant number and L_f the liquid crossing the face, so
//   rho' = rho - sum (m_hi - m_lo),  (rho u)' = (rho u) - sum (m_hi u_hi - m_lo
//   u_lo)
// and a uniform velocity stays uniform.  Near the interface the momentum flux
// is the integral over the flux volume, each piece carrying the velocity of the
// cell it lies in: a flux volume reaches beyond the upwind cell, and a single
// upwind velocity would hand its momentum to the wrong mass.  Away from it u_f
// is the Godunov face state.  Coarse faces under fine faces take the average of
// the fine fluxes per unit area, so the composite update is conservative.  With
// AMR there is one sub-step, whose fluxes use the velocity at t^n with its
// coarse/fine ghost cells (see advanceVOF).
// The result is returned as the advection term
// AofS = (u** - u^n) / dt (updateVelocity adds dt AofS).
//-----------------------------------------------------------------------------
void
PeleLM::VOFMomentumAdvTerm(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_faces)
{
  BL_PROFILE("PeleLMeX::VOFMomentumAdvTerm()");

  const amrex::Real rho_l = m_vof_rho_liq;
  const amrex::Real rho_g = m_vof_rho_gas;
  const amrex::Real dt_sub = advData->vof_dt_sub;
  const int nsub = static_cast<int>(advData->vof_lflux.size());
  AMREX_ALWAYS_ASSERT(nsub == 1 || finest_level == 0);

  amrex::Vector<amrex::MultiFab> rho(finest_level + 1);
  amrex::Vector<amrex::MultiFab> mom(finest_level + 1);
  amrex::Vector<amrex::MultiFab> vel(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    rho[lev].define(grids[lev], dmap[lev], 1, 0);
    mom[lev].define(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);
    vel[lev].define(grids[lev], dmap[lev], AMREX_SPACEDIM, 1);
    amrex::MultiFab::Copy(
      vel[lev], getLevelDataPtr(lev, AmrOldTime)->state, VELX, 0,
      AMREX_SPACEDIM, 1);
    auto const& vof_ma = advData->vof_fstart[0][lev].const_arrays();
    auto const& rho_ma = rho[lev].arrays();
    auto const& mom_ma = mom[lev].arrays();
    auto const& vel_ma = vel[lev].const_arrays();
    amrex::ParallelFor(
      rho[lev], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::Real r =
          pelelmex::vof::mixture_density(vof_ma[box_no](i, j, k), rho_l, rho_g);
        rho_ma[box_no](i, j, k) = r;
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
          mom_ma[box_no](i, j, k, n) = r * vel_ma[box_no](i, j, k, n);
        }
      });
  }
  amrex::Gpu::streamSynchronize();

  for (int isub = 0; isub < nsub; ++isub) {
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);

      // Mass and momentum crossing the faces, per unit face area
      amrex::Vector<amrex::MultiFab> mflux(finest_level + 1);
      amrex::Vector<amrex::MultiFab> pflux(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        const amrex::Real dx = geom[lev].CellSize(dir);
        const amrex::Real dxinv = geom[lev].InvCellSize(dir);
        const amrex::Real dtdx = dt_sub * dxinv;
        const auto& lflux = advData->vof_lflux[isub][lev][dir];
        mflux[lev].define(lflux.boxArray(), dmap[lev], 1, 0);
        pflux[lev].define(lflux.boxArray(), dmap[lev], AMREX_SPACEDIM, 0);
        auto const& F_mf = advData->vof_fstart[isub][lev];
        auto const& plane_mf = advData->vof_plane[isub][lev];
        auto const& back_mf = advData->vof_back[isub][lev];
        const bool cached = !advData->vof_momflux.empty();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mflux[lev]); mfi.isValid(); ++mfi) {
          const amrex::Box& fbx = mfi.validbox();
          const amrex::Box cells = amrex::grow(amrex::enclosedCells(fbx), 1);
          auto const& mf = mflux[lev].array(mfi);
          auto const& pf = pflux[lev].array(mfi);
          auto const& lf = lflux.const_array(mfi);
          auto const& um = advData->umac[lev][dir].const_array(mfi);
          auto const& face = a_faces[lev][dir].const_array(mfi);
          auto const& vl = vel[lev].const_array(mfi);
          auto const& F = F_mf.const_array(mfi);
          auto const& pl = plane_mf.const_array(mfi);
          auto const& bk = back_mf.const_array(mfi);
          auto const& mfx = cached
                              ? advData->vof_momflux[lev][dir].const_array(mfi)
                              : amrex::Array4<amrex::Real const>{};
          amrex::ParallelFor(
            fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
              const amrex::Real courant = um(iv) * dtdx;
              if (
                pelelmex::vof::interface_band(i, j, k, F) ||
                pelelmex::vof::interface_band(
                  i - e[0], j - e[1], k - AMREX_D_PICK(0, 0, e[2]), F)) {
                amrex::Real m = 0.0;
                amrex::Real p[AMREX_SPACEDIM];
                if (cached) {
                  m = mfx(iv, 0);
                  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    p[n] = mfx(iv, 1 + n);
                  }
                } else {
                  amrex::Real l = 0.0;
                  pelelmex::vof::unsplit::face_mass_momentum_flux(
                    i, j, k, dir, courant, F, pl, bk, vl, cells, rho_l, rho_g,
                    m, p, l);
                }
                mf(iv) = dx * m;
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                  pf(iv, n) = dx * p[n];
                }
              } else {
                const amrex::Real m =
                  rho_g * courant + (rho_l - rho_g) * lf(iv) * dxinv;
                mf(iv) = dx * m;
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                  pf(iv, n) = dx * m * face(iv, n);
                }
              }
            });
        }
        amrex::Gpu::streamSynchronize();
        const auto periodicity = geom[lev].periodicity();
        mflux[lev].OverrideSync(periodicity);
        pflux[lev].OverrideSync(periodicity);
      }
      for (int lev = finest_level; lev > 0; --lev) {
        amrex::average_down_faces(
          mflux[lev], mflux[lev - 1], refRatio(lev - 1), geom[lev - 1]);
        amrex::average_down_faces(
          pflux[lev], pflux[lev - 1], refRatio(lev - 1), geom[lev - 1]);
      }

      for (int lev = 0; lev <= finest_level; ++lev) {
        const amrex::Real dxinv = geom[lev].InvCellSize(dir);
        auto const& rho_ma = rho[lev].arrays();
        auto const& mom_ma = mom[lev].arrays();
        auto const& cm_ma = mflux[lev].const_arrays();
        auto const& cp_ma = pflux[lev].const_arrays();
        amrex::ParallelFor(
          rho[lev],
          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            amrex::ignore_unused(k);
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            const amrex::IntVect ivp = iv + e;
            rho_ma[box_no](iv) -=
              (cm_ma[box_no](ivp) - cm_ma[box_no](iv)) * dxinv;
            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
              mom_ma[box_no](iv, n) -=
                (cp_ma[box_no](ivp, n) - cp_ma[box_no](iv, n)) * dxinv;
            }
          });
      }
      amrex::Gpu::streamSynchronize();
    }

    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& rho_ma = rho[lev].const_arrays();
      auto const& mom_ma = mom[lev].const_arrays();
      auto const& vel_ma = vel[lev].arrays();
      amrex::ParallelFor(
        rho[lev], amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          vel_ma[box_no](i, j, k, n) =
            mom_ma[box_no](i, j, k, n) / rho_ma[box_no](i, j, k);
        });
      amrex::Gpu::streamSynchronize();
      vel[lev].FillBoundary(
        0, AMREX_SPACEDIM, amrex::IntVect(1), geom[lev].periodicity());
    }
  }

  const amrex::Real dt = m_dt;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& aofs_ma = advData->AofS[lev].arrays();
    auto const& vel_ma = vel[lev].const_arrays();
    auto const& velo_ma =
      getLevelDataPtr(lev, AmrOldTime)->state.const_arrays();
    amrex::ParallelFor(
      vel[lev], amrex::IntVect(0), AMREX_SPACEDIM,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        aofs_ma[box_no](i, j, k, VELX + n) =
          (vel_ma[box_no](i, j, k, n) - velo_ma[box_no](i, j, k, VELX + n)) /
          dt;
      });
  }
  amrex::Gpu::streamSynchronize();
}

//-----------------------------------------------------------------------------
// Face projection umac -= beta grad phi with div(beta grad phi) = div(umac), by
// the MacProjector with its solver options (mac_proj.*).  GMRES preconditioned
// by MLMG is the VOF default, for the large face-coefficient jumps of a sharp,
// high density ratio interface.
//-----------------------------------------------------------------------------
void
PeleLM::VOFFaceProjection(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_umac,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_beta)
{
  BL_PROFILE("PeleLMeX::VOFFaceProjection()");

  setupMacProjector(a_beta);
  macproj->setUMAC(a_umac);
  // Coarse faces under fine faces take the fine-face average, so that the
  // composite divergence is consistent across coarse-fine interfaces
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(a_umac[lev]), a_umac[lev - 1], refRatio(lev - 1),
      geom[lev - 1]);
  }
  // The solvers stop on their own estimate of the residual.  Convergence is
  // judged here on the divergence actually left in umac, including its mean,
  // solving again for what remains, with a warning if the target is not
  // reached: a wrong operator (a face with two coefficients) gives a converged
  // solve and a divergent umac.
  const auto max_divergence = [&]() {
    amrex::Real div = 0.0;
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab divu(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      amrex::computeDivergence(divu, GetArrOfConstPtrs(a_umac[lev]), geom[lev]);
      div = amrex::max(div, divu.norm0());
    }
    return div;
  };
  constexpr int max_solves = 10;
  amrex::Real div_left = max_divergence();
  const amrex::Real div_target =
    amrex::max(m_mac_mg_rtol * div_left, m_mac_mg_atol);
  int nsolve = 0;
  while (div_left > div_target && nsolve < max_solves) {
    solveMacProjection();
    if (m_macproj_verbose > 0) {
      amrex::Print() << "   VOF face projection: " << macproj->getNumIters()
                     << " iterations\n";
    }
    div_left = max_divergence();
    ++nsolve;
  }
  if (div_left > div_target) {
    amrex::Print() << "   VOF: WARNING face projection left a divergence of "
                   << div_left << " after " << nsolve << " solves, target "
                   << div_target << "\n";
  } else if (m_macproj_verbose > 0 && nsolve > 1) {
    amrex::Print() << "   VOF face projection: " << nsolve
                   << " solves, divergence left " << div_left << "\n";
  }
}

//-----------------------------------------------------------------------------
// Balanced-force velocity projection (Francois et al., JCP 213, 2006).
// Gravity and the pressure gradient act on the faces:
//   u_f = <u*>_f + dt g,   div(dt / rho_f grad p) = div(u_f),
//   a_f = g - grad_f p / rho_f,
// with the sharp face density rho_f of getVOFSharpFaceDensity, so that a
// fluid at rest with a planar interface normal to g gives a_f = 0 to solver
// tolerance.  The cell-centered velocity is updated with the average of
// the face accelerations of each cell, and gp holds the kinematic
// gradient g - <a_f>, i.e. grad p / rho, used as the lagged
// pressure-gradient forcing of the next step.  On entry the new velocity
// contains the lagged forcing dt (g - gp^{n-1/2}), which is removed first.
//-----------------------------------------------------------------------------
void
PeleLM::VOFVelocityProjection(const int a_initPress, const amrex::Real a_dt)
{
  BL_PROFILE("PeleLMeX::VOFVelocityProjection()");

  const auto grav = m_gravity;

  // Remove the lagged pressure-gradient / gravity acceleration
  if (a_initPress == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
      auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
      auto const& vel_ma = ldataNew_p->state.arrays();
      auto const& gp_ma = ldataOld_p->gp.const_arrays();
      amrex::ParallelFor(
        ldataNew_p->state, amrex::IntVect(0), AMREX_SPACEDIM,
        [=, dt = a_dt] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          vel_ma[box_no](i, j, k, VELX + n) -=
            dt * (grav[n] - gp_ma[box_no](i, j, k, n));
        });
    }
    amrex::Gpu::streamSynchronize();
    averageDownVelocity(AmrNewTime);
    fillPatchState(AmrNewTime);
  }

  // Face velocities before (ub) and after (umac) adding gravity
  const auto bcRecVel = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> ub(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> umac(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> beta(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box& domain = geom[lev].Domain();
    auto const& vel_ma = getLevelDataPtr(lev, AmrNewTime)->state.const_arrays();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const amrex::BoxArray fba =
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim));
      ub[lev][idim].define(fba, dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      umac[lev][idim].define(
        fba, dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      if (a_initPress != 0) {
        ub[lev][idim].setVal(0.0);
        continue;
      }
      // Dirichlet faces take the boundary velocity held in the ghost cell
      const bool periodic = geom[lev].isPeriodic(idim);
      const bool ext_lo =
        !periodic && bcRecVel[idim].lo(idim) == amrex::BCType::ext_dir;
      const bool ext_hi =
        !periodic && bcRecVel[idim].hi(idim) == amrex::BCType::ext_dir;
      const int dlo = domain.smallEnd(idim);
      const int dhi = domain.bigEnd(idim) + 1;
      auto const& ub_ma = ub[lev][idim].arrays();
      amrex::ParallelFor(
        ub[lev][idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const amrex::IntVect ivm =
            iv - amrex::IntVect::TheDimensionVector(idim);
          const auto& v = vel_ma[box_no];
          if (ext_lo && iv[idim] == dlo) {
            ub_ma[box_no](i, j, k) = v(ivm, VELX + idim);
          } else if (ext_hi && iv[idim] == dhi) {
            ub_ma[box_no](i, j, k) = v(iv, VELX + idim);
          } else {
            ub_ma[box_no](i, j, k) =
              0.5 * (v(ivm, VELX + idim) + v(iv, VELX + idim));
          }
        });
    }
  }
  amrex::Gpu::streamSynchronize();
  // Coarse faces under / at the boundary of fine grids see the fine faces,
  // as the projected umac does
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(ub[lev]), GetArrOfPtrs(ub[lev - 1]), refRatio(lev - 1),
      geom[lev - 1]);
  }

  const TimeStamp rhoTime = (a_initPress != 0) ? AmrNewTime : AmrHalfTime;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box& domain = geom[lev].Domain();
    beta[lev] = getVOFSharpFaceDensity(lev, rhoTime);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      beta[lev][idim].invert(a_dt, 0);
      // Gravity acts on all faces except those where the velocity is
      // prescribed (walls, inflow)
      const bool periodic = geom[lev].isPeriodic(idim);
      const bool fixed_lo =
        !periodic && m_phys_bc.lo(idim) != amrex::PhysBCType::outflow;
      const bool fixed_hi =
        !periodic && m_phys_bc.hi(idim) != amrex::PhysBCType::outflow;
      const int dlo = domain.smallEnd(idim);
      const int dhi = domain.bigEnd(idim) + 1;
      const amrex::Real dtg = a_dt * grav[idim];
      auto const& ub_ma = ub[lev][idim].const_arrays();
      auto const& um_ma = umac[lev][idim].arrays();
      amrex::ParallelFor(
        umac[lev][idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const bool fixed =
            (fixed_lo && iv[idim] == dlo) || (fixed_hi && iv[idim] == dhi);
          um_ma[box_no](i, j, k) = ub_ma[box_no](i, j, k) + (fixed ? 0.0 : dtg);
        });
    }
  }
  amrex::Gpu::streamSynchronize();

  VOFFaceProjection(GetVecOfArrOfPtrs(umac), beta);

  // Cell acceleration = average of the face accelerations
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldataNew_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      auto const& vel = ldataNew_p->state.array(mfi, VELX);
      auto const& gp = ldataNew_p->gp.array(mfi);
      amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> uaf{
        AMREX_D_DECL(
          umac[lev][0].const_array(mfi), umac[lev][1].const_array(mfi),
          umac[lev][2].const_array(mfi))};
      amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> ubf{
        AMREX_D_DECL(
          ub[lev][0].const_array(mfi), ub[lev][1].const_array(mfi),
          ub[lev][2].const_array(mfi))};
      const bool update_vel = (a_initPress == 0);
      amrex::ParallelFor(
        bx, [=, dt = a_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            const amrex::IntVect ivp =
              iv + amrex::IntVect::TheDimensionVector(n);
            const amrex::Real a =
              0.5 * ((uaf[n](iv) - ubf[n](iv)) + (uaf[n](ivp) - ubf[n](ivp))) /
              dt;
            if (update_vel) {
              vel(i, j, k, n) += dt * a;
            }
            gp(i, j, k, n) = grav[n] - a;
          }
        });
    }
  }
  amrex::Gpu::streamSynchronize();

  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down(
      getLevelDataPtr(lev, AmrNewTime)->gp,
      getLevelDataPtr(lev - 1, AmrNewTime)->gp, 0, AMREX_SPACEDIM,
      refRatio(lev - 1));
  }
  if (a_initPress == 0) {
    averageDownVelocity(AmrNewTime);
  }
}

//-----------------------------------------------------------------------------
// Advection: unsplit geometric fluxes (Owkes & Desjardins, JCP 270, 2014) with
// PLIC reconstruction.  All levels advance with the same dt; fine face fluxes
// are averaged onto the coarse faces before the update, so the composite update
// conserves volume across coarse/fine boundaries.
//-----------------------------------------------------------------------------
void
PeleLM::advanceVOF(std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::advanceVOF()");

  const amrex::Real vol_old =
    (m_vof_verbose > 0) ? VOFLiquidVolume(AmrNewTime) : 0.0;

  // Working state is the new VOF, which holds t^n at this point
  fillPatchVOF(AmrNewTime);

  // Largest face velocity / dx.  The step is estimated from the
  // cell-centered velocity at t^n, while the advection uses the face velocity at
  // t^{n+1/2}, which can be larger
  amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>> umac(
    finest_level + 1);
  amrex::Real umac_rate = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      umac[lev][idim] = &(advData->umac[lev][idim]);
      umac_rate = amrex::max(
        umac_rate,
        umac[lev][idim]->norm0(0, 0, true) * geom[lev].InvCellSize(idim));
    }
  }
  amrex::ParallelDescriptor::ReduceRealMax(umac_rate);

  // Sub-step the advection on one level above a Courant number of 1/2.
  // TODO: remove the sub-stepping: without it the advection stays bounded and
  // conservative for a Courant number below 1
  const amrex::Real courant_max = umac_rate * m_dt;
  // AMR already takes a single step
  const bool single_step = (max_level > 0);
  if (single_step && courant_max >= 1.0) {
    amrex::Abort(
      "VOF: the advection requires a face Courant number below 1: reduce the "
      "time step.");
  }
  const int nsub =
    single_step ? 1
                : amrex::max(1, static_cast<int>(std::ceil(courant_max / 0.5)));
  if (nsub > 1) {
    amrex::Print() << "   VOF: face Courant number " << courant_max
                   << " exceeds 0.5, advecting in " << nsub << " sub-steps\n";
  }
  const amrex::Real dt_sub = m_dt / static_cast<amrex::Real>(nsub);

  advData->vof_dt_sub = dt_sub;
  advData->vof_lflux.clear();
  advData->vof_fstart.clear();
  advData->vof_plane.clear();
  advData->vof_back.clear();
  advData->vof_momflux.clear();
  advData->vof_lflux.resize(nsub);
  advData->vof_fstart.resize(nsub);
  advData->vof_plane.resize(nsub);
  advData->vof_back.resize(nsub);
  for (int isub = 0; isub < nsub; ++isub) {
    // F with the two ghost layers of the interface band of the momentum
    // advection, at the start of this sub-step
    auto& fstart = advData->vof_fstart[isub];
    fstart.resize(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      fstart[lev].define(grids[lev], dmap[lev], 1, 2);
      amrex::MultiFab::Copy(
        fstart[lev], m_leveldata_new[lev]->vof, 0, 0, 1, 2);
    }
    advData->vof_lflux[isub].resize(finest_level + 1);
    advData->vof_plane[isub].resize(finest_level + 1);
    advData->vof_back[isub].resize(finest_level + 1);
    // With one sub-step the momentum advection starts from the velocity known
    // here, and its fluxes near the interface can reuse these cuts
    const bool with_mom = (nsub == 1) && (m_vof_advect_only == 0);
    if (with_mom) {
      advData->vof_momflux.resize(finest_level + 1);
    }
    VOFUnsplitStep(
      dt_sub, umac, advData->vof_lflux[isub], advData->vof_plane[isub],
      advData->vof_back[isub], with_mom ? &advData->vof_momflux : nullptr);
  }

  if (m_vof_verbose > 0) {
    const amrex::Real vol_new = VOFLiquidVolume(AmrNewTime);
    amrex::Print() << "   VOF: liquid volume " << vol_new
                   << ", relative change "
                   << (vol_new - vol_old) / amrex::max(vol_old, 1.0e-300)
                   << "\n";
    // The discrete divergence of the face velocities, as a change of F in a
    // pure liquid cell over this step, and the departure of F from [0,1]
    amrex::Real div_max = 0.0;
    amrex::Real f_min = 1.0e200;
    amrex::Real f_max = -1.0e200;
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab divu(grids[lev], dmap[lev], 1, 0);
      amrex::computeDivergence(divu, umac[lev], geom[lev]);
      div_max = amrex::max(div_max, divu.norm0());
      f_min = amrex::min(f_min, m_leveldata_new[lev]->vof.min(0));
      f_max = amrex::max(f_max, m_leveldata_new[lev]->vof.max(0));
    }
    amrex::Print() << "   VOF: dt*max|div umac| " << div_max * m_dt
                   << ", F below 0 by " << amrex::max(0.0, -f_min)
                   << ", above 1 by " << amrex::max(0.0, f_max - 1.0) << "\n";
  }
}

//-----------------------------------------------------------------------------
// One unsplit geometric step of the volume fraction in the new state: each face
// carries the liquid inside its flux volume, built from the mesh nodes traced
// back along the flow (see PeleLMeX_VOF_Unsplit_K.H).  a_flux is the liquid
// crossing each face per unit face area: fine faces are averaged onto the
// coarse faces below them, which conserves the volume across coarse/fine
// boundaries. The flux volumes do not tile across those boundaries, which the
// tagging keeps in pure cells.
//-----------------------------------------------------------------------------
void
PeleLM::VOFUnsplitStep(
  const amrex::Real a_dt,
  const amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_umac,
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_flux,
  amrex::Vector<amrex::MultiFab>& a_plane,
  amrex::Vector<amrex::MultiFab>& a_back,
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>* a_momflux)
{
  BL_PROFILE("PeleLMeX::VOFUnsplitStep()");

  for (int lev = 0; lev <= finest_level; ++lev) {
    VOFUnsplitFluxes(
      lev, a_dt, a_umac[lev], a_flux[lev], a_plane[lev], a_back[lev],
      (a_momflux != nullptr) ? &(*a_momflux)[lev] : nullptr);
  }
  for (int lev = finest_level; lev > 0; --lev) {
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::average_down_faces(
        a_flux[lev][dir], a_flux[lev - 1][dir], refRatio(lev - 1),
        geom[lev - 1]);
    }
  }

  // Update, Eq. (18) of the paper.  F is not clipped or limited: it conserves
  // the liquid volume, and departs from [0,1] by the errors of the fluxes
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto& vof = m_leveldata_new[lev]->vof;
    const auto dxinv = geom[lev].InvCellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      auto const& F = vof.array(mfi);
      AMREX_D_TERM(
        auto const& fx = a_flux[lev][0].const_array(mfi);
        , auto const& fy = a_flux[lev][1].const_array(mfi);
        , auto const& fz = a_flux[lev][2].const_array(mfi););
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          F(i, j, k) += AMREX_D_TERM(
            (fx(i, j, k) - fx(i + 1, j, k)) * dxinv[0],
            +(fy(i, j, k) - fy(i, j + 1, k)) * dxinv[1],
            +(fz(i, j, k) - fz(i, j, k + 1)) * dxinv[2]);
        });
    }
  }
  amrex::Gpu::streamSynchronize();

  averageDownVOF(AmrNewTime);
  fillPatchVOF(AmrNewTime);
}

void
PeleLM::VOFUnsplitFluxes(
  const int lev,
  const amrex::Real a_dt,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_umac,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
  amrex::MultiFab& plane,
  amrex::MultiFab& back,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>* a_momflux)
{
  const amrex::Box& domain = geom[lev].Domain();
  const auto periodicity = geom[lev].periodicity();
  const amrex::IntVect is_periodic(AMREX_D_DECL(
    geom[lev].isPeriodic(0), geom[lev].isPeriodic(1), geom[lev].isPeriodic(2)));
  const amrex::BoxArray nodal_ba =
    amrex::convert(grids[lev], amrex::IntVect::TheNodeVector());

  // Nodes that can be traced to.  Positions beyond the domain use its nearest
  // nodes, except across periodic boundaries.
  amrex::Box node_limit = amrex::surroundingNodes(domain);
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (is_periodic[d] != 0) {
      node_limit.grow(d, 1000000);
    }
  }

  // Node velocities in Courant numbers: each component is the average of the
  // faces normal to it that touch the node, one-sided at domain boundaries.
  // A node shared by several boxes gets one value, so that it is traced to one
  // position and the flux volumes tile across boxes.  The ghost nodes at
  // coarse/fine boundaries use the ghost faces that are available.
  amrex::MultiFab nvel(nodal_ba, dmap[lev], AMREX_SPACEDIM, 1);
  nvel.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(nvel); mfi.isValid(); ++mfi) {
    const amrex::Box nbx = amrex::grow(mfi.validbox(), 1) & node_limit;
    auto const& vel = nvel.array(mfi);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      auto const& um = a_umac[dir]->const_array(mfi);
      const amrex::Real dtdx = a_dt * geom[lev].InvCellSize(dir);
      amrex::ParallelFor(
        nbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          amrex::Real sum = 0.0;
          int count = 0;
          const int t1 = (dir + 1) % AMREX_SPACEDIM;
          const int n1 = (AMREX_SPACEDIM > 1) ? 2 : 1;
          const int t2 = (AMREX_SPACEDIM == 3) ? (dir + 2) % 3 : t1;
          const int n2 = (AMREX_SPACEDIM == 3) ? 2 : 1;
          for (int s2 = 0; s2 < n2; ++s2) {
            for (int s1 = 0; s1 < n1; ++s1) {
              amrex::IntVect f = iv;
              f[t1] -= s1;
              if (AMREX_SPACEDIM == 3) {
                f[t2] -= s2;
              }
              bool inside =
                (is_periodic[t1] != 0) ||
                (f[t1] >= domain.smallEnd(t1) && f[t1] <= domain.bigEnd(t1));
              inside =
                inside &&
                ((AMREX_SPACEDIM != 3) || (is_periodic[t2] != 0) ||
                 (f[t2] >= domain.smallEnd(t2) && f[t2] <= domain.bigEnd(t2)));
              if (inside && um.contains(f)) {
                sum += um(f);
                ++count;
              }
            }
          }
          vel(i, j, k, dir) = sum * dtdx / amrex::max(count, 1);
        });
    }
  }
  nvel.FillBoundary(periodicity);

  // Traced position of each node at the start of the step (midpoint rule),
  // in cell units
  back.define(nodal_ba, dmap[lev], 3, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(back); mfi.isValid(); ++mfi) {
    const amrex::Box& nbx = mfi.validbox();
    const amrex::Box lim = amrex::grow(nbx, 1) & node_limit;
    auto const& vel = nvel.const_array(mfi);
    auto const& bk = back.array(mfi);
    amrex::ParallelFor(nbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const amrex::Real x[3] = {
        static_cast<amrex::Real>(i), static_cast<amrex::Real>(j),
        static_cast<amrex::Real>(k)};
      amrex::Real xm[3] = {x[0], x[1], x[2]};
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        xm[d] -= 0.5 * vel(i, j, k, d);
      }
      amrex::Real um[3] = {0.0, 0.0, 0.0};
      pelelmex::vof::unsplit::interp_node_velocity(xm, vel, lim, um);
      for (int d = 0; d < 3; ++d) {
        bk(i, j, k, d) = x[d] - um[d];
      }
    });
  }

  // PLIC plane of the mixed cells, valid up to the first ghost layer
  auto& vof = m_leveldata_new[lev]->vof;
  plane.define(grids[lev], dmap[lev], 4, 1);
  plane.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(plane); mfi.isValid(); ++mfi) {
    const amrex::Box gbx = amrex::grow(mfi.validbox(), 1);
    auto const& F = vof.const_array(mfi);
    auto const& pl = plane.array(mfi);
    amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const amrex::Real f = F(i, j, k);
      if (f > pelelmex::vof::vof_tol && f < 1.0 - pelelmex::vof::vof_tol) {
        amrex::Real mx, my, mz, alpha;
        pelelmex::vof::fit_plane(i, j, k, F, mx, my, mz, alpha);
        pl(i, j, k, 0) = mx;
        pl(i, j, k, 1) = my;
        pl(i, j, k, 2) = mz;
        pl(i, j, k, 3) = alpha;
      }
    });
  }

  // Liquid crossing each face per unit face area.  With a_momflux, the mass and
  // momentum near the interface come from the same cuts of the flux volumes
  const bool with_mom = (a_momflux != nullptr);
  const amrex::Real rho_l = m_vof_rho_liq;
  const amrex::Real rho_g = m_vof_rho_gas;
  auto const& state_old = getLevelDataPtr(lev, AmrOldTime)->state;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    flux[dir].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(dir)),
      dmap[lev], 1, 0);
    if (with_mom) {
      (*a_momflux)[dir].define(
        flux[dir].boxArray(), dmap[lev], 1 + AMREX_SPACEDIM, 0);
    }
    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
    const amrex::Real dx = geom[lev].CellSize(dir);
    const amrex::Real dtdx = a_dt / dx;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(flux[dir]); mfi.isValid(); ++mfi) {
      const amrex::Box& fbx = mfi.validbox();
      const amrex::Box cells = amrex::grow(amrex::enclosedCells(fbx), 1);
      auto const& fl = flux[dir].array(mfi);
      auto const& um = a_umac[dir]->const_array(mfi);
      auto const& F = vof.const_array(mfi);
      auto const& pl = plane.const_array(mfi);
      auto const& bk = back.const_array(mfi);
      auto const& mfx =
        with_mom ? (*a_momflux)[dir].array(mfi) : amrex::Array4<amrex::Real>{};
      const amrex::Array4<amrex::Real const> vl(
        state_old.const_array(mfi), VELX);
      amrex::ParallelFor(
        fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const amrex::Real courant = um(i, j, k) * dtdx;
          if (
            with_mom &&
            (pelelmex::vof::interface_band(i, j, k, F) ||
             pelelmex::vof::interface_band(
               i - e[0], j - e[1], k - AMREX_D_PICK(0, 0, e[2]), F))) {
            amrex::Real m = 0.0;
            amrex::Real p[AMREX_SPACEDIM];
            amrex::Real l = 0.0;
            pelelmex::vof::unsplit::face_mass_momentum_flux(
              i, j, k, dir, courant, F, pl, bk, vl, cells, rho_l, rho_g, m, p,
              l);
            fl(i, j, k) = dx * l;
            mfx(i, j, k, 0) = m;
            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
              mfx(i, j, k, 1 + n) = p[n];
            }
          } else {
            fl(i, j, k) = dx * pelelmex::vof::unsplit::face_liquid_flux(
                                 i, j, k, dir, courant, F, pl, bk, cells);
          }
        });
    }
    // Faces between boxes compute the same value, up to round-off across a
    // periodic boundary: keep one, so that the volume is conserved exactly
    flux[dir].OverrideSync(periodicity);
  }
  amrex::Gpu::streamSynchronize();
}

//-----------------------------------------------------------------------------
// Prescribed velocity (peleLM.vof.advect_only): u = curl(A).  The face
// velocity is the circulation of A around the face divided by its area, so
// umac is discretely divergence-free to round-off.  In 3D the edge integrals
// use 3-point Gauss quadrature, which makes coarse face fluxes equal to the
// sum of the underlying fine face fluxes for potentials up to degree 5.
//-----------------------------------------------------------------------------
namespace {

template <typename PSF>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
prescribed_face_velocity(
  const int dir,
  const amrex::IntVect& iv,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm)
{
  const amrex::Real* plo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  amrex::Real A[3];
#if AMREX_SPACEDIM == 2
  // Stream function at the two nodes bounding the face
  const int t = 1 - dir;
  amrex::Real x[2];
  x[dir] = plo[dir] + iv[dir] * dx[dir];
  x[t] = plo[t] + iv[t] * dx[t];
  call_vof_vector_potential<PSF>(x, time, A, prob_parm);
  const amrex::Real psi_lo = A[2];
  x[t] += dx[t];
  call_vof_vector_potential<PSF>(x, time, A, prob_parm);
  const amrex::Real psi_hi = A[2];
  // u = dpsi/dy, v = -dpsi/dx
  return (dir == 0 ? 1.0 : -1.0) * (psi_hi - psi_lo) / dx[t];
#else
  // Right-handed tangential directions
  const int t1 = (dir + 1) % 3;
  const int t2 = (dir + 2) % 3;
  constexpr amrex::Real gx[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
  constexpr amrex::Real gw[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
  amrex::Real lo[3];
  for (int d = 0; d < 3; ++d) {
    lo[d] = plo[d] + iv[d] * dx[d];
  }
  // Integral of A_c along the edge parallel to c, offset by `off` (0 or 1
  // cell) in the other tangential direction o
  auto edge = [&](const int c, const int o, const int off) {
    amrex::Real x[3] = {lo[0], lo[1], lo[2]};
    x[o] += off * dx[o];
    amrex::Real sum = 0.0;
    for (int q = 0; q < 3; ++q) {
      x[c] = lo[c] + 0.5 * (1.0 + gx[q]) * dx[c];
      call_vof_vector_potential<PSF>(x, time, A, prob_parm);
      sum += gw[q] * A[c];
    }
    return sum * dx[c];
  };
  const amrex::Real circ =
    edge(t1, t2, 0) - edge(t1, t2, 1) + edge(t2, t1, 1) - edge(t2, t1, 0);
  return circ / (dx[t1] * dx[t2]);
#endif
}

} // namespace

void
PeleLM::VOFPrescribedVelocity(
  std::unique_ptr<AdvanceAdvData>& advData,
  const amrex::Real a_umac_time,
  const amrex::Real a_cc_time)
{
  BL_PROFILE("PeleLMeX::VOFPrescribedVelocity()");
  ProbParm const* lprobparm = prob_parm_d;

  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto geomdata = geom[lev].data();

    // Face velocities, including ghost faces
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto& umac = advData->umac[lev][idim];
      auto const& umac_ma = umac.arrays();
      amrex::ParallelFor(
        umac, umac.nGrowVect(),
        [umac_ma, geomdata, lprobparm, idim, a_umac_time] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k) noexcept {
          umac_ma[box_no](i, j, k) =
            prescribed_face_velocity<ProblemSpecificFunctions>(
              idim, amrex::IntVect(AMREX_D_DECL(i, j, k)), a_umac_time,
              geomdata, *lprobparm);
        });
    }

    // Cell-centered velocity at the new time, for the time step estimate
    // and the plotfiles
    auto& state = m_leveldata_new[lev]->state;
    auto const& vel_ma = state.arrays();
    amrex::ParallelFor(
      state, [vel_ma, geomdata, lprobparm, a_cc_time] AMREX_GPU_DEVICE(
               int box_no, int i, int j, int k) noexcept {
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const amrex::IntVect ivp =
            iv + amrex::IntVect::TheDimensionVector(idim);
          vel_ma[box_no](i, j, k, VELX + idim) =
            0.5 * (prescribed_face_velocity<ProblemSpecificFunctions>(
                     idim, iv, a_cc_time, geomdata, *lprobparm) +
                   prescribed_face_velocity<ProblemSpecificFunctions>(
                     idim, ivp, a_cc_time, geomdata, *lprobparm));
        }
      });
  }
  amrex::Gpu::streamSynchronize();
}

amrex::Real
PeleLM::estVOFDt(const TimeStamp a_time)
{
  amrex::Real estdt = 1.0e200;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const auto dx = geom[lev].CellSizeArray();
    const amrex::Vector<amrex::Real> u_max =
      ldata_p->state.norm0({AMREX_D_DECL(VELX, VELY, VELZ)}, 0, true, true);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      if (u_max[idim] > 0.0) {
        estdt = amrex::min(estdt, m_vof_cfl * dx[idim] / u_max[idim]);
      }
    }
  }
  amrex::ParallelDescriptor::ReduceRealMin(estdt);
  return estdt;
}

void
PeleLM::VOFTagInterface(const int lev, amrex::TagBoxArray& tags)
{
  const int maxlev =
    (m_vof_refine_maxlev < 0) ? max_level : m_vof_refine_maxlev;
  if (lev >= maxlev) {
    return;
  }

  const int nband = m_vof_refine_band;
  amrex::MultiFab vof(grids[lev], dmap[lev], 1, nband);
  fillpatch_vof(lev, getTime(lev, AmrNewTime), vof, nband);

  auto const& tag_ma = tags.arrays();
  auto const& vof_ma = vof.const_arrays();
  amrex::ParallelFor(
    tags, [tag_ma, vof_ma,
           nband] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      if (pelelmex::vof::interface_band(i, j, k, vof_ma[box_no], nband)) {
        tag_ma[box_no](i, j, k) = amrex::TagBox::SET;
      }
    });
  amrex::Gpu::streamSynchronize();
}

//-----------------------------------------------------------------------------
// Fill patch / average down
//-----------------------------------------------------------------------------
void
PeleLM::fillPatchVOF(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchVOF()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::Real time = getTime(lev, a_time);
    fillpatch_vof(lev, time, ldata_p->vof, m_nGrowVOF);
  }
}

// The wall ghost values read further than nGhost layers (VOFWallGhosts), so a
// smaller fill would see zeros there and give different values for the same
// ghost cell.  Always fill all the layers, and keep those requested.
template <typename Fill>
static void
fillVOFAllLayers(amrex::MultiFab& a_vof, const int nGhost, Fill&& fill)
{
  amrex::MultiFab full(
    a_vof.boxArray(), a_vof.DistributionMap(), 1, PeleLM::m_nGrowVOF,
    amrex::MFInfo(), a_vof.Factory());
  fill(full);
  amrex::MultiFab::Copy(a_vof, full, 0, 0, 1, nGhost);
}

void
PeleLM::fillpatch_vof(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_vof,
  const int nGhost)
{
  // The level data hold all their ghost cells at t_old and t_new, and this fill
  // is called many times in a step on the same F
  for (auto* ldata : {m_leveldata_new[lev].get(), m_leveldata_old[lev].get()}) {
    const amrex::Real t =
      (ldata == m_leveldata_new[lev].get()) ? m_t_new[lev] : m_t_old[lev];
    if (
      ldata != nullptr && a_time == t && &a_vof != &m_leveldata_new[lev]->vof &&
      &a_vof != &m_leveldata_old[lev]->vof &&
      a_vof.boxArray() == ldata->vof.boxArray() &&
      a_vof.DistributionMap() == ldata->vof.DistributionMap()) {
      amrex::MultiFab::Copy(a_vof, ldata->vof, 0, 0, 1, nGhost);
      return;
    }
  }
  if (nGhost < m_nGrowVOF) {
    fillVOFAllLayers(a_vof, nGhost, [&](amrex::MultiFab& full) {
      fillpatch_vof(lev, a_time, full, m_nGrowVOF);
    });
    return;
  }
  ProbParm const* lprobparm = prob_parm_d;
  amrex::Vector<amrex::BCRec> bcrec{m_bcrec_vof};

  using BndryFunc = amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>>>;

  if (lev == 0) {
    BndryFunc bndry_func(
      geom[lev], bcrec,
      PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>{lprobparm});
    amrex::FillPatchSingleLevel(
      a_vof, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->vof), &(m_leveldata_new[lev]->vof)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev], bndry_func, 0);
  } else {
    // Limited conservative linear interpolation keeps F within [0,1]
    BndryFunc crse_bndry_func(
      geom[lev - 1], bcrec,
      PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>{lprobparm});
    BndryFunc fine_bndry_func(
      geom[lev], bcrec,
      PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>{lprobparm});
    amrex::FillPatchTwoLevels(
      a_vof, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->vof), &(m_leveldata_new[lev - 1]->vof)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->vof), &(m_leveldata_new[lev]->vof)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
      &amrex::cell_cons_interp, bcrec, 0);
  }

  a_vof.EnforcePeriodicity(geom[lev].periodicity());
  VOFWallGhosts(lev, a_vof, nGhost);
}

void
PeleLM::fillcoarsepatch_vof(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_vof,
  const int nGhost)
{
  if (nGhost < m_nGrowVOF) {
    fillVOFAllLayers(a_vof, nGhost, [&](amrex::MultiFab& full) {
      fillcoarsepatch_vof(lev, a_time, full, m_nGrowVOF);
    });
    return;
  }
  AMREX_ASSERT(lev > 0);
  ProbParm const* lprobparm = prob_parm_d;
  amrex::Vector<amrex::BCRec> bcrec{m_bcrec_vof};

  using BndryFunc = amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>>>;
  BndryFunc crse_bndry_func(
    geom[lev - 1], bcrec,
    PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>{lprobparm});
  BndryFunc fine_bndry_func(
    geom[lev], bcrec,
    PeleLMCCFillExtDirVOF<ProblemSpecificFunctions>{lprobparm});
  amrex::InterpFromCoarseLevel(
    a_vof, amrex::IntVect(nGhost), a_time, m_leveldata_new[lev - 1]->vof, 0, 0,
    1, geom[lev - 1], geom[lev], crse_bndry_func, 0, fine_bndry_func, 0,
    refRatio(lev - 1), &amrex::cell_cons_interp, bcrec, 0);
  VOFWallGhosts(lev, a_vof, nGhost);
}

//-----------------------------------------------------------------------------
// Wall ghost cells from the extended PLIC plane of the interior interface
// (pelelmex::vof::wall_ghost_fraction): the interface meets the wall at
// whatever angle it has, as it should without surface tension.  Extrapolated
// ghost values impose 90 degrees instead, and the inconsistent near-wall
// normals unbalance gravity at a tilted interface.  Directions are processed
// in turn, so edge and corner ghosts see the ghosts of earlier directions.
//
// A ghost value depends on cells further along the wall than a_vof has ghost
// cells, so that a_vof alone would give different values in neighbouring boxes.
// The fill therefore works on a copy where the boxes extend through the walls:
// each wall ghost cell belongs to one box, its ghost layers cover the whole
// search, and they are exchanged between passes.  The result does not depend on
// the box layout.
//-----------------------------------------------------------------------------
void
PeleLM::VOFWallGhosts(const int lev, amrex::MultiFab& a_vof, const int nGhost)
{
  if (nGhost == 0) {
    return;
  }
  const auto is_wall = [&](const int dir, const int sgn) {
    const int bc = (sgn < 0) ? m_phys_bc.lo(dir) : m_phys_bc.hi(dir);
    return !geom[lev].isPeriodic(dir) &&
           (bc == BoundaryCondition::BCSlipWallAdiab ||
            bc == BoundaryCondition::BCNoSlipWallAdiab ||
            bc == BoundaryCondition::BCSlipWallIsotherm ||
            bc == BoundaryCondition::BCNoSlipWallIsotherm);
  };
  bool has_wall = false;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    has_wall = has_wall || is_wall(dir, -1) || is_wall(dir, 1);
  }
  if (!has_wall) {
    return;
  }

  // Passes beyond the first refine the ghost values at shallow contact
  // angles, where the only available planes read ghost cells
  constexpr int npass = 4;
  // The ghost layers read: the search along the wall, the reconstruction
  // stencil of the plane it finds, and the ghost layers of a_vof themselves
  constexpr int nwall = m_nGrowVOF;
  constexpr int nread =
    pelelmex::vof::wall_nsearch + pelelmex::vof::hf_half + 1;
  const amrex::Box& domain = geom[lev].Domain();
  const auto periodicity = geom[lev].periodicity();

  // Boxes extended through the walls they touch.  They stay disjoint.
  amrex::BoxList boxes;
  for (int ib = 0; ib < a_vof.boxArray().size(); ++ib) {
    const amrex::Box b = a_vof.boxArray()[ib];
    amrex::Box ext = b;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (is_wall(dir, -1) && b.smallEnd(dir) == domain.smallEnd(dir)) {
        ext.growLo(dir, nwall);
      }
      if (is_wall(dir, 1) && b.bigEnd(dir) == domain.bigEnd(dir)) {
        ext.growHi(dir, nwall);
      }
    }
    boxes.push_back(ext);
  }
  amrex::BoxArray ext_ba(std::move(boxes));
  amrex::MultiFab work(ext_ba, a_vof.DistributionMap(), 1, nread);
  amrex::MultiFab snap(ext_ba, a_vof.DistributionMap(), 1, nread);
  work.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_vof); mfi.isValid(); ++mfi) {
    const amrex::Box bx = amrex::grow(mfi.validbox(), nGhost);
    work[mfi].copy<amrex::RunOn::Device>(a_vof[mfi], bx, 0, bx, 0, 1);
  }
  work.FillBoundary(periodicity);

  // The domain and its ghost layers, without bounds in periodic directions
  amrex::Box extent = domain;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    extent.grow(dir, geom[lev].isPeriodic(dir) ? 1000000 : nwall);
  }

  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    for (int sgn = -1; sgn <= 1; sgn += 2) {
      if (!is_wall(dir, sgn)) {
        continue;
      }
      const int iwall = (sgn < 0) ? domain.smallEnd(dir) : domain.bigEnd(dir);
      for (int pass = 0; pass < npass; ++pass) {
        amrex::MultiFab::Copy(snap, work, 0, 0, 1, nread);
        const bool allow_ghost = (pass > 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(work); mfi.isValid(); ++mfi) {
          amrex::Box gbx = mfi.validbox();
          if (sgn < 0) {
            gbx.setBig(dir, iwall - 1);
          } else {
            gbx.setSmall(dir, iwall + 1);
          }
          if (!gbx.ok()) {
            continue;
          }
          const amrex::Box limit = work[mfi].box() & extent;
          auto const& F = work.array(mfi);
          auto const& S = snap.const_array(mfi);
          amrex::ParallelFor(
            gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              amrex::ignore_unused(k);
              const amrex::IntVect g(AMREX_D_DECL(i, j, k));
              F(g) = pelelmex::vof::wall_ghost_fraction(
                g, dir, iwall, sgn, limit, S, allow_ghost);
            });
        }
        amrex::Gpu::streamSynchronize();
        work.FillBoundary(periodicity);
      }
    }
  }

  // Back to the ghost cells of a_vof beyond the walls
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_vof); mfi.isValid(); ++mfi) {
    const amrex::Box bx = amrex::grow(mfi.validbox(), nGhost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        if (!is_wall(dir, sgn)) {
          continue;
        }
        amrex::Box gbx = bx;
        if (sgn < 0) {
          gbx.setBig(dir, domain.smallEnd(dir) - 1);
        } else {
          gbx.setSmall(dir, domain.bigEnd(dir) + 1);
        }
        if (gbx.ok()) {
          a_vof[mfi].copy<amrex::RunOn::Device>(work[mfi], gbx, 0, gbx, 0, 1);
        }
      }
    }
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::averageDownVOF(const TimeStamp a_time)
{
  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataFine_p = getLevelDataPtr(lev, a_time);
    auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
    amrex::average_down(
      ldataFine_p->vof, ldataCrse_p->vof, 0, 1, refRatio(lev - 1));
  }
}
