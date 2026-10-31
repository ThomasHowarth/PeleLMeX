#include <PeleLMeX.H>
#include <PeleLMeX_BCfill.H>
#include <PeleLMeX_VOF_K.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_GMRES_MLMG.H>

// Liquid volume fraction BCs.  Components are Interior, Inflow, Outflow,
// Symmetry, SlipWallAdiab, NoSlipWallAdiab, SlipWallIsoTherm,
// NoSlipWallIsoTherm.  Walls extrapolate (90 degree contact angle).
constexpr int vof_bc[] = {amrex::BCType::int_dir,  amrex::BCType::ext_dir,
                          amrex::BCType::foextrap, amrex::BCType::reflect_even,
                          amrex::BCType::foextrap, amrex::BCType::foextrap,
                          amrex::BCType::foextrap, amrex::BCType::foextrap};

void
PeleLM::VOFReadParameters()
{
  amrex::ParmParse pp("peleLM.vof");
  pp.query("cfl", m_vof_cfl);
  pp.query("refine_max_level", m_vof_refine_maxlev);
  pp.query("refine_band", m_vof_refine_band);
  pp.query("v", m_vof_verbose);
  pp.query("advect_only", m_vof_advect_only);

  m_vof_rho_liq = m_vof_rho_gas = m_rho;
  m_vof_mu_liq = m_vof_mu_gas = m_mu;
  pp.query("rho_liquid", m_vof_rho_liq);
  pp.query("rho_gas", m_vof_rho_gas);
  pp.query("mu_liquid", m_vof_mu_liq);
  pp.query("mu_gas", m_vof_mu_gas);
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
  if (m_vof_cfl <= 0.0 || m_vof_cfl > 0.5) {
    amrex::Abort(
      "peleLM.vof.cfl must be in (0, 0.5]: the split PLIC advection is "
      "only bounded and conservative for a per-sweep Courant number <= 0.5");
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
  // The PLIC stencil of the cells on either side of a face needs 2 ghosts
  constexpr int nGhost = 2;
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
  return rho_fc;
}

//-----------------------------------------------------------------------------
// Momentum advection consistent with the VOF mass transport (Rudman 1998,
// as in Kynema-SGF): with m_f the face mass crossing over the step from the
// VOF sweeps and u_f the Godunov face velocity,
//   rho^{n+1} u** = rho^n u^n - sum_f m_f u_f / dx,
//   rho^{n+1}     = rho^n     - sum_f m_f / dx,
// so that momentum is transported with the same mass as the liquid, and a
// uniform velocity stays uniform across the interface.  Within one cell of
// the interface the face velocity is the upwind cell velocity at t^n (as
// the upwind Favre flux of Kynema-SGF), bounded by the cell values where a
// higher-order state, multiplied by the liquid mass flux, could overshoot.
// The result is returned as the advection term
// AofS = (u** - u^n) / dt (updateVelocity adds dt AofS).
//-----------------------------------------------------------------------------
void
PeleLM::VOFMomentumAdvTerm(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_faces)
{
  BL_PROFILE("PeleLMeX::VOFMomentumAdvTerm()");

  // Momentum fluxes m_f u_f; coarse faces under fine faces take the average
  // of the fine products
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> momflux(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab vof(grids[lev], dmap[lev], 1, 2);
    fillpatch_vof(lev, getTime(lev, AmrOldTime), vof, 2);
    auto const& vof_ma = vof.const_arrays();
    auto const& vel_ma = getLevelDataPtr(lev, AmrOldTime)->state.const_arrays();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const auto& mflux = advData->vof_mflux[lev][idim];
      momflux[lev][idim].define(
        mflux.boxArray(), dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(),
        Factory(lev));
      auto const& mom_ma = momflux[lev][idim].arrays();
      auto const& m_ma = mflux.const_arrays();
      auto const& face_ma = a_faces[lev][idim].const_arrays();
      auto const& umac_ma = advData->umac[lev][idim].const_arrays();
      const amrex::IntVect e = amrex::IntVect::TheDimensionVector(idim);
      amrex::ParallelFor(
        momflux[lev][idim], amrex::IntVect(0), AMREX_SPACEDIM,
        [mom_ma, m_ma, face_ma, umac_ma, vof_ma, vel_ma,
         e] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const amrex::IntVect ivm = iv - e;
          amrex::Real uf = face_ma[box_no](iv, n);
          if (
            pelelmex::vof::interface_band(i, j, k, vof_ma[box_no]) ||
            pelelmex::vof::interface_band(
              i - e[0], j - e[1], k - AMREX_D_PICK(0, 0, e[2]),
              vof_ma[box_no])) {
            uf = (umac_ma[box_no](iv) >= 0.0) ? vel_ma[box_no](ivm, VELX + n)
                                              : vel_ma[box_no](iv, VELX + n);
          }
          mom_ma[box_no](iv, n) = m_ma[box_no](iv) * uf;
        });
    }
  }
  amrex::Gpu::streamSynchronize();
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(momflux[lev]), GetArrOfPtrs(momflux[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
  }

  const amrex::Real dt = m_dt;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto dxinv = geom[lev].InvCellSizeArray();
    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto const& aofs_ma = advData->AofS[lev].arrays();
    auto const& vel_ma = ldataOld_p->state.const_arrays();
    auto const& vof_ma = ldataOld_p->vof.const_arrays();
    AMREX_D_TERM(
      auto const& mx_ma = advData->vof_mflux[lev][0].const_arrays();
      , auto const& my_ma = advData->vof_mflux[lev][1].const_arrays();
      , auto const& mz_ma = advData->vof_mflux[lev][2].const_arrays();)
    AMREX_D_TERM(auto const& px_ma = momflux[lev][0].const_arrays();
                 , auto const& py_ma = momflux[lev][1].const_arrays();
                 , auto const& pz_ma = momflux[lev][2].const_arrays();)
    amrex::ParallelFor(
      advData->AofS[lev],
      [=, rho_l = m_vof_rho_liq, rho_g = m_vof_rho_gas] AMREX_GPU_DEVICE(
        int box_no, int i, int j, int k) noexcept {
        const amrex::Real rho_o =
          pelelmex::vof::mixture_density(vof_ma[box_no](i, j, k), rho_l, rho_g);
        const amrex::Real rho_n =
          rho_o -
          (AMREX_D_TERM(
            (mx_ma[box_no](i + 1, j, k) - mx_ma[box_no](i, j, k)) * dxinv[0],
            +(my_ma[box_no](i, j + 1, k) - my_ma[box_no](i, j, k)) * dxinv[1],
            +(mz_ma[box_no](i, j, k + 1) - mz_ma[box_no](i, j, k)) * dxinv[2]));
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
          const amrex::Real div_mom = AMREX_D_TERM(
            (px_ma[box_no](i + 1, j, k, n) - px_ma[box_no](i, j, k, n)) *
              dxinv[0],
            +(py_ma[box_no](i, j + 1, k, n) - py_ma[box_no](i, j, k, n)) *
              dxinv[1],
            +(pz_ma[box_no](i, j, k + 1, n) - pz_ma[box_no](i, j, k, n)) *
              dxinv[2]);
          const amrex::Real u_o = vel_ma[box_no](i, j, k, VELX + n);
          const amrex::Real u_n = (rho_o * u_o - div_mom) / rho_n;
          aofs_ma[box_no](i, j, k, VELX + n) = (u_n - u_o) / dt;
        }
      });
  }
  amrex::Gpu::streamSynchronize();
}

//-----------------------------------------------------------------------------
// Face projection umac -= beta grad phi with div(beta grad phi) = div(umac),
// the MacProjector operator, solved with GMRES preconditioned by MLMG: plain
// MLMG stalls on the large face-coefficient jumps of a sharp, high density
// ratio interface.
//-----------------------------------------------------------------------------
void
PeleLM::VOFFaceProjection(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_umac,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_beta)
{
  BL_PROFILE("PeleLMeX::VOFFaceProjection()");

  if (macproj->needInitialization()) {
    amrex::LPInfo lpInfo;
    lpInfo.setMaxCoarseningLevel(m_mac_mg_max_coarsening_level);
    macproj->initProjector(lpInfo, GetVecOfArrOfConstPtrs(a_beta));
    macproj->setDomainBC(
      getMACProjectionBC(amrex::Orientation::low),
      getMACProjectionBC(amrex::Orientation::high));
  } else {
    macproj->updateBeta(GetVecOfArrOfConstPtrs(a_beta));
  }
  auto& linop = macproj->getLinOp();
  auto& mlmg = macproj->getMLMG();
  linop.setMaxOrder(m_mac_max_order);
  // Coarse faces under fine faces take the fine-face average, so that the
  // composite divergence is consistent across coarse-fine interfaces
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(a_umac[lev]), a_umac[lev - 1], refRatio(lev - 1),
      geom[lev - 1]);
  }
  amrex::Vector<amrex::MultiFab> phi(finest_level + 1);
  amrex::Vector<amrex::MultiFab> rhs(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    rhs[lev].define(grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    phi[lev].define(grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
    phi[lev].setVal(0.0);
    amrex::computeDivergence(
      rhs[lev], GetArrOfConstPtrs(a_umac[lev]), geom[lev]);
    rhs[lev].mult(-1.0);
    linop.setLevelBC(lev, nullptr);
  }
  // GMRESMLMG does not make a singular system solvable: remove the mean
  // (round-off for a closed domain) as MLMG would
  bool singular = true;
  for (const auto& bcs :
       {getMACProjectionBC(amrex::Orientation::low),
        getMACProjectionBC(amrex::Orientation::high)}) {
    for (const auto bc : bcs) {
      if (bc == amrex::LinOpBCType::Dirichlet) {
        singular = false;
      }
    }
  }
  if (singular) {
    const amrex::Real mean =
      rhs[0].sum() / static_cast<amrex::Real>(geom[0].Domain().numPts());
    for (int lev = 0; lev <= finest_level; ++lev) {
      rhs[lev].plus(-mean, 0, 1);
    }
  }
  {
    amrex::GMRESMLMG gmres(mlmg);
    gmres.setVerbose(m_macproj_verbose);
    gmres.solve(
      GetVecOfPtrs(phi), GetVecOfConstPtrs(rhs), m_mac_mg_rtol, m_mac_mg_atol);
    if (m_macproj_verbose > 0) {
      amrex::Print() << "   VOF face projection: " << gmres.getNumIters()
                     << " GMRES iterations\n";
    }
  }
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> flux(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      flux[lev][idim].define(
        a_umac[lev][idim]->boxArray(), dmap[lev], 1, 0, amrex::MFInfo(),
        Factory(lev));
    }
  }
  mlmg.prepareForFluxes(GetVecOfConstPtrs(phi));
  mlmg.getFluxes(GetVecOfArrOfPtrs(flux), GetVecOfPtrs(phi));
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      amrex::MultiFab::Add(*a_umac[lev][idim], flux[lev][idim], 0, 0, 1, 0);
    }
  }
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      GetArrOfConstPtrs(a_umac[lev]), a_umac[lev - 1], refRatio(lev - 1),
      geom[lev - 1]);
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
// Advection: Weymouth & Yue (2010) Eulerian-implicit split sweeps with PLIC
// fluxes.  Each sweep i updates
//   F <- F + (flux_lo - flux_hi) / dx + c * (a_hi - a_lo)
// where flux = u dt * (liquid fraction of the swept volume) is the liquid
// volume crossing the face per unit face area, a = u dt / dx is the face
// Courant number and c = H(F^n - 1/2) is frozen over the step.  Storing the
// flux per unit area (a length) makes it level independent, so the fine
// fluxes can be averaged onto the coarse faces.
// The dilatation terms c * d(a)/dx sum to c * div(u) dt = 0 over the sweeps,
// so the update is conservative for a discretely divergence-free umac.
// All levels advance with the same dt; fine face fluxes are averaged onto
// the coarse faces before each update, so the composite update conserves
// volume across coarse/fine boundaries.
//-----------------------------------------------------------------------------
void
PeleLM::advanceVOF(std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::advanceVOF()");

  const amrex::Real vol_old =
    (m_vof_verbose > 0) ? VOFLiquidVolume(AmrNewTime) : 0.0;

  // Working state is the new VOF, which holds t^n at this point
  fillPatchVOF(AmrNewTime);

  amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>> umac(
    finest_level + 1);
  amrex::Vector<amrex::MultiFab> cmask(finest_level + 1);
  amrex::Real courant_max = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      umac[lev][idim] = &(advData->umac[lev][idim]);
      courant_max = amrex::max(
        courant_max, umac[lev][idim]->norm0(0, 0, true) * m_dt *
                       geom[lev].InvCellSize(idim));
    }
    cmask[lev].define(grids[lev], dmap[lev], 1, 0);
    auto const& c_ma = cmask[lev].arrays();
    auto const& vof_ma = m_leveldata_new[lev]->vof.const_arrays();
    amrex::ParallelFor(
      cmask[lev], [c_ma, vof_ma] AMREX_GPU_DEVICE(
                    int box_no, int i, int j, int k) noexcept {
        c_ma[box_no](i, j, k) = (vof_ma[box_no](i, j, k) > 0.5) ? 1.0 : 0.0;
      });
  }
  amrex::Gpu::streamSynchronize();

  // The time step is estimated from the cell-centered velocity at t^n, while
  // the sweeps use the face velocity at t^{n+1/2}
  amrex::ParallelDescriptor::ReduceRealMax(courant_max);
  if (courant_max > 1.0) {
    amrex::Abort(
      "VOF: face Courant number " + std::to_string(courant_max) +
      " exceeds 1, the PLIC swept volume is undefined; reduce amr.cfl or "
      "peleLM.vof.cfl");
  }
  if (courant_max > 0.5) {
    amrex::Print() << "   VOF: WARNING face Courant number " << courant_max
                   << " exceeds 0.5, the volume fraction may be clipped\n";
  }

  // Cycle the sweep ordering from one step to the next to avoid a
  // directional bias; tied to the step number so that restarts reproduce it
  // (m_nstep is -1 during the initial iterations)
  advData->vof_mflux.resize(finest_level + 1);
  const int step = amrex::max(m_nstep, 0);
  for (int s = 0; s < AMREX_SPACEDIM; ++s) {
    const int dir = (step + s) % AMREX_SPACEDIM;
    VOFSplitSweep(dir, umac, cmask, advData->vof_mflux);
  }

  if (m_vof_verbose > 0) {
    const amrex::Real vol_new = VOFLiquidVolume(AmrNewTime);
    amrex::Print() << "   VOF: liquid volume " << vol_new
                   << ", relative change "
                   << (vol_new - vol_old) / amrex::max(vol_old, 1.0e-300)
                   << "\n";
  }
}

void
PeleLM::VOFSplitSweep(
  const int dir,
  const amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_umac,
  const amrex::Vector<amrex::MultiFab>& a_cmask,
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_mflux)
{
  BL_PROFILE("PeleLMeX::VOFSplitSweep()");

  const amrex::Real dt = m_dt;

  // Liquid volume crossing each face per unit face area
  amrex::Vector<amrex::MultiFab> flux(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Real dtdx = dt * geom[lev].InvCellSize(dir);
    flux[lev].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(dir)),
      dmap[lev], 1, 0);
    auto const& flux_ma = flux[lev].arrays();
    auto const& umac_ma = a_umac[lev][dir]->const_arrays();
    auto const& vof_ma = m_leveldata_new[lev]->vof.const_arrays();
    amrex::ParallelFor(
      flux[lev], [flux_ma, umac_ma, vof_ma, dt, dtdx, dir] AMREX_GPU_DEVICE(
                   int box_no, int i, int j, int k) noexcept {
        const amrex::Real u = umac_ma[box_no](i, j, k);
        flux_ma[box_no](i, j, k) =
          u * dt *
          pelelmex::vof::swept_fraction(i, j, k, dir, u * dtdx, vof_ma[box_no]);
      });
  }
  amrex::Gpu::streamSynchronize();

  // Coarse faces under fine faces take the fine-face average
  for (int lev = finest_level; lev > 0; --lev) {
    amrex::average_down_faces(
      flux[lev], flux[lev - 1], refRatio(lev - 1), geom[lev - 1]);
  }

  // Mass crossing each face: the gas fills the swept volume not occupied by
  // liquid.  With c frozen over the step, the divergence of the summed sweep
  // mass fluxes reproduces the change of the mixture density.
  for (int lev = 0; lev <= finest_level; ++lev) {
    a_mflux[lev][dir].define(flux[lev].boxArray(), dmap[lev], 1, 0);
    auto const& m_ma = a_mflux[lev][dir].arrays();
    auto const& flux_ma = flux[lev].const_arrays();
    auto const& umac_ma = a_umac[lev][dir]->const_arrays();
    amrex::ParallelFor(
      a_mflux[lev][dir],
      [m_ma, flux_ma, umac_ma, dt, rho_g = m_vof_rho_gas,
       drho =
         m_vof_rho_liq -
         m_vof_rho_gas] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        m_ma[box_no](i, j, k) = rho_g * umac_ma[box_no](i, j, k) * dt +
                                drho * flux_ma[box_no](i, j, k);
      });
  }

  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Real dxinv = geom[lev].InvCellSize(dir);
    const amrex::Real dtdx = dt * dxinv;
    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
    auto const& flux_ma = flux[lev].const_arrays();
    auto const& umac_ma = a_umac[lev][dir]->const_arrays();
    auto const& c_ma = a_cmask[lev].const_arrays();
    auto const& vof_ma = m_leveldata_new[lev]->vof.arrays();
    // Volume removed by clipping to [0,1], kept for diagnostics
    amrex::MultiFab clipped(grids[lev], dmap[lev], 1, 0);
    auto const& clip_ma = clipped.arrays();
    amrex::ParallelFor(
      a_cmask[lev],
      [flux_ma, umac_ma, c_ma, vof_ma, clip_ma, dxinv, dtdx,
       e] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        const amrex::IntVect ivp = iv + e;
        const amrex::Real a_lo = umac_ma[box_no](iv) * dtdx;
        const amrex::Real a_hi = umac_ma[box_no](ivp) * dtdx;
        amrex::Real& F = vof_ma[box_no](iv);
        F += (flux_ma[box_no](iv) - flux_ma[box_no](ivp)) * dxinv +
             c_ma[box_no](iv) * (a_hi - a_lo);
        const amrex::Real Fc =
          amrex::Clamp(F, amrex::Real(0.0), amrex::Real(1.0));
        clip_ma[box_no](iv) = F - Fc;
        F = Fc;
      });
    if (m_vof_verbose > 1) {
      // Covered cells are overwritten by the average down: ignore them
      if (lev < finest_level) {
        auto const& mask_ma = m_coveredMask[lev]->const_arrays();
        amrex::ParallelFor(
          clipped, [clip_ma, mask_ma] AMREX_GPU_DEVICE(
                     int box_no, int i, int j, int k) noexcept {
            clip_ma[box_no](i, j, k) *=
              static_cast<amrex::Real>(mask_ma[box_no](i, j, k));
          });
      }
      const amrex::Real cellvol = AMREX_D_TERM(
        geom[lev].CellSize(0), *geom[lev].CellSize(1), *geom[lev].CellSize(2));
      const amrex::Real clip_max = clipped.norm0();
      const amrex::Real clip_sum = clipped.sum() * cellvol;
      if (clip_max > 1.0e-12) {
        amrex::Print() << "   VOF: sweep " << dir << " level " << lev
                       << " clipped volume " << clip_sum << " (max cell "
                       << clip_max << ")\n";
      }
    }
  }
  amrex::Gpu::streamSynchronize();

  averageDownVOF(AmrNewTime);
  fillPatchVOF(AmrNewTime);
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
    fillpatch_vof(lev, time, ldata_p->vof, m_nGrowState);
  }
}

void
PeleLM::fillpatch_vof(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_vof,
  const int nGhost)
{
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
}

void
PeleLM::fillcoarsepatch_vof(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_vof,
  const int nGhost)
{
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
