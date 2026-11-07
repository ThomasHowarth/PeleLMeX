#include <PeleLMeX.H>
#include <PeleLMeX_BCfill.H>
#include <PeleLMeX_VOF_K.H>
#include <PeleLMeX_VOF_Unsplit_K.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_GMRES_MLMG.H>

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
  amrex::ParmParse pp("peleLM.vof");
  pp.query("cfl", m_vof_cfl);
  pp.query("refine_max_level", m_vof_refine_maxlev);
  pp.query("refine_band", m_vof_refine_band);
  pp.query("v", m_vof_verbose);
  pp.query("advect_only", m_vof_advect_only);
  std::string advection = "split";
  pp.query("advection", advection);
  if (advection == "unsplit") {
    m_vof_unsplit = 1;
  } else if (advection == "split") {
    m_vof_unsplit = 0;
  } else {
    amrex::Abort("peleLM.vof.advection must be 'split' or 'unsplit'");
  }

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
  pp.query("sigma", m_vof_sigma);
  if (m_vof_sigma < 0.0) {
    amrex::Abort("peleLM.vof.sigma must be non-negative");
  }
  // Outflow band, see VOFOutflowInnerBox
  pp.query("outflow_band", m_vof_outflow_band);
  if (m_vof_outflow_band < 0) {
    amrex::Abort("peleLM.vof.outflow_band must be non-negative");
  }
  // The sharp face density balances gravity exactly at any interface
  // orientation, but on the faces where the surface tension acts it can be
  // close to rho_g, which amplifies the curvature errors by up to
  // rho_l / rho_g: the arithmetic average is the default with surface
  // tension
  std::string face_density = (m_vof_sigma > 0.0) ? "arithmetic" : "sharp";
  pp.query("face_density", face_density);
  if (face_density == "sharp") {
    m_vof_sharp_face_density = 1;
  } else if (face_density == "arithmetic") {
    m_vof_sharp_face_density = 0;
  } else {
    amrex::Abort("peleLM.vof.face_density must be 'sharp' or 'arithmetic'");
  }
  if (m_vof_verbose > 0) {
    amrex::Print() << " VOF: surface tension " << m_vof_sigma
                   << ", projection face density " << face_density
                   << ", outflow band " << m_vof_outflow_band << " cells\n";
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
  if (m_vof_unsplit != 0) {
    // Temporary, while the unsplit scheme is brought up
    if (max_level > 0) {
      amrex::Abort("peleLM.vof.advection = unsplit does not support AMR yet "
                   "(amr.max_level = 0).");
    }
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

// Liquid volume V, centroid x_c = <F x> / V, second central moments
// <F (x_d - x_c,d)^2> / V and the bounding box of the cells with F > 1/2,
// over the uncovered cells of all levels, and the interface area
// sum |grad F| dV (central differences).  With surface tension, also the
// mean curvature of the interface cells and the difference between the mean
// pressures of the pure liquid and pure gas cells.  Used by the droplet
// verification cases (centroid trajectory -> drag, moments -> oscillation
// frequency, extent and area -> deformation and breakup, pressure jump ->
// Laplace).
void
PeleLM::VOFLiquidStats(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::VOFLiquidStats()");
  // F, F x_d, F x_d^2, then interface cells and their curvature, pure
  // liquid cells and their pressure, pure gas cells and their pressure, and
  // |grad F|
  constexpr int nmom = 1 + 2 * AMREX_SPACEDIM;
  constexpr int ntot = nmom + 7;
  const bool has_st = m_vof_sigma > 0.0;
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> mom(finest_level + 1);
  amrex::Real lo[AMREX_SPACEDIM];
  amrex::Real hi[AMREX_SPACEDIM];
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    lo[d] = std::numeric_limits<amrex::Real>::max();
    hi[d] = std::numeric_limits<amrex::Real>::lowest();
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& vof = getLevelDataPtr(lev, a_time)->vof;
    const auto geomdata = geom[lev].data();
    mom[lev] = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], ntot, 0, amrex::MFInfo(), Factory(lev));
    amrex::MultiFab kappa(
      grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
    kappa.setVal(0.0);
    amrex::MultiFab F(
      grids[lev], dmap[lev], 1, m_nGrowVOF, amrex::MFInfo(), Factory(lev));
    fillpatch_vof(lev, getTime(lev, a_time), F, m_nGrowVOF);
    if (has_st) {
      VOFCurvature(lev, F, kappa);
    }
    auto const& F_ma = F.const_arrays();
    auto const& vof_ma = vof.const_arrays();
    auto const& kappa_ma = kappa.const_arrays();
    auto const& p_ma =
      getLevelDataPtr(lev, AmrNewTime)->vof_press.const_arrays();
    auto const& mom_ma = mom[lev]->arrays();
    amrex::ParallelFor(
      *mom[lev],
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
          geomdata.ProbLo(0) + (i + 0.5) * geomdata.CellSize(0),
          geomdata.ProbLo(1) + (j + 0.5) * geomdata.CellSize(1),
          geomdata.ProbLo(2) + (k + 0.5) * geomdata.CellSize(2))};
        const amrex::Real f = vof_ma[box_no](i, j, k);
        mom_ma[box_no](i, j, k, 0) = f;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          mom_ma[box_no](i, j, k, 1 + d) = f * x[d];
          mom_ma[box_no](i, j, k, 1 + AMREX_SPACEDIM + d) = f * x[d] * x[d];
        }
        const amrex::Real mixed =
          (f > pelelmex::vof::vof_tol && f < 1.0 - pelelmex::vof::vof_tol)
            ? 1.0
            : 0.0;
        const amrex::Real liq = (f >= 1.0 - pelelmex::vof::vof_tol) ? 1.0 : 0.0;
        const amrex::Real gas = (f <= pelelmex::vof::vof_tol) ? 1.0 : 0.0;
        const amrex::Real p = p_ma[box_no](i, j, k);
        mom_ma[box_no](i, j, k, nmom) = mixed;
        mom_ma[box_no](i, j, k, nmom + 1) = mixed * kappa_ma[box_no](i, j, k);
        mom_ma[box_no](i, j, k, nmom + 2) = liq;
        mom_ma[box_no](i, j, k, nmom + 3) = liq * p;
        mom_ma[box_no](i, j, k, nmom + 4) = gas;
        mom_ma[box_no](i, j, k, nmom + 5) = gas * p;
        const auto& Fa = F_ma[box_no];
        amrex::Real g2 = 0.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          const amrex::IntVect e = amrex::IntVect::TheDimensionVector(d);
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const amrex::Real g =
            0.5 * (Fa(iv + e) - Fa(iv - e)) / geomdata.CellSize(d);
          g2 += g * g;
        }
        mom_ma[box_no](i, j, k, nmom + 6) = std::sqrt(g2);
      });
    // Bounding box of the liquid cells not covered by a finer level
    const int masked = (lev < finest_level) ? 1 : 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      const amrex::Real dx = geom[lev].CellSize(d);
      const amrex::Real plo = geom[lev].ProbLo(d);
      amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax> reduce_op;
      amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
      using ReduceTuple = typename decltype(reduce_data)::Type;
      for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& F = vof.const_array(mfi);
        auto const& mask = masked != 0 ? m_coveredMask[lev]->const_array(mfi)
                                       : amrex::Array4<int const>{};
        reduce_op.eval(
          bx, reduce_data,
          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> ReduceTuple {
            const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            const bool use =
              F(i, j, k) > 0.5 && (masked == 0 || mask(i, j, k) != 0);
            if (!use) {
              return {
                std::numeric_limits<amrex::Real>::max(),
                std::numeric_limits<amrex::Real>::lowest()};
            }
            return {plo + iv[d] * dx, plo + (iv[d] + 1) * dx};
          });
      }
      const auto r = reduce_data.value();
      lo[d] = amrex::min(lo[d], amrex::get<0>(r));
      hi[d] = amrex::max(hi[d], amrex::get<1>(r));
    }
  }
  amrex::ParallelDescriptor::ReduceRealMin(lo, AMREX_SPACEDIM);
  amrex::ParallelDescriptor::ReduceRealMax(hi, AMREX_SPACEDIM);

  amrex::Real sums[ntot];
  for (int n = 0; n < ntot; ++n) {
    sums[n] = MFSum(GetVecOfConstPtrs(mom), n);
  }
  const amrex::Real vol = amrex::max(sums[0], 1.0e-300);
  amrex::Print() << "   VOF: time " << getTime(0, a_time) << " centroid";
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    amrex::Print() << " " << sums[1 + d] / vol;
  }
  amrex::Print() << " moments";
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const amrex::Real xc = sums[1 + d] / vol;
    amrex::Print() << " " << sums[1 + AMREX_SPACEDIM + d] / vol - xc * xc;
  }
  amrex::Print() << " extent";
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    amrex::Print() << " " << lo[d] << " " << hi[d];
  }
  amrex::Print() << " area " << sums[nmom + 6];
  if (has_st) {
    const auto mean = [&](const int n) {
      return sums[n + 1] / amrex::max(sums[n], 1.0e-300);
    };
    amrex::Print() << " curvature " << mean(nmom) << " pressure_jump "
                   << mean(nmom + 2) - mean(nmom + 4);
  }
  amrex::Print() << "\n";
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

//-----------------------------------------------------------------------------
// The cells of the level outside the outflow band, the m_vof_outflow_band
// cells next to each outflow boundary.  A prescribed outflow pressure cannot
// carry an interface across the boundary: the liquid reaching the band is
// removed (absorbing outflow, VOFOutflowSink) and the surface tension is
// off there.
//-----------------------------------------------------------------------------
amrex::Box
PeleLM::VOFOutflowInnerBox(const int lev) const
{
  amrex::Box inner = geom[lev].Domain();
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (geom[lev].isPeriodic(d)) {
      continue;
    }
    if (m_phys_bc.lo(d) == amrex::PhysBCType::outflow) {
      inner.growLo(d, -m_vof_outflow_band);
    }
    if (m_phys_bc.hi(d) == amrex::PhysBCType::outflow) {
      inner.growHi(d, -m_vof_outflow_band);
    }
  }
  return inner;
}

//-----------------------------------------------------------------------------
// Absorbing outflow: remove the liquid in the outflow band of each level
// (VOFOutflowInnerBox) and return the volume removed (peleLM.vof.v > 0,
// zero otherwise).
//-----------------------------------------------------------------------------
amrex::Real
PeleLM::VOFOutflowSink()
{
  const bool verbose = m_vof_verbose > 0;
  const amrex::Real vol_before = verbose ? VOFLiquidVolume(AmrNewTime) : 0.0;
  bool has_band = false;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box inner = VOFOutflowInnerBox(lev);
    if (inner == geom[lev].Domain()) {
      continue;
    }
    has_band = true;
    auto const& vof_ma = m_leveldata_new[lev]->vof.arrays();
    amrex::ParallelFor(
      m_leveldata_new[lev]->vof,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        if (!inner.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
          vof_ma[box_no](i, j, k) = 0.0;
        }
      });
  }
  amrex::Gpu::streamSynchronize();
  if (!has_band) {
    return 0.0;
  }
  averageDownVOF(AmrNewTime);
  return verbose ? vol_before - VOFLiquidVolume(AmrNewTime) : 0.0;
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFProjFaceDensity(const int lev, const TimeStamp a_time)
{
  if (m_vof_sharp_face_density != 0) {
    return getVOFSharpFaceDensity(lev, a_time);
  }
  // The volume fraction is linear in time between t^n and t^{n+1}
  const amrex::Real time = (a_time == AmrHalfTime)
                             ? 0.5 * (m_t_old[lev] + m_t_new[lev])
                             : getTime(lev, a_time);
  return getVOFFaceDensity(lev, time);
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
// Momentum advection consistent with the VOF mass transport (Rudman 1998; in
// the split form of Arrufat et al., JCP 2021): the VOF split sweeps are
// replayed on the mixture density and the momentum with the sweep face mass
// fluxes m_f,
//   rho'     = rho     - (m_hi - m_lo) / dx           + rho_c (a_hi - a_lo),
//   (rho u)' = (rho u) - (m_hi u_hi - m_lo u_lo) / dx + rho_c u_c (a_hi -
//   a_lo),
// with a = u dt / dx the face Courant number, rho_c the phase density of
// c = H(F - 1/2) and u_c the velocity, both frozen over each (sub-)step.
// rho' is then the mixture density of the swept F, the dilatation terms
// vanish over the sweeps of a (sub-)step for a divergence-free umac, and a
// uniform velocity stays uniform across the interface.
// Within one cell of the interface u_f is the upwind *intermediate* cell
// velocity (the upwind Favre flux of Kynema-SGF), bounded by the cell values
// where a higher-order state, multiplied by the liquid mass flux, could
// overshoot.  Liquid that crosses a gas cell within a step (in through one
// face in one sweep, out through another in the next) leaves with the
// velocity it came in with; upwinding with the t^n velocities instead charges
// it with the gas velocity, an error amplified by rho_l / rho_g.  Elsewhere
// u_f is the Godunov face velocity.
// Intermediate velocities are exchanged between fine-fine and periodic ghost
// cells after each sweep; coarse/fine and physical ghost cells keep t^n (the
// interface band is refined away from coarse/fine boundaries, and the wall
// mass fluxes vanish).  Coarse faces under fine faces take the average of the
// fine momentum fluxes, so the composite update is conservative.
// The result is returned as the advection term
// AofS = (u** - u^n) / dt (updateVelocity adds dt AofS).
//-----------------------------------------------------------------------------
void
PeleLM::VOFMomentumAdvTerm(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_faces)
{
  BL_PROFILE("PeleLMeX::VOFMomentumAdvTerm()");

  if (m_vof_unsplit != 0) {
    VOFMomentumAdvTermUnsplit(advData, a_faces);
    return;
  }

  const amrex::Real rho_l = m_vof_rho_liq;
  const amrex::Real rho_g = m_vof_rho_gas;
  const amrex::Real dt_sub = advData->vof_dt_sub;
  const int nsweep = static_cast<int>(advData->vof_sweep_dir.size());
  AMREX_ALWAYS_ASSERT(
    nsweep == static_cast<int>(advData->vof_cmask.size()) * AMREX_SPACEDIM);

  // Working mixture density and momentum, and the velocity with the upwind
  // ghost layer; u_c is the velocity at the start of each (sub-)step
  amrex::Vector<amrex::MultiFab> vof(finest_level + 1);
  amrex::Vector<amrex::MultiFab> rho(finest_level + 1);
  amrex::Vector<amrex::MultiFab> mom(finest_level + 1);
  amrex::Vector<amrex::MultiFab> vel(finest_level + 1);
  amrex::Vector<amrex::MultiFab> velc(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    vof[lev].define(grids[lev], dmap[lev], 1, 2);
    fillpatch_vof(lev, getTime(lev, AmrOldTime), vof[lev], 2);
    rho[lev].define(grids[lev], dmap[lev], 1, 0);
    mom[lev].define(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);
    vel[lev].define(grids[lev], dmap[lev], AMREX_SPACEDIM, 1);
    velc[lev].define(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);
    amrex::MultiFab::Copy(
      vel[lev], getLevelDataPtr(lev, AmrOldTime)->state, VELX, 0,
      AMREX_SPACEDIM, 1);
    auto const& vof_ma = vof[lev].const_arrays();
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

  for (int isweep = 0; isweep < nsweep; ++isweep) {
    const int dir = advData->vof_sweep_dir[isweep];
    const int isub = isweep / AMREX_SPACEDIM;
    const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
    if (isweep % AMREX_SPACEDIM == 0) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::MultiFab::Copy(velc[lev], vel[lev], 0, 0, AMREX_SPACEDIM, 0);
      }
    }

    // Momentum fluxes m_f u_f of this sweep
    amrex::Vector<amrex::MultiFab> momflux(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      const auto& mflux = advData->vof_mflux[isweep][lev];
      momflux[lev].define(
        mflux.boxArray(), dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(),
        Factory(lev));
      auto const& p_ma = momflux[lev].arrays();
      auto const& m_ma = mflux.const_arrays();
      auto const& face_ma = a_faces[lev][dir].const_arrays();
      auto const& umac_ma = advData->umac[lev][dir].const_arrays();
      auto const& vof_ma = vof[lev].const_arrays();
      auto const& vel_ma = vel[lev].const_arrays();
      amrex::ParallelFor(
        momflux[lev], amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const amrex::IntVect ivm = iv - e;
          amrex::Real uf = face_ma[box_no](iv, n);
          if (
            pelelmex::vof::interface_band(i, j, k, vof_ma[box_no]) ||
            pelelmex::vof::interface_band(
              i - e[0], j - e[1], k - AMREX_D_PICK(0, 0, e[2]),
              vof_ma[box_no])) {
            uf = (umac_ma[box_no](iv) >= 0.0) ? vel_ma[box_no](ivm, n)
                                              : vel_ma[box_no](iv, n);
          }
          p_ma[box_no](iv, n) = m_ma[box_no](iv) * uf;
        });
    }
    amrex::Gpu::streamSynchronize();
    for (int lev = finest_level; lev > 0; --lev) {
      amrex::average_down_faces(
        momflux[lev], momflux[lev - 1], refRatio(lev - 1), geom[lev - 1]);
    }

    // Sweep update
    for (int lev = 0; lev <= finest_level; ++lev) {
      const amrex::Real dxinv = geom[lev].InvCellSize(dir);
      const amrex::Real dtdx = dt_sub * dxinv;
      auto const& m_ma = advData->vof_mflux[isweep][lev].const_arrays();
      auto const& p_ma = momflux[lev].const_arrays();
      auto const& umac_ma = advData->umac[lev][dir].const_arrays();
      auto const& c_ma = advData->vof_cmask[isub][lev].const_arrays();
      auto const& rho_ma = rho[lev].arrays();
      auto const& mom_ma = mom[lev].arrays();
      auto const& vel_ma = vel[lev].arrays();
      auto const& velc_ma = velc[lev].const_arrays();
      amrex::ParallelFor(
        rho[lev],
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const amrex::IntVect ivp = iv + e;
          const amrex::Real dil =
            (umac_ma[box_no](ivp) - umac_ma[box_no](iv)) * dtdx;
          const amrex::Real rho_c = (c_ma[box_no](iv) > 0.5) ? rho_l : rho_g;
          const amrex::Real r = rho_ma[box_no](iv) -
                                (m_ma[box_no](ivp) - m_ma[box_no](iv)) * dxinv +
                                rho_c * dil;
          rho_ma[box_no](iv) = r;
          for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            const amrex::Real mn =
              mom_ma[box_no](iv, n) -
              (p_ma[box_no](ivp, n) - p_ma[box_no](iv, n)) * dxinv +
              rho_c * velc_ma[box_no](iv, n) * dil;
            mom_ma[box_no](iv, n) = mn;
            vel_ma[box_no](iv, n) = mn / r;
          }
        });
    }
    amrex::Gpu::streamSynchronize();
    for (int lev = 0; lev <= finest_level; ++lev) {
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
// Momentum advection with the fluxes of the unsplit VOF steps.  Each sub-step
// carries the mass that the flux volumes carry,
//   m_f = rho_g C_f + (rho_l - rho_g) L_f   (per cell volume)
// with C_f the face Courant number and L_f the liquid crossing the face, so
//   rho' = rho - sum (m_hi - m_lo),  (rho u)' = (rho u) - sum (m_hi u_hi - m_lo
//   u_lo)
// and a uniform velocity stays uniform.  Near the interface the momentum flux
// is the integral over the flux volume, each piece carrying the velocity of the
// cell it lies in: a flux volume reaches beyond the upwind cell, and a single
// upwind velocity would hand its momentum to the wrong mass.  Away from it u_f
// is the Godunov face state.  Single level for now.
//-----------------------------------------------------------------------------
void
PeleLM::VOFMomentumAdvTermUnsplit(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_faces)
{
  AMREX_ALWAYS_ASSERT(finest_level == 0);
  const int lev = 0;
  const auto periodicity = geom[lev].periodicity();
  const amrex::Real rho_l = m_vof_rho_liq;
  const amrex::Real rho_g = m_vof_rho_gas;
  const amrex::Real dt_sub = advData->vof_dt_sub;

  amrex::MultiFab rho(grids[lev], dmap[lev], 1, 0);
  amrex::MultiFab mom(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);
  amrex::MultiFab vel(grids[lev], dmap[lev], AMREX_SPACEDIM, 1);
  amrex::MultiFab::Copy(
    vel, getLevelDataPtr(lev, AmrOldTime)->state, VELX, 0, AMREX_SPACEDIM, 1);
  {
    auto const& vof_ma = advData->vof_fstart[0][lev].const_arrays();
    auto const& rho_ma = rho.arrays();
    auto const& mom_ma = mom.arrays();
    auto const& vel_ma = vel.const_arrays();
    amrex::ParallelFor(
      rho, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::Real r =
          pelelmex::vof::mixture_density(vof_ma[box_no](i, j, k), rho_l, rho_g);
        rho_ma[box_no](i, j, k) = r;
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
          mom_ma[box_no](i, j, k, n) = r * vel_ma[box_no](i, j, k, n);
        }
      });
  }
  amrex::Gpu::streamSynchronize();

  const int nsub = static_cast<int>(advData->vof_lflux.size());
  for (int isub = 0; isub < nsub; ++isub) {
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::IntVect e = amrex::IntVect::TheDimensionVector(dir);
      const amrex::Real dtdx = dt_sub * geom[lev].InvCellSize(dir);
      const auto& lflux = advData->vof_lflux[isub][lev][dir];

      // Mass and momentum crossing the faces, per cell volume
      amrex::MultiFab mflux(lflux.boxArray(), dmap[lev], 1, 0);
      amrex::MultiFab pflux(lflux.boxArray(), dmap[lev], AMREX_SPACEDIM, 0);
      auto const& F_mf = advData->vof_fstart[isub][lev];
      auto const& plane_mf = advData->vof_plane[isub][lev];
      auto const& back_mf = advData->vof_back[isub][lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(mflux); mfi.isValid(); ++mfi) {
        const amrex::Box& fbx = mfi.validbox();
        const amrex::Box cells = amrex::grow(amrex::enclosedCells(fbx), 1);
        auto const& mf = mflux.array(mfi);
        auto const& pf = pflux.array(mfi);
        auto const& lf = lflux.const_array(mfi);
        auto const& um = advData->umac[lev][dir].const_array(mfi);
        auto const& face = a_faces[lev][dir].const_array(mfi);
        auto const& vl = vel.const_array(mfi);
        auto const& F = F_mf.const_array(mfi);
        auto const& pl = plane_mf.const_array(mfi);
        auto const& bk = back_mf.const_array(mfi);
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
              pelelmex::vof::unsplit::face_mass_momentum_flux(
                i, j, k, dir, courant, F, pl, bk, vl, cells, rho_l, rho_g, m,
                p);
              mf(iv) = m;
              for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                pf(iv, n) = p[n];
              }
            } else {
              const amrex::Real m = rho_g * courant + (rho_l - rho_g) * lf(iv);
              mf(iv) = m;
              for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                pf(iv, n) = m * face(iv, n);
              }
            }
          });
      }
      amrex::Gpu::streamSynchronize();
      mflux.OverrideSync(periodicity);
      pflux.OverrideSync(periodicity);

      auto const& rho_ma = rho.arrays();
      auto const& mom_ma = mom.arrays();
      auto const& cm_ma = mflux.const_arrays();
      auto const& cp_ma = pflux.const_arrays();
      amrex::ParallelFor(
        rho, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const amrex::IntVect ivp = iv + e;
          rho_ma[box_no](iv) -= cm_ma[box_no](ivp) - cm_ma[box_no](iv);
          for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            mom_ma[box_no](iv, n) -=
              cp_ma[box_no](ivp, n) - cp_ma[box_no](iv, n);
          }
        });
      amrex::Gpu::streamSynchronize();
    }

    auto const& rho_ma = rho.const_arrays();
    auto const& mom_ma = mom.const_arrays();
    auto const& vel_ma = vel.arrays();
    amrex::ParallelFor(
      rho, amrex::IntVect(0), AMREX_SPACEDIM,
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        vel_ma[box_no](i, j, k, n) =
          mom_ma[box_no](i, j, k, n) / rho_ma[box_no](i, j, k);
      });
    amrex::Gpu::streamSynchronize();
    vel.FillBoundary(0, AMREX_SPACEDIM, amrex::IntVect(1), periodicity);
  }

  const amrex::Real dt = m_dt;
  auto const& aofs_ma = advData->AofS[lev].arrays();
  auto const& vel_ma = vel.const_arrays();
  auto const& velo_ma = getLevelDataPtr(lev, AmrOldTime)->state.const_arrays();
  amrex::ParallelFor(
    vel, amrex::IntVect(0), AMREX_SPACEDIM,
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
      aofs_ma[box_no](i, j, k, VELX + n) =
        (vel_ma[box_no](i, j, k, n) - velo_ma[box_no](i, j, k, VELX + n)) / dt;
    });
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
  const amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>>& a_beta,
  const amrex::Vector<amrex::MultiFab*>& a_phi)
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
  // GMRES stops on its own estimate of the residual in the 2-norm.  Convergence
  // is judged here on the divergence actually left in umac, solving again for
  // what remains, with a warning if the target is not reached: a wrong operator
  // (a face with two coefficients) gives a converged solve and a divergent
  // umac.
  constexpr int max_solves = 10;
  amrex::Real div_target = 0.0;
  amrex::Real div_left = 0.0;
  int nsolve = 0;
  // The pressure is the sum of the corrections of all the solves
  for (auto* phi_lev : a_phi) {
    phi_lev->setVal(0.0);
  }
  for (nsolve = 0; nsolve <= max_solves; ++nsolve) {
    amrex::Vector<amrex::MultiFab> phi(finest_level + 1);
    amrex::Vector<amrex::MultiFab> rhs(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      rhs[lev].define(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      phi[lev].define(
        grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
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
    // Measured before the mean is removed: a mean left in umac is divergence
    // too, and later solves cannot remove it
    div_left = 0.0;
    for (int lev = 0; lev <= finest_level; ++lev) {
      div_left = amrex::max(div_left, rhs[lev].norm0());
    }
    if (singular) {
      const amrex::Real mean =
        rhs[0].sum() / static_cast<amrex::Real>(geom[0].Domain().numPts());
      for (int lev = 0; lev <= finest_level; ++lev) {
        rhs[lev].plus(-mean, 0, 1);
      }
    }
    if (nsolve == 0) {
      div_target = amrex::max(m_mac_mg_rtol * div_left, m_mac_mg_atol);
    }
    if (div_left <= div_target || nsolve == max_solves) {
      break;
    }
    {
      amrex::GMRESMLMG gmres(mlmg);
      gmres.setVerbose(m_macproj_verbose);
      gmres.solve(
        GetVecOfPtrs(phi), GetVecOfConstPtrs(rhs), m_mac_mg_rtol,
        m_mac_mg_atol);
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
    if (!a_phi.empty()) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::MultiFab::Add(*a_phi[lev], phi[lev], 0, 0, 1, 0);
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
// Gravity, surface tension and the pressure gradient act on the faces:
//   u_f = <u*>_f + dt g + dt f_f / rho_f,   div(dt / rho_f grad p) = div(u_f),
//   a_f = g + (f_f - grad_f p) / rho_f,
// with the face density rho_f of getVOFProjFaceDensity and the
// surface tension f_f = sigma kappa_f grad_f F of getVOFSurfaceTension, so
// that a fluid at rest with a planar interface normal to g (sharp face
// density), or a drop of uniform curvature, gives a_f = 0 to solver
// tolerance.  The cell-centered
// velocity is updated with the average of the face accelerations of each
// cell, and gp holds g - <a_f>, i.e. (grad p - f) / rho, used as the lagged
// pressure-gradient forcing of the next step.  On entry the new velocity
// contains the lagged forcing dt (g - gp^{n-1/2}), which is removed first.
// The pressure is kept in vof_press for the plotfiles.
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
  const bool has_st = m_vof_sigma > 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box& domain = geom[lev].Domain();
    beta[lev] = getVOFProjFaceDensity(lev, rhoTime);
    // Surface tension from the volume fraction after the advection
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> st;
    if (has_st) {
      st = getVOFSurfaceTension(lev, getTime(lev, AmrNewTime));
    }
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
      auto const& beta_ma = beta[lev][idim].const_arrays();
      // beta = dt / rho_f: the surface tension increment is beta f_f
      auto const& st_ma = has_st ? st[idim].const_arrays() : beta_ma;
      amrex::ParallelFor(
        umac[lev][idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          const bool fixed =
            (fixed_lo && iv[idim] == dlo) || (fixed_hi && iv[idim] == dhi);
          amrex::Real du = dtg;
          if (has_st) {
            du += beta_ma[box_no](i, j, k) * st_ma[box_no](i, j, k);
          }
          um_ma[box_no](i, j, k) = ub_ma[box_no](i, j, k) + (fixed ? 0.0 : du);
        });
    }
  }
  amrex::Gpu::streamSynchronize();

  amrex::Vector<amrex::MultiFab*> press(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    press[lev] = &(getLevelDataPtr(lev, AmrNewTime)->vof_press);
  }
  VOFFaceProjection(GetVecOfArrOfPtrs(umac), beta, press);

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
// Interface curvature in the cells within one cell of the interface
// (interface_band), in the valid and first ghost cells of a_kappa.  a_vof
// must hold m_nGrowVOF filled ghost cells.  In order of preference:
//  1. the height-function curvature (cell_height_curvature), second-order
//     and also defined in the pure cells next to the interface, where the
//     columns hold the same heights up to a constant;
//  2. the average of the height-function curvatures of the 3^dim neighbors
//     that have one (Popinet, JCP 228, 2009), then of the 5^dim neighbors:
//     in 3D the 3x3x7 columns of the cells near the grid diagonals often
//     extend out of the 7-cell window, and a local fit there is accurate
//     but follows small interface perturbations, which drives spurious
//     currents unstable;
//  3. a least-squares parabola / paraboloid through the height and PLIC
//     interface points of the 3^dim stencil (parabolic_curvature), in the
//     valid cells;
//  4. -div(n) from the volume fraction gradient (thin, under-resolved
//     structures, e.g. ligaments about to pinch off).
// The ghost cells shared with other boxes of the level are then overwritten
// by the valid values, so that every face sees the same curvature from
// either side.  Zero away from the interface.
//-----------------------------------------------------------------------------
void
PeleLM::VOFCurvature(
  const int lev, const amrex::MultiFab& a_vof, amrex::MultiFab& a_kappa)
{
  BL_PROFILE("PeleLMeX::VOFCurvature()");
  AMREX_ALWAYS_ASSERT(a_vof.nGrow() >= m_nGrowVOF && a_kappa.nGrow() >= 1);
  const amrex::Real dxinv = geom[lev].InvCellSize(0);

  // Height-function curvature (component 0) and whether it is defined (1),
  // in the valid and first ghost cells, exchanged out to two ghost cells
  // for the neighbor averages
  amrex::MultiFab khf(a_kappa.boxArray(), a_kappa.DistributionMap(), 2, 2);
  khf.setVal(0.0);
  auto const& F_ma = a_vof.const_arrays();
  auto const& khf_ma = khf.arrays();
  amrex::ParallelFor(
    khf, amrex::IntVect(1),
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      const auto& F = F_ma[box_no];
      amrex::Real kap = 0.0;
      amrex::Real valid = 0.0;
      if (
        pelelmex::vof::interface_band(i, j, k, F) &&
        pelelmex::vof::cell_height_curvature(i, j, k, F, dxinv, kap)) {
        valid = 1.0;
      }
      khf_ma[box_no](i, j, k, 0) = valid > 0.5 ? kap : 0.0;
      khf_ma[box_no](i, j, k, 1) = valid;
    });
  amrex::Gpu::streamSynchronize();
  khf.FillBoundary(geom[lev].periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_kappa, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& gbx = mfi.growntilebox(1);
    const amrex::Box& vbx = mfi.validbox();
    auto const& F = a_vof.const_array(mfi);
    auto const& K = khf.const_array(mfi);
    auto const& kappa = a_kappa.array(mfi);
    amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real kap = 0.0;
      if (!pelelmex::vof::interface_band(i, j, k, F)) {
        kappa(i, j, k) = kap;
        return;
      }
      if (K(i, j, k, 1) > 0.5) {
        kappa(i, j, k) = K(i, j, k, 0);
        return;
      }
      // Average of the height-function curvatures within r cells
      const auto neighbor_average = [&](const int r, amrex::Real& avg) {
        amrex::Real sum = 0.0;
        int n = 0;
#if AMREX_SPACEDIM == 3
        for (int kk = -r; kk <= r; ++kk) {
#else
          {
            constexpr int kk = 0;
#endif
          for (int jj = -r; jj <= r; ++jj) {
            for (int ii = -r; ii <= r; ++ii) {
              if (
                K.contains(i + ii, j + jj, k + kk) &&
                K(i + ii, j + jj, k + kk, 1) > 0.5) {
                sum += K(i + ii, j + jj, k + kk, 0);
                ++n;
              }
            }
          }
        }
        if (n == 0) {
          return false;
        }
        avg = sum / static_cast<amrex::Real>(n);
        return true;
      };
      if (
        !neighbor_average(1, kap) && !neighbor_average(2, kap) &&
        (!vbx.contains(i, j, k) ||
         !pelelmex::vof::parabolic_curvature(i, j, k, F, dxinv, kap))) {
        // The fit reads planes in the 3^dim stencil: valid cells only
        kap = pelelmex::vof::normal_divergence_curvature(i, j, k, F, dxinv);
      }
      kappa(i, j, k) = kap;
    });
  }
  amrex::Gpu::streamSynchronize();
  a_kappa.FillBoundary(0, 1, amrex::IntVect(1), geom[lev].periodicity());
}

//-----------------------------------------------------------------------------
// Continuum surface force (Brackbill et al., JCP 100, 1992) in the
// balanced-force form of Francois et al. (JCP 213, 2006): on each face,
//   f_f = sigma kappa_f (F_i - F_{i-1}) / dx,
// the same two-point gradient as the face pressure gradient of the
// projection, so that the pressure jump sigma kappa balances it exactly
// for a uniform curvature.  kappa_f is taken from the interface (mixed)
// cells on either side: their average when both are mixed, the mixed one's
// otherwise (Popinet, JCP 228, 2009).  The pure cells next to an oblique
// interface often have no height function (in 3D, most of them) and only a
// fallback curvature, used when neither cell is mixed (interface on the
// face).  Zero on the faces of the cells within m_vof_outflow_band cells of
// an outflow boundary.
//-----------------------------------------------------------------------------
amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getVOFSurfaceTension(const int lev, const amrex::Real a_time)
{
  BL_PROFILE("PeleLMeX::getVOFSurfaceTension()");
  amrex::MultiFab F(
    grids[lev], dmap[lev], 1, m_nGrowVOF, amrex::MFInfo(), Factory(lev));
  fillpatch_vof(lev, a_time, F, m_nGrowVOF);
  amrex::MultiFab kappa(
    grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
  VOFCurvature(lev, F, kappa);

  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> st;
  auto const& F_ma = F.const_arrays();
  auto const& kappa_ma = kappa.const_arrays();
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    st[idim].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
      dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    const amrex::Real sdxinv = m_vof_sigma * geom[lev].InvCellSize(idim);
    auto const& st_ma = st[idim].arrays();
    // Cells with surface tension
    const amrex::Box inner = VOFOutflowInnerBox(lev);
    amrex::ParallelFor(
      st[idim], [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        const amrex::IntVect ivm =
          iv - amrex::IntVect::TheDimensionVector(idim);
        // The two cells of the face
        const amrex::Box fcells(ivm, iv);
        if (!inner.contains(fcells)) {
          st_ma[box_no](i, j, k) = 0.0;
          return;
        }
        const auto& Fa = F_ma[box_no];
        const auto& K = kappa_ma[box_no];
        const auto mixed = [](const amrex::Real f) {
          return f > pelelmex::vof::vof_tol && f < 1.0 - pelelmex::vof::vof_tol;
        };
        const bool mm = mixed(Fa(ivm));
        const bool mp = mixed(Fa(iv));
        const amrex::Real kf = (mm == mp) ? 0.5 * (K(ivm) + K(iv))
                               : mm       ? K(ivm)
                                          : K(iv);
        st_ma[box_no](i, j, k) = sdxinv * kf * (Fa(iv) - Fa(ivm));
      });
  }
  amrex::Gpu::streamSynchronize();
  return st;
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

  // Largest face velocity / dx.  The step is estimated from the
  // cell-centered velocity at t^n, while the sweeps use the face velocity at
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

  // The split sweeps keep F in [0,1] for a per-sweep Courant number of at
  // most 1/2: sub-step the advection when the face velocity exceeds it
  const amrex::Real courant_max = umac_rate * m_dt;
  const int nsub =
    amrex::max(1, static_cast<int>(std::ceil(courant_max / 0.5)));
  if (nsub > 1) {
    amrex::Print() << "   VOF: face Courant number " << courant_max
                   << " exceeds 0.5, advecting in " << nsub << " sub-steps\n";
  }
  const amrex::Real dt_sub = m_dt / static_cast<amrex::Real>(nsub);

  // Sweep mass fluxes and c masks are kept for the momentum advection, which
  // replays the sweeps (VOFMomentumAdvTerm)
  advData->vof_dt_sub = dt_sub;
  advData->vof_mflux.clear();
  advData->vof_sweep_dir.clear();
  advData->vof_cmask.clear();
  advData->vof_lflux.clear();
  advData->vof_fstart.clear();
  advData->vof_plane.clear();
  advData->vof_back.clear();
  if (m_vof_unsplit == 0) {
    advData->vof_cmask.resize(nsub);
  } else {
    advData->vof_lflux.resize(nsub);
    advData->vof_fstart.resize(nsub);
    advData->vof_plane.resize(nsub);
    advData->vof_back.resize(nsub);
  }
  for (int isub = 0; isub < nsub; ++isub) {
    if (m_vof_unsplit != 0) {
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
      VOFUnsplitStep(
        dt_sub, umac, advData->vof_lflux[isub][0], advData->vof_plane[isub][0],
        advData->vof_back[isub][0]);
      continue;
    }

    // c = H(F - 1/2) frozen over the (sub-)step
    auto& cmask = advData->vof_cmask[isub];
    cmask.resize(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
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

    // Cycle the sweep ordering from one step to the next to avoid a
    // directional bias; tied to the step number so that restarts reproduce
    // it (m_nstep is -1 during the initial iterations)
    const int step = amrex::max(m_nstep, 0) + isub;
    for (int s = 0; s < AMREX_SPACEDIM; ++s) {
      const int dir = (step + s) % AMREX_SPACEDIM;
      auto& mflux = advData->vof_mflux.emplace_back(finest_level + 1);
      advData->vof_sweep_dir.push_back(dir);
      for (int lev = 0; lev <= finest_level; ++lev) {
        mflux[lev].define(advData->umac[lev][dir].boxArray(), dmap[lev], 1, 0);
      }
      VOFSplitSweep(dir, dt_sub, umac, cmask, mflux);
    }
  }

  const amrex::Real vol_sink = VOFOutflowSink();

  if (m_vof_verbose > 0) {
    const amrex::Real vol_new = VOFLiquidVolume(AmrNewTime);
    amrex::Print() << "   VOF: liquid volume " << vol_new
                   << ", relative change "
                   << (vol_new - vol_old) / amrex::max(vol_old, 1.0e-300);
    if (vol_sink != 0.0) {
      amrex::Print() << ", removed at the outflow " << vol_sink;
    }
    amrex::Print() << "\n";
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
    VOFLiquidStats(AmrNewTime);
  }
}

//-----------------------------------------------------------------------------
// One unsplit geometric step of the volume fraction in the new state: each face
// carries the liquid inside its flux volume, built from the mesh nodes traced
// back along the flow (see PeleLMeX_VOF_Unsplit_K.H).  Single level for now.
//-----------------------------------------------------------------------------
void
PeleLM::VOFUnsplitStep(
  const amrex::Real a_dt,
  const amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_umac,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
  amrex::MultiFab& plane,
  amrex::MultiFab& back)
{
  BL_PROFILE("PeleLMeX::VOFUnsplitStep()");
  AMREX_ALWAYS_ASSERT(finest_level == 0);
  const int lev = 0;

  const amrex::Box& domain = geom[lev].Domain();
  const auto periodicity = geom[lev].periodicity();
  const amrex::IntVect is_periodic(AMREX_D_DECL(
    geom[lev].isPeriodic(0), geom[lev].isPeriodic(1), geom[lev].isPeriodic(2)));
  const amrex::BoxArray nodal_ba =
    amrex::convert(grids[lev], amrex::IntVect::TheNodeVector());

  // Node velocities in Courant numbers: each component is the average of the
  // faces normal to it that touch the node, one-sided at domain boundaries.
  // A node shared by several boxes gets one value, so that it is traced to one
  // position and the flux volumes tile across boxes.
  amrex::MultiFab nvel(nodal_ba, dmap[lev], AMREX_SPACEDIM, 1);
  nvel.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(nvel); mfi.isValid(); ++mfi) {
    const amrex::Box& nbx = mfi.validbox();
    auto const& vel = nvel.array(mfi);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      auto const& um = a_umac[lev][dir]->const_array(mfi);
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
              if (inside) {
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
  // in cell units.  Positions beyond the domain use its nearest nodes,
  // except across periodic boundaries.
  amrex::Box node_limit = amrex::surroundingNodes(domain);
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (is_periodic[d] != 0) {
      node_limit.grow(d, 1000000);
    }
  }
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

  // Liquid crossing each face, in cell volumes
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    flux[dir].define(
      amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(dir)),
      dmap[lev], 1, 0);
    const amrex::Real dtdx = a_dt * geom[lev].InvCellSize(dir);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(flux[dir]); mfi.isValid(); ++mfi) {
      const amrex::Box& fbx = mfi.validbox();
      const amrex::Box cells = amrex::grow(amrex::enclosedCells(fbx), 1);
      auto const& fl = flux[dir].array(mfi);
      auto const& um = a_umac[lev][dir]->const_array(mfi);
      auto const& F = vof.const_array(mfi);
      auto const& pl = plane.const_array(mfi);
      auto const& bk = back.const_array(mfi);
      amrex::ParallelFor(
        fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          fl(i, j, k) = pelelmex::vof::unsplit::face_liquid_flux(
            i, j, k, dir, um(i, j, k) * dtdx, F, pl, bk, cells);
        });
    }
    // Faces between boxes compute the same value, up to round-off across a
    // periodic boundary: keep one, so that the volume is conserved exactly
    flux[dir].OverrideSync(periodicity);
  }
  amrex::Gpu::streamSynchronize();

  // Update, Eq. (18) of the paper.  F is not clipped or limited: it conserves
  // the liquid volume, and departs from [0,1] by the errors of the fluxes
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(vof); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.validbox();
    auto const& F = vof.array(mfi);
    AMREX_D_TERM(auto const& fx = flux[0].const_array(mfi);
                 , auto const& fy = flux[1].const_array(mfi);
                 , auto const& fz = flux[2].const_array(mfi););
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      F(i, j, k) += AMREX_D_TERM(
        fx(i, j, k) - fx(i + 1, j, k), +fy(i, j, k) - fy(i, j + 1, k),
        +fz(i, j, k) - fz(i, j, k + 1));
    });
  }
  amrex::Gpu::streamSynchronize();

  fillPatchVOF(AmrNewTime);
}

void
PeleLM::VOFSplitSweep(
  const int dir,
  const amrex::Real a_dt,
  const amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_umac,
  const amrex::Vector<amrex::MultiFab>& a_cmask,
  amrex::Vector<amrex::MultiFab>& a_mflux)
{
  BL_PROFILE("PeleLMeX::VOFSplitSweep()");

  const amrex::Real dt = a_dt;

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

  // Mass crossing each face in this sweep: the gas fills the swept volume
  // not occupied by liquid.  With the dilatation term rho_c (a_hi - a_lo),
  // rho_c = rho_l c + rho_g (1 - c), the sweep mass fluxes reproduce the
  // mixture density of the swept F
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& m_ma = a_mflux[lev].arrays();
    auto const& flux_ma = flux[lev].const_arrays();
    auto const& umac_ma = a_umac[lev][dir]->const_arrays();
    amrex::ParallelFor(
      a_mflux[lev],
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
        amrex::ignore_unused(k);
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
    // Clipping only compensates round-off when the sweeps are bounded
    // (per-sweep Courant number <= 1/2, divergence-free umac): report
    // anything larger.  Covered cells are overwritten by the average down.
    if (lev < finest_level) {
      auto const& mask_ma = m_coveredMask[lev]->const_arrays();
      amrex::ParallelFor(
        clipped, [clip_ma, mask_ma] AMREX_GPU_DEVICE(
                   int box_no, int i, int j, int k) noexcept {
          clip_ma[box_no](i, j, k) *=
            static_cast<amrex::Real>(mask_ma[box_no](i, j, k));
        });
    }
    const amrex::Real clip_max = clipped.norm0();
    if (clip_max > 1.0e-12) {
      const amrex::Real cellvol = AMREX_D_TERM(
        geom[lev].CellSize(0), *geom[lev].CellSize(1), *geom[lev].CellSize(2));
      amrex::Print() << "   VOF: WARNING sweep " << dir << " level " << lev
                     << " clipped volume " << clipped.sum() * cellvol
                     << " (max cell " << clip_max << ")\n";
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
    // Capillary waves (Brackbill et al., JCP 100, 1992): explicit surface
    // tension is stable for dt < sqrt(rho_avg dx^3 / (2 pi sigma))
    if (m_vof_sigma > 0.0) {
      const amrex::Real rho_avg = 0.5 * (m_vof_rho_liq + m_vof_rho_gas);
      estdt = amrex::min(
        estdt, std::sqrt(
                 rho_avg * dx[0] * dx[0] * dx[0] /
                 (2.0 * amrex::Math::pi<amrex::Real>() * m_vof_sigma)));
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
  VOFWallGhosts(lev, a_vof, nGhost);
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
