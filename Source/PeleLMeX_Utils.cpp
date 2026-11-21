#include <AMReX_buildInfo.H>
#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <hydro_utils.H>
#include <limits>
#include <memory>
#include <sstream>
#ifdef PELE_USE_PLASMA
#include <PeleLMeX_EF_Constants.H>
#endif
#ifdef PELE_USE_CMAKE
#include "PeleGitHashes.H"
#endif

void
writeBuildInfo()
{
  // build information
  std::cout << PrettyLine;
  std::cout << " PeleLMeX Build Information\n";
  std::cout << PrettyLine;

  std::cout << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
  std::cout << "build machine: " << amrex::buildInfoGetBuildMachine() << "\n";
  std::cout << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
  std::cout << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

  std::cout << "\n";

  std::cout << "COMP:          " << amrex::buildInfoGetComp() << "\n";
  std::cout << "COMP version:  " << amrex::buildInfoGetCompVersion() << "\n";

  std::cout << "\n";

  std::cout << "C++ compiler:  " << amrex::buildInfoGetCXXName() << "\n";
  std::cout << "C++ flags:     " << amrex::buildInfoGetCXXFlags() << "\n";

  std::cout << "\n";

  std::cout << "Link flags:    " << amrex::buildInfoGetLinkFlags() << "\n";
  std::cout << "Libraries:     " << amrex::buildInfoGetLibraries() << "\n";

  std::cout << "\n";

  for (int n = 1; n <= amrex::buildInfoGetNumModules(); ++n) {
    std::cout << amrex::buildInfoGetModuleName(n) << ": "
              << amrex::buildInfoGetModuleVal(n) << "\n";
  }

  std::cout << "\n";

#ifdef PELE_USE_CMAKE
  const std::string pele_hash = PeleBuildInfo::PeleLMeX_git_hash;
  const std::string amrex_hash = PeleBuildInfo::AMReX_git_hash;
  const std::string pelephysics_hash = PeleBuildInfo::PelePhysics_git_hash;
  const std::string amrex_hydro_hash = PeleBuildInfo::AMReXHydro_git_hash;
  const std::string sundials_hash = PeleBuildInfo::SUNDIALS_git_hash;
#else
  const std::string pele_hash = amrex::buildInfoGetGitHash(1);
  const std::string amrex_hash = amrex::buildInfoGetGitHash(2);
  const std::string pelephysics_hash = amrex::buildInfoGetGitHash(3);
  const std::string amrex_hydro_hash = amrex::buildInfoGetGitHash(4);
  const std::string sundials_hash = amrex::buildInfoGetGitHash(5);
#endif

  if (!pele_hash.empty()) {
    std::cout << "PeleLMeX     git hash: " << pele_hash << "\n";
  }
  if (!amrex_hash.empty()) {
    std::cout << "AMReX        git hash: " << amrex_hash << "\n";
  }
  if (!pelephysics_hash.empty()) {
    std::cout << "PelePhysics  git hash: " << pelephysics_hash << "\n";
  }
  if (!amrex_hydro_hash.empty()) {
    std::cout << "AMReX-Hydro  git hash: " << amrex_hydro_hash << "\n";
  }
  if (!sundials_hash.empty()) {
    std::cout << "SUNDIALS     git hash: " << sundials_hash << "\n";
  }

  const std::string buildgithash = amrex::buildInfoGetBuildGitHash();
  const std::string buildgitname = amrex::buildInfoGetBuildGitName();
  if (!buildgithash.empty()) {
    std::cout << buildgitname << " git hash: " << buildgithash << "\n";
  }

  std::cout << "\n\n";
}

void
PeleLM::fluxDivergence(
  const amrex::Vector<amrex::MultiFab*>& a_divergence,
  const int div_comp,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_fluxes,
  const int flux_comp,
  const int ncomp,
  const int intensiveFluxes,
  amrex::Real scale)
{
  fluxDivergence(
    a_divergence, div_comp, a_fluxes, flux_comp, {}, 0, ncomp, intensiveFluxes,
    scale);
}

// a_EBfluxes: empty, or the EB face fluxes of each level (intensive only)
void
PeleLM::fluxDivergence(
  const amrex::Vector<amrex::MultiFab*>& a_divergence,
  const int div_comp,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_fluxes,
  const int flux_comp,
  const amrex::Vector<amrex::MultiFab*>& a_EBfluxes,
  const int ebflux_comp,
  const int ncomp,
  const int intensiveFluxes,
  const amrex::Real scale)
{
  BL_PROFILE("PeleLMeX::fluxDivergence()");
  AMREX_ASSERT(a_EBfluxes.empty() || intensiveFluxes != 0);
  for (int lev = 0; lev <= finest_level; ++lev) {
    fluxDivergenceLevel(
      lev, *a_divergence[lev], div_comp, a_fluxes[lev], flux_comp,
      a_EBfluxes.empty() ? nullptr : a_EBfluxes[lev], ebflux_comp, ncomp,
      intensiveFluxes, scale);
  }
}

void
PeleLM::fluxDivergenceLevel(
  const int lev,
  amrex::MultiFab& a_divergence,
  const int div_comp,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const int flux_comp,
  const amrex::MultiFab* a_EBfluxes,
  const int ebflux_comp,
  const int ncomp,
  const int intensiveFluxes,
  const amrex::Real scale)
{
  if (intensiveFluxes != 0) { // Fluxes are intensive -> need area scaling
    intFluxDivergenceLevel(
      lev, a_divergence, div_comp, a_fluxes, flux_comp, ncomp, scale,
      a_EBfluxes, ebflux_comp);
  } else { // Fluxes are extensive
    extFluxDivergenceLevel(
      lev, a_divergence, div_comp, a_fluxes, flux_comp, ncomp, scale);
  }
}

void
PeleLM::fluxDivergenceRD(
  const amrex::Vector<const amrex::MultiFab*>& a_state,
  const int state_comp,
  const amrex::Vector<amrex::MultiFab*>& a_divergence,
  const int div_comp,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_fluxes,
  const int flux_comp,
  const amrex::Vector<amrex::MultiFab*>& a_EBfluxes,
  const int ebflux_comp,
  const int ncomp,
  const int intensiveFluxes,
  const amrex::BCRec* state_bc_d,
  const amrex::Real& scale,
  const amrex::Real& a_dt)
{
  BL_PROFILE("PeleLMeX::fluxDivergenceRD()");
#ifdef AMREX_USE_EB
  for (int lev = 0; lev <= finest_level; ++lev) {
    //----------------------------------------------------------------
    // Use a temporary MF to hold divergence before redistribution
    constexpr int nGrow_divTmp = 3;
    amrex::MultiFab divTmp(
      grids[lev], dmap[lev], ncomp, nGrow_divTmp, amrex::MFInfo(),
      EBFactory(lev));
    divTmp.setVal(0.0);
    fluxDivergenceLevel(
      lev, divTmp, 0, a_fluxes[lev], flux_comp,
      a_EBfluxes.empty() ? nullptr : a_EBfluxes[lev], ebflux_comp, ncomp,
      intensiveFluxes, scale);

    // Need FillBoundary before redistribution
    divTmp.FillBoundary(geom[lev].periodicity());

    // Redistribute diffusion term
    redistribute(
      lev, a_dt, divTmp, 0, *a_divergence[lev], div_comp, *a_state[lev],
      state_comp, ncomp, state_bc_d, geom[lev], m_diff_redist_type);
  }
#else
  amrex::ignore_unused(a_state);
  amrex::ignore_unused(state_comp);
  amrex::ignore_unused(state_bc_d);
  amrex::ignore_unused(a_dt);
  amrex::ignore_unused(a_EBfluxes);
  amrex::ignore_unused(ebflux_comp);
  fluxDivergence(
    a_divergence, div_comp, a_fluxes, flux_comp, ncomp, intensiveFluxes, scale);
#endif
}

void
PeleLM::extFluxDivergenceLevel(
  const int lev,
  amrex::MultiFab& a_divergence,
  const int div_comp,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const int flux_comp,
  const int ncomp,
  const amrex::Real scale)
{

  AMREX_ASSERT(a_divergence.nComp() >= div_comp + ncomp);

  // Get the volume
  amrex::MultiFab volume(grids[lev], dmap[lev], 1, 0);
  geom[lev].GetVolume(volume);

#ifdef AMREX_USE_EB
  auto const& ebfact = EBFactory(lev);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_divergence, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& bx = mfi.tilebox();
    AMREX_D_TERM(
      auto const& fluxX = a_fluxes[0]->const_array(mfi, flux_comp);
      , auto const& fluxY = a_fluxes[1]->const_array(mfi, flux_comp);
      , auto const& fluxZ = a_fluxes[2]->const_array(mfi, flux_comp););
    auto const& divergence = a_divergence.array(mfi, div_comp);
    auto const& vol = volume.const_array(mfi);
#ifdef AMREX_USE_EB
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& flag = flagfab.const_array();

    if (flagfab.getType(bx) == amrex::FabType::covered) { // Covered boxes
      amrex::ParallelFor(
        bx, ncomp,
        [divergence] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
          divergence(i, j, k, n) = 0.0;
        });
    } else if (flagfab.getType(bx) != amrex::FabType::regular) { // EB
                                                                 // containing
                                                                 // boxes
      auto vfrac = ebfact.getVolFrac().const_array(mfi);
      amrex::ParallelFor(
        bx, [flag, divergence, ncomp, vol, scale, vfrac, fluxX, fluxY
#if AMREX_SPACEDIM == 3
             ,
             fluxZ
#endif
      ] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (flag(i, j, k).isCovered()) { // Covered cell
            for (int n = 0; n < ncomp; ++n) {
              divergence(i, j, k, n) = 0.0;
            }
          } else { // Regular or cut cell
            extFluxDivergence_K(
              i, j, k, ncomp, AMREX_D_DECL(fluxX, fluxY, fluxZ), vol, scale,
              divergence);
            if (!flag(i, j, k).isRegular()) { // Cut cell
              const amrex::Real vfracinv = 1.0 / vfrac(i, j, k);
              for (int n = 0; n < ncomp; ++n) {
                divergence(i, j, k, n) *= vfracinv;
              }
            }
          }
        });
    } else // Regular boxes
#endif
    {
      amrex::ParallelFor(
        bx, [ncomp, vol, scale, divergence, fluxX, fluxY
#if AMREX_SPACEDIM == 3
             ,
             fluxZ
#endif
      ] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          extFluxDivergence_K(
            i, j, k, ncomp, AMREX_D_DECL(fluxX, fluxY, fluxZ), vol, scale,
            divergence);
        });
    }
  }
}

// Divergence of intensive (per unit area) face fluxes. a_EBfluxes, if not
// null, holds the flux through the EB face of cut cells (EB builds only).
void
PeleLM::intFluxDivergenceLevel(
  const int lev,
  amrex::MultiFab& a_divergence,
  const int div_comp,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const int flux_comp,
  const int ncomp,
  const amrex::Real scale,
  const amrex::MultiFab* a_EBfluxes,
  const int ebflux_comp)
{

  AMREX_ASSERT(a_divergence.nComp() >= div_comp + ncomp);

  // Get the volume
  amrex::MultiFab volume(grids[lev], dmap[lev], 1, 0);
  geom[lev].GetVolume(volume);

  // Get areas
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
    Geom(lev).CellSizeArray();
#if (AMREX_SPACEDIM == 2)
  amrex::MultiFab mf_ax, mf_ay;
  if (geom[lev].IsRZ()) {
    geom[lev].GetFaceArea(mf_ax, grids[lev], dmap[lev], 0, 0);
    geom[lev].GetFaceArea(mf_ay, grids[lev], dmap[lev], 1, 0);
  }
  amrex::Real areax = dx[1];
  amrex::Real areay = dx[0];
#elif (AMREX_SPACEDIM == 3)
  amrex::Real areax = dx[1] * dx[2];
  amrex::Real areay = dx[0] * dx[2];
  amrex::Real areaz = dx[0] * dx[1];
#endif

  // Get areafrac if EB
#ifdef AMREX_USE_EB
  auto const& ebfact = EBFactory(lev);
  amrex::Array<const amrex::MultiCutFab*, AMREX_SPACEDIM> areafrac;
  areafrac = ebfact.getAreaFrac();
  const auto* eb_area = &(ebfact.getBndryArea());
  const bool has_ebflux = (a_EBfluxes != nullptr);
#else
  AMREX_ASSERT(a_EBfluxes == nullptr);
  amrex::ignore_unused(a_EBfluxes, ebflux_comp);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_divergence, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& bx = mfi.tilebox();
    AMREX_D_TERM(
      auto const& fluxX = a_fluxes[0]->const_array(mfi, flux_comp);
      , auto const& fluxY = a_fluxes[1]->const_array(mfi, flux_comp);
      , auto const& fluxZ = a_fluxes[2]->const_array(mfi, flux_comp););
    auto const& divergence = a_divergence.array(mfi, div_comp);
    auto const& vol = volume.const_array(mfi);

#ifdef AMREX_USE_EB
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& flag = flagfab.const_array();

    if (flagfab.getType(bx) == amrex::FabType::covered) { // Covered boxes
      amrex::ParallelFor(
        bx, ncomp,
        [divergence] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
          divergence(i, j, k, n) = 0.0;
        });
    } else if (flagfab.getType(bx) != amrex::FabType::regular) { // EB
                                                                 // containing
                                                                 // boxes
      auto vfrac = ebfact.getVolFrac().const_array(mfi);
      AMREX_D_TERM(
        const auto& afrac_x = areafrac[0]->array(mfi);
        , const auto& afrac_y = areafrac[1]->array(mfi);
        , const auto& afrac_z = areafrac[2]->array(mfi););
      // Point at something when there is no EB flux
      auto const& ebflux =
        has_ebflux ? a_EBfluxes->const_array(mfi, ebflux_comp) : vol;
      const auto& ebarea = eb_area->array(mfi);
      amrex::ParallelFor(
        bx, [flag, divergence, vol, scale, ncomp, has_ebflux, ebflux, ebarea,
             dx, vfrac, fluxX, areax, afrac_x, fluxY, areay, afrac_y
#if (AMREX_SPACEDIM == 3)
             ,
             fluxZ, areaz, afrac_z
#endif
      ] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (flag(i, j, k).isCovered()) {
            for (int n = 0; n < ncomp; ++n) {
              divergence(i, j, k, n) = 0.0;
            }
          } else if (flag(i, j, k).isRegular()) {
            intFluxDivergence_K(
              i, j, k, ncomp, AMREX_D_DECL(fluxX, fluxY, fluxZ),
              AMREX_D_DECL(areax, areay, areaz), vol, scale, divergence);
          } else {
            const amrex::Real vfracinv = 1.0 / vfrac(i, j, k);
            if (has_ebflux) {
              EB_intFluxDivergence_K(
                i, j, k, ncomp, AMREX_D_DECL(fluxX, fluxY, fluxZ),
                AMREX_D_DECL(afrac_x, afrac_y, afrac_z),
                AMREX_D_DECL(areax, areay, areaz), ebflux, ebarea, vol, dx[0],
                scale, divergence);
            } else {
              EB_intFluxDivergence_K(
                i, j, k, ncomp, AMREX_D_DECL(fluxX, fluxY, fluxZ),
                AMREX_D_DECL(afrac_x, afrac_y, afrac_z),
                AMREX_D_DECL(areax, areay, areaz), vol, scale, divergence);
            }
            for (int n = 0; n < ncomp; ++n) {
              divergence(i, j, k, n) *= vfracinv;
            }
          }
        });
    } else // Regular boxes
#endif
    {
#if (AMREX_SPACEDIM == 2)
      if (geom[lev].IsRZ()) {
        amrex::Array4<amrex::Real const> const& ax = mf_ax.const_array(mfi);
        amrex::Array4<amrex::Real const> const& ay = mf_ay.const_array(mfi);
        amrex::ParallelFor(
          bx, [ncomp, ax, ay, fluxX, fluxY, vol, scale,
               divergence] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            intFluxDivergence_rz_K(
              i, j, k, ncomp, fluxX, fluxY, ax, ay, vol, scale, divergence);
          });
      } else
#endif
      {
        amrex::ParallelFor(
          bx, [ncomp, vol, scale, divergence, fluxX, areax, fluxY, areay
#if (AMREX_SPACEDIM == 3)
               ,
               fluxZ, areaz
#endif
        ] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            intFluxDivergence_K(
              i, j, k, ncomp, AMREX_D_DECL(fluxX, fluxY, fluxZ),
              AMREX_D_DECL(areax, areay, areaz), vol, scale, divergence);
          });
      }
    }
  }
}

void
PeleLM::advFluxDivergence(
  const int a_lev,
  amrex::MultiFab& a_divergence,
  const int div_comp,
  amrex::MultiFab& a_divu,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const int flux_comp,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_faceState,
  const int face_comp,
  const int ncomp,
  int const* l_conserv_d,
  const amrex::Real scale,
  const bool fluxes_are_area_weighted,
  const amrex::MultiFab* a_EBvelocity,
  const amrex::MultiFab* a_EBvalue) const
{
  BL_PROFILE("PeleLMeX::advFluxDivergence()");

  AMREX_ASSERT(a_divergence.nComp() >= div_comp + ncomp);
  AMREX_ASSERT(a_fluxes[0]->nComp() >= flux_comp + ncomp);
  AMREX_ASSERT(a_faceState[0]->nComp() >= face_comp + ncomp);
  // The EB wall velocity and value are given together or not at all
  AMREX_ASSERT((a_EBvelocity == nullptr) == (a_EBvalue == nullptr));

#if (AMREX_SPACEDIM == 2)
  amrex::MultiFab mf_ax, mf_ay;
  if (geom[a_lev].IsRZ()) {
    geom[a_lev].GetFaceArea(mf_ax, grids[a_lev], dmap[a_lev], 0, 0);
    geom[a_lev].GetFaceArea(mf_ay, grids[a_lev], dmap[a_lev], 1, 0);
  }
#else
  amrex::ignore_unused(a_lev);
#endif
#ifndef AMREX_USE_EB
  amrex::ignore_unused(a_EBvelocity, a_EBvalue);
#endif

#ifdef AMREX_USE_EB
  auto const& ebfact = EBFactory(a_lev);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_divergence, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {

    amrex::Box const& bx = mfi.tilebox();

    // Get the divergence
    auto const& div_arr = a_divergence.array(mfi, div_comp);
    AMREX_D_TERM(
      auto const& fx = a_fluxes[0]->const_array(mfi, flux_comp);
      , auto const& fy = a_fluxes[1]->const_array(mfi, flux_comp);
      , auto const& fz = a_fluxes[2]->const_array(mfi, flux_comp);)

#ifdef AMREX_USE_EB
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& vfrac_arr = ebfact.getVolFrac().const_array(mfi);
    if (
      flagfab.getType(bx) == amrex::FabType::singlevalued &&
      (a_EBvelocity != nullptr)) {
      // Cut cells with an EB velocity (moving/porous walls)
      HydroUtils::EB_ComputeDivergence(
        bx, div_arr, AMREX_D_DECL(fx, fy, fz), vfrac_arr, ncomp, geom[a_lev],
        scale, fluxes_are_area_weighted, a_EBvelocity->const_array(mfi),
        a_EBvalue->const_array(mfi),
        ebfact.getMultiEBCellFlagFab().const_array(mfi),
        ebfact.getBndryArea().const_array(mfi),
        ebfact.getBndryNormal().const_array(mfi));
    } else if (flagfab.getType(bx) == amrex::FabType::singlevalued) {
      HydroUtils::EB_ComputeDivergence(
        bx, div_arr, AMREX_D_DECL(fx, fy, fz), vfrac_arr, ncomp, geom[a_lev],
        scale, fluxes_are_area_weighted);
    } else if (flagfab.getType(bx) == amrex::FabType::regular)
#endif
    {
      HydroUtils::ComputeDivergence(
        bx, div_arr, AMREX_D_DECL(fx, fy, fz), ncomp, geom[a_lev], scale,
        fluxes_are_area_weighted);
    }

    // If convective, we define u dot grad q = div (u q) - q div(u)
    // averaging face and t^{n+1/2} q to the cell center
    auto const& divu_arr = a_divu.const_array(mfi);
    AMREX_D_TERM(
      auto const& facex = a_faceState[0]->const_array(mfi, face_comp);
      , auto const& facey = a_faceState[1]->const_array(mfi, face_comp);
      , auto const& facez = a_faceState[2]->const_array(mfi, face_comp);)

#ifdef AMREX_USE_EB
    if (flagfab.getType(bx) == amrex::FabType::covered) {
      AMREX_PARALLEL_FOR_4D(
        bx, ncomp, i, j, k, n, { div_arr(i, j, k, n) = 0.0; });
    } else if (flagfab.getType(bx) == amrex::FabType::singlevalued) {
      AMREX_D_TERM(
        auto const& apx_arr = ebfact.getAreaFrac()[0]->const_array(mfi);
        , auto const& apy_arr = ebfact.getAreaFrac()[1]->const_array(mfi);
        , auto const& apz_arr = ebfact.getAreaFrac()[2]->const_array(mfi););
      amrex::ParallelFor(
        bx, ncomp,
        [l_conserv_d, vfrac_arr, div_arr, divu_arr, apx_arr, facex, apy_arr,
         facey
#if (AMREX_SPACEDIM == 3)
         ,
         apz_arr, facez
#endif
      ] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
          if ((l_conserv_d[n] == 0) && vfrac_arr(i, j, k) > 0.) {
            const amrex::Real qwsum = AMREX_D_TERM(
              apx_arr(i, j, k) * facex(i, j, k, n) +
                apx_arr(i + 1, j, k) * facex(i + 1, j, k, n),
              +apy_arr(i, j, k) * facey(i, j, k, n) +
                apy_arr(i, j + 1, k) * facey(i, j + 1, k, n),
              +apz_arr(i, j, k) * facez(i, j, k, n) +
                apz_arr(i, j, k + 1) * facez(i, j, k + 1, n));
            const amrex::Real areasum = AMREX_D_TERM(
              apx_arr(i, j, k) + apx_arr(i + 1, j, k),
              +apy_arr(i, j, k) + apy_arr(i, j + 1, k),
              +apz_arr(i, j, k) + apz_arr(i, j, k + 1));
            // Note that because we define adv update as MINUS div(u q), here we
            // add q div (u)
            div_arr(i, j, k, n) += qwsum / areasum * divu_arr(i, j, k);
          }
        });
    } else
#endif
    {
#if (AMREX_SPACEDIM == 2)
      if (geom[a_lev].IsRZ()) {
        amrex::Array4<amrex::Real const> const& ax = mf_ax.const_array(mfi);
        amrex::Array4<amrex::Real const> const& ay = mf_ay.const_array(mfi);
        amrex::ParallelFor(
          bx, ncomp,
          [l_conserv_d, ax, ay, facex, facey, divu_arr,
           div_arr] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            if (l_conserv_d[n] == 0) {
              amrex::Real qavg = AMREX_D_TERM(
                ax(i, j, k) * facex(i, j, k, n) +
                  ax(i + 1, j, k) * facex(i + 1, j, k, n),
                +ay(i, j, k) * facey(i, j, k, n) +
                  ay(i, j + 1, k) * facey(i, j + 1, k, n), );
              const amrex::Real areasum =
                ax(i, j, k) + ax(i + 1, j, k) + ay(i, j, k) + ay(i, j + 1, k);
              qavg /= areasum;
              // Note that because we define adv update as MINUS div(u q), here
              // we add q div (u)
              div_arr(i, j, k, n) += qavg * divu_arr(i, j, k);
            }
          });
      } else
#endif
      {
        amrex::ParallelFor(
          bx, ncomp,
          [l_conserv_d, div_arr, divu_arr, facex, facey
#if (AMREX_SPACEDIM == 3)
           ,
           facez
#endif
        ] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            if (l_conserv_d[n] == 0) {
              amrex::Real qavg = AMREX_D_TERM(
                facex(i, j, k, n) + facex(i + 1, j, k, n),
                +facey(i, j, k, n) + facey(i, j + 1, k, n),
                +facez(i, j, k, n) + facez(i, j, k + 1, n));
              AMREX_D_PICK(qavg *= 0.5;, qavg *= 0.25;, qavg /= 6.0;)
              // Note that because we define adv update as MINUS div(u q), here
              // we add q div (u)
              div_arr(i, j, k, n) += qavg * divu_arr(i, j, k);
            }
          });
      }
    }
  }
}

void
PeleLM::floorSpecies(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::floorSpecies()");
  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);
  if (m_floor_species == 0) {
    return;
  }

  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, a_time);
    auto const& sma = ldata_p->state.arrays();
    auto const* leosparm = eos_parms.device_parm();

    amrex::ParallelFor(
      ldata_p->state, [sma, leosparm] AMREX_GPU_DEVICE(
                        int box_no, int i, int j, int k) noexcept {
        fabMinMax(
          i, j, k, NUM_SPECIES, 0.0, AMREX_REAL_MAX,
          amrex::Array4<amrex::Real>(sma[box_no], FIRSTSPEC));
#ifdef PELE_USE_PLASMA
        fabMinMax(
          i, j, k, 1, 0.0, AMREX_REAL_MAX,
          amrex::Array4<amrex::Real>(sma[box_no], NE));
#endif
        // Update density and RhoH accordingly ...
        amrex::Real massfrac[NUM_SPECIES] = {0.0};
        amrex::Real massdens[NUM_SPECIES] = {0.0};
        amrex::Real rhoinv, h_cgs = 0.0;
        for (int n = 0; n < NUM_SPECIES; ++n) {
          massdens[n] = sma[box_no](i, j, k, FIRSTSPEC + n);
        }
        auto eos = pele::physics::PhysicsType::eos(leosparm);
        eos.RY2RRinvY(
          massdens, sma[box_no](i, j, k, DENSITY), rhoinv, massfrac);
        eos.TY2H(sma[box_no](i, j, k, TEMP), massfrac, h_cgs);
        sma[box_no](i, j, k, RHOH) =
          h_cgs * 1.0e-4 * sma[box_no](i, j, k, DENSITY);
      });
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::resetCoveredMask()
{
  BL_PROFILE("PeleLMeX::resetCoveredMask()");
  if (m_resetCoveredMask != 0) {

    if (m_verbose != 0) {
      amrex::Print() << " Resetting fine-covered cells mask \n";
    }

    for (int lev = 0; lev < finest_level; ++lev) {
      // Set a fine-covered mask: 1 where uncovered, 0 under level lev+1
      *m_coveredMask[lev] = amrex::makeFineMask(
        grids[lev], dmap[lev], grids[lev + 1], ref_ratio[lev], 1, 0);
      amrex::BoxArray baf = grids[lev + 1];
      baf.coarsen(ref_ratio[lev]);

      //----------------------------------------------------------------------------
      // Setup a BoxArray for the chemistry

      // Get an uncovered BoxArray
      amrex::BoxArray baCompDom = complementIn(geom[lev].Domain(), baf);
      amrex::BoxArray baUnCovered = intersect(baCompDom, grids[lev]);
      // Chop in smaller boxes if triggered
      if (m_max_grid_size_chem.min() > 0) {
        baUnCovered.maxSize(m_max_grid_size_chem);
      }

      // Assemble a BoxArray with covered and uncovered ones + flags
      amrex::BoxList bl(grids[lev].ixType());
      bl.reserve(baUnCovered.size() + baf.size());
      m_baChemFlag[lev].resize(baUnCovered.size() + baf.size());
      int bxIdx = 0;
      for (int i = 0, n = static_cast<int>(baUnCovered.size()); i < n;
           ++i) { // Append uncovered boxes
        bl.push_back(baUnCovered[i]);
        m_baChemFlag[lev][bxIdx] = 1;
        bxIdx += 1;
      }
      for (int i = 0, n = static_cast<int>(baf.size()); i < n;
           ++i) { // Append covered boxes
        bl.push_back(baf[i]);
        m_baChemFlag[lev][bxIdx] = 0;
        bxIdx += 1;
      }
      m_baChem[lev] = std::make_unique<amrex::BoxArray>(std::move(bl));
      m_dmapChem[lev] =
        std::make_unique<amrex::DistributionMapping>(*m_baChem[lev]);

      // Load balancing of the chemistry DMap
      if (m_doLoadBalance != 0) {
        loadBalanceChemLev(lev);
      }
    }

    // Set a BoxArray for the chemistry on the finest level too
    m_baChem[finest_level] =
      std::make_unique<amrex::BoxArray>(grids[finest_level]);
    if (m_max_grid_size_chem.min() > 0) {
      m_baChem[finest_level]->maxSize(m_max_grid_size_chem);
    }
    m_baChemFlag[finest_level].resize(m_baChem[finest_level]->size());
    std::fill(
      m_baChemFlag[finest_level].begin(), m_baChemFlag[finest_level].end(), 1);
    m_dmapChem[finest_level] =
      std::make_unique<amrex::DistributionMapping>(*m_baChem[finest_level]);

    if ((m_doLoadBalance != 0) && m_max_grid_size_chem.min() > 0) {
      loadBalanceChemLev(finest_level);
    }

    // Switch off trigger
    m_resetCoveredMask = 0;

    // The uncovered volume depends on the hierarchy (detJ sampled per
    // level under mapping, vfrac per level under EB): recompute it.
    m_uncoveredVol = -1.0;

  } else {
    // Just load balance the chem. distribution map
    if (m_doLoadBalance != 0) {
      loadBalanceChem();
    }
  }

  //----------------------------------------------------------------------------
  // Need to compute the uncovered volume
  if (m_uncoveredVol < 0.0) {
    amrex::Vector<amrex::MultiFab> dummy;
    dummy.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      dummy.emplace_back(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), *m_factory[lev]);
      dummy[lev].setVal(1.0);
    }
    m_uncoveredVol = MFSum(GetVecOfConstPtrs(dummy), 0);
  }
}

void
PeleLM::loadBalanceChem()
{

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Finest grid uses AmrCore DM unless different max grid size specified.
    // Keep the AmrCore DM.
    if (lev == finest_level && m_max_grid_size_chem.min() < 0) {
      continue;
    }
    loadBalanceChemLev(lev);
  }
}

void
PeleLM::loadBalanceChemLev(const int a_lev)
{
  amrex::LayoutData<amrex::Real> new_cost(*m_baChem[a_lev], *m_dmapChem[a_lev]);
  computeCosts(a_lev, new_cost, m_loadBalanceCostChem);

  // Use efficiency: average MPI rank cost / max cost
  amrex::Real currentEfficiency = 0.0;
  amrex::Real testEfficiency = 0.0;

  amrex::DistributionMapping test_dmap;
  // Build the test dmap, w/o braodcasting
  if (m_loadBalanceMethodChem == LoadBalanceMethod::SFC) {
    test_dmap = amrex::DistributionMapping::makeSFC(
      new_cost, currentEfficiency, testEfficiency, false,
      amrex::ParallelDescriptor::IOProcessorNumber());

  } else if (m_loadBalanceMethodChem == LoadBalanceMethod::Knapsack) {

    const amrex::Real navg =
      static_cast<amrex::Real>(m_baChem[a_lev]->size()) /
      static_cast<amrex::Real>(amrex::ParallelDescriptor::NProcs());
    const int nmax = static_cast<int>(amrex::max<amrex::Real>(
      std::round(m_loadBalanceKSfactor * navg), std::ceil(navg)));
    test_dmap = amrex::DistributionMapping::makeKnapSack(
      new_cost, currentEfficiency, testEfficiency, nmax, false,
      amrex::ParallelDescriptor::IOProcessorNumber());
  }

  // IO proc determine if the test dmap offers significant improvements
  int updateDmap = 0;
  if (
    (m_loadBalanceEffRatioThreshold > 0.0) &&
    (amrex::ParallelDescriptor::MyProc() ==
     amrex::ParallelDescriptor::IOProcessorNumber())) {
    updateDmap = static_cast<int>(
      testEfficiency > m_loadBalanceEffRatioThreshold * currentEfficiency);
  }
  amrex::ParallelDescriptor::Bcast(
    &updateDmap, 1, amrex::ParallelDescriptor::IOProcessorNumber());

  if (m_verbose > 2 && (updateDmap != 0)) {
    amrex::Print() << "   Old Chem LoadBalancing efficiency on a_lev " << a_lev
                   << ": " << currentEfficiency << "\n"
                   << "   New Chem LoadBalancing efficiency: " << testEfficiency
                   << " \n";
  }

  // Bcast the test dmap if better
  if (updateDmap != 0) {
    amrex::Vector<int> pmap;
    if (
      amrex::ParallelDescriptor::MyProc() ==
      amrex::ParallelDescriptor::IOProcessorNumber()) {
      pmap = test_dmap.ProcessorMap();
    } else {
#pragma GCC diagnostic ignored "-Wnull-dereference"
      pmap.resize(static_cast<std::size_t>(m_baChem[a_lev]->size()));
    }
    amrex::ParallelDescriptor::Bcast(
      pmap.data(), pmap.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    if (
      amrex::ParallelDescriptor::MyProc() !=
      amrex::ParallelDescriptor::IOProcessorNumber()) {
      test_dmap = amrex::DistributionMapping(pmap);
    }
    m_dmapChem[a_lev] = std::make_unique<amrex::DistributionMapping>(test_dmap);
  }
}

// Return a unique_ptr with the entire derive
std::unique_ptr<amrex::MultiFab>
PeleLM::derive(
  const std::string& a_name,
  const amrex::Real a_time,
  const int lev,
  const int nGrow)
{
  BL_PROFILE("PeleLMeX::derive()");
  AMREX_ASSERT(nGrow >= 0);

  std::unique_ptr<amrex::MultiFab> mf;

  bool itexists = derive_lst.canDerive(a_name) || isStateVariable(a_name) ||
                  isReactVariable(a_name);

  if (!itexists) {
    amrex::Error("PeleLM::derive(): unknown variable: " + a_name);
  }

  const PeleLMDeriveRec* rec = derive_lst.get(a_name);

  if (rec != nullptr) { // This is a derived variable
    mf = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], rec->numDerive(), nGrow, amrex::MFInfo(),
      Factory(lev));
    std::unique_ptr<amrex::MultiFab> statemf =
      fillPatchState(lev, a_time, m_nGrowState);
    // Get pressure: TODO no fillpatch for pressure just yet, simply get new
    // state
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    std::unique_ptr<amrex::MultiFab> reactmf;
    if (m_incompressible == 0 && m_do_react != 0) {
      reactmf = fillPatchReact(lev, a_time, nGrow);
    } else {
      // No reaction data exists; derives that read it (HeatRelease) get
      // zeros of the right width rather than an unrelated array.
      reactmf = std::make_unique<amrex::MultiFab>(
        grids[lev], dmap[lev], nCompIR(), nGrow, amrex::MFInfo(), Factory(lev));
      reactmf->setVal(0.0);
    }
    auto stateBCs = fetchBCRecArray(VELX, NVAR);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.growntilebox(nGrow);
      amrex::FArrayBox& derfab = (*mf)[mfi];
      amrex::FArrayBox const& statefab = (*statemf)[mfi];
      amrex::FArrayBox const& reactfab = (*reactmf)[mfi];
      amrex::FArrayBox const& pressfab = ldata_p->press[mfi];
      rec->derFunc()(
        this, bx, derfab, 0, rec->numDerive(), statefab, reactfab, pressfab,
        geom[lev], a_time, stateBCs, lev);
    }
  } else if (isStateVariable(a_name)) { // This is a state variable
    mf = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
    int idx = stateVariableIndex(a_name);
    std::unique_ptr<amrex::MultiFab> statemf =
      fillPatchState(lev, a_time, nGrow);
    amrex::MultiFab::Copy(*mf, *statemf, idx, 0, 1, nGrow);
  } else { // This is a reaction variable
    mf = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
    int idx = reactVariableIndex(a_name);
    std::unique_ptr<amrex::MultiFab> reactmf =
      fillPatchReact(lev, a_time, nGrow);
    amrex::MultiFab::Copy(*mf, *reactmf, idx, 0, 1, nGrow);
  }

  return mf;
}

// Return a unique_ptr with only the required component of a derive
std::unique_ptr<amrex::MultiFab>
PeleLM::deriveComp(
  const std::string& a_name,
  const amrex::Real a_time,
  const int lev,
  const int nGrow)
{
  BL_PROFILE("PeleLMeX::deriveComp()");
  const PeleLMDeriveRec* rec = derive_lst.get(a_name);
  if (rec == nullptr) {
    // State and reaction variables are single components already
    return derive(a_name, a_time, lev, nGrow);
  }
  const int derComp = rec->variableComp(a_name);
  if (derComp < 0) {
    amrex::Error(
      "PeleLM::deriveComp(): unknown derive component: " + a_name + " of " +
      rec->variableName(1000));
  }
  const std::unique_ptr<amrex::MultiFab> all =
    derive(a_name, a_time, lev, nGrow);
  auto mf = std::make_unique<amrex::MultiFab>(
    grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
  amrex::MultiFab::Copy(*mf, *all, derComp, 0, 1, nGrow);
  return mf;
}

void
PeleLM::initProgressVariable()
{
  amrex::Vector<std::string> varNames;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    varNames, &(eos_parms.host_parm()));
  varNames.push_back("temp");

  amrex::ParmParse pp("peleLM");
  std::string Cformat;
  int hasUserC = static_cast<int>(pp.contains("progressVariable.format"));
  if (hasUserC != 0) {
    pp.query("progressVariable.format", Cformat);
    if (Cformat == "Cantera") { // use a Cantera-like format with
                                // <entry>:<weight>, default to 0.0
      // Weights
      amrex::Vector<std::string> stringIn;
      amrex::Vector<amrex::Real> weightsIn(NUM_SPECIES + 1, 0.0);
      int entryCount = pp.countval("progressVariable.weights");
      stringIn.resize(entryCount);
      pp.getarr("progressVariable.weights", stringIn, 0, entryCount);
      parseVars(varNames, stringIn, weightsIn);

      // Cold side/Hot side
      amrex::Vector<amrex::Real> coldState(NUM_SPECIES + 1, 0.0);
      entryCount = pp.countval("progressVariable.coldState");
      stringIn.resize(entryCount);
      pp.getarr("progressVariable.coldState", stringIn, 0, entryCount);
      parseVars(varNames, stringIn, coldState);
      amrex::Vector<amrex::Real> hotState(NUM_SPECIES + 1, 0.0);
      entryCount = pp.countval("progressVariable.hotState");
      stringIn.resize(entryCount);
      pp.getarr("progressVariable.hotState", stringIn, 0, entryCount);
      parseVars(varNames, stringIn, hotState);
      m_C0 = 0.0;
      m_C1 = 0.0;
      for (int i = 0; i < NUM_SPECIES + 1; ++i) {
        m_Cweights[i] = weightsIn[i];
        m_C0 += coldState[i] * m_Cweights[i];
        m_C1 += hotState[i] * m_Cweights[i];
      }
    } else if (Cformat == "RealList") { // use a list of amrex::Real. MUST
                                        // contains an entry
                                        // for each species+Temp
      // Weights
      amrex::Vector<amrex::Real> weightsIn;
      int entryCount = pp.countval("progressVariable.weights");
      AMREX_ALWAYS_ASSERT(entryCount == NUM_SPECIES + 1);
      weightsIn.resize(entryCount);
      pp.getarr("progressVariable.weights", weightsIn, 0, entryCount);
      for (int i = 0; i < NUM_SPECIES + 1; ++i) {
        m_Cweights[i] = weightsIn[i];
      }
      // Cold side/Hot side
      entryCount = pp.countval("progressVariable.coldState");
      AMREX_ALWAYS_ASSERT(entryCount == NUM_SPECIES + 1);
      amrex::Vector<amrex::Real> coldState(entryCount);
      pp.getarr("progressVariable.coldState", coldState, 0, entryCount);
      entryCount = pp.countval("progressVariable.hotState");
      AMREX_ALWAYS_ASSERT(entryCount == NUM_SPECIES + 1);
      amrex::Vector<amrex::Real> hotState(entryCount);
      pp.getarr("progressVariable.hotState", hotState, 0, entryCount);
      m_C0 = 0.0;
      m_C1 = 0.0;
      for (int i = 0; i < NUM_SPECIES + 1; ++i) {
        m_C0 += coldState[i] * m_Cweights[i];
        m_C1 += hotState[i] * m_Cweights[i];
      }
    } else {
      amrex::Abort(
        "Unknown progressVariable.format ! Should be 'Cantera' or 'RealList'");
    }
    pp.query("progressVariable.revert", m_Crevert);
  }
}

void
PeleLM::parseVars(
  const amrex::Vector<std::string>& a_varsNames,
  const amrex::Vector<std::string>& a_stringIn,
  amrex::Vector<amrex::Real>& a_rVars)
{
  const int varCountIn = static_cast<int>(a_stringIn.size());

  // For each entry in the user-provided composition, parse name and value
  std::string delimiter = ":";
  for (int i = 0; i < varCountIn; ++i) {
    long unsigned sep = a_stringIn[i].find(delimiter);
    if (sep == std::string::npos) {
      amrex::Abort(
        "Error parsing '" + a_stringIn[i] + "' --> unable to find delimiter :");
    }
    std::string varNameIn = a_stringIn[i].substr(0, sep);
    amrex::Real value =
      std::stod(a_stringIn[i].substr(sep + 1, a_stringIn[i].length()));
    int foundIt = 0;
    for (int k = 0; k < a_varsNames.size(); ++k) {
      if (varNameIn == a_varsNames[k]) {
        a_rVars[k] = value;
        foundIt = 1;
      }
    }
    if (foundIt == 0) {
      amrex::Abort(
        "Error parsing '" + a_stringIn[i] +
        "' --> unable to match to any provided variable name");
    }
  }
}

amrex::Real
PeleLM::MLNorm0(const amrex::Vector<const amrex::MultiFab*>& a_MF)
{
  return MLNorm0(a_MF, 0, 1)[0];
}

amrex::Vector<amrex::Real>
PeleLM::MLNorm0(
  const amrex::Vector<const amrex::MultiFab*>& a_MF,
  const int startcomp,
  const int ncomp)
{
  BL_PROFILE("PeleLMeX::MLNorm0()");
  AMREX_ASSERT(a_MF[0]->nComp() >= startcomp + ncomp);
  amrex::Vector<amrex::Real> r(ncomp);
  for (int n = 0; n < ncomp; ++n) {
    r[n] = 0.0;
  }
  for (int lev = 0; lev < a_MF.size(); ++lev) {
    if (lev != finest_level) {
      for (int n = 0; n < ncomp; ++n) {
        r[n] = amrex::max<amrex::Real>(
          r[n], a_MF[lev]->norm0(*m_coveredMask[lev], startcomp + n, 0, true));
      }
    } else {
      for (int n = 0; n < ncomp; ++n) {
        r[n] = amrex::max<amrex::Real>(
          r[n], a_MF[lev]->norm0(startcomp + n, 0, true, true));
      }
    }
  }
  amrex::ParallelDescriptor::ReduceRealMax(r.data(), ncomp);
  return r;
}

namespace {
// Index of the component named a_name in a (index, name) list, -1 if absent
int
findComponent(
  const std::list<std::tuple<int, std::string>>& a_components,
  std::string_view a_name)
{
  int idx = -1;
  for (const auto& component : a_components) {
    if (std::get<1>(component) == a_name) {
      idx = std::get<0>(component);
    }
  }
  return idx;
}
} // namespace

bool
PeleLM::isStateVariable(std::string_view a_name)
{
  return findComponent(stateComponents, a_name) >= 0;
}

bool
PeleLM::isReactVariable(std::string_view a_name)
{
  return findComponent(reactComponents, a_name) >= 0;
}

int
PeleLM::stateVariableIndex(std::string_view a_name)
{
  const int idx = findComponent(stateComponents, a_name);
  if (idx < 0) {
    amrex::Error(
      "PeleLM::stateVariableIndex(): unknown State variable: " +
      static_cast<std::string>(a_name));
  }
  return idx;
}

std::string
PeleLM::stateVariableName(int a_index)
{
  if (a_index < 0 || a_index >= NVAR) {
    amrex::Error(
      "PeleLM::stateVariableName(): invalid state index: " +
      std::to_string(a_index));
  }

  std::string var_name;
  for (const auto& stateComponent : stateComponents) {
    if (std::get<0>(stateComponent) == a_index) {
      var_name = std::get<1>(stateComponent);
    }
  }
  return var_name;
}

int
PeleLM::reactVariableIndex(std::string_view a_name)
{
  const int idx = findComponent(reactComponents, a_name);
  if (idx < 0) {
    amrex::Error(
      "PeleLM::reactVariableIndex(): unknown Reaction variable: " +
      static_cast<std::string>(a_name));
  }
  return idx;
}

amrex::Vector<int>
PeleLM::fetchAdvTypeArray(int scomp, int ncomp)
{
  amrex::Vector<int> types(ncomp);
  for (int comp = 0; comp < ncomp; ++comp) {
    types[comp] = m_AdvTypeState[scomp + comp];
  }
  return types;
}

amrex::Vector<int>
PeleLM::fetchAdvTypeAuxArray(int scomp, int ncomp)
{
  amrex::Vector<int> types(ncomp);
  for (int comp = 0; comp < ncomp; ++comp) {
    types[comp] = m_AdvTypeAux[scomp + comp];
  }
  return types;
}

void
PeleLM::defineCellVolume(
  const int lev,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm,
  const amrex::FabFactory<amrex::FArrayBox>& factory)
{
  // Geometry volume (Xi cell, or RZ) x detJ (mapping) x vfrac (EB).
  AMREX_ASSERT(lev >= 0 && lev < m_cellVol.size());
  m_cellVol[lev].define(ba, dm, 1, 0, amrex::MFInfo(), factory);
  geom[lev].GetVolume(m_cellVol[lev]);
  if (m_mesh_mapping) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      (m_mesh_map != nullptr) && !m_mesh_map->detJ_cc(lev).empty(),
      "defineCellVolume: mesh-mapping metrics must be defined on this "
      "level before the cell volume is built");
    m_mesh_map->multiply_by_detJ_cc(lev, m_cellVol[lev], 0, 1);
  }
#ifdef AMREX_USE_EB
  auto const& ebfact = dynamic_cast<amrex::EBFArrayBoxFactory const&>(factory);
  amrex::MultiFab::Multiply(m_cellVol[lev], ebfact.getVolFrac(), 0, 0, 1, 0);
#endif
}

bool
PeleLM::domainHasOpenFace() const
{
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (geom[0].isPeriodic(idim)) {
      continue;
    }
    for (const int bc : {m_phys_bc.lo(idim), m_phys_bc.hi(idim)}) {
      if (
        bc == BoundaryCondition::BCInflow ||
        bc == BoundaryCondition::BCOutflow) {
        return true;
      }
    }
  }
  return false;
}

amrex::Real
PeleLM::domainFaceFlux(
  const amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_umac)
{
  // Geometry face areas (x EB area fractions); a coarse boundary face
  // whose cell is covered by the finer level is skipped (MFSum's mask).
  // Under mapping umac is already (J/fac) u, i.e. the projector's own
  // boundary term.
  amrex::Real net = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const bool has_mask = (lev != finest_level);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      if (geom[lev].isPeriodic(idim)) {
        continue;
      }
      amrex::MultiFab area;
      geom[lev].GetFaceArea(area, grids[lev], dmap[lev], idim, 0);
#ifdef AMREX_USE_EB
      // EB area fraction: cut boxes carry one, covered boxes get zero.
      {
        auto const& ebfact = EBFactory(lev);
        auto const& areafrac = ebfact.getAreaFrac();
        auto const& flags = ebfact.getMultiEBCellFlagFab();
        for (amrex::MFIter mfi(area); mfi.isValid(); ++mfi) {
          if (areafrac[idim]->ok(mfi)) {
            area[mfi].mult<amrex::RunOn::Device>(
              (*areafrac[idim])[mfi], 0, 0, 1);
          } else if (
            flags[mfi].getType(mfi.validbox()) == amrex::FabType::covered) {
            area[mfi].setVal<amrex::RunOn::Device>(0.0);
          }
        }
        amrex::Gpu::streamSynchronize();
      }
#endif
      const amrex::Box faceDomain = amrex::convert(
        geom[lev].Domain(), amrex::IntVect::TheDimensionVector(idim));
      const int lo = faceDomain.smallEnd(idim);
      const int hi = faceDomain.bigEnd(idim);
      auto const& fma = a_umac[lev][idim]->const_arrays();
      auto const& ama = area.const_arrays();
      auto const& mma = has_mask ? m_coveredMask[lev]->const_arrays()
                                 : amrex::MultiArray4<int const>{};
      const amrex::Real s = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum>{}, amrex::TypeList<amrex::Real>{},
        *a_umac[lev][idim], amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
          -> amrex::GpuTuple<amrex::Real> {
          const int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
          amrex::Real sgn = 0.0; // outward normal is -e_idim at lo
          amrex::IntVect cell(AMREX_D_DECL(i, j, k));
          if (idx == lo) {
            sgn = -1.0;
          } else if (idx == hi) {
            sgn = 1.0;
            cell[idim] -= 1; // the cell this high face bounds
          } else {
            return {0.0};
          }
          amrex::Real w = ama[box_no](i, j, k);
          if (has_mask) {
            w *= static_cast<amrex::Real>(mma[box_no](cell));
          }
          return {sgn * w * fma[box_no](i, j, k)};
        });
      net += s;
    }
  }
  amrex::ParallelDescriptor::ReduceRealSum(net);
  return net;
}

amrex::Real
PeleLM::ebInflowFlux(const PeleLM::TimeStamp a_time)
{
#ifdef AMREX_USE_EB
  if (m_useEBinflow == 0) {
    return 0.0;
  }
  // AMReX adds (u_eb.n) * barea * |n/dx| / vfrac to the divergence of a cut
  // cell; times the cell volume (vfrac * dV) that is the outward volume flux
  // through the embedded surface of that cell.
  amrex::Real net = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const bool has_mask = (lev != finest_level);
    auto ebvel = getEBState(lev, VELX, AMREX_SPACEDIM, a_time);
    auto const& ebfact = EBFactory(lev);
    auto const& flags = ebfact.getMultiEBCellFlagFab();
    auto const& bnorm = ebfact.getBndryNormal();
    auto const& barea = ebfact.getBndryArea();
    const auto dxinv = geom[lev].InvCellSizeArray();
    const amrex::Real dV = AMREX_D_TERM(
      geom[lev].CellSize(0), *geom[lev].CellSize(1), *geom[lev].CellSize(2));
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (amrex::MFIter mfi(*ebvel); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      if (flags[mfi].getType(bx) != amrex::FabType::singlevalued) {
        continue;
      }
      auto const& vel = ebvel->const_array(mfi);
      auto const& flag = flags.const_array(mfi);
      auto const& nrm = bnorm.const_array(mfi);
      auto const& area = barea.const_array(mfi);
      auto const& mask = has_mask ? m_coveredMask[lev]->const_array(mfi)
                                  : amrex::Array4<int const>{};
      reduce_op.eval(
        bx, reduce_data,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> ReduceTuple {
          if (!flag(i, j, k).isSingleValued()) {
            return {0.0};
          }
          if (has_mask && mask(i, j, k) == 0) {
            return {0.0};
          }
          const amrex::Real u_dot_n = AMREX_D_TERM(
            vel(i, j, k, 0) * nrm(i, j, k, 0),
            +vel(i, j, k, 1) * nrm(i, j, k, 1),
            +vel(i, j, k, 2) * nrm(i, j, k, 2));
          const amrex::Real dl_eb_inv = std::sqrt(AMREX_D_TERM(
            nrm(i, j, k, 0) * dxinv[0] * nrm(i, j, k, 0) * dxinv[0],
            +nrm(i, j, k, 1) * dxinv[1] * nrm(i, j, k, 1) * dxinv[1],
            +nrm(i, j, k, 2) * dxinv[2] * nrm(i, j, k, 2) * dxinv[2]));
          return {u_dot_n * area(i, j, k) * dl_eb_inv * dV};
        });
    }
    ReduceTuple hv = reduce_data.value(reduce_op);
    net += amrex::get<0>(hv);
  }
  amrex::ParallelDescriptor::ReduceRealSum(net);
  if (m_debug_divu_mismatch != 0) {
    amrex::Print() << "   ... EB inflow volume flux (outward +): " << net
                   << "\n";
  }
  return net;
#else
  amrex::ignore_unused(a_time);
  return 0.0;
#endif
}

void
PeleLM::checkProjectionCompatibility(
  const std::string& a_who,
  const amrex::Vector<const amrex::MultiFab*>& a_rhs,
  const amrex::Real a_bdry_flux,
  const amrex::Real a_rhs_sign,
  const bool a_rhs_is_rz_scaled)
{
  // Solvability of  div(beta grad phi) = div(U) + sign . rhs  with
  // all-Neumann/periodic BCs:  F_bdry(U) + sign . sum(rhs dV) = 0.  dV is
  // the projector's measure (see the header), NOT m_cellVol, which carries
  // detJ: under mapping the RHS is already J-scaled.
  if (!compatibilityCheckEnabled()) {
    return;
  }
  amrex::Real sum_rhs = 0.0;
  amrex::Real sum_abs = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab dv(grids[lev], dmap[lev], 1, 0);
    if (a_rhs_is_rz_scaled) {
      const amrex::Real* dx = geom[lev].CellSize();
      dv.setVal(AMREX_D_TERM(dx[0], *dx[1], *dx[2]));
    } else {
      geom[lev].GetVolume(dv);
    }
#ifdef AMREX_USE_EB
    amrex::MultiFab::Multiply(dv, EBFactory(lev).getVolFrac(), 0, 0, 1, 0);
#endif
    auto const& rma = a_rhs[lev]->const_arrays();
    auto const& vma = dv.const_arrays();
    const bool has_mask = (lev != finest_level);
    auto const& mma = has_mask ? m_coveredMask[lev]->const_arrays()
                               : amrex::MultiArray4<int const>{};
    const auto r = amrex::ParReduce(
      amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
      amrex::TypeList<amrex::Real, amrex::Real>{}, *a_rhs[lev],
      amrex::IntVect(0),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
        -> amrex::GpuTuple<amrex::Real, amrex::Real> {
        amrex::Real v = rma[box_no](i, j, k) * vma[box_no](i, j, k);
        if (has_mask) {
          v *= static_cast<amrex::Real>(mma[box_no](i, j, k));
        }
        return {v, std::abs(v)};
      });
    sum_rhs += amrex::get<0>(r);
    sum_abs += amrex::get<1>(r);
  }
  amrex::ParallelDescriptor::ReduceRealSum(sum_rhs);
  amrex::ParallelDescriptor::ReduceRealSum(sum_abs);

  const amrex::Real resid = a_bdry_flux + a_rhs_sign * sum_rhs;
  const amrex::Real scale =
    amrex::max(sum_abs, std::abs(a_bdry_flux), amrex::Real(1.0e-300));
  const amrex::Real rel = std::abs(resid) / scale;
  if (m_verbose >= 2) {
    amrex::Print() << "   ... " << a_who
                   << " compatibility: |F_bdry + sign.sum(rhs)| / scale = "
                   << rel << " (F_bdry = " << a_bdry_flux
                   << ", sum(rhs) = " << sum_rhs << ")\n";
  }
  // !(rel <= tol) so that a NaN residual also trips the gate.
  if (
    (m_debug_check_compatibility != 0) && !(rel <= m_debug_compatibility_tol)) {
    std::ostringstream msg;
    msg << a_who << ": closed-chamber projection RHS is incompatible (relative "
        << "residual " << std::scientific << rel
        << " > peleLM.debug_compatibility_tol = " << m_debug_compatibility_tol
        << ")";
    amrex::Abort(msg.str());
  }
}

void
PeleLM::checkMACDivuMismatch(
  const int a_lev,
  const std::string& a_who,
  const amrex::MultiFab& a_divu,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_umac)
{
  // Diagnostic only.  Two numbers per level:
  //  1. divu - div(umac), the divu handed to the Godunov prediction minus
  //     the divergence of the face velocity it is paired with, in valid
  //     cells that are regular (no EB cut) and not covered by a finer
  //     level: min, max and plain cell average.
  //  2. At non-periodic domain faces, the jump divu(ghost) - divu(interior)
  //     between the first ghost cell and the cell inside it, which the
  //     prediction reads through the transverse terms: largest |jump|.
  if (m_debug_divu_mismatch == 0 || m_incompressible != 0) {
    return;
  }
  if (m_mesh_mapping || geom[a_lev].IsRZ()) {
    amrex::Print() << "   ... DIVUCHECK " << a_who
                   << ": skipped (mesh mapping or RZ)\n";
    return;
  }
  AMREX_ALWAYS_ASSERT(a_divu.nGrow() >= 1);

  const auto dxinv = geom[a_lev].InvCellSizeArray();
  const amrex::Box& domain = geom[a_lev].Domain();
  const amrex::IntVect dlo = domain.smallEnd();
  const amrex::IntVect dhi = domain.bigEnd();
  amrex::GpuArray<int, AMREX_SPACEDIM> isper{AMREX_D_DECL(
    geom[a_lev].isPeriodic(0), geom[a_lev].isPeriodic(1),
    geom[a_lev].isPeriodic(2))};
  const bool has_mask = (a_lev != finest_level);
  constexpr amrex::Real big = std::numeric_limits<amrex::Real>::max();

  amrex::ReduceOps<
    amrex::ReduceOpMax, amrex::ReduceOpMin, amrex::ReduceOpSum,
    amrex::ReduceOpSum, amrex::ReduceOpMax, amrex::ReduceOpMax>
    reduce_op;
  amrex::ReduceData<
    amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
    amrex::Real>
    reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  for (amrex::MFIter mfi(a_divu); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.validbox();
    auto const& divu = a_divu.const_array(mfi);
    AMREX_D_TERM(
      auto const& umac = a_umac[0]->const_array(mfi);
      , auto const& vmac = a_umac[1]->const_array(mfi);
      , auto const& wmac = a_umac[2]->const_array(mfi);)
    auto const& mask = has_mask ? m_coveredMask[a_lev]->const_array(mfi)
                                : amrex::Array4<int const>{};
#ifdef AMREX_USE_EB
    auto const& flag =
      EBFactory(a_lev).getMultiEBCellFlagFab().const_array(mfi);
#endif
    reduce_op.eval(
      bx, reduce_data,
      [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> ReduceTuple {
        bool use = true;
        if (has_mask) {
          use = (mask(i, j, k) != 0);
        }
#ifdef AMREX_USE_EB
        use = use && flag(i, j, k).isRegular();
#endif
        if (!use) {
          return {-big, big, 0.0, 0.0, 0.0, 0.0};
        }
        const amrex::Real divum = AMREX_D_TERM(
          (umac(i + 1, j, k) - umac(i, j, k)) * dxinv[0],
          +(vmac(i, j + 1, k) - vmac(i, j, k)) * dxinv[1],
          +(wmac(i, j, k + 1) - wmac(i, j, k)) * dxinv[2]);
        const amrex::Real d = divu(i, j, k) - divum;
        // Ghost-layer jump at non-periodic domain faces
        amrex::Real jump = 0.0;
        const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          if (isper[idim] != 0) {
            continue;
          }
          if (iv[idim] == dlo[idim]) {
            amrex::IntVect ivg(iv);
            ivg[idim] -= 1;
            jump = amrex::max(jump, std::abs(divu(ivg) - divu(iv)));
          }
          if (iv[idim] == dhi[idim]) {
            amrex::IntVect ivg(iv);
            ivg[idim] += 1;
            jump = amrex::max(jump, std::abs(divu(ivg) - divu(iv)));
          }
        }
        return {d, d, d, 1.0, std::abs(divu(i, j, k)), jump};
      });
  }

  ReduceTuple hv = reduce_data.value(reduce_op);
  amrex::Real dmax = amrex::get<0>(hv);
  amrex::Real dmin = amrex::get<1>(hv);
  amrex::Real dsum = amrex::get<2>(hv);
  amrex::Real cnt = amrex::get<3>(hv);
  amrex::Real smax = amrex::get<4>(hv);
  amrex::Real jmax = amrex::get<5>(hv);
  amrex::ParallelDescriptor::ReduceRealMax(dmax);
  amrex::ParallelDescriptor::ReduceRealMin(dmin);
  amrex::ParallelDescriptor::ReduceRealSum(dsum);
  amrex::ParallelDescriptor::ReduceRealSum(cnt);
  amrex::ParallelDescriptor::ReduceRealMax(smax);
  amrex::ParallelDescriptor::ReduceRealMax(jmax);

  if (cnt < 0.5) {
    amrex::Print() << "   ... DIVUCHECK lev " << a_lev << " " << a_who
                   << ": no regular uncovered cells\n";
    return;
  }
  amrex::Print().SetPrecision(10)
    << "   ... DIVUCHECK lev " << a_lev << " " << a_who
    << ": divu - div(umac): min = " << dmin << ", max = " << dmax
    << ", mean = " << dsum / cnt << ", max|divu| = " << smax
    << ", max ghost-layer jump = " << jmax << "\n";
}

amrex::Real
PeleLM::MFSum(const amrex::Vector<const amrex::MultiFab*>& a_mf, const int comp)
{
  BL_PROFILE("PeleLMeX::MFSum()");
  // Physical-volume integral over uncovered cells; m_cellVol carries detJ
  // and the EB volume fraction (see defineCellVolume).

  amrex::Real volwgtsum = 0.0;

  for (int lev = 0; lev <= finest_level; ++lev) {
    AMREX_ASSERT(!m_cellVol[lev].empty());
    AMREX_ASSERT(m_cellVol[lev].boxArray() == a_mf[lev]->boxArray());

    amrex::Real sm = 0.0;
    if (lev != finest_level) {
      sm = amrex::ReduceSum(
        *a_mf[lev], m_cellVol[lev], *m_coveredMask[lev], 0,
        [comp] AMREX_GPU_HOST_DEVICE(
          amrex::Box const& bx, amrex::Array4<amrex::Real const> const& mf_arr,
          amrex::Array4<amrex::Real const> const& vol_arr,
          amrex::Array4<int const> const& covered_arr) -> amrex::Real {
          amrex::Real sum = 0.0;
          AMREX_LOOP_3D(bx, i, j, k, {
            sum += mf_arr(i, j, k, comp) * vol_arr(i, j, k) *
                   static_cast<amrex::Real>(covered_arr(i, j, k));
          });
          return sum;
        });
    } else {
      sm = amrex::ReduceSum(
        *a_mf[lev], m_cellVol[lev], 0,
        [comp] AMREX_GPU_HOST_DEVICE(
          amrex::Box const& bx, amrex::Array4<amrex::Real const> const& mf_arr,
          amrex::Array4<amrex::Real const> const& vol_arr) -> amrex::Real {
          amrex::Real sum = 0.0;
          AMREX_LOOP_3D(
            bx, i, j, k, { sum += mf_arr(i, j, k, comp) * vol_arr(i, j, k); });
          return sum;
        });
    }

    volwgtsum += sm;
  } // lev

  amrex::ParallelDescriptor::ReduceRealSum(volwgtsum);

  return volwgtsum;
}

void
PeleLM::setTypicalValues(const TimeStamp a_time, const int is_init)
{
  // Get state Max/Min
  const int nstate = (m_incompressible != 0) ? AMREX_SPACEDIM : NVAR;
  // getStateVect returns aliases: keep them alive while reducing
  const auto stateVect = getStateVect(a_time);
  auto stateMax = MLmax(GetVecOfConstPtrs(stateVect), 0, nstate);
  auto stateMin = MLmin(GetVecOfConstPtrs(stateVect), 0, nstate);

  // Fill typical values vector
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    typical_values[idim] = amrex::max<amrex::Real>(
      stateMax[VELX + idim], std::abs(stateMin[VELX + idim]));
  }

  if (m_incompressible == 0) {
    // Average between max/min
    typical_values[DENSITY] = 0.5 * (stateMax[DENSITY] + stateMin[DENSITY]);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      typical_values[FIRSTSPEC + n] =
        0.5 * (stateMax[FIRSTSPEC + n] + stateMin[FIRSTSPEC + n]) /
        typical_values[DENSITY];
    }
    typical_values[RHOH] =
      0.5 * (stateMax[RHOH] + stateMin[RHOH]) / typical_values[DENSITY];
    typical_values[TEMP] = 0.5 * (stateMax[TEMP] + stateMin[TEMP]);
    typical_values[RHORT] = m_pOld;
#ifdef PELE_USE_PLASMA
    typical_values[NE] = 0.5 * (stateMax[NE] + stateMin[NE]);
#endif
#if NUM_ODE > 0
    for (int n = 0; n < NUM_ODE; ++n) {
      typical_values[FIRSTODE + n] =
        0.5 * (stateMax[FIRSTODE + n] + stateMin[FIRSTODE + n]);
    }
#endif

    // Pass into chemsitry if requested
    updateTypicalValuesChem();
  }

  if ((is_init != 0) || m_verbose > 1) {
    amrex::Print() << PrettyLine;
    amrex::Print() << " Typical values: " << '\n';
    amrex::Print() << "\tVelocity: ";
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      amrex::Print() << typical_values[idim] << ' ';
    }
    amrex::Print() << '\n';
    if (m_incompressible == 0) {
      amrex::Print() << "\tDensity:  " << typical_values[DENSITY] << '\n';
      amrex::Print() << "\tTemp:     " << typical_values[TEMP] << '\n';
      amrex::Print() << "\tH:        " << typical_values[RHOH] << '\n';
      amrex::Vector<std::string> spec_names;
      pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
        spec_names, &(eos_parms.host_parm()));
      for (int n = 0; n < NUM_SPECIES; ++n) {
        amrex::Print() << "\tY_" << spec_names[n]
                       << std::setw(
                            amrex::max<int>(
                              0, static_cast<int>(8 - spec_names[n].length())))
                       << std::left << ":" << typical_values[FIRSTSPEC + n]
                       << '\n';
      }
#ifdef PELE_USE_PLASMA
      amrex::Print() << "\tnE:       " << typical_values[NE] << '\n';
#endif
#if NUM_ODE > 0
      for (int n = 0; n < NUM_ODE; ++n) {
        amrex::Print() << "\t" << m_ode_names[n]
                       << std::setw(
                            amrex::max<int>(
                              0,
                              static_cast<int>(10 - m_ode_names[n].length())))
                       << std::left << ":" << typical_values[FIRSTODE + n]
                       << '\n';
      }
#endif
    }
    amrex::Print() << PrettyLine;
  }
}

void
PeleLM::updateTypicalValuesChem()
{
  if ((m_useTypValChem != 0) && (m_do_react != 0)) {
    if (m_verbose > 2) {
      amrex::Print() << " Update chemistry typical values \n";
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
      amrex::Vector<amrex::Real> typical_values_chem;
      typical_values_chem.resize(NUM_SPECIES + 1);
      for (int i = 0; i < NUM_SPECIES; ++i) {
        typical_values_chem[i] = amrex::max<amrex::Real>(
          m_typicalYvalMin * typical_values[DENSITY] * 1.E-3,
          typical_values[FIRSTSPEC + i] * typical_values[DENSITY] *
            1.E-3); // CGS -> MKS conversion
      }
      typical_values_chem[NUM_SPECIES] = typical_values[TEMP];
#ifdef PELE_USE_PLASMA
      auto const* leosparm = &eos_parms.host_parm();
      auto eos = pele::physics::PhysicsType::eos(leosparm);
      amrex::Real mwt[NUM_SPECIES] = {0.0};
      eos.molecular_weight(mwt);
      typical_values_chem[E_ID] =
        typical_values[NE] / Na * mwt[E_ID] * 1.0e-6 * 1.0e-2;
#endif
      m_reactor->set_typ_vals_ode(typical_values_chem);
    }
  }
}

namespace pelelmex_utils_detail {
// Local reduction (Op = amrex::ReduceOpMax or ReduceOpMin) of component comp
// over the cells where a_mask is non-zero and, in EB builds, not covered.
// a_skip is the value an excluded cell contributes.
template <typename Op>
amrex::Real
maskedReduce(
  const amrex::MultiFab& a_MF,
  const amrex::iMultiFab& a_mask,
  const int comp,
  const amrex::Real a_skip)
{
  auto const& ma = a_MF.const_arrays();
  auto const& mask = a_mask.const_arrays();
#ifdef AMREX_USE_EB
  if (a_MF.hasEBFabFactory()) {
    const auto& ebfactory =
      dynamic_cast<amrex::EBFArrayBoxFactory const&>(a_MF.Factory());
    auto const& flagsma = ebfactory.getMultiEBCellFlagFab().const_arrays();
    return amrex::ParReduce(
      amrex::TypeList<Op>{}, amrex::TypeList<amrex::Real>{}, a_MF,
      amrex::IntVect(0),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
        -> amrex::GpuTuple<amrex::Real> {
        if (flagsma[box_no](i, j, k).isCovered() || !mask[box_no](i, j, k)) {
          return a_skip;
        }
        return ma[box_no](i, j, k, comp);
      });
  }
#endif
  return amrex::ParReduce(
    amrex::TypeList<Op>{}, amrex::TypeList<amrex::Real>{}, a_MF,
    amrex::IntVect(0),
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
      -> amrex::GpuTuple<amrex::Real> {
      if (!mask[box_no](i, j, k)) {
        return a_skip;
      }
      return ma[box_no](i, j, k, comp);
    });
}
} // namespace pelelmex_utils_detail

namespace pelelmex_utils_detail {
// MultiLevel max (Op = ReduceOpMax) or min (ReduceOpMin) of each component,
// excluding EB-covered/fine-covered cells
template <typename Op>
amrex::Vector<amrex::Real>
mlReduce(
  const amrex::Vector<const amrex::MultiFab*>& a_MF,
  const amrex::Vector<std::unique_ptr<amrex::iMultiFab>>& a_coveredMask,
  const int finest_level,
  const int scomp,
  const int ncomp)
{
  AMREX_ASSERT(a_MF[0]->nComp() >= scomp + ncomp);
  constexpr bool is_max = std::is_same_v<Op, amrex::ReduceOpMax>;
  constexpr amrex::Real skip = is_max ? AMREX_REAL_LOWEST : AMREX_REAL_MAX;

  amrex::Vector<amrex::Real> r(ncomp, skip);
  for (int lev = 0; lev < a_MF.size(); ++lev) {
    for (int n = 0; n < ncomp; ++n) {
      if constexpr (is_max) {
        const amrex::Real levmax =
          (lev != finest_level) ? maskedReduce<Op>(
                                    *a_MF[lev], *a_coveredMask[lev], scomp + n,
                                    std::numeric_limits<amrex::Real>::lowest())
                                : a_MF[lev]->max(scomp + n, 0, true);
        r[n] = amrex::max<amrex::Real>(r[n], levmax);
      } else {
        const amrex::Real levmin =
          (lev != finest_level) ? maskedReduce<Op>(
                                    *a_MF[lev], *a_coveredMask[lev], scomp + n,
                                    std::numeric_limits<amrex::Real>::max())
                                : a_MF[lev]->min(scomp + n, 0, true);
        r[n] = amrex::min<amrex::Real>(r[n], levmin);
      }
    }
  }

  if constexpr (is_max) {
    amrex::ParallelDescriptor::ReduceRealMax(r.data(), ncomp);
  } else {
    amrex::ParallelDescriptor::ReduceRealMin(r.data(), ncomp);
  }
  return r;
}
} // namespace pelelmex_utils_detail

amrex::Vector<amrex::Real>
PeleLM::MLmax(
  const amrex::Vector<const amrex::MultiFab*>& a_MF,
  const int scomp,
  const int ncomp)
{
  BL_PROFILE("PeleLMeX::MLmax()");
  return pelelmex_utils_detail::mlReduce<amrex::ReduceOpMax>(
    a_MF, m_coveredMask, finest_level, scomp, ncomp);
}

amrex::Vector<amrex::Real>
PeleLM::MLmin(
  const amrex::Vector<const amrex::MultiFab*>& a_MF,
  const int scomp,
  const int ncomp)
{
  BL_PROFILE("PeleLMeX::MLmin()");
  return pelelmex_utils_detail::mlReduce<amrex::ReduceOpMin>(
    a_MF, m_coveredMask, finest_level, scomp, ncomp);
}

void
PeleLM::checkMemory(const std::string& a_message) const
{
  if (m_checkMem == 0) {
    return;
  }

  const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
#ifdef AMREX_USE_GPU
  amrex::Long free_mem_avail =
    amrex::Gpu::Device::freeMemAvailable() / (1024 * 1024);
  amrex::ParallelDescriptor::ReduceLongMin(free_mem_avail, IOProc);
  amrex::Print() << "     [" << a_message << "] GPU mem. avail. (MB) "
                 << free_mem_avail << "\n";
#else
  // amrex::MultiFab memory usage
  amrex::Long max_fab_megabytes =
    amrex::TotalBytesAllocatedInFabsHWM() / (1024 * 1024);
  amrex::ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);
  amrex::Print() << "     [" << a_message << "] MFs mem. allocated (MB) "
                 << max_fab_megabytes << "\n";
#endif
}

void
PeleLM::initMixtureFraction()
{
  // set up a few variables
  auto const* leosparm = &eos_parms.host_parm();
  auto eos = pele::physics::PhysicsType::eos(leosparm);
  amrex::Vector<std::string> specNames;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    specNames, leosparm);
  amrex::ParmParse pp("peleLM");

  // Do simpler things for some EOS
  if (pele::physics::PhysicsType::eos_type::identifier() == "GammaLaw") {
    // Do nothing - Bilger mixture fraction has no meaning here
    // error will be raised if user tries to derive mixfrac because Zfu is
    // negative
    return;
  }
  if (pele::physics::PhysicsType::eos_type::identifier() == "Manifold") {
    // Just take a mixture fraction if it is a manifold parameter
    // otherwise do nothing (an error will later be raised if the user tries to
    // derive it) also raise an error if the user requests it and it is not
    // found
    std::string mixfrac_name = "ZMIX";
    bool requested_mixfrac =
      pp.contains("mixtureFraction.manifoldParameterName");
    pp.query("mixtureFraction.manifoldParameterName", mixfrac_name);
    bool found = false;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      if (specNames[n] == mixfrac_name) {
        if (!found) {
          found = true;
          spec_Bilger_fact[n] = 1.0;
        } else {
          amrex::Abort(
            "initMixtureFraction: requested manifold parameter "
            "found multiple times");
        }
      } else {
        spec_Bilger_fact[n] = 0.0;
      }
    }
    if (found) {
      Zfu = 1.0;
      Zox = 0.0;
    } else if (requested_mixfrac) {
      amrex::Abort(
        "initMixtureFraction: requested manifold parameter not found");
    }
    return;
  }

  // Otherwise - compute Bilger weights for detailed chemistry

  amrex::Real YF[NUM_SPECIES], YO[NUM_SPECIES];
  for (int i = 0; i < NUM_SPECIES; ++i) {
    YF[i] = 0.0;
    YO[i] = 0.0;
    if (specNames[i] == "O2") {
      YO[i] = 0.233;
    }
    if (specNames[i] == "N2") {
      YO[i] = 0.767;
    }
    if (i == fuelID) {
      YF[i] = 1.0;
    }
  }

  // Overwrite with user-defined value if provided in input file
  std::string MFformat;
  int hasUserMF = static_cast<int>(pp.contains("mixtureFraction.format"));
  if (hasUserMF != 0) {
    pp.query("mixtureFraction.format", MFformat);
    std::string MFCompoType;
    pp.query("mixtureFraction.type", MFCompoType);
    if (MFformat == "Cantera") { // use a Cantera-like format with
      // <SpeciesName>:<Value>, default in 0.0
      amrex::Vector<std::string> compositionIn;
      int entryCount = pp.countval("mixtureFraction.oxidTank");
      compositionIn.resize(entryCount);
      pp.getarr("mixtureFraction.oxidTank", compositionIn, 0, entryCount);
      parseComposition(compositionIn, MFCompoType, YO);
      entryCount = pp.countval("mixtureFraction.fuelTank");
      compositionIn.resize(entryCount);
      pp.getarr("mixtureFraction.fuelTank", compositionIn, 0, entryCount);
      parseComposition(compositionIn, MFCompoType, YF);
    } else if (MFformat == "RealList") { // use a list of amrex::Real. MUST
                                         // contains an entry
      // for each species in the mixture
      auto readTank = [&pp](const std::string& a_key, amrex::Real* a_out) {
        const int entryCount = pp.countval(a_key.c_str());
        AMREX_ALWAYS_ASSERT(entryCount == NUM_SPECIES);
        amrex::Vector<amrex::Real> compositionIn(NUM_SPECIES);
        pp.getarr(a_key.c_str(), compositionIn, 0, NUM_SPECIES);
        for (int i = 0; i < NUM_SPECIES; ++i) {
          a_out[i] = compositionIn[i];
        }
      };
      if (MFCompoType == "mass") {
        readTank("mixtureFraction.oxidTank", YO);
        readTank("mixtureFraction.fuelTank", YF);
      } else if (MFCompoType == "mole") {
        amrex::Real XF[NUM_SPECIES], XO[NUM_SPECIES];
        readTank("mixtureFraction.oxidTank", XO);
        readTank("mixtureFraction.fuelTank", XF);
        eos.X2Y(XO, YO);
        eos.X2Y(XF, YF);
      } else {
        amrex::Abort(
          "Unknown mixtureFraction.type ! Should be 'mass' or 'mole'");
      }
    } else {
      amrex::Abort(
        "Unknown mixtureFraction.format ! Should be 'Cantera' or 'RealList'");
    }
  }
  if (fuelID < 0 && (hasUserMF != 0)) {
    amrex::Print()
      << " Mixture fraction definition lacks fuelID: consider using "
         "peleLM.fuel_name keyword \n";
  }

  // Detailed chem - compute Bilger coefficients
  // Only interested in CHON -in that order. Compute Bilger weights
  amrex::Real atwCHON[4] = {0.0};
  pele::physics::eos::atomic_weightsCHON<pele::physics::PhysicsType::eos_type>(
    atwCHON);
  Beta_mix[0] = (atwCHON[0] != 0.0) ? 2.0 / atwCHON[0] : 0.0;
  Beta_mix[1] = (atwCHON[1] != 0.0) ? 1.0 / (2.0 * atwCHON[1]) : 0.0;
  Beta_mix[2] = (atwCHON[2] != 0.0) ? -1.0 / atwCHON[2] : 0.0;
  Beta_mix[3] = 0.0;

  // Compute each species weight for the Bilger formulation based on elemental
  // compo Only interested in CHON -in that order.
  int ecompCHON[NUM_SPECIES * 4];
  pele::physics::eos::element_compositionCHON<
    pele::physics::PhysicsType::eos_type>(ecompCHON);
  amrex::Real mwt[NUM_SPECIES];
  eos.molecular_weight(mwt);
  Zfu = 0.0;
  Zox = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i) {
    spec_Bilger_fact[i] = 0.0;
    for (int k = 0; k < 4; ++k) {
      spec_Bilger_fact[i] +=
        Beta_mix[k] * (ecompCHON[i * 4 + k] * atwCHON[k] / mwt[i]);
    }
    Zfu += spec_Bilger_fact[i] * YF[i];
    Zox += spec_Bilger_fact[i] * YO[i];
  }
}

void
PeleLM::parseComposition(
  const amrex::Vector<std::string>& compositionIn,
  const std::string& compositionType,
  amrex::Real* massFrac)
{
  amrex::Vector<amrex::Real> compoIn(NUM_SPECIES, 0.0);

  // Parse the <SpeciesName>:<Value> entries
  amrex::Vector<std::string> specNames;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    specNames, &(eos_parms.host_parm()));
  parseVars(specNames, compositionIn, compoIn);

  // Ensure that it sums to 1.0:
  amrex::Real sum = 0.0;
  for (amrex::Real k : compoIn) {
    sum += k;
  }
  for (amrex::Real& k : compoIn) {
    k /= sum;
  }

  // Fill the massFrac array, convert from mole fraction if necessary
  if (compositionType == "mass") { // mass
    for (int i = 0; i < NUM_SPECIES; ++i) {
      massFrac[i] = compoIn[i];
    }
  } else if (compositionType == "mole") { // mole
    auto const* leosparm = &eos_parms.host_parm();
    auto eos = pele::physics::PhysicsType::eos(leosparm);
    eos.X2Y(compoIn.data(), massFrac);
  } else {
    amrex::Abort("Unknown mixtureFraction.type ! Should be 'mass' or 'mole'");
  }
}

#ifdef AMREX_USE_EB
// Extend the cell-centered based signed distance function
void
PeleLM::extendSignedDistance(
  amrex::MultiFab* a_signDist, const amrex::Real a_extendFactor)
{
  BL_PROFILE("PeleLMeX::extendSignedDistance()");
  // This is a not-so-pretty piece of code that'll take AMReX cell-averaged
  // signed distance and propagates it manually up to the point where we need to
  // have it for derefining.
  const amrex::Real maxSignedDist = a_signDist->max(0);
  const auto& ebfactory =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(a_signDist->Factory());
  const auto& flags = ebfactory.getMultiEBCellFlagFab();
  const int nGrowFac = flags.nGrow() + 1;

  // First set the region far away at the max value we need
  auto const& sd_cc_ma = a_signDist->arrays();
  const auto& dx = geom[0].CellSizeArray();
  const amrex::Real sd_cc_fact = nGrowFac * dx[0] * a_extendFactor;
  amrex::ParallelFor(
    *a_signDist, a_signDist->nGrowVect(),
    [sd_cc_ma, maxSignedDist,
     sd_cc_fact] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      if (sd_cc_ma[box_no](i, j, k) > maxSignedDist - 1e-12) {
        sd_cc_ma[box_no](i, j, k) = sd_cc_fact;
      }
    });
  amrex::Gpu::streamSynchronize();
  // Iteratively compute the distance function in boxes, propagating across
  // boxes using ghost cells If needed, increase the number of loop to extend
  // the reach of the distance function
  constexpr int nMaxLoop = 4;
  for (int dloop = 1; dloop <= nMaxLoop; ++dloop) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*a_signDist, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const amrex::Box& gbx = grow(bx, 1);
      if (flags[mfi].getType(gbx) == amrex::FabType::covered) {
        continue;
      }
      auto const& sd_cc = a_signDist->array(mfi);
      amrex::ParallelFor(
        bx, [gbx, dx, a_extendFactor, sd_cc,
             maxSignedDist] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const auto glo = amrex::lbound(gbx);
          const auto ghi = amrex::ubound(gbx);
          amrex::Real extendedDist = dx[0] * a_extendFactor;
          if (sd_cc(i, j, k) >= maxSignedDist - 1e-12) {
            amrex::Real closestEBDist = 1e12;
            for (int kk = glo.z; kk <= ghi.z; ++kk) {
              for (int jj = glo.y; jj <= ghi.y; ++jj) {
                for (int ii = glo.x; ii <= ghi.x; ++ii) {
                  if ((i != ii) || (j != jj) || (k != kk)) {
                    if (sd_cc(ii, jj, kk) > 0.0) {
                      amrex::Real distToCell = std::sqrt(AMREX_D_TERM(
                        ((i - ii) * dx[0] * (i - ii) * dx[0]),
                        +((j - jj) * dx[1] * (j - jj) * dx[1]),
                        +((k - kk) * dx[2] * (k - kk) * dx[2])));
                      amrex::Real distToEB = distToCell + sd_cc(ii, jj, kk);
                      if (distToEB < closestEBDist) {
                        closestEBDist = distToEB;
                      }
                    }
                  }
                }
              }
            }
            if (closestEBDist < 1e10) {
              sd_cc(i, j, k) = closestEBDist;
            } else {
              sd_cc(i, j, k) = extendedDist;
            }
          }
        });
    }
    a_signDist->FillBoundary(geom[0].periodicity());
  }
}
#endif
