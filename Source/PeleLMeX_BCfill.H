#ifndef PELELM_BCFILL
#define PELELM_BCFILL
#include <PeleLMeX_Index.H>
#include <PeleLMeX_MeshMapEvaluator.H>
#include <pelelmex_prob.H>
#include <PMFData.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

// Compute the boundary-face coordinate array x[] that gets passed to
// the user's bcnormal.  The boundary-direction component (d == idir)
// is taken at the actual physical face; the tangential components
// (d != idir) are taken at the ghost-cell centre (which has the same
// tangential index as the matching interior cell).
//
// When the supplied MeshMapEvaluator is Kind::Identity, x[idir] is
// `prob_lo[idir] + face_idx*dx[idir]` and x[d!=idir] is
// `prob_lo[d] + (iv[d]+0.5)*dx[d]`.  When a mapping is active the
// returned values are the physical-space coordinates under that
// mapping, so user bcnormal code that branches on x[] sees the
// physical face position rather than the xi-space position.
//
// Note: this corrects a latent half-cell offset in the legacy
// computation, which used the ghost-cell centre for all components
// (i.e. `x[idir] = prob_lo + (iv[idir]+0.5)*dx`, half a cell outside
// the actual wall).  User code that branched on `x[idir]` via a
// loose threshold (e.g. `x[idir] > 0.99*L`) is unaffected; code that
// compares to an exact half-cell offset will see the wall position.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
pelelm_bc_face_x(
  amrex::IntVect const& iv,
  int idir,
  int sgn,
  amrex::GeometryData const& geom,
  MeshMapEvaluator const& mmap,
  amrex::Real x[AMREX_SPACEDIM]) noexcept
{
  // sgn = +1 means low side: ghost cell at iv[idir] = domlo-1, face at
  //                          integer index iv[idir] + 1 = domlo.
  // sgn = -1 means high side: ghost cell at iv[idir] = domhi+1, face at
  //                          integer index iv[idir] = domhi+1.
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (d == idir) {
      const int face_idx = (sgn > 0) ? iv[d] + 1 : iv[d];
      x[d] = mmap.x_phys_face(d, face_idx, geom);
    } else {
      x[d] = x_phys_cc_periodic(mmap, d, iv[d], geom);
    }
  }
}

// Members and helpers shared by the ext_dir fill functors that call the
// problem's bcnormal (T is the problem's ProblemSpecificFunctions).
template <typename T>
struct PeleLMCCFillExtDirBase
{
  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{}; // defaults to Identity (no mesh mapping)

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirBase(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  // Call bcnormal for ghost cell iv beyond the (idir, sgn) face (sgn = +1
  // low side, -1 high side), passing it the state in the first cell inside
  // the domain. bcnormal() is defined in pelelmex_prob.H in the problem
  // directory in /Exec.
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void bcnormal_face(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& state,
    const int idir,
    const int sgn,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    amrex::Real s_ext[NVAR]) const
  {
    amrex::Real x[AMREX_SPACEDIM];
    pelelm_bc_face_x(iv, idir, sgn, geom, m_map_eval, x);

    auto iv_in = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_in[0] = iv[0];, iv_in[1] = iv[1];, iv_in[2] = iv[2];);
    iv_in[idir] += sgn;
    const auto s_in = state.cellData(iv_in[0], iv_in[1], iv_in[2]);

    T::bcnormal(x, s_in, s_ext, idir, sgn, time, geom, *lprobparm, lpmfdata);
  }
};

template <typename T>
struct PeleLMCCFillExtDirState : PeleLMCCFillExtDirBase<T>
{
  int m_do_turbInflow = 0;
  int m_do_inlet_from_plane = 0;
  int m_inlet_from_plane_dir = -1;
  // When != 0 (peleLM.recycling_mode = full), the pre-loaded ghost value on
  // the recycling face is the complete inflow velocity and is preserved
  // against whatever bcnormal writes to the velocity components there.
  int m_recycling_full = 0;

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirState(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    int do_turbInflow,
    int do_inletFromPlane,
    int inlet_plane_dir,
    int recycling_full,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : PeleLMCCFillExtDirBase<T>(a_prob_parm, a_pmf_data, a_nAux, a_map_eval),
      m_do_turbInflow(do_turbInflow),
      m_do_inlet_from_plane(do_inletFromPlane),
      m_inlet_from_plane_dir(inlet_plane_dir),
      m_recycling_full(recycling_full)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& state,
    const int /*dcomp*/,
    const int numcomp,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    const int ncomp = amrex::min<int>(numcomp, NVAR);
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      if (iv[idir] < domlo[idir]) {
        fillFace(iv, state, ncomp, geom, time, bcr, idir, 1);
      }
      if (iv[idir] > domhi[idir]) {
        fillFace(iv, state, ncomp, geom, time, bcr, idir, -1);
      }
    }
  }

  // Fill the ext_dir components of ghost cell iv beyond the (idir, sgn)
  // face. sgn = +1 is the low side, -1 the high side. bcnormal returns
  // every state component at once, so it is called once per face and its
  // result is copied into each component that has an ext_dir BC there.
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void fillFace(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& state,
    const int ncomp,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int idir,
    const int sgn) const
  {
    const int bcidx = (sgn > 0) ? idir : idir + AMREX_SPACEDIM;

    bool any_ext_dir = false;
    for (int n = 0; n < ncomp; ++n) {
      any_ext_dir =
        any_ext_dir || (bcr[n].data()[bcidx] == amrex::BCType::ext_dir);
    }
    if (!any_ext_dir) {
      return;
    }

    amrex::Real s_ext[NVAR] = {0.0};

    // If using TurbInflow or inlet_from_plane, pass in the sampled data.
    // User can overwrite or increment, as needed. In full recycling
    // mode the pre-loaded value is the complete inflow velocity and is
    // kept regardless of what bcnormal does to it.
    bool keep_recycled[NVAR] = {false};
    if (
      m_do_turbInflow != 0 ||
      ((m_do_inlet_from_plane != 0) && (idir == m_inlet_from_plane_dir))) {
      for (int n = VELX; n < amrex::min<int>(ncomp, DENSITY); ++n) {
        if (bcr[n].data()[bcidx] != amrex::BCType::ext_dir) {
          continue;
        }
        if (!isnan(state(iv, VELX + n))) { // During fillcoarsepatch, get here
                                           // with nan. TODO: find a better fix
          s_ext[VELX + n] = state(iv, VELX + n);
          keep_recycled[n] = (m_recycling_full != 0) &&
                             (m_do_inlet_from_plane != 0) &&
                             (idir == m_inlet_from_plane_dir);
        } else {
          s_ext[VELX + n] = 0;
        }
      }
    }

    this->bcnormal_face(iv, state, idir, sgn, geom, time, s_ext);

    for (int n = 0; n < ncomp; ++n) {
      if (
        (bcr[n].data()[bcidx] == amrex::BCType::ext_dir) && !keep_recycled[n]) {
        state(iv, n) = s_ext[n];
      }
    }
  }
};

template <typename T>
struct PeleLMCCFillExtDirAux : PeleLMCCFillExtDirBase<T>
{
  using PeleLMCCFillExtDirBase<T>::PeleLMCCFillExtDirBase;

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& aux,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

#if AMREX_SPACEDIM == 2
    auto aux_ext = aux.cellData(iv[0], iv[1], 0);
#else
    auto aux_ext = aux.cellData(iv[0], iv[1], iv[2]);
#endif

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      // side 0: low (sgn = +1), side 1: high (sgn = -1)
      for (int side = 0; side < 2; ++side) {
        const int sgn = 1 - 2 * side;
        const bool outside =
          (side == 0) ? (iv[idir] < domlo[idir]) : (iv[idir] > domhi[idir]);
        if (!outside) {
          continue;
        }
        for (int n = 0; n < this->m_nAux; ++n) {
          if (
            bcr[n].data()[idir + side * AMREX_SPACEDIM] ==
            amrex::BCType::ext_dir) {
            amrex::Real x[AMREX_SPACEDIM];
            pelelm_bc_face_x(iv, idir, sgn, geom, this->m_map_eval, x);
            // bcnormal_aux() is defined in pelelmex_prob.H in problem
            // directory in /Exec
            T::bcnormal_aux(
              x, aux_ext, idir, sgn, time, geom, *this->lprobparm,
              this->lpmfdata);
          }
        }
      }
    }
  }
};

template <typename T>
struct PeleLMCCFillExtDirSpec : PeleLMCCFillExtDirBase<T>
{
  using PeleLMCCFillExtDirBase<T>::PeleLMCCFillExtDirBase;

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& rhoY,
    const int dcomp,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Faces this ghost cell lies beyond on which each species has an ext_dir
    // BC, as a bit mask (bit 2*idir for the low side, 2*idir+1 for the high
    // side). Species usually share one mask, but they can differ, e.g. for
    // charged species under electrode BCs.
    int mask[NUM_SPECIES];
    for (int n = 0; n < NUM_SPECIES; ++n) {
      mask[n] = 0;
      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        if (
          (iv[idir] < domlo[idir]) &&
          (bcr[n].data()[idir] == amrex::BCType::ext_dir)) {
          mask[n] |= 1 << (2 * idir);
        } else if (
          (iv[idir] > domhi[idir]) &&
          (bcr[n].data()[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir)) {
          mask[n] |= 1 << (2 * idir + 1);
        }
      }
    }

    // bcnormal handles all the state components at once, so it is called
    // once per face for each distinct mask, and its result is copied into
    // every species with that mask. Each mask keeps its own s_ext history
    // across the faces of a corner ghost cell, exactly as a per-species
    // loop would, since bcnormal may use its input (e.g. as an initial
    // guess) or leave some entries untouched on a given face.
    for (int n = 0; n < NUM_SPECIES; ++n) {
      bool seen = (mask[n] == 0);
      for (int k = 0; k < n && !seen; ++k) {
        seen = (mask[k] == mask[n]);
      }
      if (seen) {
        continue;
      }

      amrex::Real s_ext[NVAR] = {0.0};

      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        int sgn = 0;
        if (((mask[n] >> (2 * idir)) & 1) != 0) {
          sgn = 1;
        } else if (((mask[n] >> (2 * idir + 1)) & 1) != 0) {
          sgn = -1;
        } else {
          continue;
        }

        this->bcnormal_face(iv, rhoY, idir, sgn, geom, time, s_ext);

        for (int k = n; k < NUM_SPECIES; ++k) {
          if (mask[k] == mask[n]) {
            rhoY(iv, dcomp + k) = s_ext[FIRSTSPEC + k];
          }
        }
      }
    }
  }
};

// ext_dir fill of a single state component, Comp (an index into the full
// state, e.g. DENSITY or TEMP), stored at dcomp of the destination.
template <typename T, int Comp>
struct PeleLMCCFillExtDirScalar : PeleLMCCFillExtDirBase<T>
{
  using PeleLMCCFillExtDirBase<T>::PeleLMCCFillExtDirBase;

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& state,
    const int dcomp,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();
    const int* bc = bcr->data();

    // bcnormal handles all the state components at once
    amrex::Real s_ext[NVAR] = {0.0};

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      int sgn = 0;
      if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
        sgn = 1;
      } else if (
        (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
        (iv[idir] > domhi[idir])) {
        sgn = -1;
      }
      if (sgn != 0) {
        this->bcnormal_face(iv, state, idir, sgn, geom, time, s_ext);
        state(iv, dcomp) = s_ext[Comp];
      }
    }
  }
};

template <typename T>
using PeleLMCCFillExtDirDens = PeleLMCCFillExtDirScalar<T, DENSITY>;
template <typename T>
using PeleLMCCFillExtDirRhoH = PeleLMCCFillExtDirScalar<T, RHOH>;
template <typename T>
using PeleLMCCFillExtDirTemp = PeleLMCCFillExtDirScalar<T, TEMP>;
#ifdef PELE_USE_PLASMA
template <typename T>
using PeleLMCCFillExtDirnE = PeleLMCCFillExtDirScalar<T, NE>;
template <typename T>
using PeleLMCCFillExtDirPhiV = PeleLMCCFillExtDirScalar<T, PHIV>;
#endif

struct PeleLMCCFillExtDirDummy
{
  const int m_nAux;

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirDummy(int a_nAux) : m_nAux(a_nAux) {}

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& /*iv*/,
    amrex::Array4<amrex::Real> const& /*dummy*/,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& /*geom*/,
    const amrex::Real /*time*/,
    const amrex::BCRec* /*bcr*/,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
  }
};

//
// A dummy function because FillPatch requires something to exist for filling
// dirichlet boundary conditions, even if we know we cannot have an ext_dir BC.
// u_mac BCs are only either periodic (amrex::BCType::int_dir) or first order
// extrapolation (amrex::BCType::foextrap).
//
struct umacFill
{
  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& /*iv*/,
    amrex::Array4<amrex::Real> const& /*dummy*/,
    const int /*dcomp*/,
    const int numcomp,
    amrex::GeometryData const& /*geom*/,
    const amrex::Real /*time*/,
    const amrex::BCRec* bcr,
    const int bcomp,
    const int /*orig_comp*/) const
  {
    // Abort if this function is expected to fill an ext_dir BC.
    for (int n = bcomp; n < bcomp + numcomp; ++n) {
      const amrex::BCRec& bc = bcr[n];
      if (AMREX_D_TERM(
            bc.lo(0) == amrex::BCType::ext_dir ||
              bc.hi(0) == amrex::BCType::ext_dir,
            || bc.lo(1) == amrex::BCType::ext_dir ||
              bc.hi(1) == amrex::BCType::ext_dir,
            || bc.lo(2) == amrex::BCType::ext_dir ||
              bc.hi(2) == amrex::BCType::ext_dir)) {
        amrex::Abort("PeleLM::umacFill: umac should not have BCType::ext_dir");
      }
    }
  }
};
#endif
