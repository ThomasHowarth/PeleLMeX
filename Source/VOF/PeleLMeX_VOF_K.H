#ifndef PELELM_VOF_K_H
#define PELELM_VOF_K_H

// Geometric PLIC volume-of-fluid kernels.
//
// The interface in a cell is the plane m.x = alpha in unit-cube cell
// coordinates, with the liquid on the side m.x <= alpha (m points out of
// the liquid).  Normals come from the Mixed Youngs-Centered scheme
// (Aulisa et al., JCP 225, 2007) and the volume <-> plane-constant
// relations are the analytic ones of Scardovelli & Zaleski (JCP 164, 2000).
//
// Ported from Kynema-SGF (formerly AMR-Wind), src/equation_systems/vof,
// BSD 3-Clause License, Copyright (c) 2020, Alliance for Energy Innovation
// and Regents of the University of California.  Changes: 2D variants,
// per-face flux evaluation, PeleLMeX conventions.

#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <limits>

namespace pelelmex::vof {

using namespace amrex::literals;

// Cells within this of 0 or 1 count as pure.  It must stay well above the noise
// in F (about 1e-11): otherwise a cell flips between mixed and pure from one
// neighbour to the next, which breaks the symmetry of the flow.
static constexpr amrex::Real vof_tol = 1.0e-8_rt;

// Half-length of the height-function columns (7 cells); the volume fraction
// carries one more ghost cell so that planes can be fitted in the first
// ghost layer
static constexpr int hf_half = 3;
static constexpr int nghost = hf_half + 1;
// Cells searched along a wall for the interface meeting it, on either side of
// a wall ghost cell (see wall_ghost_fraction)
static constexpr int wall_nsearch = 8;

/**
 * \brief Analytic plane constant alpha such that the region
 *        m1*x1 + m2*x2 + m3*x3 <= alpha of the unit cube has volume volF.
 *        Requires m1,m2,m3 >= 0 and m1+m2+m3 = 1.  Any one (2D) or two
 *        (grid-aligned) coefficients may be zero.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
volume_intercept(
  const amrex::Real b1,
  const amrex::Real b2,
  const amrex::Real b3,
  const amrex::Real volF)
{
  constexpr amrex::Real tiny = std::numeric_limits<amrex::Real>::epsilon();

  // Order coefficients: m1 <= m2 <= m3
  amrex::Real m1 = amrex::min(b1, b2);
  amrex::Real m3 = amrex::max(b1, b2);
  amrex::Real m2 = b3;
  if (m2 < m1) {
    const amrex::Real tmp = m1;
    m1 = m2;
    m2 = tmp;
  } else if (m2 > m3) {
    const amrex::Real tmp = m3;
    m3 = m2;
    m2 = tmp;
  }

  // Volume at the breakpoints V1 < V2 < V3
  const amrex::Real m12 = m1 + m2;
  const amrex::Real pr = 6.0_rt * m1 * m2 * m3;
  // Without a corner region (m1 = 0, a 2D or grid-aligned plane), V1 = 0
  const amrex::Real V1 = (pr > 0.0_rt) ? m1 * m1 * m1 / pr : 0.0_rt;
  const amrex::Real V2 = V1 + 0.5_rt * (m2 - m1) / (m3 + tiny);
  amrex::Real mm = 0.0_rt;
  amrex::Real V3 = 0.0_rt;
  if (m3 < m12) {
    mm = m3;
    V3 = (m3 * m3 * (3.0_rt * m12 - m3) + m1 * m1 * (m1 - 3.0_rt * m3) +
          m2 * m2 * (m2 - 3.0_rt * m3)) /
         pr;
  } else {
    mm = m12;
    V3 = 0.5_rt * mm / (m3 + tiny);
  }

  // Use symmetry to work with ch <= 1/2
  const amrex::Real ch = amrex::min(volF, 1.0_rt - volF);

  amrex::Real alpha = 0.0_rt;
  if (ch < V1) {
    alpha = std::cbrt(pr * ch);
  } else if (ch < V2) {
    alpha = 0.5_rt * (m1 + std::sqrt(m1 * m1 + 8.0_rt * m2 * m3 * (ch - V1)));
  } else if (ch < V3) {
    const amrex::Real p = 2.0_rt * m1 * m2;
    const amrex::Real q = 1.5_rt * m1 * m2 * (m12 - 2.0_rt * m3 * ch);
    const amrex::Real p12 = std::sqrt(amrex::max(0.0_rt, p));
    const amrex::Real pp = p * p12;
    const amrex::Real teta =
      std::acos(
        (pp > 0.0_rt) ? amrex::Clamp(q / pp, -1.0_rt, 1.0_rt) : 1.0_rt) /
      3.0_rt;
    const amrex::Real cs = std::cos(teta);
    alpha = p12 * (std::sqrt(3.0_rt * (1.0_rt - cs * cs)) - cs) + m12;
  } else if (m12 < m3) {
    alpha = m3 * ch + 0.5_rt * mm;
  } else {
    const amrex::Real p = m1 * (m2 + m3) + m2 * m3 - 0.25_rt;
    const amrex::Real q = 1.5_rt * m1 * m2 * m3 * (0.5_rt - ch);
    const amrex::Real p12 = std::sqrt(amrex::max(0.0_rt, p));
    const amrex::Real pp = p * p12;
    const amrex::Real teta =
      std::acos(
        (pp > 0.0_rt) ? amrex::Clamp(q / pp, -1.0_rt, 1.0_rt) : 1.0_rt) /
      3.0_rt;
    const amrex::Real cs = std::cos(teta);
    alpha = p12 * (std::sqrt(3.0_rt * (1.0_rt - cs * cs)) - cs) + 0.5_rt;
  }

  if (volF > 0.5_rt) {
    alpha = 1.0_rt - alpha;
  }
  return alpha;
}

/**
 * \brief Liquid volume fraction of the sub-box [r0, r0+dr0] x [0,1] x [0,1]
 *        of the unit cube cut by m1*x1 + m2*x2 + m3*x3 <= alpha.  The
 *        result is normalized by the sub-box volume dr0.  Coefficients may
 *        have any sign; m3 = 0 gives the 2D result.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
cut_volume(
  const amrex::Real m1,
  const amrex::Real m2,
  const amrex::Real m3,
  const amrex::Real alpha,
  const amrex::Real r0,
  const amrex::Real dr0)
{
  constexpr amrex::Real tiny = std::numeric_limits<amrex::Real>::epsilon();

  // Move origin to r0, then reflect so that all coefficients are positive
  amrex::Real al = alpha - m1 * r0;
  al += amrex::max(0.0_rt, -m1 * dr0) + amrex::max(0.0_rt, -m2) +
        amrex::max(0.0_rt, -m3);

  // Normalize to the unit cube: n1*y1 + n2*y2 + n3*y3 = al, sum(n) = 1
  amrex::Real tmp =
    amrex::max(tiny, std::abs(m1) * dr0 + std::abs(m2) + std::abs(m3));
  const amrex::Real n1 = std::abs(m1) / tmp;
  const amrex::Real n2 = std::abs(m2) / tmp;
  const amrex::Real n3 = std::abs(m3) / tmp;
  al = amrex::max(0.0_rt, amrex::min(1.0_rt, al / tmp));
  const amrex::Real al0 = amrex::min(al, 1.0_rt - al);

  // Order coefficients: b1 <= b2 <= b3
  amrex::Real b1 = amrex::min(n1 * dr0, n2);
  amrex::Real b3 = amrex::max(n1 * dr0, n2);
  amrex::Real b2 = n3;
  if (b2 < b1) {
    tmp = b1;
    b1 = b2;
    b2 = tmp;
  } else if (b2 > b3) {
    tmp = b3;
    b3 = b2;
    b2 = tmp;
  }
  const amrex::Real b12 = b1 + b2;

  // Scardovelli & Zaleski (JCP 2000), as arranged by Kawano (C&F 2016)
  const amrex::Real a = al0;
  amrex::Real v = 0.0_rt;
  if (a > 0.0_rt) {
    if (a < b1) {
      v = a * a * a / (6.0_rt * b1 * b2 * b3);
    } else if (a < b2) {
      v =
        a * (a - b1) / (2.0_rt * b2 * b3) + b1 * b1 / (6.0_rt * b2 * b3 + tiny);
    } else if (a < amrex::min(b12, b3)) {
      v = (a * a * (3.0_rt * b12 - a) + b1 * b1 * (b1 - 3.0_rt * a) +
           b2 * b2 * (b2 - 3.0_rt * a)) /
          (6.0_rt * b1 * b2 * b3);
    } else if (b3 < b12) {
      v = (a * a * (3.0_rt - 2.0_rt * a) + b1 * b1 * (b1 - 3.0_rt * a) +
           b2 * b2 * (b2 - 3.0_rt * a) + b3 * b3 * (b3 - 3.0_rt * a)) /
          (6.0_rt * b1 * b2 * b3);
    } else {
      v = (a - 0.5_rt * b12) / b3;
    }
  }

  return (al <= 0.5_rt) ? v : 1.0_rt - v;
}

#if AMREX_SPACEDIM == 3
/**
 * \brief Youngs' finite-difference normal on the 27-point stencil, scaled
 *        by -32 (i.e. m = -32 grad F).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
youngs_normal(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Real& mx,
  amrex::Real& my,
  amrex::Real& mz)
{
  amrex::Real mm1 = F(i - 1, j - 1, k - 1) + F(i - 1, j - 1, k + 1) +
                    F(i - 1, j + 1, k - 1) + F(i - 1, j + 1, k + 1) +
                    2.0_rt * (F(i - 1, j - 1, k) + F(i - 1, j + 1, k) +
                              F(i - 1, j, k - 1) + F(i - 1, j, k + 1)) +
                    4.0_rt * F(i - 1, j, k);
  amrex::Real mm2 = F(i + 1, j - 1, k - 1) + F(i + 1, j - 1, k + 1) +
                    F(i + 1, j + 1, k - 1) + F(i + 1, j + 1, k + 1) +
                    2.0_rt * (F(i + 1, j - 1, k) + F(i + 1, j + 1, k) +
                              F(i + 1, j, k - 1) + F(i + 1, j, k + 1)) +
                    4.0_rt * F(i + 1, j, k);
  mx = mm1 - mm2;

  mm1 = F(i - 1, j - 1, k - 1) + F(i - 1, j - 1, k + 1) +
        F(i + 1, j - 1, k - 1) + F(i + 1, j - 1, k + 1) +
        2.0_rt * (F(i - 1, j - 1, k) + F(i + 1, j - 1, k) + F(i, j - 1, k - 1) +
                  F(i, j - 1, k + 1)) +
        4.0_rt * F(i, j - 1, k);
  mm2 = F(i - 1, j + 1, k - 1) + F(i - 1, j + 1, k + 1) +
        F(i + 1, j + 1, k - 1) + F(i + 1, j + 1, k + 1) +
        2.0_rt * (F(i - 1, j + 1, k) + F(i + 1, j + 1, k) + F(i, j + 1, k - 1) +
                  F(i, j + 1, k + 1)) +
        4.0_rt * F(i, j + 1, k);
  my = mm1 - mm2;

  mm1 = F(i - 1, j - 1, k - 1) + F(i - 1, j + 1, k - 1) +
        F(i + 1, j - 1, k - 1) + F(i + 1, j + 1, k - 1) +
        2.0_rt * (F(i - 1, j, k - 1) + F(i + 1, j, k - 1) + F(i, j - 1, k - 1) +
                  F(i, j + 1, k - 1)) +
        4.0_rt * F(i, j, k - 1);
  mm2 = F(i - 1, j - 1, k + 1) + F(i - 1, j + 1, k + 1) +
        F(i + 1, j - 1, k + 1) + F(i + 1, j + 1, k + 1) +
        2.0_rt * (F(i - 1, j, k + 1) + F(i + 1, j, k + 1) + F(i, j - 1, k + 1) +
                  F(i, j + 1, k + 1)) +
        4.0_rt * F(i, j, k + 1);
  mz = mm1 - mm2;
}

/**
 * \brief Mixed Youngs-Centered normal (3D).  Returns an L1-normalized
 *        normal pointing out of the liquid.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
myc_normal(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Real& mx,
  amrex::Real& my,
  amrex::Real& mz)
{
  // m[c][d]: candidate c (0-2 centered height-function schemes along x, y,
  // z, 3 Youngs), component d
  amrex::Real m[4][3];

  // Centered scheme with x as the height direction:
  //   sgn(mx) X = my Y + mz Z + alpha
  amrex::Real m1 = F(i - 1, j, k - 1) + F(i - 1, j, k + 1) +
                   F(i - 1, j - 1, k) + F(i - 1, j + 1, k) + F(i - 1, j, k);
  amrex::Real m2 = F(i + 1, j, k - 1) + F(i + 1, j, k + 1) +
                   F(i + 1, j - 1, k) + F(i + 1, j + 1, k) + F(i + 1, j, k);
  m[0][0] = (m1 > m2) ? 1.0_rt : -1.0_rt;
  m1 = F(i - 1, j - 1, k) + F(i + 1, j - 1, k) + F(i, j - 1, k);
  m2 = F(i - 1, j + 1, k) + F(i + 1, j + 1, k) + F(i, j + 1, k);
  m[0][1] = 0.5_rt * (m1 - m2);
  m1 = F(i - 1, j, k - 1) + F(i + 1, j, k - 1) + F(i, j, k - 1);
  m2 = F(i - 1, j, k + 1) + F(i + 1, j, k + 1) + F(i, j, k + 1);
  m[0][2] = 0.5_rt * (m1 - m2);

  // y as the height direction
  m1 = F(i - 1, j - 1, k) + F(i - 1, j + 1, k) + F(i - 1, j, k);
  m2 = F(i + 1, j - 1, k) + F(i + 1, j + 1, k) + F(i + 1, j, k);
  m[1][0] = 0.5_rt * (m1 - m2);
  m1 = F(i, j - 1, k - 1) + F(i, j - 1, k + 1) + F(i + 1, j - 1, k) +
       F(i - 1, j - 1, k) + F(i, j - 1, k);
  m2 = F(i, j + 1, k - 1) + F(i, j + 1, k + 1) + F(i + 1, j + 1, k) +
       F(i - 1, j + 1, k) + F(i, j + 1, k);
  m[1][1] = (m1 > m2) ? 1.0_rt : -1.0_rt;
  m1 = F(i, j - 1, k - 1) + F(i, j, k - 1) + F(i, j + 1, k - 1);
  m2 = F(i, j - 1, k + 1) + F(i, j, k + 1) + F(i, j + 1, k + 1);
  m[1][2] = 0.5_rt * (m1 - m2);

  // z as the height direction
  m1 = F(i - 1, j, k - 1) + F(i - 1, j, k + 1) + F(i - 1, j, k);
  m2 = F(i + 1, j, k - 1) + F(i + 1, j, k + 1) + F(i + 1, j, k);
  m[2][0] = 0.5_rt * (m1 - m2);
  m1 = F(i, j - 1, k - 1) + F(i, j - 1, k + 1) + F(i, j - 1, k);
  m2 = F(i, j + 1, k - 1) + F(i, j + 1, k + 1) + F(i, j + 1, k);
  m[2][1] = 0.5_rt * (m1 - m2);
  m1 = F(i - 1, j, k - 1) + F(i + 1, j, k - 1) + F(i, j - 1, k - 1) +
       F(i, j + 1, k - 1) + F(i, j, k - 1);
  m2 = F(i - 1, j, k + 1) + F(i + 1, j, k + 1) + F(i, j - 1, k + 1) +
       F(i, j + 1, k + 1) + F(i, j, k + 1);
  m[2][2] = (m1 > m2) ? 1.0_rt : -1.0_rt;

  // L1-normalize the centered candidates and keep the one whose height
  // direction dominates
  for (int c = 0; c < 3; ++c) {
    const amrex::Real t =
      std::abs(m[c][0]) + std::abs(m[c][1]) + std::abs(m[c][2]);
    for (int d = 0; d < 3; ++d) {
      m[c][d] /= t;
    }
  }
  int cn = 0;
  amrex::Real t0 = std::abs(m[0][0]);
  if (std::abs(m[1][1]) > t0) {
    t0 = std::abs(m[1][1]);
    cn = 1;
  }
  if (std::abs(m[2][2]) > t0) {
    cn = 2;
  }

  // Youngs candidate
  constexpr amrex::Real tiny = std::numeric_limits<amrex::Real>::epsilon();
  youngs_normal(i, j, k, F, m[3][0], m[3][1], m[3][2]);
  amrex::Real t =
    std::abs(m[3][0]) + std::abs(m[3][1]) + std::abs(m[3][2]) + tiny;
  for (int d = 0; d < 3; ++d) {
    m[3][d] /= t;
  }

  // Youngs wins when its largest component is smaller than the centered
  // scheme's height component (interface not aligned with a grid plane)
  t = amrex::max(std::abs(m[3][0]), std::abs(m[3][1]), std::abs(m[3][2]));
  if (std::abs(m[cn][cn]) > t && t > 0.0_rt) {
    cn = 3;
  }

  mx = m[cn][0];
  my = m[cn][1];
  mz = m[cn][2];
}
#else
/**
 * \brief Youngs' finite-difference normal on the 9-point stencil, scaled
 *        by -8 (i.e. m = -8 grad F).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
youngs_normal(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Real& mx,
  amrex::Real& my,
  amrex::Real& mz)
{
  mx = (F(i - 1, j - 1, k) + 2.0_rt * F(i - 1, j, k) + F(i - 1, j + 1, k)) -
       (F(i + 1, j - 1, k) + 2.0_rt * F(i + 1, j, k) + F(i + 1, j + 1, k));
  my = (F(i - 1, j - 1, k) + 2.0_rt * F(i, j - 1, k) + F(i + 1, j - 1, k)) -
       (F(i - 1, j + 1, k) + 2.0_rt * F(i, j + 1, k) + F(i + 1, j + 1, k));
  mz = 0.0_rt;
}

/**
 * \brief Mixed Youngs-Centered normal (2D).  Returns an L1-normalized
 *        normal pointing out of the liquid.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
myc_normal(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Real& mx,
  amrex::Real& my,
  amrex::Real& mz)
{
  // m[c][d]: candidate c (0-1 centered height-function schemes along x, y,
  // 2 Youngs), component d
  amrex::Real m[3][2];

  // Column / row sums
  const amrex::Real cL =
    F(i - 1, j - 1, k) + F(i - 1, j, k) + F(i - 1, j + 1, k);
  const amrex::Real cR =
    F(i + 1, j - 1, k) + F(i + 1, j, k) + F(i + 1, j + 1, k);
  const amrex::Real rB =
    F(i - 1, j - 1, k) + F(i, j - 1, k) + F(i + 1, j - 1, k);
  const amrex::Real rT =
    F(i - 1, j + 1, k) + F(i, j + 1, k) + F(i + 1, j + 1, k);

  // x as the height direction: sgn(mx) X = my Y + alpha
  m[0][0] = (cL > cR) ? 1.0_rt : -1.0_rt;
  m[0][1] = 0.5_rt * (rB - rT);
  // y as the height direction: sgn(my) Y = mx X + alpha
  m[1][0] = 0.5_rt * (cL - cR);
  m[1][1] = (rB > rT) ? 1.0_rt : -1.0_rt;

  for (int c = 0; c < 2; ++c) {
    const amrex::Real t = std::abs(m[c][0]) + std::abs(m[c][1]);
    m[c][0] /= t;
    m[c][1] /= t;
  }
  int cn = (std::abs(m[1][1]) > std::abs(m[0][0])) ? 1 : 0;

  constexpr amrex::Real tiny = std::numeric_limits<amrex::Real>::epsilon();
  amrex::Real mzy = 0.0_rt;
  youngs_normal(i, j, k, F, m[2][0], m[2][1], mzy);
  amrex::Real t = std::abs(m[2][0]) + std::abs(m[2][1]) + tiny;
  m[2][0] /= t;
  m[2][1] /= t;

  t = amrex::max(std::abs(m[2][0]), std::abs(m[2][1]));
  if (std::abs(m[cn][cn]) > t && t > 0.0_rt) {
    cn = 2;
  }

  mx = m[cn][0];
  my = m[cn][1];
  mz = 0.0_rt;
}
#endif

/**
 * \brief Height-function normal with height direction d: the liquid heights
 *        of the 7-cell columns through (i,j,k) and its neighbors in the
 *        tangential directions, differenced centrally.  Exact for a planar
 *        interface whose slope relative to d is at most 1, which holds for
 *        the dominant normal direction.  Returns false (m untouched) when a
 *        column does not span from pure liquid to pure gas in the same
 *        orientation as the others.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
height_normal(
  const int i,
  const int j,
  const int k,
  const int d,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Real m[3])
{
  amrex::ignore_unused(k);
  constexpr amrex::Real pure_tol = 1.0e-6_rt;
  const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
  const amrex::IntVect ed = amrex::IntVect::TheDimensionVector(d);
  // Orientation from the center column: +1 liquid on the low side
  const amrex::Real f_lo = F(iv - hf_half * ed);
  const amrex::Real f_hi = F(iv + hf_half * ed);
  amrex::Real sigma = 0.0_rt;
  if (f_lo >= 1.0_rt - pure_tol && f_hi <= pure_tol) {
    sigma = 1.0_rt;
  } else if (f_lo <= pure_tol && f_hi >= 1.0_rt - pure_tol) {
    sigma = -1.0_rt;
  } else {
    return false;
  }
  const auto column = [&](const amrex::IntVect& c, amrex::Real& H) {
    const amrex::Real a = F(c - hf_half * ed);
    const amrex::Real b = F(c + hf_half * ed);
    const amrex::Real a_pure = (sigma > 0.0_rt) ? 1.0_rt - a : a;
    const amrex::Real b_pure = (sigma > 0.0_rt) ? b : 1.0_rt - b;
    if (a_pure > pure_tol || b_pure > pure_tol) {
      return false;
    }
    H = 0.0_rt;
    for (int s = -hf_half; s <= hf_half; ++s) {
      H += F(c + s * ed);
    }
    return true;
  };
  amrex::Real mt[3] = {0.0_rt, 0.0_rt, 0.0_rt};
  for (int t = 0; t < AMREX_SPACEDIM; ++t) {
    if (t == d) {
      continue;
    }
    const amrex::IntVect et = amrex::IntVect::TheDimensionVector(t);
    amrex::Real Hm = 0.0_rt;
    amrex::Real Hp = 0.0_rt;
    if (!column(iv - et, Hm) || !column(iv + et, Hp)) {
      return false;
    }
    // m = grad(x_d - h) (sigma = 1) or grad(h - x_d): m_t = (H- - H+) / 2
    mt[t] = 0.5_rt * (Hm - Hp);
  }
  mt[d] = sigma;
  for (int a = 0; a < 3; ++a) {
    m[a] = mt[a];
  }
  return true;
}

/**
 * \brief Reconstruct the PLIC plane m.x = alpha in cell (i,j,k), in unit-cube
 *        cell coordinates with sum|m| = 1.  The normal is the height-function
 *        one along the dominant direction of the Mixed Youngs-Centered
 *        normal, or the latter where the heights are not defined (thin or
 *        strongly curved structures).  Reads up to hf_half cells away.  Only
 *        meaningful for vof_tol < F < 1 - vof_tol.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
fit_plane(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Real& mx,
  amrex::Real& my,
  amrex::Real& mz,
  amrex::Real& alpha)
{
  myc_normal(i, j, k, F, mx, my, mz);
  {
    const amrex::Real am[3] = {std::abs(mx), std::abs(my), std::abs(mz)};
    int d = (am[1] > am[0]) ? 1 : 0;
    if (AMREX_SPACEDIM == 3 && am[2] > am[d]) {
      d = 2;
    }
    amrex::Real mh[3];
    if (height_normal(i, j, k, d, F, mh)) {
      mx = mh[0];
      my = mh[1];
      mz = mh[2];
    }
  }

  // An isolated blob can produce a null normal: pick an arbitrary one
  amrex::Real norm = std::abs(mx) + std::abs(my) + std::abs(mz);
  if (norm <= std::numeric_limits<amrex::Real>::epsilon()) {
    mx = 1.0_rt;
    my = 0.0_rt;
    mz = 0.0_rt;
    norm = 1.0_rt;
  }
  mx /= norm;
  my /= norm;
  mz /= norm;

  alpha =
    volume_intercept(std::abs(mx), std::abs(my), std::abs(mz), F(i, j, k));

  // Back from the reflected (all positive) frame to the cell frame
  alpha +=
    amrex::min(0.0_rt, mx) + amrex::min(0.0_rt, my) + amrex::min(0.0_rt, mz);
}

/**
 * \brief Liquid volume fraction of cell (i,j,k) for an implicit surface
 *        phi(x) > 0 inside the liquid, by midpoint sampling on nsub^dim
 *        sub-cells.  Intended for problem initialization.
 */
template <typename Phi>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
cell_fraction(
  const int i,
  const int j,
  const int k,
  amrex::GeometryData const& geomdata,
  Phi const& phi,
  const int nsub = 8)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real h = 1.0_rt / static_cast<amrex::Real>(nsub);
  int inside = 0;
#if AMREX_SPACEDIM == 3
  for (int kk = 0; kk < nsub; ++kk) {
    const amrex::Real z = prob_lo[2] + (k + (kk + 0.5_rt) * h) * dx[2];
#else
  amrex::ignore_unused(k);
  {
    const amrex::Real z = 0.0_rt;
#endif
    for (int jj = 0; jj < nsub; ++jj) {
      const amrex::Real y = prob_lo[1] + (j + (jj + 0.5_rt) * h) * dx[1];
      for (int ii = 0; ii < nsub; ++ii) {
        const amrex::Real x = prob_lo[0] + (i + (ii + 0.5_rt) * h) * dx[0];
        if (phi(x, y, z) > 0.0_rt) {
          ++inside;
        }
      }
    }
  }
  return static_cast<amrex::Real>(inside) /
         static_cast<amrex::Real>(AMREX_D_TERM(nsub, *nsub, *nsub));
}

/**
 * \brief True if cell (i,j,k) is within nband cells of the interface.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
interface_band(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  const int nband = 1)
{
  amrex::Real Fmax = 0.0_rt;
  amrex::Real Fmin = 1.0_rt;
#if AMREX_SPACEDIM == 3
  for (int kk = -nband; kk <= nband; ++kk) {
#else
  {
    constexpr int kk = 0;
#endif
    for (int jj = -nband; jj <= nband; ++jj) {
      for (int ii = -nband; ii <= nband; ++ii) {
        const amrex::Real f = F(i + ii, j + jj, k + kk);
        if (f > vof_tol && f < 1.0_rt - vof_tol) {
          return true;
        }
        Fmax = amrex::max(Fmax, f);
        Fmin = amrex::min(Fmin, f);
      }
    }
  }
  // Interface lying exactly on a cell face
  return (Fmax - Fmin) > 0.5_rt;
}

/**
 * \brief Exact liquid volume fraction of cell (i,j,k) for the half space
 *        n.x <= d (physical coordinates, n need not be normalized).
 *        Intended for problem initialization.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
plane_fraction(
  const int i,
  const int j,
  const int k,
  amrex::GeometryData const& geomdata,
  const amrex::Real n[AMREX_SPACEDIM],
  const amrex::Real d)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  const int idx[3] = {i, j, k};
  amrex::Real m[3] = {0.0_rt, 0.0_rt, 0.0_rt};
  amrex::Real alpha = d;
  amrex::Real norm = 0.0_rt;
  for (int a = 0; a < AMREX_SPACEDIM; ++a) {
    m[a] = n[a] * dx[a];
    alpha -= n[a] * (prob_lo[a] + idx[a] * dx[a]);
    norm += std::abs(m[a]);
  }
  if (norm <= std::numeric_limits<amrex::Real>::min()) {
    return (alpha >= 0.0_rt) ? 1.0_rt : 0.0_rt;
  }
  return amrex::Clamp(
    cut_volume(
      m[0] / norm, m[1] / norm, m[2] / norm, alpha / norm, 0.0_rt, 1.0_rt),
    0.0_rt, 1.0_rt);
}

/**
 * \brief Volume fraction of the ghost cell g beyond a wall normal to dir, by
 *        extending the PLIC plane of the interior interface: a free contact
 *        line, exact for a planar interface and for one uniform normal to the
 *        wall.  Used when the interface meets
 *        the wall within nsearch cells (tangentially) of g.  The plane is
 *        that of the nearest interface cell, 1 to hf_half layers inside the
 *        wall, whose reconstruction does not read ghost cells: height
 *        columns normal to the wall reach hf_half cells, the other stencils
 *        one.  With nsearch = 8 such a plane exists for contact angles
 *        down to about 20 degrees from the wall.  Below, if allow_ghost, the
 *        nearest interface cell of layers 0 to hf_half is used although its
 *        reconstruction reads the current ghost values: iterating the fill
 *        converges to the extended plane, which is a fixed point.
 *        Otherwise the wall-adjacent value is copied (90 degrees).
 * \param iwall index along dir of the wall-adjacent interior cell
 * \param sgn   -1 for a low wall, +1 for a high wall
 * \param limit cells of F that hold data and may be searched
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
wall_ghost_fraction(
  const amrex::IntVect& g,
  const int dir,
  const int iwall,
  const int sgn,
  const amrex::Box& limit,
  amrex::Array4<amrex::Real const> const& F,
  const bool allow_ghost)
{
  constexpr int nsearch = wall_nsearch;
  constexpr int nfar = 1000;
  const amrex::Box rbox = amrex::grow(limit, -hf_half);
  amrex::IntVect iv_w = g;
  iv_w[dir] = iwall;

  const auto is_mixed = [&F](const amrex::IntVect& c) {
    const amrex::Real f = F(c);
    return f > vof_tol && f < 1.0_rt - vof_tol;
  };

  amrex::IntVect off_lo(-nsearch);
  amrex::IntVect off_hi(nsearch);
  off_lo[dir] = 0;
  off_hi[dir] = 0;
#if AMREX_SPACEDIM == 2
  const int b_lo = 0;
  const int b_hi = 0;
#else
  const int b_lo = off_lo[2];
  const int b_hi = off_hi[2];
#endif

  // Most ghost cells are far from any contact line: no search for those
  bool contact = false;
  for (int b = b_lo; b <= b_hi && !contact; ++b) {
    for (int a1 = off_lo[1]; a1 <= off_hi[1] && !contact; ++a1) {
      for (int a0 = off_lo[0]; a0 <= off_hi[0] && !contact; ++a0) {
        const amrex::IntVect cw =
          iv_w + amrex::IntVect(AMREX_D_DECL(a0, a1, b));
        contact = limit.contains(cw) && is_mixed(cw);
      }
    }
  }
  if (!contact) {
    return F(iv_w);
  }

  int ref_dist = nfar;
  amrex::IntVect ref = iv_w;
  // Fallback reading ghost cells
  int any_dist = nfar;
  amrex::IntVect any_ref = iv_w;
  for (int b = b_lo; b <= b_hi; ++b) {
    for (int a1 = off_lo[1]; a1 <= off_hi[1]; ++a1) {
      for (int a0 = off_lo[0]; a0 <= off_hi[0]; ++a0) {
        const amrex::IntVect off(AMREX_D_DECL(a0, a1, b));
        const int tdist =
          AMREX_D_TERM(std::abs(a0), +std::abs(a1), +std::abs(b));
        const amrex::IntVect cw = iv_w + off;
        for (int layer = 0; layer <= hf_half; ++layer) {
          amrex::IntVect c = cw;
          c[dir] = iwall - layer * sgn;
          const int dist = tdist + layer;
          if (dist >= ref_dist || !rbox.contains(c) || !is_mixed(c)) {
            continue;
          }
          if (dist < any_dist) {
            any_dist = dist;
            any_ref = c;
          }
          if (layer == 0) {
            continue;
          }
          if (layer < hf_half) {
            // Height direction of fit_plane: wall-normal columns would read
            // ghost cells
            amrex::Real mm[3];
            const auto cell = c.dim3();
            myc_normal(cell.x, cell.y, cell.z, F, mm[0], mm[1], mm[2]);
            bool normal_dominant = true;
            for (int t = 0; t < AMREX_SPACEDIM; ++t) {
              if (t != dir && std::abs(mm[t]) > std::abs(mm[dir])) {
                normal_dominant = false;
              }
            }
            if (normal_dominant) {
              continue;
            }
          }
          ref_dist = dist;
          ref = c;
        }
      }
    }
  }
  if (ref_dist == nfar && allow_ghost) {
    ref_dist = any_dist;
    ref = any_ref;
  }
  if (ref_dist == nfar) {
    return F(iv_w);
  }

  amrex::Real m[3] = {0.0_rt, 0.0_rt, 0.0_rt};
  amrex::Real alpha = 0.0_rt;
  const auto cell = ref.dim3();
  fit_plane(cell.x, cell.y, cell.z, F, m[0], m[1], m[2], alpha);
  // Plane constant in the unit-cube coordinates of g and of the wall cell.  The
  // ghost takes the change of the plane's volume from the wall cell: the same
  // as the plane's volume at g for a planar interface, and the value of the
  // wall cell where the plane does not vary across the wall, whichever cell the
  // plane comes from, so that a flow uniform normal to the wall stays uniform.
  amrex::Real alpha_g = alpha;
  amrex::Real alpha_w = alpha;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    alpha_g -= m[d] * static_cast<amrex::Real>(g[d] - ref[d]);
    alpha_w -= m[d] * static_cast<amrex::Real>(iv_w[d] - ref[d]);
  }
  const amrex::Real V_g = cut_volume(m[0], m[1], m[2], alpha_g, 0.0_rt, 1.0_rt);
  const amrex::Real V_w = cut_volume(m[0], m[1], m[2], alpha_w, 0.0_rt, 1.0_rt);
  return amrex::Clamp(F(iv_w) + V_g - V_w, 0.0_rt, 1.0_rt);
}

/**
 * \brief Liquid fraction of the half segment from the center of cell
 *        (i,j,k) to the center of its face in direction dir on side sgn
 *        (+1 high, -1 low), from the cell's own PLIC plane.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
half_segment_fraction(
  const int i,
  const int j,
  const int k,
  const int dir,
  const int sgn,
  amrex::Array4<amrex::Real const> const& F)
{
  const amrex::Real Fc = F(i, j, k);
  if (Fc <= vof_tol) {
    return 0.0_rt;
  }
  if (Fc >= 1.0_rt - vof_tol) {
    return 1.0_rt;
  }
  amrex::Real m[3], alpha;
  fit_plane(i, j, k, F, m[0], m[1], m[2], alpha);
  // psi = m.x - alpha < 0 in the liquid, linear along the half segment
  const amrex::Real psi_c = 0.5_rt * (m[0] + m[1] + m[2]) - alpha;
  const amrex::Real psi_f = psi_c + 0.5_rt * sgn * m[dir];
  if (psi_c <= 0.0_rt && psi_f <= 0.0_rt) {
    return 1.0_rt;
  }
  if (psi_c > 0.0_rt && psi_f > 0.0_rt) {
    return 0.0_rt;
  }
  const amrex::Real t = psi_c / (psi_c - psi_f);
  return (psi_c <= 0.0_rt) ? t : 1.0_rt - t;
}

/**
 * \brief Liquid fraction of the segment joining the centers of the cells on
 *        either side of the low face of (i,j,k) in direction dir.  Exact for
 *        a planar interface, which makes rho_face * g a discrete gradient
 *        (hydrostatic balance) when rho_face = rho_g + (rho_l - rho_g) * this.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
face_segment_fraction(
  const int i,
  const int j,
  const int k,
  const int dir,
  amrex::Array4<amrex::Real const> const& F)
{
  const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
  const amrex::IntVect ivm = iv - amrex::IntVect::TheDimensionVector(dir);
#if AMREX_SPACEDIM == 2
  const int km = 0;
#else
  const int km = ivm[2];
#endif
  return 0.5_rt * (half_segment_fraction(ivm[0], ivm[1], km, dir, 1, F) +
                   half_segment_fraction(i, j, k, dir, -1, F));
}

/**
 * \brief Mixture property (density, viscosity) of a cell with liquid fraction
 *        F: arithmetic or harmonic volume average of the phase values
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mixture(
  const amrex::Real F,
  const amrex::Real liq,
  const amrex::Real gas,
  const int harmonic = 0)
{
  const amrex::Real f = amrex::Clamp(F, 0.0_rt, 1.0_rt);
  if (harmonic != 0) {
    return liq * gas / (f * gas + (1.0_rt - f) * liq);
  }
  return f * liq + (1.0_rt - f) * gas;
}

//-----------------------------------------------------------------------------
// Unsplit geometric flux kernels for the PLIC volume of fluid, after Owkes &
// Desjardins (JCP 270, 2014).
//
// The liquid crossing a face in a step is the liquid inside the flux volume,
// the region bounded by the face and its corners traced back along the flow.
// Everything here works in cell units: cell (i,j,k) is the unit cube
// [i,i+1] x [j,j+1] x [k,k+1], and face velocities are Courant numbers.  In 2D
// the mesh is one cell thick in z, so that one kernel serves both dimensions.
//
// Differences from the paper, none of which change the method:
// - The flux volume is the signed sum of the tetrahedra joining a corner of the
//   face to the triangles of its boundary, rather than the tabulated partition
//   of its Fig. 9.  The two are equivalent: any closed oriented triangulated
//   surface gives the right signed integral, including crossed flux volumes.
// - The side faces are shared by the flux volumes of neighboring faces, so they
//   must be triangulated identically by each of them.  The diagonal joins the
//   traced lower node of the mesh edge to its upper node, which depends only
//   on the mesh edge.  This is what makes the fluxes tile without overlaps.
// - Simplices are cut using an explicit stack, for GPU use, and the normals
//   are those of pelelmex::vof::fit_plane rather than ELVIRA, as the paper
//   allows.
//-----------------------------------------------------------------------------

struct Vec3
{
  amrex::Real x[3];
};

struct Tet
{
  Vec3 v[4];
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Vec3
lerp(const Vec3& a, const Vec3& b, const amrex::Real t) noexcept
{
  return Vec3{
    {a.x[0] + t * (b.x[0] - a.x[0]), a.x[1] + t * (b.x[1] - a.x[1]),
     a.x[2] + t * (b.x[2] - a.x[2])}};
}

/**
 * \brief Signed volume of the tetrahedron with apex o over the triangle
 *        (a,b,c): positive when the triangle's right-handed normal points away
 *        from o.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
signed_volume(
  const Vec3& o, const Vec3& a, const Vec3& b, const Vec3& c) noexcept
{
  const amrex::Real ax = a.x[0] - o.x[0], ay = a.x[1] - o.x[1],
                    az = a.x[2] - o.x[2];
  const amrex::Real bx = b.x[0] - o.x[0], by = b.x[1] - o.x[1],
                    bz = b.x[2] - o.x[2];
  const amrex::Real cx = c.x[0] - o.x[0], cy = c.x[1] - o.x[1],
                    cz = c.x[2] - o.x[2];
  return (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) +
          az * (bx * cy - by * cx)) /
         6.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
abs_volume(const Tet& t) noexcept
{
  return std::abs(signed_volume(t.v[0], t.v[1], t.v[2], t.v[3]));
}

/** \brief Tetrahedra filling the prism with triangles p, q and edges p-q */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
prism_to_tets(
  const Vec3& p0,
  const Vec3& p1,
  const Vec3& p2,
  const Vec3& q0,
  const Vec3& q1,
  const Vec3& q2,
  Tet* out) noexcept
{
  out[0] = Tet{{p0, p1, p2, q2}};
  out[1] = Tet{{p0, p1, q2, q1}};
  out[2] = Tet{{p0, q0, q1, q2}};
}

/**
 * \brief Split a tetrahedron by the plane where the vertex values d vanish.
 *
 * Pieces with d <= 0 go to neg and those with d > 0 to pos.
 *
 * \param n_neg,n_pos number of pieces written, at most 3 each
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
split_tet(
  const Tet& t,
  const amrex::Real* d,
  Tet* neg,
  int& n_neg,
  Tet* pos,
  int& n_pos) noexcept
{
  int in[4];
  int out[4];
  int ni = 0;
  int no = 0;
  for (int v = 0; v < 4; ++v) {
    if (d[v] <= 0.0_rt) {
      in[ni++] = v;
    } else {
      out[no++] = v;
    }
  }
  n_neg = 0;
  n_pos = 0;
  if (no == 0) {
    neg[0] = t;
    n_neg = 1;
    return;
  }
  if (ni == 0) {
    pos[0] = t;
    n_pos = 1;
    return;
  }

  // Point where the edge joining an inside and an outside vertex crosses
  auto cross = [&](const int a, const int b) {
    return lerp(t.v[a], t.v[b], d[a] / (d[a] - d[b]));
  };
  if (ni == 1) {
    const int a = in[0];
    const Vec3 p0 = cross(a, out[0]);
    const Vec3 p1 = cross(a, out[1]);
    const Vec3 p2 = cross(a, out[2]);
    neg[0] = Tet{{t.v[a], p0, p1, p2}};
    n_neg = 1;
    prism_to_tets(t.v[out[0]], t.v[out[1]], t.v[out[2]], p0, p1, p2, pos);
    n_pos = 3;
  } else if (no == 1) {
    const int a = out[0];
    const Vec3 p0 = cross(in[0], a);
    const Vec3 p1 = cross(in[1], a);
    const Vec3 p2 = cross(in[2], a);
    pos[0] = Tet{{t.v[a], p0, p1, p2}};
    n_pos = 1;
    prism_to_tets(t.v[in[0]], t.v[in[1]], t.v[in[2]], p0, p1, p2, neg);
    n_neg = 3;
  } else {
    const int a = in[0];
    const int b = in[1];
    const int c = out[0];
    const int e = out[1];
    const Vec3 ac = cross(a, c);
    const Vec3 ae = cross(a, e);
    const Vec3 bc = cross(b, c);
    const Vec3 be = cross(b, e);
    prism_to_tets(t.v[a], ac, ae, t.v[b], bc, be, neg);
    n_neg = 3;
    prism_to_tets(t.v[c], ac, bc, t.v[e], ae, be, pos);
    n_pos = 3;
  }
}

/**
 * \brief Volume of the part of a tetrahedron on the liquid side of the plane
 *        m.(x - x_lo) <= alpha of the cell with low corner x_lo.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
tet_liquid_volume(
  const Tet& t,
  const int i,
  const int j,
  const int k,
  const amrex::Real* plane) noexcept
{
  amrex::Real d[4];
  for (int v = 0; v < 4; ++v) {
    d[v] = plane[0] * (t.v[v].x[0] - i) + plane[1] * (t.v[v].x[1] - j) +
           plane[2] * (t.v[v].x[2] - k) - plane[3];
  }
  Tet neg[3];
  Tet pos[3];
  int n_neg = 0;
  int n_pos = 0;
  split_tet(t, d, neg, n_neg, pos, n_pos);
  amrex::Real vol = 0.0_rt;
  for (int n = 0; n < n_neg; ++n) {
    vol += abs_volume(neg[n]);
  }
  return vol;
}

// Tetrahedra held while cutting by the mesh planes.  Each cut replaces one
// tetrahedron by at most 6 and a depth-first path crosses few planes for a
// Courant number below 1: 64 is well above what can be reached.
static constexpr int cut_stack_size = 64;

/**
 * \brief Cut a tetrahedron along the mesh planes into one piece per cell and
 *        call visit(piece, volume, i, j, k) on each.
 *
 * \param bx cells where the data are valid; pieces beyond it are charged to
 *           the nearest cell inside it
 */
template <typename Visit>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
for_each_cell_piece(
  const Tet& tet, const amrex::Box& bx, Visit&& visit) noexcept
{
  constexpr amrex::Real eps = 1.0e-12_rt;
  Tet stack[cut_stack_size];
  int top = 0;
  stack[top++] = tet;
  const auto lo = amrex::lbound(bx);
  const auto hi = amrex::ubound(bx);

  while (top > 0) {
    const Tet t = stack[--top];

    // The first mesh plane crossing this tetrahedron, if any.  The z planes
    // play no part in 2D, whose mesh is one cell thick.
    int cut_dir = -1;
    amrex::Real cut_at = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM && cut_dir < 0; ++d) {
      amrex::Real tmin = t.v[0].x[d];
      amrex::Real tmax = t.v[0].x[d];
      for (int v = 1; v < 4; ++v) {
        tmin = amrex::min(tmin, t.v[v].x[d]);
        tmax = amrex::max(tmax, t.v[v].x[d]);
      }
      const amrex::Real p = std::floor(tmin + eps) + 1.0_rt;
      if (tmax > p + eps) {
        cut_dir = d;
        cut_at = p;
      }
    }

    if (cut_dir >= 0 && top + 6 <= cut_stack_size) {
      const amrex::Real d[4] = {
        t.v[0].x[cut_dir] - cut_at, t.v[1].x[cut_dir] - cut_at,
        t.v[2].x[cut_dir] - cut_at, t.v[3].x[cut_dir] - cut_at};
      Tet neg[3];
      Tet pos[3];
      int n_neg = 0;
      int n_pos = 0;
      split_tet(t, d, neg, n_neg, pos, n_pos);
      for (int n = 0; n < n_neg; ++n) {
        stack[top++] = neg[n];
      }
      for (int n = 0; n < n_pos; ++n) {
        stack[top++] = pos[n];
      }
      continue;
    }

    // Within a single cell, found from the centroid
    const amrex::Real vol = abs_volume(t);
    if (vol <= 0.0_rt) {
      continue;
    }
    int c[3] = {0, 0, 0};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      c[d] = static_cast<int>(std::floor(
        0.25_rt * (t.v[0].x[d] + t.v[1].x[d] + t.v[2].x[d] + t.v[3].x[d])));
    }
    const int ci = amrex::Clamp(c[0], lo.x, hi.x);
    const int cj = amrex::Clamp(c[1], lo.y, hi.y);
    const int ck = amrex::Clamp(c[2], lo.z, hi.z);
    visit(t, vol, ci, cj, ck);
  }
}

/** \brief Liquid volume of a piece of tetrahedron lying within cell (i,j,k). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
piece_liquid(
  const Tet& t,
  const amrex::Real vol,
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Array4<amrex::Real const> const& plane) noexcept
{
  const amrex::Real f = F(i, j, k);
  if (f <= vof_tol) {
    return 0.0_rt;
  }
  if (f >= 1.0_rt - vof_tol) {
    return vol;
  }
  const amrex::Real pl[4] = {
    plane(i, j, k, 0), plane(i, j, k, 1), plane(i, j, k, 2), plane(i, j, k, 3)};
  return tet_liquid_volume(t, i, j, k, pl);
}

/**
 * \brief Apex o of the two tetrahedra (o,a,c,e) and (o,a,b,c) added on a
 *        traced face (a,b,c,e) to give them a total signed volume vcor.  o lies
 *        over the barycenter of the face; the volume is linear in o along dir.
 *
 * \return false if the face has no extent normal to dir (no correction)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
correction_apex(
  const Vec3& a,
  const Vec3& b,
  const Vec3& c,
  const Vec3& e,
  const int dir,
  const amrex::Real vcor,
  Vec3& o) noexcept
{
  const Vec3* tri[2][3] = {{&a, &c, &e}, {&a, &b, &c}}; // +dir normals
  for (int d = 0; d < 3; ++d) {
    o.x[d] = 0.25_rt * (a.x[d] + b.x[d] + c.x[d] + e.x[d]);
  }
  amrex::Real s0 = 0.0_rt; // 6 V at o_dir = 0
  amrex::Real sd = 0.0_rt; // d(6 V)/d(o_dir)
  for (auto& n : tri) {
    const Vec3 &p = *n[0], &q = *n[1], &r = *n[2];
    const amrex::Real u[3] = {
      q.x[0] - p.x[0], q.x[1] - p.x[1], q.x[2] - p.x[2]};
    const amrex::Real w[3] = {
      r.x[0] - p.x[0], r.x[1] - p.x[1], r.x[2] - p.x[2]};
    const amrex::Real nrm[3] = {
      u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
      u[0] * w[1] - u[1] * w[0]};
    // V = (p - o).N / 6
    for (int d = 0; d < 3; ++d) {
      const amrex::Real od = (d == dir) ? 0.0_rt : o.x[d];
      s0 += (p.x[d] - od) * nrm[d];
    }
    sd -= nrm[dir];
  }
  if (std::abs(sd) <= 1.0e-10_rt) {
    return false;
  }
  o.x[dir] = (6.0_rt * vcor - s0) / sd;
  return true;
}

// Tetrahedra of a flux volume: the fan and the two of the correction
static constexpr int max_flux_tets = 10;

/**
 * \brief Signed tetrahedra making up the flux volume of the low face of cell
 *        (i,j,k) in direction dir.
 *
 * \param back    traced position of each mesh node at the start of the step,
 *                in cell units (3 components)
 * \param courant the face velocity times dt/dx, which the flux volume is
 *                corrected to match
 * \return the number of tetrahedra in tets
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
flux_volume_tets(
  const int i,
  const int j,
  const int k,
  const int dir,
  const amrex::Real courant,
  amrex::Array4<amrex::Real const> const& back,
  Tet* tets) noexcept
{
  // Right-handed tangential directions.  The face spans 0..1 along each, and
  // in 2D the node arrays have no extent in z: the k+1 nodes are the k nodes
  // shifted by one cell.
  const int t1 = (dir + 1) % 3;
  const int t2 = (dir + 2) % 3;

  int off1[3] = {0, 0, 0};
  int off2[3] = {0, 0, 0};
  off1[t1] = 1;
  off2[t2] = 1;

  auto node = [&](const int s1, const int s2) {
    return Vec3{
      {static_cast<amrex::Real>(i + s1 * off1[0] + s2 * off2[0]),
       static_cast<amrex::Real>(j + s1 * off1[1] + s2 * off2[1]),
       static_cast<amrex::Real>(k + s1 * off1[2] + s2 * off2[2])}};
  };
  auto traced = [&](const int s1, const int s2) {
#if AMREX_SPACEDIM == 2
    // Nodes of a 2D mesh: z stays as it was, and only the in-plane offset
    // selects the node
    const int ni = i + s1 * off1[0] + s2 * off2[0];
    const int nj = j + s1 * off1[1] + s2 * off2[1];
    const amrex::Real z =
      static_cast<amrex::Real>(k + s1 * off1[2] + s2 * off2[2]);
    return Vec3{{back(ni, nj, 0, 0), back(ni, nj, 0, 1), z}};
#else
    const int ni = i + s1 * off1[0] + s2 * off2[0];
    const int nj = j + s1 * off1[1] + s2 * off2[1];
    const int nk = k + s1 * off1[2] + s2 * off2[2];
    return Vec3{
      {back(ni, nj, nk, 0), back(ni, nj, nk, 1), back(ni, nj, nk, 2)}};
#endif
  };

  // Face corners counterclockwise about +dir, and their traced positions
  const Vec3 a = node(0, 0);
  const Vec3 b = node(1, 0);
  const Vec3 c = node(1, 1);
  const Vec3 e = node(0, 1);
  const Vec3 ap = traced(0, 0);
  const Vec3 bp = traced(1, 0);
  const Vec3 cp = traced(1, 1);
  const Vec3 ep = traced(0, 1);

  // Fan of the boundary from a.  With outward triangles the signed volumes
  // add up to the flux volume, positive for a flow along +dir.  Triangles
  // touching a vanish: the face itself and the side faces through a.  The
  // diagonals of the side faces depend only on the mesh edge (see above).
  constexpr int ntri = 8;
  const Vec3* tri[ntri][3] = {
    {&ap, &bp, &b},  // side a-b, diagonal a' - b
    {&e, &ep, &ap},  // side e-a, diagonal e - a'
    {&b, &bp, &c},   // side b-c, diagonal b' - c
    {&bp, &cp, &c},  //
    {&c, &cp, &ep},  // side c-e, diagonal e' - c
    {&c, &ep, &e},   //
    {&ap, &ep, &cp}, // traced face, diagonal a' - c'
    {&ap, &cp, &bp}};

  amrex::Real vol = 0.0_rt;
  for (int n = 0; n < ntri; ++n) {
    vol += signed_volume(a, *tri[n][0], *tri[n][1], *tri[n][2]);
  }

  // Extra volume so that the flux volume matches the face velocity exactly:
  // two tetrahedra from the traced face to a point o above its barycenter, in
  // the direction of dir (Eq. 27-28).  Their volume is linear in o along dir.
  const Vec3* ttri[2][3] = {{&ap, &cp, &ep}, {&ap, &bp, &cp}}; // +dir normals
  Vec3 o;
  const bool has_cor = correction_apex(ap, bp, cp, ep, dir, courant - vol, o);

  int ntet = 0;
  for (int n = 0; n < ntri; ++n) {
    tets[ntet++] = Tet{{a, *tri[n][0], *tri[n][1], *tri[n][2]}};
  }
  if (has_cor) {
    for (auto& n : ttri) {
      tets[ntet++] = Tet{{o, *n[0], *n[1], *n[2]}};
    }
  }
  return ntet;
}

/**
 * \brief Whether the cells [lo, hi], limited to bx, are all gas or all liquid.
 *
 * \return true if they are pure, with a_liquid set to 1 or 0 accordingly
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
pure_cells(
  const int* lo,
  const int* hi,
  amrex::Array4<amrex::Real const> const& F,
  const amrex::Box& bx,
  bool& a_liquid) noexcept
{
  const auto blo = amrex::lbound(bx);
  const auto bhi = amrex::ubound(bx);
  bool all_gas = true;
  bool all_liq = true;
#if AMREX_SPACEDIM == 2
  const int klo = 0;
  const int khi = 0;
#else
  const int klo = amrex::Clamp(lo[2], blo.z, bhi.z);
  const int khi = amrex::Clamp(hi[2], blo.z, bhi.z);
#endif
  const int ilo = amrex::Clamp(lo[0], blo.x, bhi.x);
  const int ihi = amrex::Clamp(hi[0], blo.x, bhi.x);
  const int jlo = amrex::Clamp(lo[1], blo.y, bhi.y);
  const int jhi = amrex::Clamp(hi[1], blo.y, bhi.y);
  for (int kk = klo; kk <= khi; ++kk) {
    for (int jj = jlo; jj <= jhi; ++jj) {
      for (int ii = ilo; ii <= ihi; ++ii) {
        const amrex::Real f = F(ii, jj, kk);
        all_gas = all_gas && (f <= vof_tol);
        all_liq = all_liq && (f >= 1.0_rt - vof_tol);
      }
      if (!all_gas && !all_liq) {
        return false;
      }
    }
  }
  a_liquid = all_liq;
  return true;
}

/**
 * \brief Whether a flux volume lies in pure cells only, from its bounding box:
 *        it then needs no cutting.
 *
 * \param liquid its signed liquid volume, if so
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
pure_flux_volume(
  const Tet* tets,
  const int ntet,
  const amrex::Real courant,
  amrex::Array4<amrex::Real const> const& F,
  const amrex::Box& bx,
  amrex::Real& liquid) noexcept
{
  amrex::Real pmin[3] = {
    tets[0].v[0].x[0], tets[0].v[0].x[1], tets[0].v[0].x[2]};
  amrex::Real pmax[3] = {pmin[0], pmin[1], pmin[2]};
  for (int n = 0; n < ntet; ++n) {
    for (const auto& vert : tets[n].v) {
      for (int d = 0; d < 3; ++d) {
        pmin[d] = amrex::min(pmin[d], vert.x[d]);
        pmax[d] = amrex::max(pmax[d], vert.x[d]);
      }
    }
  }
  int lo[3];
  int hi[3];
  for (int d = 0; d < 3; ++d) {
    lo[d] = static_cast<int>(std::floor(pmin[d]));
    hi[d] = static_cast<int>(std::floor(pmax[d]));
  }
  bool is_liquid = false;
  if (!pure_cells(lo, hi, F, bx, is_liquid)) {
    return false;
  }
  liquid = is_liquid ? courant : 0.0_rt;
  return true;
}

/**
 * \brief Mass and momentum in the flux volume of the low face of cell (i,j,k)
 *        in direction dir, per cell volume.  Each piece of the flux volume
 *        carries the density of its phase and the velocity of the cell it lies
 *        in, so that a uniform velocity stays uniform and momentum moves with
 *        the mass it belongs to.
 *
 * \param vel  cell-centered velocity at the start of the step
 * \param bx   cells where F, plane and vel are valid
 * \param mass signed mass crossing the face
 * \param mom  signed momentum crossing the face
 * \param liquid signed liquid volume crossing the face
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
face_mass_momentum_flux(
  const int i,
  const int j,
  const int k,
  const int dir,
  const amrex::Real courant,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Array4<amrex::Real const> const& plane,
  amrex::Array4<amrex::Real const> const& back,
  amrex::Array4<amrex::Real const> const& vel,
  const amrex::Box& bx,
  const amrex::Real rho_l,
  const amrex::Real rho_g,
  amrex::Real& mass,
  amrex::Real* mom,
  amrex::Real& liquid) noexcept
{
  Tet tets[max_flux_tets];
  const int ntet = flux_volume_tets(i, j, k, dir, courant, back, tets);
  amrex::Real pure_liquid = 0.0_rt;
  const bool pure = pure_flux_volume(tets, ntet, courant, F, bx, pure_liquid);
  liquid = 0.0_rt;
  mass = 0.0_rt;
  for (int n = 0; n < AMREX_SPACEDIM; ++n) {
    mom[n] = 0.0_rt;
  }
  for (int n = 0; n < ntet; ++n) {
    const Tet& tet = tets[n];
    const amrex::Real v = signed_volume(tet.v[0], tet.v[1], tet.v[2], tet.v[3]);
    if (v == 0.0_rt) {
      continue;
    }
    const amrex::Real s = (v > 0.0_rt) ? 1.0_rt : -1.0_rt;
    amrex::Real tet_liquid = 0.0_rt;
    for_each_cell_piece(
      tet, bx,
      [&](
        const Tet& t, const amrex::Real vol, int ci, int cj, int ck) noexcept {
        const amrex::Real l = piece_liquid(t, vol, ci, cj, ck, F, plane);
        tet_liquid += l;
        const amrex::Real m = s * (rho_l * l + rho_g * (vol - l));
        mass += m;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          mom[d] += m * vel(ci, cj, ck, d);
        }
      });
    liquid += s * tet_liquid;
  }
  if (pure) {
    liquid = pure_liquid;
  }
}

/**
 * \brief Velocity at a position, in Courant numbers, by multilinear
 *        interpolation of the node velocities.  Positions beyond the node box
 *        use its nearest nodes.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
interp_node_velocity(
  const amrex::Real* x,
  amrex::Array4<amrex::Real const> const& vel,
  const amrex::Box& nbx,
  amrex::Real* u) noexcept
{
  const auto lo = amrex::lbound(nbx);
  const auto hi = amrex::ubound(nbx);
  const int nlo[3] = {lo.x, lo.y, lo.z};
  const int nhi[3] = {hi.x, hi.y, hi.z};
  int base[3] = {0, 0, 0};
  amrex::Real w[3] = {0.0_rt, 0.0_rt, 0.0_rt};
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const int b = static_cast<int>(std::floor(x[d]));
    base[d] = amrex::Clamp(b, nlo[d], amrex::max(nlo[d], nhi[d] - 1));
    w[d] = amrex::Clamp(x[d] - base[d], 0.0_rt, 1.0_rt);
  }
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    u[d] = 0.0_rt;
  }
#if AMREX_SPACEDIM == 2
  for (int s1 = 0; s1 <= 1; ++s1) {
    for (int s0 = 0; s0 <= 1; ++s0) {
      const amrex::Real wt =
        (s0 != 0 ? w[0] : 1.0_rt - w[0]) * (s1 != 0 ? w[1] : 1.0_rt - w[1]);
      for (int d = 0; d < 2; ++d) {
        u[d] += wt * vel(base[0] + s0, base[1] + s1, 0, d);
      }
    }
  }
#else
  for (int s2 = 0; s2 <= 1; ++s2) {
    for (int s1 = 0; s1 <= 1; ++s1) {
      for (int s0 = 0; s0 <= 1; ++s0) {
        const amrex::Real wt = (s0 != 0 ? w[0] : 1.0_rt - w[0]) *
                               (s1 != 0 ? w[1] : 1.0_rt - w[1]) *
                               (s2 != 0 ? w[2] : 1.0_rt - w[2]);
        for (int d = 0; d < 3; ++d) {
          u[d] += wt * vel(base[0] + s0, base[1] + s1, base[2] + s2, d);
        }
      }
    }
  }
#endif
}

} // namespace pelelmex::vof

#endif
