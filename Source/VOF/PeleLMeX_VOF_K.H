#ifndef PELELM_VOF_K_H
#define PELELM_VOF_K_H

// Geometric PLIC volume-of-fluid kernels.
//
// The interface in a cell is the plane m.x = alpha in unit-cube cell
// coordinates, with the liquid on the side m.x <= alpha (m points out of
// the liquid).  Normals come from the Mixed Youngs-Centered scheme
// (Aulisa et al., JCP 225, 2007) and the volume <-> plane-constant
// relations are the analytic ones of Scardovelli & Zaleski (JCP 164, 2000).
// Advection is the directionally split Eulerian-implicit scheme of
// Weymouth & Yue (JCP 229, 2010), which conserves volume to round-off for a
// discretely divergence-free face velocity.
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

// Volume fraction below which a cell is considered pure gas / above
// (1 - tol) pure liquid
static constexpr amrex::Real vof_tol = 1.0e-12_rt;

// Half-length of the height-function columns (7 cells); the volume fraction
// carries one more ghost cell so that planes can be fitted in the first
// ghost layer
static constexpr int hf_half = 3;
static constexpr int nghost = hf_half + 1;

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
  const amrex::Real pr = amrex::max(6.0_rt * m1 * m2 * m3, tiny);
  const amrex::Real V1 = m1 * m1 * m1 / pr;
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
    const amrex::Real teta =
      std::acos(amrex::max(-1.0_rt, amrex::min(1.0_rt, q / (p * p12 + tiny)))) /
      3.0_rt;
    const amrex::Real cs = std::cos(teta);
    alpha = p12 * (std::sqrt(3.0_rt * (1.0_rt - cs * cs)) - cs) + m12;
  } else if (m12 < m3) {
    alpha = m3 * ch + 0.5_rt * mm;
  } else {
    const amrex::Real p = m1 * (m2 + m3) + m2 * m3 - 0.25_rt;
    const amrex::Real q = 1.5_rt * m1 * m2 * m3 * (0.5_rt - ch);
    const amrex::Real p12 = std::sqrt(amrex::max(0.0_rt, p));
    const amrex::Real teta =
      std::acos(amrex::max(-1.0_rt, amrex::min(1.0_rt, q / (p * p12 + tiny)))) /
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
  t = amrex::max(
    std::abs(m[3][0]), amrex::max(std::abs(m[3][1]), std::abs(m[3][2])));
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
 * \brief Height-function curvature with height direction d (Cummins et al.,
 *        C&S 34, 2005): the liquid heights H of the 7-cell columns through
 *        (i,j,k) and its 3 (2D) or 3x3 (3D) tangential neighbors,
 *        differenced centrally,
 *          2D: kappa = -H'' / (1 + H'^2)^(3/2)
 *          3D: kappa = -(Hxx (1 + Hy^2) + Hyy (1 + Hx^2) - 2 Hxy Hx Hy)
 *                      / (1 + Hx^2 + Hy^2)^(3/2)
 *        divided by dx.  kappa is the sum of the principal curvatures,
 *        positive for a drop, so that p_l - p_g = sigma kappa.  The sign is
 *        independent of the column orientation: with the liquid on the high
 *        side the interface height is 7 - H.  Second-order accurate for a
 *        smooth interface whose slope relative to d is at most 1.  Returns
 *        false when a column does not span from pure liquid to pure gas in
 *        the orientation of the center column.  Reads hf_half cells along d
 *        and one cell in the tangential directions.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
height_curvature(
  const int i,
  const int j,
  const int k,
  const int d,
  amrex::Array4<amrex::Real const> const& F,
  const amrex::Real dxinv,
  amrex::Real& kappa)
{
  amrex::ignore_unused(k);
  constexpr amrex::Real pure_tol = 1.0e-6_rt;
  const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
  const amrex::IntVect ed = amrex::IntVect::TheDimensionVector(d);
  const amrex::Real f_lo = F(iv - hf_half * ed);
  const amrex::Real f_hi = F(iv + hf_half * ed);
  bool liquid_lo = false;
  if (f_lo >= 1.0_rt - pure_tol && f_hi <= pure_tol) {
    liquid_lo = true;
  } else if (!(f_lo <= pure_tol && f_hi >= 1.0_rt - pure_tol)) {
    return false;
  }
  const auto column = [&](const amrex::IntVect& c, amrex::Real& H) {
    const amrex::Real a = F(c - hf_half * ed);
    const amrex::Real b = F(c + hf_half * ed);
    const amrex::Real a_pure = liquid_lo ? 1.0_rt - a : a;
    const amrex::Real b_pure = liquid_lo ? b : 1.0_rt - b;
    if (a_pure > pure_tol || b_pure > pure_tol) {
      return false;
    }
    H = 0.0_rt;
    for (int s = -hf_half; s <= hf_half; ++s) {
      H += F(c + s * ed);
    }
    return true;
  };
#if AMREX_SPACEDIM == 2
  const amrex::IntVect et = amrex::IntVect::TheDimensionVector(1 - d);
  amrex::Real Hm = 0.0_rt;
  amrex::Real H0 = 0.0_rt;
  amrex::Real Hp = 0.0_rt;
  if (!column(iv - et, Hm) || !column(iv, H0) || !column(iv + et, Hp)) {
    return false;
  }
  const amrex::Real Hx = 0.5_rt * (Hp - Hm);
  const amrex::Real Hxx = Hp - 2.0_rt * H0 + Hm;
  const amrex::Real g = 1.0_rt + Hx * Hx;
  kappa = -Hxx / (g * std::sqrt(g)) * dxinv;
#else
  const amrex::IntVect e1 = amrex::IntVect::TheDimensionVector((d + 1) % 3);
  const amrex::IntVect e2 = amrex::IntVect::TheDimensionVector((d + 2) % 3);
  amrex::Real H[3][3];
  for (int a = -1; a <= 1; ++a) {
    for (int b = -1; b <= 1; ++b) {
      if (!column(iv + a * e1 + b * e2, H[a + 1][b + 1])) {
        return false;
      }
    }
  }
  const amrex::Real Hx = 0.5_rt * (H[2][1] - H[0][1]);
  const amrex::Real Hy = 0.5_rt * (H[1][2] - H[1][0]);
  const amrex::Real Hxx = H[2][1] - 2.0_rt * H[1][1] + H[0][1];
  const amrex::Real Hyy = H[1][2] - 2.0_rt * H[1][1] + H[1][0];
  const amrex::Real Hxy = 0.25_rt * (H[2][2] - H[2][0] - H[0][2] + H[0][0]);
  const amrex::Real g = 1.0_rt + Hx * Hx + Hy * Hy;
  kappa = -(Hxx * (1.0_rt + Hy * Hy) + Hyy * (1.0_rt + Hx * Hx) -
            2.0_rt * Hxy * Hx * Hy) /
          (g * std::sqrt(g)) * dxinv;
#endif
  return true;
}

/**
 * \brief Height-function curvature of cell (i,j,k), trying the height
 *        directions in decreasing order of the Mixed Youngs-Centered normal
 *        components.  Returns false when no direction has valid heights
 *        (thin or under-resolved structures).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
cell_height_curvature(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  const amrex::Real dxinv,
  amrex::Real& kappa)
{
  amrex::Real m[3];
  myc_normal(i, j, k, F, m[0], m[1], m[2]);
  int order[AMREX_SPACEDIM];
  for (int a = 0; a < AMREX_SPACEDIM; ++a) {
    order[a] = a;
  }
  for (int a = 1; a < AMREX_SPACEDIM; ++a) {
    for (int b = a; b > 0 && std::abs(m[order[b]]) > std::abs(m[order[b - 1]]);
         --b) {
      const int tmp = order[b];
      order[b] = order[b - 1];
      order[b - 1] = tmp;
    }
  }
  for (int a = 0; a < AMREX_SPACEDIM; ++a) {
    if (height_curvature(i, j, k, order[a], F, dxinv, kappa)) {
      return true;
    }
  }
  return false;
}

/**
 * \brief Curvature kappa = -div(n), n = grad F / |grad F|, with the normals
 *        at the cell nodes from the 2^dim surrounding cells.  First-order
 *        and noisy: only a last resort where no height function is defined.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
normal_divergence_curvature(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  const amrex::Real dxinv)
{
  amrex::ignore_unused(k);
  constexpr amrex::Real tiny = 1.0e-12_rt;
  // Unit normal at node (i+a, j+b, k+c), a,b,c in {0,1}
  const auto node_normal =
    [&](const int a, const int b, const int c, amrex::Real n[3]) {
      amrex::ignore_unused(c);
      n[0] = n[1] = n[2] = 0.0_rt;
#if AMREX_SPACEDIM == 2
      const int ii = i + a;
      const int jj = j + b;
      n[0] = 0.5_rt * (F(ii, jj, k) + F(ii, jj - 1, k) - F(ii - 1, jj, k) -
                       F(ii - 1, jj - 1, k));
      n[1] = 0.5_rt * (F(ii, jj, k) + F(ii - 1, jj, k) - F(ii, jj - 1, k) -
                       F(ii - 1, jj - 1, k));
#else
      const int ii = i + a;
      const int jj = j + b;
      const int kk = k + c;
      for (int q = 0; q < 2; ++q) {
        for (int r = 0; r < 2; ++r) {
          n[0] += F(ii, jj - q, kk - r) - F(ii - 1, jj - q, kk - r);
          n[1] += F(ii - q, jj, kk - r) - F(ii - q, jj - 1, kk - r);
          n[2] += F(ii - q, jj - r, kk) - F(ii - q, jj - r, kk - 1);
        }
      }
      for (int q = 0; q < 3; ++q) {
        n[q] *= 0.25_rt;
      }
#endif
      const amrex::Real nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      for (int q = 0; q < 3; ++q) {
        n[q] = (nn > tiny) ? n[q] / nn : 0.0_rt;
      }
    };
  amrex::Real div = 0.0_rt;
  amrex::Real n[3];
#if AMREX_SPACEDIM == 2
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      node_normal(a, b, 0, n);
      div += 0.5_rt * ((2 * a - 1) * n[0] + (2 * b - 1) * n[1]);
    }
  }
#else
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        node_normal(a, b, c, n);
        div += 0.25_rt *
               ((2 * a - 1) * n[0] + (2 * b - 1) * n[1] + (2 * c - 1) * n[2]);
      }
    }
  }
#endif
  return -div * dxinv;
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
 * \brief Interface point of the 7-cell column through cell (i,j,k) along d:
 *        when the column spans pure liquid to pure gas and its liquid height
 *        puts the interface within cell (i,j,k), returns true with z, the
 *        offset of the interface from the cell center along d in cell units,
 *        and s = +1 (liquid below, the normal out of the liquid along +d) or
 *        -1.  Second-order accurate for a smooth interface whose slope
 *        relative to d is at most 1.  Reads hf_half cells along d.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
height_point(
  const int i,
  const int j,
  const int k,
  const int d,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Real& z,
  amrex::Real& s)
{
  amrex::ignore_unused(k);
  constexpr amrex::Real pure_tol = 1.0e-6_rt;
  const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
  const amrex::IntVect ed = amrex::IntVect::TheDimensionVector(d);
  const amrex::Real f_lo = F(iv - hf_half * ed);
  const amrex::Real f_hi = F(iv + hf_half * ed);
  if (f_lo >= 1.0_rt - pure_tol && f_hi <= pure_tol) {
    s = 1.0_rt;
  } else if (f_lo <= pure_tol && f_hi >= 1.0_rt - pure_tol) {
    s = -1.0_rt;
  } else {
    return false;
  }
  amrex::Real H = 0.0_rt;
  for (int q = -hf_half; q <= hf_half; ++q) {
    H += F(iv + q * ed);
  }
  z = s * (H - (hf_half + 0.5_rt));
  return std::abs(z) <= 0.5_rt;
}

/**
 * \brief Curvature of a least-squares parabola (2D) or paraboloid (3D)
 *        through interface points around cell (i,j,k) (Popinet, JCP 228,
 *        2009), for the mixed cells without a height-function curvature.  The
 *        points are those of the single height columns (height_point) of the
 *        3^dim stencil in every direction, second-order accurate and robust to
 *        small interface perturbations; the projections of the cell centers
 *        on the PLIC planes of the mixed cells of the stencil are added only
 *        when there are fewer than fit_min_points height points.  In the
 *        frame of the PLIC normal n of (i,j,k), pointing out of the liquid,
 *        z = a0 + a1 x + a2 x^2 (2D) or
 *        z = a0 + a1 x + a2 y + a3 x^2 + a4 y^2 + a5 x y (3D), and kappa is
 *        minus the curvature of that graph at the origin.  Points of the
 *        other side of a thin structure (facing away from n) are left out.
 *        Returns false when there are too few points or they do not
 *        determine the fit.  Reads up to hf_half + 1 cells away: only for
 *        cells whose 3^dim stencil lies within the first ghost layer.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
parabolic_curvature(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real const> const& F,
  const amrex::Real dxinv,
  amrex::Real& kappa)
{
  amrex::ignore_unused(k);
  constexpr int np = (AMREX_SPACEDIM == 2) ? 3 : 6;
  // At least twice the number of unknowns from the heights alone
  constexpr int fit_min_points = 2 * np;
  const auto mixed = [](const amrex::Real f) {
    return f > vof_tol && f < 1.0_rt - vof_tol;
  };
  if (!mixed(F(i, j, k))) {
    return false;
  }
  // PLIC interface point of a mixed cell, relative to the center of (i,j,k),
  // in cell units, and the unit normal of its plane
  const auto plic_point = [&](
                            const int a, const int b, const int c,
                            amrex::Real p[3], amrex::Real nrm[3]) {
    amrex::Real m[3], alpha;
    fit_plane(i + a, j + b, k + c, F, m[0], m[1], m[2], alpha);
    const amrex::Real l2 = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    const amrex::Real dist = (0.5_rt * (m[0] + m[1] + m[2]) - alpha) / l2;
    const int off[3] = {a, b, c};
    for (int d = 0; d < 3; ++d) {
      nrm[d] = m[d] / l2;
      p[d] = off[d] - dist * nrm[d];
    }
  };
  amrex::Real n[3], p0[3];
  plic_point(0, 0, 0, p0, n);
  // Tangent basis t1, t2 orthogonal to n
  amrex::Real t1[3] = {0.0_rt, 0.0_rt, 0.0_rt};
  amrex::Real t2[3] = {0.0_rt, 0.0_rt, 0.0_rt};
#if AMREX_SPACEDIM == 2
  t1[0] = -n[1];
  t1[1] = n[0];
#else
  {
    // Cross n with the axis least aligned with it
    int a = 0;
    if (std::abs(n[1]) < std::abs(n[a])) {
      a = 1;
    }
    if (std::abs(n[2]) < std::abs(n[a])) {
      a = 2;
    }
    amrex::Real e[3] = {0.0_rt, 0.0_rt, 0.0_rt};
    e[a] = 1.0_rt;
    t1[0] = n[1] * e[2] - n[2] * e[1];
    t1[1] = n[2] * e[0] - n[0] * e[2];
    t1[2] = n[0] * e[1] - n[1] * e[0];
    const amrex::Real l =
      std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    for (int d = 0; d < 3; ++d) {
      t1[d] /= l;
    }
    t2[0] = n[1] * t1[2] - n[2] * t1[1];
    t2[1] = n[2] * t1[0] - n[0] * t1[2];
    t2[2] = n[0] * t1[1] - n[1] * t1[0];
  }
#endif
  // Normal equations A^T A c = A^T z
  amrex::Real M[np][np + 1];
  for (int r = 0; r < np; ++r) {
    for (int c = 0; c <= np; ++c) {
      M[r][c] = 0.0_rt;
    }
  }
  int npts = 0;
  const auto add_point = [&](const amrex::Real p[3]) {
    const amrex::Real x = p[0] * t1[0] + p[1] * t1[1] + p[2] * t1[2];
    const amrex::Real z = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
#if AMREX_SPACEDIM == 2
    const amrex::Real phi[np] = {1.0_rt, x, x * x};
#else
    const amrex::Real y = p[0] * t2[0] + p[1] * t2[1] + p[2] * t2[2];
    const amrex::Real phi[np] = {1.0_rt, x, y, x * x, y * y, x * y};
#endif
    for (int r = 0; r < np; ++r) {
      for (int q = 0; q < np; ++q) {
        M[r][q] += phi[r] * phi[q];
      }
      M[r][np] += phi[r] * z;
    }
    ++npts;
  };
  // Height points
#if AMREX_SPACEDIM == 3
  for (int c = -1; c <= 1; ++c) {
#else
  {
    constexpr int c = 0;
#endif
    for (int b = -1; b <= 1; ++b) {
      for (int a = -1; a <= 1; ++a) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          amrex::Real z, sg;
          if (
            !height_point(i + a, j + b, k + c, d, F, z, sg) ||
            sg * n[d] <= 0.0_rt) {
            continue;
          }
          amrex::Real p[3] = {
            static_cast<amrex::Real>(a), static_cast<amrex::Real>(b),
            static_cast<amrex::Real>(c)};
          p[d] += z;
          add_point(p);
        }
      }
    }
  }
  // PLIC points when the heights are too few
  if (npts < fit_min_points) {
#if AMREX_SPACEDIM == 3
    for (int c = -1; c <= 1; ++c) {
#else
    {
      constexpr int c = 0;
#endif
      for (int b = -1; b <= 1; ++b) {
        for (int a = -1; a <= 1; ++a) {
          if (!mixed(F(i + a, j + b, k + c))) {
            continue;
          }
          amrex::Real p[3], nb[3];
          plic_point(a, b, c, p, nb);
          if (nb[0] * n[0] + nb[1] * n[1] + nb[2] * n[2] <= 0.0_rt) {
            continue;
          }
          add_point(p);
        }
      }
    }
  }
  if (npts < np + 1) {
    return false;
  }
  // Gaussian elimination with partial pivoting; a small pivot relative to
  // the number of points means the points do not determine the fit
  const amrex::Real pivot_min = 1.0e-3_rt * static_cast<amrex::Real>(npts);
  for (int col = 0; col < np; ++col) {
    int piv = col;
    for (int r = col + 1; r < np; ++r) {
      if (std::abs(M[r][col]) > std::abs(M[piv][col])) {
        piv = r;
      }
    }
    if (std::abs(M[piv][col]) < pivot_min) {
      return false;
    }
    if (piv != col) {
      for (int q = 0; q <= np; ++q) {
        const amrex::Real tmp = M[col][q];
        M[col][q] = M[piv][q];
        M[piv][q] = tmp;
      }
    }
    for (int r = col + 1; r < np; ++r) {
      const amrex::Real f = M[r][col] / M[col][col];
      for (int q = col; q <= np; ++q) {
        M[r][q] -= f * M[col][q];
      }
    }
  }
  amrex::Real cf[np];
  for (int r = np - 1; r >= 0; --r) {
    amrex::Real sum = M[r][np];
    for (int q = r + 1; q < np; ++q) {
      sum -= M[r][q] * cf[q];
    }
    cf[r] = sum / M[r][r];
  }
#if AMREX_SPACEDIM == 2
  const amrex::Real g = 1.0_rt + cf[1] * cf[1];
  kappa = -2.0_rt * cf[2] / (g * std::sqrt(g)) * dxinv;
#else
  const amrex::Real g = 1.0_rt + cf[1] * cf[1] + cf[2] * cf[2];
  kappa = -(2.0_rt * cf[3] * (1.0_rt + cf[2] * cf[2]) +
            2.0_rt * cf[4] * (1.0_rt + cf[1] * cf[1]) -
            2.0_rt * cf[5] * cf[1] * cf[2]) /
          (g * std::sqrt(g)) * dxinv;
#endif
  return true;
}

/**
 * \brief Liquid fraction of the volume swept through the low face of cell
 *        (i,j,k) in direction dir during one split sweep, given the face
 *        Courant number a = u_face * dt / dx.  The swept region lies in the
 *        upwind cell.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
swept_fraction(
  const int i,
  const int j,
  const int k,
  const int dir,
  const amrex::Real a,
  amrex::Array4<amrex::Real const> const& F)
{
  amrex::ignore_unused(k);
  if (a == 0.0_rt) {
    return 0.0_rt;
  }
  const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
  const amrex::IntVect up =
    (a > 0.0_rt) ? iv - amrex::IntVect::TheDimensionVector(dir) : iv;
#if AMREX_SPACEDIM == 2
  const int iu = up[0], ju = up[1], ku = 0;
#else
  const int iu = up[0], ju = up[1], ku = up[2];
#endif
  const amrex::Real Fu = F(iu, ju, ku);
  if (Fu <= vof_tol) {
    return 0.0_rt;
  }
  if (Fu >= 1.0_rt - vof_tol) {
    return 1.0_rt;
  }

  amrex::Real mx, my, mz, alpha;
  fit_plane(iu, ju, ku, F, mx, my, mz, alpha);

  // Positive velocity sweeps the high end [1-a, 1] of the upwind cell,
  // negative velocity the low end [0, -a]
  const amrex::Real r0 = (a > 0.0_rt) ? 1.0_rt - a : 0.0_rt;
  const amrex::Real dr0 = std::abs(a);
  if (dir == 0) {
    return cut_volume(mx, my, mz, alpha, r0, dr0);
  }
  if (dir == 1) {
    return cut_volume(my, mz, mx, alpha, r0, dr0);
  }
  return cut_volume(mz, mx, my, alpha, r0, dr0);
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
 * \brief Liquid volume fraction of cell (i,j,k) for a smooth implicit
 *        surface phi(x) > 0 inside the liquid: phi is linearized on each of
 *        the nsub^dim sub-cells (central-difference gradient) and the
 *        fraction of the sub-cell below the plane is exact.  The error is
 *        O(kappa h) relative per cut sub-cell and varies smoothly along the
 *        interface, whereas the sampling noise of cell_fraction, O(1/nsub^2)
 *        per cell, is amplified by 1/dx in the height-function curvature.
 *        Intended for the initialization of curved interfaces with surface
 *        tension.
 */
template <typename Phi>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_cell_fraction(
  const int i,
  const int j,
  const int k,
  amrex::GeometryData const& geomdata,
  Phi const& phi,
  const int nsub = 8)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  const amrex::Real s = 1.0_rt / static_cast<amrex::Real>(nsub);
  amrex::Real h[3] = {AMREX_D_DECL(s * dx[0], s * dx[1], s * dx[2])};
#if AMREX_SPACEDIM == 2
  h[2] = 0.0_rt;
#endif
  amrex::Real sum = 0.0_rt;
#if AMREX_SPACEDIM == 3
  for (int kk = 0; kk < nsub; ++kk) {
    const amrex::Real z = prob_lo[2] + (k + (kk + 0.5_rt) * s) * dx[2];
#else
  amrex::ignore_unused(k);
  {
    const amrex::Real z = 0.0_rt;
#endif
    for (int jj = 0; jj < nsub; ++jj) {
      const amrex::Real y = prob_lo[1] + (j + (jj + 0.5_rt) * s) * dx[1];
      for (int ii = 0; ii < nsub; ++ii) {
        const amrex::Real x = prob_lo[0] + (i + (ii + 0.5_rt) * s) * dx[0];
        const amrex::Real pc = phi(x, y, z);
        // Gradient by central differences over a fraction of the sub-cell
        constexpr amrex::Real eps = 1.0e-3_rt;
        amrex::Real m[3] = {0.0_rt, 0.0_rt, 0.0_rt};
        m[0] = -(phi(x + eps * h[0], y, z) - phi(x - eps * h[0], y, z)) /
               (2.0_rt * eps);
        m[1] = -(phi(x, y + eps * h[1], z) - phi(x, y - eps * h[1], z)) /
               (2.0_rt * eps);
#if AMREX_SPACEDIM == 3
        m[2] = -(phi(x, y, z + eps * h[2]) - phi(x, y, z - eps * h[2])) /
               (2.0_rt * eps);
#endif
        // phi > 0  <=>  m.xi < pc + (m0 + m1 + m2) / 2, xi in the unit
        // sub-cell, with m = -h grad phi
        const amrex::Real norm =
          std::abs(m[0]) + std::abs(m[1]) + std::abs(m[2]);
        if (norm <= std::abs(pc) * 1.0e-12_rt || norm == 0.0_rt) {
          sum += (pc > 0.0_rt) ? 1.0_rt : 0.0_rt;
          continue;
        }
        const amrex::Real alpha = pc + 0.5_rt * (m[0] + m[1] + m[2]);
        sum += amrex::Clamp(
          cut_volume(
            m[0] / norm, m[1] / norm, m[2] / norm, alpha / norm, 0.0_rt,
            1.0_rt),
          0.0_rt, 1.0_rt);
      }
    }
  }
  return sum / static_cast<amrex::Real>(AMREX_D_TERM(nsub, *nsub, *nsub));
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
 *        line, exact for a planar interface.  Used when the interface meets
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
 * \param fbox  box of the array F
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
wall_ghost_fraction(
  const amrex::IntVect& g,
  const int dir,
  const int iwall,
  const int sgn,
  const amrex::Box& fbox,
  amrex::Array4<amrex::Real const> const& F,
  const bool allow_ghost)
{
  constexpr int nsearch = 8;
  constexpr int nfar = 1000;
  const amrex::Box rbox = amrex::grow(fbox, -hf_half);
  amrex::IntVect iv_w = g;
  iv_w[dir] = iwall;

  const auto is_mixed = [&F](const amrex::IntVect& c) {
    const amrex::Real f = F(c);
    return f > vof_tol && f < 1.0_rt - vof_tol;
  };

  bool contact = false;
  int ref_dist = nfar;
  amrex::IntVect ref = iv_w;
  // Fallback reading ghost cells
  int any_dist = nfar;
  amrex::IntVect any_ref = iv_w;
  amrex::IntVect off_lo(-nsearch);
  amrex::IntVect off_hi(nsearch);
  off_lo[dir] = 0;
  off_hi[dir] = 0;
#if AMREX_SPACEDIM == 2
  for (int b = 0; b <= 0; ++b) {
#else
  for (int b = off_lo[2]; b <= off_hi[2]; ++b) {
#endif
    for (int a1 = off_lo[1]; a1 <= off_hi[1]; ++a1) {
      for (int a0 = off_lo[0]; a0 <= off_hi[0]; ++a0) {
        const amrex::IntVect off(AMREX_D_DECL(a0, a1, b));
        const int tdist =
          AMREX_D_TERM(std::abs(a0), +std::abs(a1), +std::abs(b));
        const amrex::IntVect cw = iv_w + off;
        if (fbox.contains(cw) && is_mixed(cw)) {
          contact = true;
        }
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
#if AMREX_SPACEDIM == 2
            myc_normal(c[0], c[1], 0, F, mm[0], mm[1], mm[2]);
#else
            myc_normal(c[0], c[1], c[2], F, mm[0], mm[1], mm[2]);
#endif
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
  if (!contact || ref_dist == nfar) {
    return F(iv_w);
  }

  amrex::Real m[3] = {0.0_rt, 0.0_rt, 0.0_rt};
  amrex::Real alpha = 0.0_rt;
#if AMREX_SPACEDIM == 2
  fit_plane(ref[0], ref[1], 0, F, m[0], m[1], m[2], alpha);
#else
  fit_plane(ref[0], ref[1], ref[2], F, m[0], m[1], m[2], alpha);
#endif
  // Plane constant in the unit-cube coordinates of g
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    alpha -= m[d] * static_cast<amrex::Real>(g[d] - ref[d]);
  }
  return amrex::Clamp(
    cut_volume(m[0], m[1], m[2], alpha, 0.0_rt, 1.0_rt), 0.0_rt, 1.0_rt);
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
 * \brief Volume-weighted mixture density of a cell with liquid fraction F
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mixture_density(
  const amrex::Real F, const amrex::Real rho_l, const amrex::Real rho_g)
{
  const amrex::Real f = amrex::Clamp(F, 0.0_rt, 1.0_rt);
  return f * rho_l + (1.0_rt - f) * rho_g;
}

/**
 * \brief Mixture viscosity of a cell with liquid fraction F: arithmetic
 *        or harmonic volume average of the phase viscosities
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mixture_viscosity(
  const amrex::Real F,
  const amrex::Real mu_l,
  const amrex::Real mu_g,
  const int harmonic)
{
  const amrex::Real f = amrex::Clamp(F, 0.0_rt, 1.0_rt);
  if (harmonic != 0) {
    return mu_l * mu_g / (f * mu_g + (1.0_rt - f) * mu_l);
  }
  return f * mu_l + (1.0_rt - f) * mu_g;
}

} // namespace pelelmex::vof

#endif
