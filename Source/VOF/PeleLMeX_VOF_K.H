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
 * \brief Reconstruct the PLIC plane m.x = alpha in cell (i,j,k), in unit-cube
 *        cell coordinates with sum|m| = 1.  Only meaningful for
 *        vof_tol < F < 1 - vof_tol.
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
