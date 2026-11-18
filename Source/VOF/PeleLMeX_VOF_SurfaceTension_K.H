#ifndef PELELM_VOF_SURFACETENSION_K_H
#define PELELM_VOF_SURFACETENSION_K_H

// Surface tension kernels of the VOF solver: the interface curvature and the
// initialization of curved interfaces.  The plane geometry, the normals and
// the height-function constants are those of the AMReX-Hydro PLIC library
// (hydro_plic_K.H, hydro_plic_normals_K.H).

#include <hydro_plic_K.H>
#include <hydro_plic_normals_K.H>
#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <cmath>

namespace pelelmex::vof {

using namespace amrex::literals;

// The PLIC normal of the solver (vof_reconstruction = "myc_height" in
// PeleLMeX_VOF.cpp), for the interface points of parabolic_curvature
using VOFNormal = PLIC::HeightFunctionMYC;

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
 *        the orientation of the center column, as PLIC::height_normal does.
 *        Reads PLIC::hf_half cells along d and one cell in the tangential
 *        directions.
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
  constexpr int hf_half = PLIC::hf_half;
  constexpr amrex::Real pure_tol = PLIC::hf_pure_tol;
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
 *        components (PLIC::myc_normal).  Returns false when no direction has
 *        valid heights (thin or under-resolved structures).
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
  PLIC::myc_normal(i, j, k, F, m[0], m[1], m[2]);
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
 * \brief Interface point of the 7-cell column through cell (i,j,k) along d:
 *        when the column spans pure liquid to pure gas and its liquid height
 *        puts the interface within cell (i,j,k), returns true with z, the
 *        offset of the interface from the cell center along d in cell units,
 *        and s = +1 (liquid below, the normal out of the liquid along +d) or
 *        -1.  Second-order accurate for a smooth interface whose slope
 *        relative to d is at most 1.  Reads PLIC::hf_half cells along d.
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
  constexpr int hf_half = PLIC::hf_half;
  constexpr amrex::Real pure_tol = PLIC::hf_pure_tol;
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
 *        on the PLIC planes (PLIC::fit_plane) of the mixed cells of the
 *        stencil are added only when there are fewer than fit_min_points
 *        height points.  In the frame of the PLIC normal n of (i,j,k),
 *        pointing out of the liquid, z = a0 + a1 x + a2 x^2 (2D) or
 *        z = a0 + a1 x + a2 y + a3 x^2 + a4 y^2 + a5 x y (3D), and kappa is
 *        minus the curvature of that graph at the origin.  Points of the
 *        other side of a thin structure (facing away from n) are left out.
 *        Returns false when there are too few points or they do not
 *        determine the fit.  Reads up to PLIC::hf_half + 1 cells away: only
 *        for cells whose 3^dim stencil lies within the first ghost layer.
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
  if (!PLIC::is_mixed(F(i, j, k))) {
    return false;
  }
  // PLIC interface point of a mixed cell, relative to the center of (i,j,k),
  // in cell units, and the unit normal of its plane
  const auto plic_point = [&](
                            const int a, const int b, const int c,
                            amrex::Real p[3], amrex::Real nrm[3]) {
    amrex::Real m[3], alpha;
    PLIC::fit_plane<VOFNormal>(i + a, j + b, k + c, F, m[0], m[1], m[2], alpha);
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
  amrex::ignore_unused(t2);
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
          if (!PLIC::is_mixed(F(i + a, j + b, k + c))) {
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
 * \brief Liquid volume fraction of cell (i,j,k) for a smooth implicit
 *        surface phi(x) > 0 inside the liquid: phi is linearized on each of
 *        the nsub^dim sub-cells (central-difference gradient) and the
 *        fraction of the sub-cell below the plane is exact
 *        (PLIC::cut_volume).  The error is O(kappa h) relative per cut
 *        sub-cell and varies smoothly along the interface, whereas the
 *        sampling noise of PLIC::cell_fraction, O(1/nsub^2) per cell, is
 *        amplified by 1/dx in the height-function curvature.  Intended for
 *        the initialization of curved interfaces with surface tension.
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
          PLIC::cut_volume(
            m[0] / norm, m[1] / norm, m[2] / norm, alpha / norm, 0.0_rt,
            1.0_rt),
          0.0_rt, 1.0_rt);
      }
    }
  }
  return sum / static_cast<amrex::Real>(AMREX_D_TERM(nsub, *nsub, *nsub));
}

} // namespace pelelmex::vof

#endif
