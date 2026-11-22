#ifndef AMREX_MLABECCECLAP_2D_K_H
#define AMREX_MLABECCECLAP_2D_K_H

namespace amrex {

constexpr Real eps = 1.0e-10;

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getFaceStateUpwind(
  Real const& edgeVel, Real const& leftState, Real const& rightState)
{
  Real edstate = 0.0;
  if (edgeVel > eps) {
    edstate = leftState;
  }
  if (edgeVel < -eps) {
    edstate = rightState;
  }
  if (fabs(edgeVel) <= eps) {
    edstate = 0.5 * (leftState + rightState);
  }
  return edstate;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

  Real coeff = 0.0;
  if (leftedgeVel * rightedgeVel < 0.0) { // Case where velocity change sign,
                                          // use centered + art. visc
    Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
    coeff += 0.5 * (rightedgeVel - leftedgeVel); // centered part
    coeff += maxvel;                             // art. visc.
  } else { // same sign velocity, use upwind
    if (leftedgeVel < -eps)
      coeff -= leftedgeVel;
    else if (fabs(leftedgeVel) <= eps)
      coeff -= 0.5 * leftedgeVel;
    if (rightedgeVel > eps)
      coeff += rightedgeVel;
    else if (fabs(rightedgeVel) <= eps)
      coeff += 0.5 * rightedgeVel;
  }

  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getLeftCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

  Real coeff = 0.0;
  if (leftedgeVel * rightedgeVel < 0.0) {
    Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
    coeff -= 0.5 * leftedgeVel;
    coeff -= 0.5 * maxvel;
  } else {
    if (leftedgeVel > eps)
      coeff -= leftedgeVel;
    else if (fabs(leftedgeVel) < eps)
      coeff -= 0.5 * leftedgeVel;
  }
  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getRightCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

  Real coeff = 0.0;
  if (leftedgeVel * rightedgeVel < 0.0) {
    Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
    coeff += 0.5 * rightedgeVel;
    coeff += 0.5 * maxvel;
  } else {
    if (rightedgeVel < -eps)
      coeff += rightedgeVel;
    else if (fabs(rightedgeVel) < eps)
      coeff += 0.5 * rightedgeVel;
  }
  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getCellUpwindCoeff(
  Real const& leftedgeVel,
  Real const& rightedgeVel,
  Real const& relaxleft,
  Real const& relaxright)
{

  Real coeff = 0.0;
  if (
    leftedgeVel * relaxleft * rightedgeVel * relaxright <
    0.0) { // Case where velocity change sign, use centered + art. visc
    Real maxvel =
      std::max(fabs(leftedgeVel * relaxleft), fabs(rightedgeVel * relaxright));
    coeff += 0.5 * (rightedgeVel * relaxright -
                    leftedgeVel * relaxleft); // centered part
    coeff += maxvel;                          // art. visc.
  } else {                                    // same sign velocity, use upwind
    if (leftedgeVel < -eps)
      coeff -= leftedgeVel * relaxleft;
    else if (fabs(leftedgeVel) < eps)
      coeff -= 0.5 * leftedgeVel * relaxleft;
    if (rightedgeVel > eps)
      coeff += rightedgeVel * relaxright;
    else if (fabs(rightedgeVel) < eps)
      coeff += 0.5 * rightedgeVel * relaxright;
  }

  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getFaceUpwindFlux(
  Real const& edgeVel, Real const& leftState, Real const& rightState)
{

  Real Flux = 0.0;
  if (edgeVel > eps)
    Flux = edgeVel * leftState;
  else if (edgeVel < -eps)
    Flux = edgeVel * rightState;
  else
    Flux = edgeVel * 0.5 * (leftState + rightState);
  return Flux;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_adotx(
  Box const& box,
  Array4<Real> const& y,
  Array4<Real const> const& x,
  Array4<Real const> const& a,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
  Real alpha,
  Real beta,
  Real eta,
  int ncomp) noexcept
{
  const Real dhxsq = beta * dxinv[0] * dxinv[0];
  const Real dhysq = beta * dxinv[1] * dxinv[1];
  const Real dhx = eta * dxinv[0];
  const Real dhy = eta * dxinv[1];

  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        y(i, j, 0, n) =
          alpha * a(i, j, 0) * x(i, j, 0, n) -
          dhxsq * (bX(i + 1, j, 0, n) * (x(i + 1, j, 0, n) - x(i, j, 0, n)) -
                   bX(i, j, 0, n) * (x(i, j, 0, n) - x(i - 1, j, 0, n))) -
          dhysq * (bY(i, j + 1, 0, n) * (x(i, j + 1, 0, n) - x(i, j, 0, n)) -
                   bY(i, j, 0, n) * (x(i, j, 0, n) - x(i, j - 1, 0, n))) -
          dhx * (cX(i + 1, j, 0, n) *
                   getFaceStateUpwind(
                     cX(i + 1, j, 0, n), x(i, j, 0, n), x(i + 1, j, 0, n)) -
                 cX(i, j, 0, n) *
                   getFaceStateUpwind(
                     cX(i, j, 0, n), x(i - 1, j, 0, n), x(i, j, 0, n))) -
          dhy * (cY(i, j + 1, 0, n) *
                   getFaceStateUpwind(
                     cY(i, j + 1, 0, n), x(i, j, 0, n), x(i, j + 1, 0, n)) -
                 cY(i, j, 0, n) *
                   getFaceStateUpwind(
                     cY(i, j, 0, n), x(i, j - 1, 0, n), x(i, j, 0, n)));
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_normalize(
  Box const& box,
  Array4<Real> const& x,
  Array4<Real const> const& a,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
  Real alpha,
  Real beta,
  Real eta,
  int ncomp) noexcept
{
  const Real dhxsq = beta * dxinv[0] * dxinv[0];
  const Real dhysq = beta * dxinv[1] * dxinv[1];
  const Real dhx = eta * dxinv[0];
  const Real dhy = eta * dxinv[1];

  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        x(i, j, 0, n) /=
          alpha * a(i, j, 0) + dhxsq * (bX(i, j, 0, n) + bX(i + 1, j, 0, n)) +
          dhysq * (bY(i, j, 0, n) + bY(i, j + 1, 0, n)) -
          dhx * getCellUpwindCoeff(cX(i, j, 0, n), cX(i + 1, j, 0, n)) -
          dhy * getCellUpwindCoeff(cY(i, j, 0, n), cY(i, j + 1, 0, n));
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_x(
  Box const& box,
  Array4<Real> const& fx,
  Array4<Real const> const& sol,
  Array4<Real const> const& bx,
  Array4<Real const> const& cx,
  Real fac,
  Real facc,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        fx(i, j, 0, n) =
          -fac * bx(i, j, 0, n) * (sol(i, j, 0, n) - sol(i - 1, j, 0, n)) -
          facc * getFaceUpwindFlux(
                   cx(i, j, 0, n), sol(i - 1, j, 0, n), sol(i, j, 0, n));
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_xface(
  Box const& box,
  Array4<Real> const& fx,
  Array4<Real const> const& sol,
  Array4<Real const> const& bx,
  Array4<Real const> const& cx,
  Real fac,
  Real facc,
  int xlen,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int j = lo.y; j <= hi.y; ++j) {
      int i = lo.x;
      fx(i, j, 0, n) =
        -fac * bx(i, j, 0, n) * (sol(i, j, 0, n) - sol(i - 1, j, 0, n)) -
        facc * getFaceUpwindFlux(
                 cx(i, j, 0, n), sol(i - 1, j, 0, n), sol(i, j, 0, n));
      i += xlen;
      fx(i, j, 0, n) =
        -fac * bx(i, j, 0, n) * (sol(i, j, 0, n) - sol(i - 1, j, 0, n)) -
        facc * getFaceUpwindFlux(
                 cx(i, j, 0, n), sol(i - 1, j, 0, n), sol(i, j, 0, n));
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_y(
  Box const& box,
  Array4<Real> const& fy,
  Array4<Real const> const& sol,
  Array4<Real const> const& by,
  Array4<Real const> const& cy,
  Real fac,
  Real facc,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        fy(i, j, 0, n) =
          -fac * by(i, j, 0, n) * (sol(i, j, 0, n) - sol(i, j - 1, 0, n)) -
          facc * getFaceUpwindFlux(
                   cy(i, j, 0, n), sol(i, j - 1, 0, n), sol(i, j, 0, n));
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_yface(
  Box const& box,
  Array4<Real> const& fy,
  Array4<Real const> const& sol,
  Array4<Real const> const& by,
  Array4<Real const> const& cy,
  Real fac,
  Real facc,
  int ylen,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    int j = lo.y;
    AMREX_PRAGMA_SIMD
    for (int i = lo.x; i <= hi.x; ++i) {
      fy(i, j, 0, n) =
        -fac * by(i, j, 0, n) * (sol(i, j, 0, n) - sol(i, j - 1, 0, n)) -
        facc * getFaceUpwindFlux(
                 cy(i, j, 0, n), sol(i, j - 1, 0, n), sol(i, j, 0, n));
    }
    j += ylen;
    AMREX_PRAGMA_SIMD
    for (int i = lo.x; i <= hi.x; ++i) {
      fy(i, j, 0, n) =
        -fac * by(i, j, 0, n) * (sol(i, j, 0, n) - sol(i, j - 1, 0, n)) -
        facc * getFaceUpwindFlux(
                 cy(i, j, 0, n), sol(i, j - 1, 0, n), sol(i, j, 0, n));
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
abeccec_gsrb(
  Box const& box,
  Array4<Real> const& phi,
  Array4<Real const> const& rhs,
  Real alpha,
  Real omega,
  Array4<Real const> const& a,
  Real dhxsq,
  Real dhysq,
  Real dhx,
  Real dhy,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<int const> const& m0,
  Array4<int const> const& m2,
  Array4<int const> const& m1,
  Array4<int const> const& m3,
  Array4<Real const> const& f0,
  Array4<Real const> const& f2,
  Array4<Real const> const& f1,
  Array4<Real const> const& f3,
  Box const& vbox,
  int redblack,
  int nc) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);
  const auto vlo = amrex::lbound(vbox);
  const auto vhi = amrex::ubound(vbox);

  for (int n = 0; n < nc; ++n) {
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        if ((i + j + redblack) % 2 == 0) {
          Real cf0 =
            (i == vlo.x and m0(vlo.x - 1, j, 0) > 0) ? f0(vlo.x, j, 0, n) : 0.0;
          Real cf1 =
            (j == vlo.y and m1(i, vlo.y - 1, 0) > 0) ? f1(i, vlo.y, 0, n) : 0.0;
          Real cf2 =
            (i == vhi.x and m2(vhi.x + 1, j, 0) > 0) ? f2(vhi.x, j, 0, n) : 0.0;
          Real cf3 =
            (j == vhi.y and m3(i, vhi.y + 1, 0) > 0) ? f3(i, vhi.y, 0, n) : 0.0;

          Real gamma =
            alpha * a(i, j, 0) + dhxsq * (bX(i, j, 0, n) + bX(i + 1, j, 0, n)) +
            dhysq * (bY(i, j, 0, n) + bY(i, j + 1, 0, n)) -
            dhx * getCellUpwindCoeff(cX(i, j, 0, n), cX(i + 1, j, 0, n)) -
            dhy * getCellUpwindCoeff(cY(i, j, 0, n), cY(i, j + 1, 0, n));

          Real g_m_d =
            gamma - (dhxsq * (bX(i, j, 0, n) * cf0 + bX(i + 1, j, 0, n) * cf2) +
                     dhysq * (bY(i, j, 0, n) * cf1 + bY(i, j + 1, 0, n) * cf3));

          Real rho =
            dhxsq * (bX(i, j, 0, n) * phi(i - 1, j, 0, n) +
                     bX(i + 1, j, 0, n) * phi(i + 1, j, 0, n)) +
            dhysq * (bY(i, j, 0, n) * phi(i, j - 1, 0, n) +
                     bY(i, j + 1, 0, n) * phi(i, j + 1, 0, n)) +
            dhx * (getLeftCellUpwindCoeff(cX(i, j, 0, n), cX(i + 1, j, 0, n)) *
                     phi(i - 1, j, 0, n) +
                   getRightCellUpwindCoeff(cX(i, j, 0, n), cX(i + 1, j, 0, n)) *
                     phi(i + 1, j, 0, n)) +
            dhy * (getLeftCellUpwindCoeff(cY(i, j, 0, n), cY(i, j + 1, 0, n)) *
                     phi(i, j - 1, 0, n) +
                   getRightCellUpwindCoeff(cY(i, j, 0, n), cY(i, j + 1, 0, n)) *
                     phi(i, j + 1, 0, n));

          Real res = rhs(i, j, 0, n) - (gamma * phi(i, j, 0, n) - rho);
          phi(i, j, 0, n) = phi(i, j, 0, n) + omega / g_m_d * res;
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
abeccec_getdiag(
  Box const& box,
  Real alpha,
  Array4<Real const> const& a,
  Real dhxsq,
  Real dhysq,
  Real dhx,
  Real dhy,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  int nc,
  Array4<Real> const& diag) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < nc; ++n) {
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        diag(i, j, 0) =
          alpha * a(i, j, 0) - dhxsq * (-bX(i, j, 0, n) - bX(i + 1, j, 0, n)) -
          dhysq * (-bY(i, j, 0, n) - bY(i, j + 1, 0, n)) -
          dhx * getCellUpwindCoeff(cX(i, j, 0, n), cX(i + 1, j, 0, n)) -
          dhy * getCellUpwindCoeff(cY(i, j, 0, n), cY(i, j + 1, 0, n));
      }
    }
  }
}

} // namespace amrex
#endif
