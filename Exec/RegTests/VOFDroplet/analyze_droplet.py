#!/usr/bin/env python3
"""Post-processing of the VOFDroplet verification cases.

Reads the ``VOF: time ...`` lines printed with ``peleLM.vof.v = 1`` (liquid
centroid, second central moments, extent of the F > 1/2 cells, mean interface
curvature and pressure jump) and, for the spurious currents, the
``temporals/tempExtremas`` file (``peleLM.do_extremas = 1``).

  static    Laplace pressure jump and curvature errors against sigma kappa,
            kappa = (dim - 1) / R, and the spurious currents as a capillary
            number Ca = max|u| mu_l / sigma
  oscillate period and damping of the mode-2 oscillation, from a damped
            cosine fit of the moment difference I_xx - I_yy, against the
            inviscid Rayleigh / Lamb frequency
              2D: omega^2 = (n^3 - n) sigma / ((rho_l + rho_g) R^3)
              3D: omega^2 = n (n - 1) (n + 1) (n + 2) sigma
                            / (((n + 1) rho_l + n rho_g) R^3)
            with n = 2.  omega_0 = sqrt(omega_d^2 + gamma^2) removes the
            viscous shift of a linear damped oscillator.

Usage:
  analyze_droplet.py static    run.log --dim 2 --radius 0.25 --sigma 1 --mu-liquid 1e-2
  analyze_droplet.py oscillate run.log --dim 2 --radius 0.2 --sigma 1 \\
                     --rho-liquid 1 --rho-gas 1e-3 [--max-period-error 0.05]
Exits with 1 when a given tolerance is exceeded.
"""
import argparse
import math
import re
import sys

import numpy as np


def read_stats(fname, dim):
    """Rows of (time, centroid[dim], moments[dim], extent[2 dim], kappa, dp)."""
    num = r"(\S+)"
    pat = re.compile(
        r"VOF: time " + num + r" centroid((?: \S+){%d}) moments((?: \S+){%d})"
        r" extent((?: \S+){%d}) area \S+"
        r"(?: curvature (\S+) pressure_jump (\S+))?"
        % (dim, dim, 2 * dim)
    )
    rows = []
    with open(fname) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            row = [float(m.group(1))]
            for g in (2, 3, 4):
                row += [float(v) for v in m.group(g).split()]
            row += [float(m.group(5)) if m.group(5) else math.nan,
                    float(m.group(6)) if m.group(6) else math.nan]
            rows.append(row)
    if not rows:
        sys.exit(f"no 'VOF: time' lines in {fname}: set peleLM.vof.v = 1")
    # Several lines share a time during the initial iterations: keep the last
    data = {}
    for r in rows:
        data[r[0]] = r
    return np.array([data[t] for t in sorted(data)])


def max_velocity(fname, dim):
    umax = []
    with open(fname) as f:
        next(f)
        for line in f:
            cols = line.strip().split(",")
            umax.append(max(abs(float(c)) for c in cols[2:2 + 2 * dim]))
    return np.array(umax)


def static(args, failures):
    d = read_stats(args.log, args.dim)
    kappa = (args.dim - 1) / args.radius
    dp_exact = args.sigma * kappa
    k_err = abs(d[-1, -2] - kappa) / kappa
    p_err = abs(d[-1, -1] - dp_exact) / dp_exact
    print(f"curvature      {d[-1, -2]:.8f} (exact {kappa:.8f}), "
          f"relative error {k_err:.3e}")
    print(f"pressure jump  {d[-1, -1]:.8f} (exact {dp_exact:.8f}), "
          f"relative error {p_err:.3e}")
    if args.max_pressure_error is not None and p_err > args.max_pressure_error:
        failures.append(f"pressure jump error {p_err:.3e} > "
                        f"{args.max_pressure_error:.1e}")
    try:
        umax = max_velocity(args.extremas, args.dim)
    except OSError:
        return
    if args.mu_liquid is not None:
        ca = umax * args.mu_liquid / args.sigma
        print(f"capillary number of the spurious currents: max {ca.max():.3e}"
              f", final {ca[-1]:.3e}")
    print(f"max |u|: over the run {umax.max():.3e}, final {umax[-1]:.3e}")
    if args.max_velocity is not None and umax.max() > args.max_velocity:
        failures.append(f"max |u| {umax.max():.3e} > {args.max_velocity:.1e}")


def oscillate(args, failures):
    d = read_stats(args.log, args.dim)
    t = d[:, 0]
    # I_xx minus the mean of the transverse moments
    mom = d[:, 1 + args.dim:1 + 2 * args.dim]
    s = mom[:, 0] - mom[:, 1:].mean(axis=1)
    # Successive extrema -> half periods and logarithmic decrement
    ext = [i for i in range(1, len(s) - 1)
           if (s[i] - s[i - 1]) * (s[i + 1] - s[i]) < 0.0]
    if len(ext) < 3:
        sys.exit("fewer than 3 extrema of I_xx - I_yy: run longer")
    te, se = [], []
    for i in ext:
        # Parabola through the three points around the extremum
        a, b, c = np.polyfit(t[i - 1:i + 2] - t[i], s[i - 1:i + 2], 2)
        dt = -b / (2.0 * a)
        te.append(t[i] + dt)
        se.append(c - b * b / (4.0 * a))
    te, se = np.array(te), np.array(se)
    half = np.diff(te)
    omega_d = math.pi / half.mean()
    amp = np.abs(se)
    gamma = -np.polyfit(te, np.log(amp), 1)[0] if len(amp) > 2 else 0.0
    omega_0 = math.sqrt(omega_d ** 2 + gamma ** 2)

    n = 2
    rl, rg, R, sig = args.rho_liquid, args.rho_gas, args.radius, args.sigma
    if args.dim == 2:
        omega_th = math.sqrt((n ** 3 - n) * sig / ((rl + rg) * R ** 3))
    else:
        omega_th = math.sqrt(n * (n - 1) * (n + 1) * (n + 2) * sig /
                             (((n + 1) * rl + n * rg) * R ** 3))
    err = abs(omega_0 - omega_th) / omega_th
    print(f"extrema {len(te)}, half periods {np.array2string(half, precision=5)}")
    print(f"damped omega {omega_d:.5f}, damping rate {gamma:.5f}")
    print(f"omega_0 {omega_0:.5f}, Rayleigh {omega_th:.5f}, relative error "
          f"{err:.3e} (period {2 * math.pi / omega_0:.5f} vs "
          f"{2 * math.pi / omega_th:.5f})")
    if args.max_period_error is not None and err > args.max_period_error:
        failures.append(f"frequency error {err:.3e} > "
                        f"{args.max_period_error:.1e}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=["static", "oscillate"])
    parser.add_argument("log")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--rho-liquid", type=float, default=1.0)
    parser.add_argument("--rho-gas", type=float, default=1.0e-3)
    parser.add_argument("--mu-liquid", type=float, default=None)
    parser.add_argument("--extremas", default="temporals/tempExtremas")
    parser.add_argument("--max-pressure-error", type=float, default=None)
    parser.add_argument("--max-velocity", type=float, default=None)
    parser.add_argument("--max-period-error", type=float, default=None)
    args = parser.parse_args()

    failures = []
    {"static": static, "oscillate": oscillate}[args.case](args, failures)
    for f in failures:
        print("FAIL: " + f)
    if failures:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
