#!/usr/bin/env python3
"""Post-processing of the droplet breakup runs (run_weber_sweep.sh).

Reads the ``VOF: time ...`` lines that ``peleLM.vof.v = 1`` prints after each
step (liquid volume, centroid, second moments, extent of the F > 1/2 cells,
interface area) and tabulates, against the deformation time
t* = t U / D sqrt(rho_g / rho_l):

  - the cross-stream extent D_cross / D (mean of the y and z extents in 3D)
    and the streamwise extent D_stream / D of the liquid,
  - the interface area A / A_0,
  - the centroid velocity u_d / U and the drag coefficient from the
    momentum balance of the liquid,
        rho_l V du_d/dt = 1/2 rho_g C_D A_ref (U - u_d)^2,
    with A_ref the initial frontal area (pi D^2 / 4 in 3D, D per unit depth
    in 2D); u_d and du_d/dt come from a least-squares parabola through the
    centroid positions within +-`--window`/2 (in t*) of each time.  The
    impulsive start gives the drop an initial added-mass velocity: C_D is
    only meaningful after t* ~ 0.1.

For 3D runs the summary lists, for reference, the breakup-time correlations of
Pilch & Erdman (Int. J. Multiphase Flow 13, 1987) for Oh < 0.1: initiation
T_ini = 1.9 (We - 12)^(-1/4) (1 + 2.2 Oh^1.6) and total breakup time T_b
(piecewise in We), and the regime boundaries of Guildenbecher, Lopez-Rivera &
Sojka (Exp. Fluids 46, 2009): vibrational We < ~11, bag 11-35, multimode
35-80, sheet thinning 80-350.  These are measured on drops in shock tubes and
air jets; the 2D (planar) runs are liquid cylinders and are only qualitative.

  python3 analyze_breakup.py We*/run.log --dim 3 [--density-ratio 1000]
  python3 analyze_breakup.py water_We*/run.log --dim 3 --case water

With ``--case water`` (input.*d-water, SI units) the defaults are those of a
2 mm water drop in air, the surface tension is fixed and the air velocity of
each run is U = sqrt(We sigma / (rho_g D)), with We from the directory name.
"""
import argparse
import math
import re
import sys

import numpy as np


def read_stats(fname, dim):
    pat = re.compile(
        r"VOF: time (\S+) centroid((?: \S+){%d}) moments(?: \S+){%d}"
        r" extent((?: \S+){%d}) area (\S+)" % (dim, dim, 2 * dim))
    rows = {}
    with open(fname) as f:
        for line in f:
            m = pat.search(line)
            if m:
                t = float(m.group(1))
                rows[t] = ([t] + [float(v) for v in m.group(2).split()] +
                           [float(v) for v in m.group(3).split()] +
                           [float(m.group(4))])
    return np.array([rows[t] for t in sorted(rows)])


def read_input(fname, key):
    """Last value of `key = value` echoed by ParmParse in the log, if any."""
    val = None
    pat = re.compile(r"\b" + re.escape(key) + r"\s*=\s*(\S+)")
    with open(fname) as f:
        for line in f:
            m = pat.search(line)
            if m:
                val = m.group(1)
    return val


def pilch_erdman(we, oh):
    if we <= 12.0:
        return math.nan, math.nan
    t_ini = 1.9 * (we - 12.0) ** -0.25 * (1.0 + 2.2 * oh ** 1.6)
    if we < 18.0:
        t_b = 6.0 * (we - 12.0) ** -0.25
    elif we < 45.0:
        t_b = 2.45 * (we - 12.0) ** 0.25
    elif we < 351.0:
        t_b = 14.1 * (we - 12.0) ** -0.25
    elif we < 2670.0:
        t_b = 0.766 * (we - 12.0) ** 0.25
    else:
        t_b = 5.5
    return t_ini, t_b


def regime(we):
    for bound, name in ((11.0, "vibrational/deformation"), (35.0, "bag"),
                        (80.0, "multimode"), (350.0, "sheet thinning")):
        if we < bound:
            return name
    return "catastrophic"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--case", choices=("nondim", "water"),
                        default="nondim",
                        help="input.*d (We set by sigma) or input.*d-water "
                        "(SI, We set by the air velocity)")
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--velocity", type=float, default=None,
                        help="water: default from the directory name")
    parser.add_argument("--rho-gas", type=float, default=None)
    parser.add_argument("--density-ratio", type=float, default=None)
    parser.add_argument("--mu-liquid", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None,
                        help="nondim: default from the run directory name "
                        "We<value>")
    parser.add_argument("--every", type=float, default=0.25,
                        help="t* interval of the table")
    parser.add_argument("--window", type=float, default=0.25,
                        help="t* width of the centroid fits")
    args = parser.parse_args()
    water = args.case == "water"
    defaults = (dict(diameter=2.0e-3, rho_gas=1.204, density_ratio=998.2 / 1.204,
                     mu_liquid=1.002e-3, sigma=0.0728) if water else
                dict(diameter=1.0, velocity=1.0, rho_gas=1.0,
                     density_ratio=1000.0, mu_liquid=5.0e-2))
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    dim, D = args.dim, args.diameter
    rho_g = args.rho_gas
    rho_l = rho_g * args.density_ratio
    vol0 = math.pi * D ** 3 / 6.0 if dim == 3 else math.pi * D ** 2 / 4.0
    aref = math.pi * D ** 2 / 4.0 if dim == 3 else D
    summary = []

    for log in args.logs:
        d = read_stats(log, dim)
        if len(d) < 3:
            print(f"{log}: fewer than 3 'VOF: time' lines, skipped")
            continue
        m = re.search(r"We([0-9.eE+-]+)", log)
        we_dir = float(m.group(1)) if m else math.nan
        sigma, U = args.sigma, args.velocity
        if U is None:
            U = math.sqrt(we_dir * sigma / (rho_g * D))
        if sigma is None:
            sigma = rho_g * U ** 2 * D / we_dir
        we = rho_g * U ** 2 * D / sigma
        tscale = D / U * math.sqrt(args.density_ratio)
        oh = args.mu_liquid / math.sqrt(rho_l * sigma * D)

        t = d[:, 0]
        ts = t / tscale
        xc = d[:, 1]
        ext = d[:, 1 + dim:1 + 3 * dim].reshape(-1, dim, 2)
        width = ext[:, :, 1] - ext[:, :, 0]
        stream = width[:, 0] / D
        cross = width[:, 1:].mean(axis=1) / D
        area = d[:, -1] / d[0, -1]
        ud = np.full(len(t), math.nan)
        ad = np.full(len(t), math.nan)
        half = 0.5 * args.window * tscale
        for i in range(len(t)):
            sel = np.abs(t - t[i]) <= half
            if sel.sum() >= 5:
                c = np.polyfit(t[sel] - t[i], xc[sel], 2)
                ud[i], ad[i] = c[1], 2.0 * c[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            cd = rho_l * vol0 * ad / (0.5 * rho_g * aref * (U - ud) ** 2)

        units = f", U = {U:.4g} m/s, t_scale = {tscale * 1e3:.4g} ms" \
            if water else ""
        print(f"\n=== {log}: We = {we:.4g}, Oh = {oh:.3g}{units}, "
              f"{regime(we)} (Guildenbecher et al. 2009 map)")
        print(f"{'t*':>6} {'D_cross/D':>10} {'D_stream/D':>11} {'A/A0':>7} "
              f"{'x_c/D':>7} {'u_d/U':>7} {'C_D':>7}")
        for tt in np.arange(0.0, ts[-1] + 1.0e-12, args.every):
            i = min(np.searchsorted(ts, tt), len(ts) - 1)
            print(f"{ts[i]:6.2f} {cross[i]:10.3f} {stream[i]:11.3f} "
                  f"{area[i]:7.3f} {xc[i] / D:7.3f} {ud[i] / U:7.4f} "
                  f"{cd[i]:7.3f}")
        imax = int(np.argmax(cross))
        late = (ts > 0.1) & np.isfinite(cd)
        cd_mean = cd[late].mean() if late.any() else math.nan
        summary.append((we, oh, ts[-1], cross[imax], ts[imax], area.max(),
                        cd_mean))

    if not summary:
        sys.exit(1)
    print("\n=== Summary")
    hdr = (f"{'We':>7} {'Oh':>7} {'t*_end':>7} {'max D_cross/D':>14} "
           f"{'at t*':>6} {'max A/A0':>9} {'mean C_D':>9}")
    if dim == 3:
        hdr += f" {'T_ini PE87':>11} {'T_b PE87':>9}"
    print(hdr)
    for we, oh, tend, cmax, tcmax, amax, cd in sorted(summary):
        line = (f"{we:7.3g} {oh:7.3g} {tend:7.2f} {cmax:14.3f} {tcmax:6.2f} "
                f"{amax:9.3f} {cd:9.3f}")
        if dim == 3:
            t_ini, t_b = pilch_erdman(we, oh)
            line += f" {t_ini:11.2f} {t_b:9.2f}"
        print(line)


if __name__ == "__main__":
    main()
