#!/usr/bin/env python3
"""Checks for the volume-of-fluid regression tests.

Reads the run log (``peleLM.vof.v = 1``) and the ``temporals/tempExtremas``
file (``peleLM.do_extremas = 1``) written in the working directory, and fails
if:
  - the volume fraction was clipped (``VOF: WARNING`` in the log),
  - the liquid volume changed by more than ``--volume-rtol`` in any step,
  - the largest velocity magnitude component exceeds ``--max-velocity``,
  - any velocity component departs from ``--uniform-velocity`` by more than
    ``--uniform-atol``,
  - ``--require-substeps`` is given and the advection was never sub-stepped.
"""
import argparse
import re
import sys


def read_log(fname):
    warnings = []
    changes = []
    substeps = 0
    pattern = re.compile(r"VOF: liquid volume \S+, relative change (\S+)")
    with open(fname) as f:
        for line in f:
            if "VOF: WARNING" in line:
                warnings.append(line.strip())
            elif "sub-steps" in line and "VOF:" in line:
                substeps += 1
            m = pattern.search(line)
            if m:
                changes.append(float(m.group(1)))
    return warnings, changes, substeps


def read_velocity_extrema(fname, ndim):
    """(min, max) pairs of each velocity component, over all steps."""
    vmin = [float("inf")] * ndim
    vmax = [float("-inf")] * ndim
    with open(fname) as f:
        next(f)  # header
        for line in f:
            cols = line.strip().split(",")
            # iter, time, then min/max of each velocity component
            for d in range(ndim):
                vmin[d] = min(vmin[d], float(cols[2 + 2 * d]))
                vmax[d] = max(vmax[d], float(cols[3 + 2 * d]))
    return vmin, vmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="run log")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--volume-rtol", type=float, default=1.0e-13)
    parser.add_argument("--max-velocity", type=float, default=None)
    parser.add_argument("--uniform-velocity", type=float, nargs="+", default=None)
    parser.add_argument("--uniform-atol", type=float, default=1.0e-12)
    parser.add_argument("--require-substeps", action="store_true")
    parser.add_argument("--extremas", default="temporals/tempExtremas")
    args = parser.parse_args()

    failures = []
    warnings, changes, substeps = read_log(args.log)

    if not changes:
        failures.append("no 'VOF: liquid volume' lines: set peleLM.vof.v = 1")
    for w in warnings:
        failures.append("clipping: " + w)
    if changes:
        worst = max(abs(c) for c in changes)
        print(f"steps {len(changes)}, max |relative volume change| {worst:.3e}")
        if worst > args.volume_rtol:
            failures.append(
                f"liquid volume changed by {worst:.3e} > {args.volume_rtol:.1e}"
            )
    print(f"sub-stepped steps {substeps}")
    if args.require_substeps and substeps == 0:
        failures.append("the VOF advection was never sub-stepped")

    if args.max_velocity is not None or args.uniform_velocity is not None:
        vmin, vmax = read_velocity_extrema(args.extremas, args.dim)
        print("velocity range " + ", ".join(
            f"[{lo:.6e}, {hi:.6e}]" for lo, hi in zip(vmin, vmax)))
        if args.max_velocity is not None:
            umax = max(max(abs(lo), abs(hi)) for lo, hi in zip(vmin, vmax))
            if umax > args.max_velocity:
                failures.append(
                    f"max |u| {umax:.3e} > {args.max_velocity:.1e}"
                )
        if args.uniform_velocity is not None:
            for d, (lo, hi) in enumerate(zip(vmin, vmax)):
                u0 = args.uniform_velocity[d]
                dev = max(abs(lo - u0), abs(hi - u0))
                if dev > args.uniform_atol:
                    failures.append(
                        f"velocity {d} departs from {u0} by {dev:.3e}"
                    )

    for f in failures:
        print("FAIL: " + f)
    if failures:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
