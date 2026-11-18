#!/bin/bash
# Weber number sweep of the droplet breakup case: one run directory per
# Weber number.  Extra arguments are passed to every run as ParmParse
# overrides, e.g. a coarser test: amr.max_level=3.
#
#   CASE=nondim (default): input.${DIM}d, D = U = rho_g = 1, the surface
#     tension sigma = rho_g U^2 D / We = 1 / We sets the Weber number.
#   CASE=water: input.${DIM}d-water, a 2 mm water drop in air (SI units);
#     the air velocity U = sqrt(We sigma / (rho_g D)) sets the Weber number,
#     and the stop, plot and checkpoint times are rescaled with
#     t_scale = D / U sqrt(rho_l / rho_g).
#
#   DIM=2|3 (default 3)   NP: MPI ranks per run (default 8)
#   WE: Weber numbers (default "5 15 40 120")
#   MPIRUN: launcher (default "mpirun -np $NP")
#   PARALLEL=1: run the Weber numbers concurrently
#
#   ./run_weber_sweep.sh                       # 3D, We = 5 15 40 120
#   DIM=2 NP=8 PARALLEL=1 ./run_weber_sweep.sh amr.max_level=3
#   CASE=water DIM=2 ./run_weber_sweep.sh
#
# Then: python3 analyze_breakup.py We*/run.log --dim $DIM [--case water]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIM="${DIM:-3}"
CASE="${CASE:-nondim}"
NP="${NP:-8}"
WE="${WE:-5 15 40 120}"
MPIRUN="${MPIRUN:-mpirun -np ${NP}}"
PARALLEL="${PARALLEL:-0}"
exe="$(ls "${here}"/PeleLMeX${DIM}d.*.ex 2>/dev/null | head -n 1 || true)"
if [ -z "${exe}" ]; then
  echo "No PeleLMeX${DIM}d executable in ${here}: run 'make DIM=${DIM}' first" >&2
  exit 1
fi

case "${CASE}" in
  nondim) input="${here}/input.${DIM}d"; prefix="We" ;;
  water)  input="${here}/input.${DIM}d-water"; prefix="water_We" ;;
  *) echo "Unknown CASE=${CASE} (nondim or water)" >&2; exit 1 ;;
esac

for we in ${WE}; do
  if [ "${CASE}" = water ]; then
    # Keep in sync with input.*d-water
    read -r -a over <<< "$(awk -v we="${we}" 'BEGIN {
      D = 2.0e-3; rl = 998.2; rg = 1.204; s = 0.0728
      U = sqrt(we * s / (rg * D)); ts = D / U * sqrt(rl / rg)
      printf "prob.velocity=%.8g amr.stop_time=%.8g amr.plot_per=%.8g", U, 6 * ts, 0.1 * ts
      printf " amr.check_per=%.8g amr.init_dt=%.8g", ts, 1.0e-3 * D / U }')"
  else
    over=("peleLM.vof.sigma=$(awk -v we="${we}" 'BEGIN { printf "%.12g", 1.0 / we }')")
  fi
  dir="${prefix}${we}"
  mkdir -p "${dir}"
  echo "We = ${we}: ${over[*]}, running in ${dir}"
  (
    cd "${dir}"
    ${MPIRUN} "${exe}" "${input}" "${over[@]}" "$@" > run.log 2>&1
  ) &
  if [ "${PARALLEL}" != "1" ]; then
    wait
  fi
done
wait
