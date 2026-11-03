# VOFDroplet

Verification of the VOF surface tension (`peleLM.vof.sigma`): incompressible
drops with a density ratio of 1000 and no gravity.

| Input               | Case                                                                 |
|---------------------|----------------------------------------------------------------------|
| `input.2d-static`   | Drop at rest, R = 0.25: Laplace pressure jump sigma / R, spurious currents |
| `input.3d-static`   | Same in 3D (jump 2 sigma / R), 2 levels                               |
| `input.2d-oscillate`| Mode-2 oscillation, R = 0.2, 5% amplitude: Rayleigh frequency          |
| `vof-breakup-2d.inp`| Drop in an impulsively started gas stream (CI only, see `Exec/Production/DropletBreakup`) |

```
make -j
mpirun -np 4 ./PeleLMeX2d.gnu.MPI.ex input.2d-static
python3 analyze_droplet.py static run.log --dim 2 --radius 0.25 --sigma 1 --mu-liquid 1e-2
mpirun -np 4 ./PeleLMeX2d.gnu.MPI.ex input.2d-oscillate
python3 analyze_droplet.py oscillate run.log --dim 2 --radius 0.2 --sigma 1
```

`analyze_droplet.py` reads the `VOF: time ...` lines of the log
(`peleLM.vof.v = 1`) and `temporals/tempExtremas`.

## Results

Static drop, 2D, `input.2d-static` to t = 0.5 (about 3 capillary times
sqrt(rho_l R^3 / sigma)), single level:

| Grid | R / dx | Pressure jump error | max \|u\| over the run | \|u\| at t = 0.5 |
|------|--------|---------------------|------------------------|------------------|
| 32²  | 8      | 7.7e-3              | 2.5e-3                 | 1.6e-4           |
| 64²  | 16     | 1.9e-3              | 7.1e-4                 | 8.0e-5           |
| 128² | 32     | 4.7e-4              | 1.7e-4                 | 1.9e-5           |
| 256² | 64     | 1.2e-4              | 5.8e-5                 | 4.0e-6           |

The pressure jump converges at second order and the spurious currents at
second order or better; at 128² the capillary number of the currents,
max|u| mu_l / sigma, is 1.7e-6.  With the exact curvature imposed the
velocity stays at 2e-11 (balanced force).  A 32² base grid with one level of
refinement reproduces the 64² values.

Static drop, 3D, `input.3d-static` (32³ base grid and one level, R / dx = 16
at the interface), 300 steps to t = 0.16: max |u| stays between 5e-4 and
4e-3 with no growth, the curvature error of every interface cell is below
0.3% and the pressure jump is within 0.1% of 2 sigma / R.  With the exact
curvature imposed the velocity stays at 1e-12.  In 3D about 2% of the
interface cells (near the grid diagonals) have no height function in their
3x3x3 neighborhood and take the average of the 5x5x5 one: a local fit there
is as accurate initially but follows small interface perturbations, and the
spurious currents grow without bound.

Oscillating drop, 2D, `input.2d-oscillate` to t = 1 (4.3 periods): the
frequency, corrected for the damping, is within 0.5% of the inviscid
Rayleigh value at 32², 64² and 128² (grid converged).  The departure is
physical: it falls to 0.28% and 0.22% at amplitudes 0.025 and 0.0125
(second order in the amplitude), and to 0.14% with a 5 times smaller
viscosity at amplitude 0.0125.

## Notes

- The projections use the arithmetic face density with surface tension (see
  `peleLM.vof.face_density` in the controls documentation): with the sharp
  face density the static drop at a density ratio of 1000 is unstable.
- The drops are initialized with `pelelmex::vof::smooth_cell_fraction`.  The
  sampling noise of `cell_fraction` (32² samples per cell) caps the curvature
  error at about 2% on 128² and stalls the convergence.

## Regression tests

`vof-staticdrop-2d.inp`, `vof-staticdrop-3d.inp` and `vof-breakup-2d.inp` are
registered with CTest (`ctest -L vof`, configure with `-DPELE_ENABLE_VOF=ON`).
Besides the volume conservation and clipping checks of `Tests/test_vof.py`,
the static drops bound the velocity and the error of the pressure jump
(`--laplace`), and the breakup case bounds the velocity.
