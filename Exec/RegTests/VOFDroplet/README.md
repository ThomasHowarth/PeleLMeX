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
| 32²  | 8      | 7.7e-3              | 2.5e-3                 | 2.5e-4           |
| 64²  | 16     | 1.9e-3              | 7.1e-4                 | 7.1e-5           |
| 128² | 32     | 4.7e-4              | 1.7e-4                 | 1.6e-5           |

The pressure jump converges at second order and the spurious currents at
second order or better; at 128² the capillary number of the currents,
max|u| mu_l / sigma, is 1.7e-6.  The liquid volume is conserved to 1e-15 per
step.

Static drop, 3D, `input.3d-static` (32³ base grid and one level, R / dx = 16
at the interface), 150 steps to t = 0.077: max |u| peaks at 4.3e-3 and is
1.8e-3 at the end, and the mean curvature and the pressure jump are within
0.1% of 2 / R and 2 sigma / R.  In 3D about 2% of the interface cells (near
the grid diagonals) have no height function in their 3x3x3 neighborhood and
take the average of the 5x5x5 one: a local fit there is as accurate
initially but follows small interface perturbations, and the spurious
currents grow without bound.

Oscillating drop, 2D, `input.2d-oscillate` to t = 1 (4.3 periods): the
frequency, corrected for the damping, is within 0.6% of the inviscid
Rayleigh value at 32², 64² and 128² (0.53%, 0.24%, 0.32%).

These figures are for the unsplit advection.  With the former split
advection they were the same to two digits, except the final velocity of
the static drop at 32² (1.9e-4) and the oscillation frequency (0.77%, 0.10%
and 0.35% from the Rayleigh value).

## Notes

- The projections use the arithmetic face density with surface tension (see
  `peleLM.vof.face_density` in the controls documentation): with the sharp
  face density the static drop at a density ratio of 1000 is unstable.
- The drops are initialized with `pelelmex::vof::smooth_cell_fraction`
  (`Source/VOF/PeleLMeX_VOF_SurfaceTension_K.H`).  The sampling noise of
  `PLIC::cell_fraction` (32² samples per cell) caps the curvature error at
  about 2% on 128² and stalls the convergence.
- The curvature kernels (height functions, parabolic fit) are in
  `Source/VOF/PeleLMeX_VOF_SurfaceTension_K.H`; the normals, plane
  reconstruction and height-function constants they use are those of the
  AMReX-Hydro PLIC library.

## Regression tests

`vof-staticdrop-2d.inp`, `vof-staticdrop-3d.inp` and `vof-breakup-2d.inp` are
registered with CTest (`ctest -L vof`, configure with `-DPELE_ENABLE_VOF=ON`).
Besides the volume conservation and [0, 1] bound checks of `Tests/test_vof.py`,
the static drops bound the velocity and the error of the pressure jump
(`--laplace`), and the breakup case bounds the velocity.
