# VOFAdvection

Verification cases for the geometric PLIC volume-of-fluid advection
(`USE_VOF = TRUE`, `Source/VOF`).  All cases run on the unit square / cube
in incompressible mode.

| Input                | Case                                                     | Velocity                    |
|----------------------|----------------------------------------------------------|-----------------------------|
| `input.2d-zalesak`   | Zalesak slotted disk, one solid-body revolution, 3 levels | prescribed (`advect_only`)  |
| `input.2d-vortex`    | Circle in the time-reversed single vortex, T = 4, 3 levels | prescribed (`advect_only`)  |
| `input.2d-translate` | Circle carried diagonally across a periodic box, 2 levels | Navier-Stokes (MAC projected) |
| `input.3d-deform`    | Sphere in LeVeque's deformation field, T = 3, 3 levels    | prescribed (`advect_only`)  |

With `peleLM.vof.advect_only = 1` the face velocities are the discrete curl
of `vof_vector_potential` in `pelelmex_prob.H`, so they are divergence free
to round-off and consistent across coarse/fine faces; the momentum update is
skipped.  The prescribed cases return to their initial shape at
`t = prob.period`, so the shape error is the L1 difference between the first
and last plotfiles.

```
make -j                     # DIM = 2 by default; DIM=3 for input.3d-deform
mpirun -np 4 ./PeleLMeX2d.gnu.MPI.ex input.2d-zalesak
```

`peleLM.vof.v = 1` prints the liquid volume and its relative change every
step, and how far the volume fraction leaves [0, 1].

## Expected results

Results are for the default unsplit advection.  Liquid volume is conserved to
round-off (relative change per step of order 1e-15 or below) in all cases, with
and without AMR, and the volume fraction leaves [0, 1] by round-off only.  A
restarted Zalesak run with 3 levels reproduces the composite solution of a
continuous run exactly, although the plotfiles can differ in their box layout;
with a projected velocity restarts agree to the linear solver tolerance.

L1 shape error `sum |F(T) - F(0)| dV` on uniform grids (`amr.max_level = 0`;
single vortex with `amr.fixed_dt = 0.5/n`):

| n     | Zalesak  | order | single vortex | order |
|-------|----------|-------|---------------|-------|
| 64²   | 2.96e-3  |       | 1.87e-3       |       |
| 128²  | 1.03e-3  | 1.52  | 4.68e-4       | 2.00  |
| 256²  | 4.68e-4  | 1.14  | 1.39e-4       | 1.75  |

The single vortex converges at close to second order; Zalesak's disk is
limited to about first order by the unresolved slot corners, as expected
for PLIC schemes.

## Regression tests

The `vof-*.inp` inputs are short versions of these cases registered with
CTest when configuring with `-DPELE_ENABLE_VOF=ON` (`ctest -L vof`).
`Tests/test_vof.py` checks physical invariants rather than gold files: liquid
volume conserved to 1e-13 per step, the volume fraction within 1e-9 of [0, 1],
and for `vof-translate-2d`, a velocity that stays exactly uniform.
