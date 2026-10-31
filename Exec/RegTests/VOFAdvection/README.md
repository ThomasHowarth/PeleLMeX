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
step; `peleLM.vof.v = 2` also reports any volume removed by clipping the
volume fraction to [0, 1] on uncovered cells.

## Expected results

Liquid volume is conserved to round-off (relative change per step of order
1e-16) in all cases, with and without AMR, and checkpoint restarts reproduce
continuous runs bit for bit.

L1 shape error `sum |F(T) - F(0)| dV` on uniform grids (`amr.max_level = 0`;
single vortex with `amr.fixed_dt = 0.5/n`):

| n     | Zalesak  | order | single vortex | order |
|-------|----------|-------|---------------|-------|
| 64²   | 3.14e-3  |       | 1.93e-3       |       |
| 128²  | 1.05e-3  | 1.58  | 4.90e-4       | 1.98  |
| 256²  | 4.77e-4  | 1.14  | 1.43e-4       | 1.78  |

The single vortex converges at close to second order; Zalesak's disk is
limited to about first order by the unresolved slot corners, as expected
for PLIC schemes.
