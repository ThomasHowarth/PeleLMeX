# VOFTwoPhase

Incompressible two-phase cases (`USE_VOF = TRUE`) in which the density and
viscosity are derived from the liquid volume fraction,
`rho = F rho_l + (1 - F) rho_g`, with gravity along -y where set.

| Input               | Case                                                            |
|---------------------|-----------------------------------------------------------------|
| `input.2d-pool`     | Water layer at rest under air; `prob.tilt` tilts the surface     |
| `input.2d-rt`       | Rayleigh-Taylor instability, rho 3:1, Re = 3000, 3 levels         |
| `input.2d-dambreak` | Water column collapse (Martin & Moyce 1952), rho 1000:1.2          |
| `input.2d-translate`| Drop carried by a uniform flow through a periodic box, 2 levels   |

```
make -j
mpirun -np 4 ./PeleLMeX2d.gnu.MPI.ex input.2d-pool
```

## Current status

Gravity and the pressure gradient are balanced on the faces with a sharp
(PLIC) face density, and momentum is advected with the VOF mass fluxes (see
the volume of fluid section of the controls documentation).  The liquid
volume is conserved to round-off in all cases.

- `input.2d-pool` (grid-aligned surface): velocity stays at 1e-12.
- `input.2d-pool prob.tilt=30 peleLM.gravity="-4.905 -8.49571"` (surface
  at 30 degrees, gravity normal to it): the velocity stays below 1e-7 m/s
  for any tilt from 5 to 89 degrees, with and without AMR.  The interface
  meets the side walls at whatever angle it has (free contact line, see the
  controls documentation); extrapolated wall ghost values would impose 90
  degrees and drive currents of order 1e-2 m/s.
- `input.2d-rt` with `prob.amplitude=0.01 amr.n_cell="64 256" amr.max_level=0`:
  the amplitude of the first Fourier mode of the interface height follows
  the viscous linear theory, `cosh(sigma t)`, to 0.5% at t = 0.4 and 2.1% at
  t = 0.8 (the departure grows with the nonlinear saturation).  The
  amplitude must be measured on the mode-1 Fourier coefficient: a two-point
  difference is dominated by the PLIC height quantization at small
  amplitudes.
- `input.2d-dambreak` runs to t = 0.4 s through the wall impact and run-up.
  The surge front is ahead of the Martin & Moyce data by about 0.3 column
  widths with a similar speed, as is usual for simulations that ignore the
  gate opening.  It also runs with `amr.cfl=0.9`.
- A uniform flow carrying a 1000:1 drop
  (`input.2d-translate peleLM.vof.rho_liquid=1000`) stays exactly uniform, and
  its volume fraction and velocity are identical to those of the equal-density
  case.

## Regression tests

The `vof-*.inp` inputs are short versions of these cases registered with
CTest when configuring with `-DPELE_ENABLE_VOF=ON` (`ctest -L vof`).
`Tests/test_vof.py` checks physical invariants rather than gold files: liquid
volume conserved to 1e-13 per step, the volume fraction within 1e-9 of [0, 1],
and for the tilted pools, a velocity below 1e-8 m/s (without the free contact
line it reaches 6e-3 m/s within 20 steps).  `vof-dambreak-3d` and
`vof-dambreak-amr-3d` check that w stays at round-off in an extruded flow, and
`vof-translate-2d` that a uniform velocity stays uniform across a 1000:1
interface, with regridding.  The geometric advection itself (shape errors in
the single vortex, Zalesak's disk and the 3D deformation, independence of the
box layout and of the number of ranks) is verified in AMReX-Hydro, by
`Tests/PLIC`.

Surface tension is the next step.
