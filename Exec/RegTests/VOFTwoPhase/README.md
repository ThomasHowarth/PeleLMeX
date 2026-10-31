# VOFTwoPhase

Incompressible two-phase cases (`USE_VOF = TRUE`) in which the density and
viscosity are derived from the liquid volume fraction,
`rho = F rho_l + (1 - F) rho_g`, with gravity along -y.

| Input               | Case                                                            |
|---------------------|-----------------------------------------------------------------|
| `input.2d-pool`     | Water layer at rest under air; `prob.tilt` tilts the surface     |
| `input.2d-rt`       | Rayleigh-Taylor instability, rho 3:1, Re = 3000, 3 levels         |
| `input.2d-dambreak` | Water column collapse (Martin & Moyce 1952), rho 1000:1.2          |

```
make -j
mpirun -np 4 ./PeleLMeX2d.gnu.MPI.ex input.2d-pool
```

## Current status

The variable-density path (MAC and nodal projections, gravity and pressure
gradient divided by the mixture density, implicit viscous solve with
`rho^{n+1/2}`) reproduces the constant-density results bit for bit when
`rho_liquid = rho_gas`, and a uniform flow carrying a 1000:1 drop stays
exactly uniform.  The liquid volume is conserved to round-off in all cases.

Gravity is not yet *balanced* at the interface: the cell-centered gravity
and pressure-gradient terms of the approximate projection only cancel when
the interface is aligned with the grid.

- `input.2d-pool` (grid-aligned surface): velocity stays at 1e-12.
- `input.2d-pool prob.tilt=30 peleLM.gravity="-4.905 -8.49571"
  amr.max_level=0` (surface at 30 degrees, gravity normal to it): spurious
  velocities of 0.08 m/s (rho 3:1) and 0.4 m/s (rho 1000:1.2) after 0.2 s.
- `input.2d-rt` with `prob.amplitude=0.01`: the linear growth rate is about
  15% below sqrt(A k g), independently of the grid and the time step.
- `input.2d-dambreak` is unstable at this density ratio.

A balanced-force (face-based) treatment of gravity and of the pressure
gradient, and momentum advection consistent with the VOF mass fluxes, are
the next steps.
