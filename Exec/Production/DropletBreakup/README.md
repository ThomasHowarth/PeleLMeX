# DropletBreakup

Aerodynamic (secondary) breakup of a liquid drop by an impulsively started gas
stream, at several gas Weber numbers: a validation case for the VOF solver
with surface tension (`USE_VOF = TRUE`, `peleLM.vof.sigma`).  The surface
tension discretization itself is verified in `Exec/RegTests/VOFDroplet`
(Laplace pressure and Rayleigh oscillation); this case compares the
breakup morphology and time scales with the experimental regime map.

## Setup

A drop of diameter D, initially at rest, meets a uniform gas stream of
velocity U (Inflow at x-lo, Outflow at x-hi, slip side walls, no gravity).
In the input units D = U = rho_g = 1:

| Parameter                           | Value                                    |
|-------------------------------------|------------------------------------------|
| density ratio rho_l / rho_g         | 1000                                     |
| viscosity ratio mu_l / mu_g         | 50                                       |
| gas Reynolds number rho_g U D / mu_g| 1000                                     |
| Weber number rho_g U^2 D / sigma    | set by `peleLM.vof.sigma = 1 / We`       |
| Ohnesorge number mu_l / sqrt(rho_l sigma D) | 0.05 sqrt(We / 1000) < 0.02      |
| time scale                          | t* = t U / D sqrt(rho_g / rho_l) = t / 31.6 |
| domain (3D)                         | 40 D x 8 D x 8 D, drop at 4 D from the inlet |
| resolution                          | 160 x 32 x 32 base grid, 4 levels: D / dx = 64 at the interface |

`input.2d` is the planar equivalent (a liquid cylinder, 40 D x 16 D), cheaper
but only qualitative: the regime boundaries of cylinders differ from those of
drops.

The Weber numbers of the sweep, with the expected regime for Oh < 0.1
(Guildenbecher, Lopez-Rivera & Sojka, Exp. Fluids 46, 2009) and the breakup
times of Pilch & Erdman (Int. J. Multiphase Flow 13, 1987):

| We  | Regime                      | Initiation T_ini | Total breakup T_b |
|-----|-----------------------------|------------------|-------------------|
| 5   | vibrational / deformation (no breakup) | -     | -                 |
| 15  | bag                         | 1.44             | 4.56              |
| 40  | multimode (bag and stamen)  | 0.83             | 5.64              |
| 120 | sheet thinning (stripping)  | 0.59             | 4.37              |

T_ini = 1.9 (We - 12)^(-1/4) (1 + 2.2 Oh^1.6), and T_b piecewise in We (see
`analyze_breakup.py`), both in units of t*.

### Water drop in air (`input.2d-water`, `input.3d-water`)

The same setup with real fluids in SI units: a 2 mm water drop in air at
20 C (rho_l = 998.2 kg/m3, mu_l = 1.002e-3 Pa s, sigma = 0.0728 N/m;
rho_g = 1.204 kg/m3, mu_g = 1.81e-5 Pa s), so that the density ratio (829),
the viscosity ratio (55) and the Ohnesorge number (0.0026) match water / air
experiments.  As in an air-jet experiment, the Weber number is set by the air
velocity, so the gas Reynolds number rises with it:

| We  | U (m/s) | Re_g | t_scale = D / U sqrt(rho_l / rho_g) (ms) |
|-----|---------|------|------------------------------------------|
| 5   | 12.29   | 1640 | 4.684                                    |
| 15  | 21.30   | 2830 | 2.704                                    |
| 40  | 34.78   | 4630 | 1.656                                    |
| 120 | 60.23   | 8010 | 0.956                                    |

The input files are set to We = 15; `CASE=water ./run_weber_sweep.sh` sets
`prob.velocity` for each Weber number and rescales the stop (t* = 6), plot
(t* = 0.1) and checkpoint (t* = 1) times.  The domain is the same in units of
D (80 mm x 16 mm x 16 mm in 3D).  Gravity across the stream (Bond number
0.54) is in the inputs but commented out: the drop falls about 0.6 D by
t* = 6 at We = 15.  At We = 120 the gas boundary layer on the drop
(~ D / sqrt(Re_g) = D / 90) is under-resolved at D / dx = 64: use
`amr.max_level=5` (D / dx = 128) if the stripping rate matters.

## Running

```
make -j DIM=3                        # or DIM=2
NP=256 ./run_weber_sweep.sh          # We = 5 15 40 120, one directory each
python3 analyze_breakup.py We*/run.log --dim 3

CASE=water NP=256 ./run_weber_sweep.sh     # water_We5 ... water_We120
python3 analyze_breakup.py water_We*/run.log --dim 3 --case water
```

`run_weber_sweep.sh` sets `peleLM.vof.sigma = 1 / We` for each Weber number
(`WE="..."` to change the list) and passes any extra arguments as overrides,
e.g. `amr.max_level=3` for a test at D / dx = 32.  `MPIRUN` sets the launcher
(e.g. `MPIRUN="srun -n 256"`), `PARALLEL=1` runs the Weber numbers
concurrently.  Plotfiles are written every 0.1 t* and checkpoints every t*.

`analyze_breakup.py` reads the `VOF: time ...` lines that `peleLM.vof.v = 1`
prints after each step and tabulates against t*: the cross-stream and
streamwise extents of the liquid (bounding box of the F > 1/2 cells), the
interface area, the centroid velocity and the drag coefficient from the
momentum balance of the liquid.  The breakup morphology itself (bag, stamen,
rim, ligaments, stripped sheets) is read from the plotfiles (`vof`, the
`vof = 0.5` isosurface).

## What to compare

- The regime at each Weber number: no breakup at We = 5 (the drop deforms
  into an oscillating disc), a bag blown downstream from a rim at We = 15, a
  bag with a central stamen at We = 40, sheets stripped from the rim at
  We = 120.  The critical Weber number is 11 +- 2 at low Oh.
- The initiation and total breakup times against Pilch & Erdman (their
  correlations scatter by about +-30% around the data).
- The cross-stream deformation before breakup (about 2 D for bag breakup)
  and the drag coefficient of the flattened drop.

## Caveats

- Incompressible, with an impulsive start instead of a shock or a jet
  shear layer; the gas Reynolds number (1000) is at the low end of the
  experiments.
- The density ratio is 1000 in `input.*d` (water / air: 830, as in
  `input.*d-water`).
- Bag and sheet films thinner than the cell size rupture numerically in
  VOF: the rupture time and the fragment sizes are resolution dependent,
  the initiation time and the deformation much less so.  Check the
  sensitivity with `amr.max_level=3` (D / dx = 32) against the default.
- Slip side walls block 1.2% of the 3D cross section (6% in 2D).
- The drop travels 20-30 D by t* = 6.  Liquid reaching the last
  `peleLM.vof.outflow_band` cells before the outflow is removed (the
  `VOF: liquid volume` line prints the removed volume): a prescribed outflow
  pressure cannot carry an interface across the boundary.
