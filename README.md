# Kolmogorov HD Simulation

This branch is focused on the incompressible hydrodynamic Kolmogorov-flow
benchmark from arXiv:2507.08972v2. It uses the external `MHDFlows` checkout
through `HDSolver.jl`, with no magnetic field and no compressibility.

The external solver checkout is intentionally not edited by these scripts.
Any compatibility changes used by the runner are installed in memory at runtime.

## Setup

Install Julia, then clone or unpack the upstream `MHDFlows` package into:

```text
MHDFlows_dev-main/
```

That folder is ignored by Git because it is external code. If you keep the
package elsewhere, set `mhdflows_project` in `config.kolmogorov.local.toml`, or
set `MHDFLOWS_PROJECT` before running the scripts.

Instantiate the solver environment once:

```powershell
julia --project=.\MHDFlows_dev-main -e 'using Pkg; Pkg.instantiate()'
```

## Configure

Copy the tracked example config to your ignored local config:

```powershell
Copy-Item .\config.kolmogorov.example.toml .\config.kolmogorov.local.toml
```

Important defaults:

| Parameter | Default |
| --- | --- |
| Domain | `[0, 1]^2` |
| Resolution | `128 x 128 x 2` |
| Force | `f_x = 0.1 sin(4 pi y), f_y = 0, f_z = 0` |
| Reynolds number | `1e6` |
| Viscosity | `1e-6` |
| End time | `5.0` |
| Snapshot interval | `0.5` |

`nz = 2` is a degenerate z embedding. The initial condition and forcing are
z-independent, but the upstream grid constructor requires even grid sizes in
every direction.

The HD solver receives the viscosity as `nu`, so the effective Reynolds number
is `1 / viscosity`. The `reynolds_number` config entry is used to default
`viscosity` when no explicit viscosity is set. If both are set and disagree,
the runner warns and the solver still uses `viscosity`.

## Run

Run with your local config:

```powershell
julia .\run_kolmogorov_hd.jl --config .\config.kolmogorov.local.toml
```

Or run directly from the tracked example config:

```powershell
julia .\run_kolmogorov_hd.jl --config .\config.kolmogorov.example.toml
```

Optional positional overrides:

```text
julia run_kolmogorov_hd.jl [--config config.kolmogorov.example.toml] [nx] [end_time] [force_amplitude] [viscosity] [tag_suffix] [fixed_dt] [snapshot_dt] [seed]
```

The config defaults to `plot_after_run = true` and `reuse_existing_data = true`,
so rerunning the same case tag will reuse existing CSV/snapshots and regenerate
figures instead of repeating the simulation.

Short smoke runs such as `end_time = 0.001` only test plumbing. They are not
long enough to produce developed turbulent structures like the paper figures.

## Plot

To plot an existing case manually:

```powershell
julia .\plot_kolmogorov_hd.jl --config .\config.kolmogorov.example.toml .\outputs\<case-tag>
```

The plotting script writes:

```text
figures/vorticity_snapshots.png
figures/vy_vs_vx_snapshots.png
figures/energy_enstrophy_history.png
figures/energy_spectrum_final.png
```

## Outputs

Each case is written under the configured `output_root`, normally
`outputs/<case-tag>/`:

```text
analysis/case_metadata.toml
analysis/energy_enstrophy_history.csv
figures/*.png
snapshots/state_t_*.h5
```

Generated outputs are ignored by Git because HDF5 snapshots can become large.
`MHDFlows_dev-main/` is also ignored because it is an external package checkout,
not original code from this repository. Local task notes and
`config.kolmogorov.local.toml` are ignored as machine-local working files.
