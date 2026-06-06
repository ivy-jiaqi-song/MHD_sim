# Kolmogorov HD Simulation

This branch is focused on the hydrodynamic Kolmogorov-flow benchmark from
arXiv:2507.08972v2. It has two package-specific backends:

- `mhdflows`: the original incompressible HD run through the external
  `MHDFlows` checkout and `HDSolver.jl`.
- `athena`: an Athena++ non-magnetic low-Mach isothermal-hydro sanity-check
  proxy.

Athena++ is a compressible hydro/MHD code, so the Athena backend is not an
exact incompressible projection solve. It uses a higher isothermal sound speed
to keep Mach number and density variation small, and it is labeled separately
in output metadata under `outputs/athena/`.

The external solver checkouts are intentionally not edited by these scripts.
MHDFlows compatibility changes are installed in memory at runtime. The Athena
backend copies the Athena++ checkout into an ignored `athena_work/` tree, then
injects the local problem generator into that copy.

## Setup

Install Julia for the MHDFlows backend, then clone or unpack the upstream
`MHDFlows` package into:

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

For the Athena backend, clone or unpack Athena++ and point the config or
environment to it:

```powershell
$env:ATHENA_PROJECT = "D:\path\to\athena"
```

The default config uses `athena_project = "../athena"`, matching this workspace.
On the remote machine, the Athena runner needs Python plus the compiler toolchain
used by Athena++ (`make`, `g++` by default).

## Configure

Copy the tracked example config to your ignored local config:

```powershell
Copy-Item .\config.kolmogorov.example.toml .\config.kolmogorov.local.toml
```

Important defaults:

| Parameter | Default |
| --- | --- |
| Domain | `[0, 1]^2` |
| Resolution | `128 x 128` |
| MHDFlows z embedding | `nz = 2` |
| Athena z embedding | `nx3 = 1` |
| Force | `f_x = 0.1 sin(4 pi y), f_y = 0, f_z = 0` |
| Reynolds number | `1e6` |
| Viscosity | `1e-6` |
| End time | `5.0` |
| Snapshot interval | `0.5` |
| Athena sound speed | `10.0` |

For MHDFlows, `nz = 2` is a degenerate z embedding. The initial condition and
forcing are z-independent, but the upstream grid constructor requires even grid
sizes in every direction.

The MHDFlows HD solver receives the viscosity as `nu`, so the effective Reynolds
number is `1 / viscosity`. The `reynolds_number` config entry is used to default
`viscosity` when no explicit viscosity is set. If both are set and disagree, the
runner warns and the solver still uses `viscosity`.

Generated cases are split by backend when `output_by_backend = true`:

```text
outputs/mhdflows/<case-tag>/
outputs/athena/<case-tag>/
```

For the Athena low-Mach approximation, `athena_iso_sound_speed` controls how
close the compressible run stays to incompressible behavior. Larger values
reduce density variation but also reduce Athena's CFL timestep.

## Run

Use the package-selecting wrapper:

```powershell
bash .\run.sh --backend mhdflows --config .\config.kolmogorov.local.toml
bash .\run.sh --backend athena --config .\config.kolmogorov.local.toml
```

Or run the original MHDFlows Julia entrypoint directly:

```powershell
julia .\run_kolmogorov_hd.jl --config .\config.kolmogorov.local.toml
```

Optional positional overrides:

```text
./run.sh --backend <mhdflows|athena> [--config config.kolmogorov.example.toml] [nx] [end_time] [force_amplitude] [viscosity] [tag_suffix] [fixed_dt] [snapshot_dt] [seed]
```

The config defaults to `plot_after_run = true` and `reuse_existing_data = true`,
so rerunning the same case tag will reuse existing data and regenerate figures
instead of repeating the simulation.

Short smoke runs such as `end_time = 0.001` only test plumbing. They are not
long enough to produce developed turbulent structures like the paper figures.

## Plot

To plot an existing MHDFlows case manually:

```powershell
julia .\plot_kolmogorov_hd.jl --config .\config.kolmogorov.example.toml .\outputs\mhdflows\<case-tag>
```

To plot an existing Athena case manually:

```powershell
python .\backends\athena\plot_athena_kolmogorov_hd.py .\outputs\athena\<case-tag>
```

The plotting scripts write:

```text
figures/vorticity_snapshots.png
figures/vy_vs_vx_snapshots.png
figures/energy_enstrophy_history.png
figures/energy_spectrum_final.png
```

## Outputs

Each case is written under the configured `output_root`, normally
`outputs/<backend>/<case-tag>/`:

```text
analysis/case_metadata.toml
analysis/energy_enstrophy_history.csv
figures/*.png
snapshots/*
```

MHDFlows snapshots are HDF5 files. Athena snapshots are raw Athena++ VTK files,
and its runner converts the Athena history file into the shared
`analysis/energy_enstrophy_history.csv` schema before plotting. Athena analysis
CSV files also include low-Mach diagnostics: density mean/min/max, fractional
density RMS variation, and max Mach number.

Generated outputs are ignored by Git because snapshots can become large.
`MHDFlows_dev-main/` and `athena_work/` are ignored because they are external or
generated solver code, not original code from this repository. Local task notes
and `config.kolmogorov.local.toml` are ignored as machine-local working files.
