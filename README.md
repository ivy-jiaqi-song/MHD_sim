# Compressible MHD Simulation Activator

This workspace is a compact launcher for a three-dimensional, isothermal,
compressible MHD turbulence simulation. It expects a local checkout of the
external `MHDFlows` package and writes snapshots plus an energy-history
stability diagnostic.

The activator intentionally does not contain mode decomposition. Snapshots are
kept as HDF5 files so a separate analysis project can consume them later.

## Setup

Install Julia, then clone or unpack the upstream `MHDFlows` package into:

```text
MHDFlows_dev-main/
```

That folder is intentionally ignored by Git because it is external code. If you
keep the package elsewhere, set `mhdflows_project` in `config.local.toml`, or
set `MHDFLOWS_PROJECT` before running the scripts.

Instantiate the solver environment once:

```powershell
julia --project=.\MHDFlows_dev-main -e 'using Pkg; Pkg.instantiate()'
```

The solver supports a single CPU or a single CUDA-capable GPU. The launcher
selects a functional GPU when available and falls back to the CPU otherwise.
For CPU-only machines, the runner installs a small runtime compatibility shim
in memory. It does not edit the external `MHDFlows` checkout on disk.

## Configure

Copy the tracked example config to your ignored local config:

```powershell
Copy-Item .\config.example.toml .\config.local.toml
```

Edit `config.local.toml` for your machine and run settings:

```toml
mhdflows_project = "MHDFlows_dev-main"
output_root = "outputs"
device = "auto" # auto, cpu, or gpu

nx = 128
end_time = 60.0
forcing_power = 8000.0
viscosity = 0.01
resistivity = 0.01
snapshot_dt = 5.0
seed = 1234
```

`config.local.toml` is ignored by Git, so local machine paths and experiment
notes stay out of GitHub.

## Run

Start a simulation with the defaults:

```powershell
julia .\run_simulation.jl
```

Optional positional overrides preserve the order used by the original script:

```text
julia run_simulation.jl [--config config.local.toml] [nx] [end_time] [forcing_power] [viscosity] [resistivity] [tag_suffix] [fixed_dt] [snapshot_dt] [seed]
```

Set `fixed_dt` to `0` to use the solver's adaptive CFL timestep.

The default model is a periodic `128^3` compressible MHD box with:

| Parameter | Default |
| --- | --- |
| Box size | `2pi` |
| Sound speed | `sqrt(2)` |
| Viscosity and resistivity | `0.01`, `0.01` |
| Mean magnetic field | `(1, 0, 0)` |
| Forcing wavenumber | `2` |
| Forcing power | `8000` |
| End time | `60` |
| Snapshot interval | `5` |
| Random seed | `1234` |

## Outputs

Each case is written under the configured `output_root`, normally
`outputs/<case-tag>/`:

```text
analysis/case_metadata.toml
analysis/energy_history.csv
figures/energy_history_support.png
snapshots/state_t_*.h5
```

The CSV is checkpointed while the simulation runs. The PNG is generated after
integration finishes and compares kinetic energy, fluctuating magnetic energy,
and their sum. Its lower panel normalizes each curve against the final 25% of
the sampled time range to make late-time stability easier to inspect.

![Example energy-history stability figure](./docs/energy_history_support_example.png)

Regenerate the figure for the newest completed case:

```powershell
julia .\plot_energy_history.jl
```

Or provide a specific case directory:

```powershell
julia .\plot_energy_history.jl .\outputs\<case-tag>
```

## Repository Notes

Generated outputs are ignored by Git because HDF5 snapshots can become large.
`MHDFlows_dev-main/` is also ignored because it is an external package checkout,
not original code from this repository. `config.local.toml` and `task*.md` are
ignored because they are machine-local working notes/settings.
