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

The normal runner is foreground: it prints progress in the current terminal and
returns only when the simulation finishes. For long runs, start it in the
background and write logs under `logs/`.

On Windows PowerShell:

```powershell
.\run_background.ps1 -Config .\config.local.toml
```

For a quick background smoke test with positional overrides:

```powershell
.\run_background.ps1 -Config .\config.local.toml -- 8 0.001 10 0.01 0.01 smoke 0.001 0.01 1234
```

On Linux/macOS or a remote shell:

```bash
./run_background.sh --config config.local.toml
```

Both wrappers print a process ID and log file path. The `logs/` directory is
ignored by Git.

Device selection comes from `device` in `config.local.toml`:

```toml
device = "auto" # auto, cpu, or gpu
```

`gpu` means the `GPU()` device used by `MHDFlows`/`FourierFlows` with CUDA.jl.
`auto` uses GPU only when CUDA is functional, otherwise CPU.

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

`energy_history.csv` also records basic Mach diagnostics for follow-up
statistics. The runner uses the density-weighted turbulent velocity

```text
u_rms = sqrt(<rho |u - <u>_rho|^2> / <rho>)
```

and the configured isothermal sound speed:

```text
M_s = u_rms / c_s
```

For Alfvenic Mach numbers, the relevant definition is velocity divided by an
Alfven speed, not `<B> / B`:

```text
v_A(B_ref) = B_ref / sqrt(<rho>)
M_A(B_ref) = u_rms / v_A(B_ref)
```

This matches the code normalization used by the existing magnetic energy
diagnostic, `0.5 * <|B|^2>`.

The CSV records three useful choices for `B_ref`:

| Column | Magnetic reference |
| --- | --- |
| `alfven_mach_mean` | `|<B>|`, the guide/mean field |
| `alfven_mach_total` | `sqrt(<|B|^2>)`, mean plus fluctuations |
| `alfven_mach_fluct` | `sqrt(<|B - <B>|^2>)`, fluctuations only |

It also records `velocity_rms`, `velocity_fluct_rms`, `sonic_mach`,
`sonic_mach_total`, `magnetic_mean_strength`, `magnetic_rms_total`,
`magnetic_rms_fluct`, and the corresponding Alfven speeds. `NaN` means the
chosen magnetic reference is zero, for example the fluctuating field at the
initial snapshot.

Existing `.h5` snapshots are sufficient for post-processing these diagnostics:
they contain `gas_density`, `i_velocity`, `j_velocity`, `k_velocity`,
`i_mag_field`, `j_mag_field`, `k_mag_field`, and `time`. For sonic Mach, also
read `sound_speed` from `analysis/case_metadata.toml` or from the config used
for that run.

To compare runs, change the input parameters and let the diagnostics measure
the resulting state. `sound_speed` directly changes `M_s`; larger `c_s` lowers
`M_s` for the same turbulent velocity. `mean_field` directly changes the
mean-field Alfven speed; a stronger guide field lowers `alfven_mach_mean`.
`forcing_power` changes the driven turbulent velocity and therefore changes
both `M_s` and `M_A`. `viscosity` and `resistivity` mainly change dissipation,
Reynolds number, and saturation behavior, so they affect Mach numbers
indirectly rather than serving as clean Mach-number knobs.

![Example energy-history stability figure](./docs/energy_history_support_example.png)

Regenerate the figure for the newest completed case:

```powershell
julia .\plot_energy_history.jl
```

Or provide a specific case directory:

```powershell
julia .\plot_energy_history.jl .\outputs\<case-tag>
```

## 2D Kolmogorov HD Side Run

The side-task runner `run_kolmogorov_hd.jl` sets up the incompressible
hydrodynamic Kolmogorov-flow benchmark from arXiv:2507.08972v2 without editing
the external `MHDFlows_dev-main/` checkout. It uses `HDSolver.jl` through
`MHDFlows.Problem` with no magnetic field and no compressibility. Because the
underlying `FourierFlows.ThreeDGrid` requires even grid sizes in every
direction, the runner uses `nz = 2` as a degenerate z embedding while keeping
the initial condition and forcing z-independent. It injects the steady force

```text
f_x = 0.1 sin(4 pi y), f_y = 0, f_z = 0
```

on `[0, 1]^2` with `Re = 1e6`.

The config sets `disable_package_dealiasing = true` because the upstream HD
problem constructor always builds a grid with the default dealiasing policy.
That default is tuned for the usual `2*pi` box and removes the low physical
modes when the paper's `[0, 1]^2` domain is used directly. The runner installs
this as an in-memory shim only; it does not edit `MHDFlows_dev-main/`.

Run with the tracked example config:

```powershell
julia .\run_kolmogorov_hd.jl --config .\config.kolmogorov.example.toml
```

For local changes, copy it to `config.kolmogorov.local.toml`; that file is
ignored by Git. Outputs go to `outputs/<case-tag>/analysis/` and
`outputs/<case-tag>/snapshots/`.

## Repository Notes

Generated outputs are ignored by Git because HDF5 snapshots can become large.
`MHDFlows_dev-main/` is also ignored because it is an external package checkout,
not original code from this repository. `config.local.toml` and `task*.md` are
ignored because they are machine-local working notes/settings.
