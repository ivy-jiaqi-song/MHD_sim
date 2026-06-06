#!/usr/bin/env python3
"""Build and run the Athena++ Kolmogorov-flow sanity-check backend."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PGEN_SOURCE = SCRIPT_DIR / "kolmogorov_hd.cpp"


def load_toml(path: Path) -> dict:
    try:
        import tomllib

        with path.open("rb") as handle:
            return tomllib.load(handle)
    except ModuleNotFoundError:
        return parse_flat_toml(path)


def parse_flat_toml(path: Path) -> dict:
    settings = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
            settings[key] = [parse_value(item) for item in items]
        else:
            settings[key] = parse_value(value)
    return settings


def parse_value(value: str):
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def default_config_path() -> Path:
    local = REPO_ROOT / "config.kolmogorov.local.toml"
    return local if local.is_file() else REPO_ROOT / "config.kolmogorov.example.toml"


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(os.path.expanduser(str(path)))
    return expanded.resolve() if expanded.is_absolute() else (REPO_ROOT / expanded).resolve()


def get_config(settings: dict, key: str, default):
    return settings[key] if key in settings else default


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def float_tag(value: float) -> str:
    return f"{float(value):.4g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class AthenaConfig:
    output_root: Path
    athena_project: Path
    athena_work_root: Path
    name: str
    description: str
    nx: int
    ny: int
    domain_length: float
    reynolds_number: float
    viscosity: float
    force_amplitude: float
    force_mode_y: int
    initial_velocity_rms: float
    initial_modes: int
    end_time: float
    max_steps: int
    fixed_dt: float
    diagnostics_sample_every: int
    snapshot_dt: float
    plot_after_run: bool
    reuse_existing_data: bool
    seed: int
    tag_suffix: str
    iso_sound_speed: float
    cfl_number: float
    configure_args: tuple[str, ...]
    make_jobs: int
    prepare_only: bool
    skip_build: bool
    skip_run: bool

    @property
    def effective_reynolds(self) -> float:
        return 1.0 / self.viscosity if self.viscosity > 0 else float("inf")

    @property
    def case_tag(self) -> str:
        base = (
            f"{self.name}_n{self.nx}x{self.ny}_Re{float_tag(self.effective_reynolds)}"
            f"_A{float_tag(self.force_amplitude)}_k{self.force_mode_y}_T{float_tag(self.end_time)}"
        )
        return base if not self.tag_suffix else f"{base}_{self.tag_suffix}"

    @property
    def case_dir(self) -> Path:
        return self.output_root / self.case_tag

    @property
    def snapshot_dir(self) -> Path:
        return self.case_dir / "snapshots"

    @property
    def analysis_dir(self) -> Path:
        return self.case_dir / "analysis"

    @property
    def figure_dir(self) -> Path:
        return self.case_dir / "figures"

    @property
    def athena_work_dir(self) -> Path:
        return self.athena_work_root / "kolmogorov_hd"


def config_from_sources(settings: dict, positionals: list[str]) -> AthenaConfig:
    if len(positionals) > 8:
        raise SystemExit("Expected at most 8 positional overrides. Run with --help for usage.")

    output_root = resolve_repo_path(str(get_config(settings, "output_root", "outputs")))
    if as_bool(get_config(settings, "output_by_backend", True)):
        output_root = output_root / "athena"

    re_configured = float(get_config(settings, "reynolds_number", 1.0e6))
    viscosity = float(get_config(settings, "viscosity", 1.0 / re_configured))
    nx = int(get_config(settings, "nx", 128))
    ny = int(get_config(settings, "ny", nx))
    fixed_dt = float(get_config(settings, "fixed_dt", 1.0e-3))
    snapshot_dt = float(get_config(settings, "snapshot_dt", 0.5))
    seed = int(get_config(settings, "seed", 1234))

    if len(positionals) >= 1:
        nx = ny = int(positionals[0])
    if len(positionals) >= 2:
        end_time = float(positionals[1])
    else:
        end_time = float(get_config(settings, "end_time", 5.0))
    if len(positionals) >= 3:
        force_amplitude = float(positionals[2])
    else:
        force_amplitude = float(get_config(settings, "force_amplitude", 0.1))
    if len(positionals) >= 4:
        viscosity = float(positionals[3])
    if len(positionals) >= 5:
        tag_suffix = positionals[4]
    else:
        tag_suffix = str(get_config(settings, "tag_suffix", ""))
    if len(positionals) >= 6:
        fixed_dt = float(positionals[5])
    if len(positionals) >= 7:
        snapshot_dt = float(positionals[6])
    if len(positionals) >= 8:
        seed = int(positionals[7])

    configured_athena = str(get_config(settings, "athena_project", "../athena"))
    athena_project = resolve_repo_path(os.environ.get("ATHENA_PROJECT", configured_athena))
    athena_work_root = resolve_repo_path(str(get_config(settings, "athena_work_root", "athena_work")))
    configure_args = tuple(str(arg) for arg in get_config(settings, "athena_configure_args", ["--eos=isothermal", "--flux=hlle"]))

    return AthenaConfig(
        output_root=output_root,
        athena_project=athena_project,
        athena_work_root=athena_work_root,
        name=str(get_config(settings, "name", "kolmogorov_hd")),
        description=str(get_config(settings, "description", "Athena++ Kolmogorov-flow sanity check")),
        nx=nx,
        ny=ny,
        domain_length=float(get_config(settings, "domain_length", 1.0)),
        reynolds_number=re_configured,
        viscosity=viscosity,
        force_amplitude=force_amplitude,
        force_mode_y=int(get_config(settings, "force_mode_y", 2)),
        initial_velocity_rms=float(get_config(settings, "initial_velocity_rms", 1.0e-3)),
        initial_modes=int(get_config(settings, "initial_modes", 4)),
        end_time=end_time,
        max_steps=int(get_config(settings, "max_steps", 100000)),
        fixed_dt=fixed_dt,
        diagnostics_sample_every=int(get_config(settings, "diagnostics_sample_every", 10)),
        snapshot_dt=snapshot_dt,
        plot_after_run=as_bool(get_config(settings, "plot_after_run", True)),
        reuse_existing_data=as_bool(get_config(settings, "reuse_existing_data", True)),
        seed=seed,
        tag_suffix=tag_suffix,
        iso_sound_speed=float(get_config(settings, "athena_iso_sound_speed", 10.0)),
        cfl_number=float(get_config(settings, "athena_cfl_number", 0.3)),
        configure_args=configure_args,
        make_jobs=int(get_config(settings, "athena_make_jobs", os.cpu_count() or 1)),
        prepare_only=as_bool(get_config(settings, "athena_prepare_only", False)),
        skip_build=as_bool(get_config(settings, "athena_skip_build", False)),
        skip_run=as_bool(get_config(settings, "athena_skip_run", False)),
    )


def ensure_athena_project(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"Athena++ project not found at {path}. Set athena_project or ATHENA_PROJECT.")
    if not (path / "configure.py").is_file():
        raise SystemExit(f"Athena++ project is missing configure.py: {path}")
    if not (path / "src" / "pgen").is_dir():
        raise SystemExit(f"Athena++ project is missing src/pgen: {path}")


def ignore_athena_copy(dirname: str, names: list[str]) -> set[str]:
    ignored = {".git", "obj", "bin", "configure.log"}
    if "Makefile" in names:
        ignored.add("Makefile")
    return ignored.intersection(names)


def prepare_athena_tree(cfg: AthenaConfig) -> None:
    ensure_athena_project(cfg.athena_project)
    if not cfg.athena_work_dir.exists():
        cfg.athena_work_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying Athena++ to ignored work tree: {cfg.athena_work_dir}")
        shutil.copytree(cfg.athena_project, cfg.athena_work_dir, ignore=ignore_athena_copy)
    shutil.copy2(PGEN_SOURCE, cfg.athena_work_dir / "src" / "pgen" / "kolmogorov_hd.cpp")


def case_has_data(cfg: AthenaConfig) -> bool:
    return (
        (cfg.analysis_dir / "case_metadata.toml").is_file()
        and any(cfg.snapshot_dir.glob("*.hst"))
        and any(cfg.snapshot_dir.glob("*.vtk"))
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def toml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_metadata(cfg: AthenaConfig) -> Path:
    path = cfg.analysis_dir / "case_metadata.toml"
    domain_volume = cfg.domain_length * cfg.domain_length * cfg.domain_length
    rows = {
        "name": cfg.name,
        "backend": "athena",
        "package": "Athena++",
        "description": cfg.description,
        "athena_project": str(cfg.athena_project),
        "athena_work_dir": str(cfg.athena_work_dir),
        "output_root": str(cfg.output_root),
        "nx": cfg.nx,
        "ny": cfg.ny,
        "nz": 1,
        "domain_length": cfg.domain_length,
        "domain_volume": domain_volume,
        "eos": "isothermal",
        "iso_sound_speed": cfg.iso_sound_speed,
        "reynolds_number_configured": cfg.reynolds_number,
        "reynolds_number_effective": cfg.effective_reynolds,
        "viscosity": cfg.viscosity,
        "force_amplitude": cfg.force_amplitude,
        "force_mode_y": cfg.force_mode_y,
        "force_form": "f_x = force_amplitude * sin(2*pi*force_mode_y*y/domain_length), f_y = 0, f_z = 0",
        "initial_velocity_rms": cfg.initial_velocity_rms,
        "initial_modes": cfg.initial_modes,
        "cfl_number": cfg.cfl_number,
        "end_time": cfg.end_time,
        "max_steps": cfg.max_steps,
        "diagnostics_sample_every": cfg.diagnostics_sample_every,
        "snapshot_dt": cfg.snapshot_dt,
        "plot_after_run": cfg.plot_after_run,
        "reuse_existing_data": cfg.reuse_existing_data,
        "seed": cfg.seed,
        "note": "Athena++ is compressible hydro; this is a low-Mach isothermal, non-magnetic proxy sanity check, not an exact incompressible projection solve.",
    }
    lines = []
    for key in sorted(rows):
        value = rows[key]
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = toml_quote(str(value))
        lines.append(f"{key} = {rendered}")
    write_text(path, "\n".join(lines) + "\n")
    return path


def athinput_text(cfg: AthenaConfig) -> str:
    problem_id = (cfg.snapshot_dir / "athena_kolmogorov").resolve()
    history_dt = max(cfg.fixed_dt * cfg.diagnostics_sample_every, 1.0e-12)
    return f"""<comment>
problem   = Athena++ Kolmogorov HD sanity check
configure = --prob=kolmogorov_hd --eos=isothermal

<job>
problem_id = {problem_id}

<output1>
file_type   = hst
dt          = {history_dt:.16g}
data_format = %24.16e

<output2>
file_type = vtk
variable  = v
dt        = {cfg.snapshot_dt:.16g}

<time>
cfl_number = {cfg.cfl_number:.16g}
nlim       = {cfg.max_steps}
tlim       = {cfg.end_time:.16g}
integrator = vl2
xorder     = 2
ncycle_out = 10

<mesh>
nx1        = {cfg.nx}
x1min      = 0.0
x1max      = {cfg.domain_length:.16g}
ix1_bc     = periodic
ox1_bc     = periodic

nx2        = {cfg.ny}
x2min      = 0.0
x2max      = {cfg.domain_length:.16g}
ix2_bc     = periodic
ox2_bc     = periodic

nx3        = 1
x3min      = 0.0
x3max      = {cfg.domain_length:.16g}
ix3_bc     = periodic
ox3_bc     = periodic

refinement = none

<meshblock>
nx1 = {cfg.nx}
nx2 = {cfg.ny}
nx3 = 1

<hydro>
gamma           = 1.6666666666666667
iso_sound_speed = {cfg.iso_sound_speed:.16g}
nu_iso         = {cfg.viscosity:.16g}

<problem>
density              = 1.0
pressure             = 1.0
force_amplitude      = {cfg.force_amplitude:.16g}
force_mode_y         = {cfg.force_mode_y}
initial_velocity_rms = {cfg.initial_velocity_rms:.16g}
initial_modes        = {cfg.initial_modes}
seed                 = {cfg.seed}
"""


def run_command(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def configure_and_build(cfg: AthenaConfig) -> None:
    configure = [sys.executable, "configure.py", "--prob=kolmogorov_hd", *cfg.configure_args]
    run_command(configure, cfg.athena_work_dir)
    if not cfg.skip_build:
        run_command(["make", f"-j{cfg.make_jobs}"], cfg.athena_work_dir)


def run_athena(cfg: AthenaConfig, input_path: Path) -> None:
    executable = cfg.athena_work_dir / "bin" / "athena"
    if not executable.is_file():
        raise SystemExit(f"Athena++ executable not found: {executable}. Build first or unset athena_skip_build.")
    run_command([str(executable), "-i", str(input_path)], cfg.athena_work_dir)


def plot_case(cfg: AthenaConfig) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    from plot_athena_kolmogorov_hd import plot_athena_kolmogorov_hd

    paths = plot_athena_kolmogorov_hd(cfg.case_dir)
    print(f"Vorticity snapshots figure: {paths['vorticity']}")
    print(f"v_y vs v_x snapshots figure: {paths['velocity_phase']}")
    print(f"Energy/enstrophy figure: {paths['history']}")
    print(f"Final spectrum figure: {paths['spectrum']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true", help="write Athena work tree/input/metadata but do not configure, build, run, or plot")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("positionals", nargs="*")
    args = parser.parse_args()

    config_path = (args.config.resolve() if args.config else default_config_path().resolve())
    settings = load_toml(config_path)
    cfg = config_from_sources(settings, args.positionals)
    if args.prepare_only:
        cfg = cfg.__class__(**{**cfg.__dict__, "prepare_only": True})
    if args.skip_build:
        cfg = cfg.__class__(**{**cfg.__dict__, "skip_build": True})
    if args.skip_run:
        cfg = cfg.__class__(**{**cfg.__dict__, "skip_run": True})

    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)

    print("Running Athena++ Kolmogorov HD sanity-check backend")
    print(f"Config file: {config_path}")
    print(f"Athena++ project: {cfg.athena_project}")
    print(f"Athena++ work tree: {cfg.athena_work_dir}")
    print(f"Case directory: {cfg.case_dir}")
    print("Note: Athena++ is compressible; this backend uses low-Mach isothermal HD as a proxy sanity check.")

    prepare_athena_tree(cfg)
    metadata_path = write_metadata(cfg)
    input_path = cfg.analysis_dir / "athinput.kolmogorov_hd"
    write_text(input_path, athinput_text(cfg))
    print(f"Metadata: {metadata_path}")
    print(f"Athena input: {input_path}")

    if cfg.prepare_only:
        print("Prepare-only mode requested; stopping before configure/build/run.")
        return

    if args.plot_only:
        plot_case(cfg)
        return

    if cfg.reuse_existing_data and case_has_data(cfg):
        print("Existing Athena++ Kolmogorov HD data found; skipping simulation")
        if cfg.plot_after_run:
            plot_case(cfg)
        return

    configure_and_build(cfg)
    if cfg.skip_run:
        print("athena_skip_run requested; stopping after configure/build.")
        return

    run_athena(cfg, input_path)
    if cfg.plot_after_run:
        plot_case(cfg)


if __name__ == "__main__":
    main()
