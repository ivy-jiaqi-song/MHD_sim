#!/usr/bin/env python3
"""Plot Athena++ Kolmogorov-flow outputs into the same figure set as MHDFlows."""

from __future__ import annotations

import argparse
import math
import re
import struct
from pathlib import Path

import numpy as np


SNAPSHOT_PANEL_COUNT = 6
VTK_TIME_RE = re.compile(r"time=([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)")


def parse_simple_toml(path: Path) -> dict:
    settings = {}
    if not path.exists():
        return settings
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value.startswith('"') and value.endswith('"'):
            settings[key] = value[1:-1]
        elif value.lower() in {"true", "false"}:
            settings[key] = value.lower() == "true"
        else:
            try:
                settings[key] = int(value)
            except ValueError:
                try:
                    settings[key] = float(value)
                except ValueError:
                    settings[key] = value
    return settings


def parse_hst(path: Path) -> tuple[list[str], np.ndarray]:
    header = None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and "[1]=time" in stripped:
            header = stripped
        elif not stripped.startswith("#"):
            rows.append([float(value) for value in stripped.split()])
    if header is None:
        raise RuntimeError(f"Could not find Athena++ history header in {path}")
    names_by_index = {
        int(index): name for index, name in re.findall(r"\[(\d+)\]=(\S+)", header)
    }
    names = [names_by_index[i] for i in range(1, max(names_by_index) + 1)]
    return names, np.array(rows, dtype=float)


def history_column(names: list[str], data: np.ndarray, name: str, default=0.0) -> np.ndarray:
    if name not in names:
        if np.isscalar(default):
            return np.full(data.shape[0], default, dtype=float)
        return np.asarray(default, dtype=float)
    return data[:, names.index(name)]


def write_analysis_csv(case_dir: Path, metadata: dict) -> Path:
    hst_files = sorted((case_dir / "snapshots").glob("*.hst"))
    if not hst_files:
        raise RuntimeError(f"No Athena++ .hst file found in {case_dir / 'snapshots'}")
    names, data = parse_hst(hst_files[0])
    volume = float(metadata.get("domain_volume", 1.0))
    volume = volume if volume > 0 else 1.0

    time = history_column(names, data, "time")
    mass = history_column(names, data, "mass", volume)
    density_mean = mass / volume
    kinetic = (
        history_column(names, data, "1-KE")
        + history_column(names, data, "2-KE")
        + history_column(names, data, "3-KE")
    )
    enstrophy = history_column(names, data, "enstrophy") / volume
    velocity_rms = np.sqrt(np.maximum(2.0 * kinetic / np.maximum(mass, 1.0e-300), 0.0))
    max_velocity = np.sqrt(np.maximum(history_column(names, data, "maxvel2"), 0.0))
    divergence_rms = np.sqrt(np.maximum(history_column(names, data, "div2") / volume, 0.0))
    forcing_power = history_column(names, data, "forcepow") / volume
    density_min = history_column(names, data, "rho_min", density_mean)
    density_max = history_column(names, data, "rho_max", density_mean)
    density_squared_mean = history_column(names, data, "rho2", density_mean * density_mean * volume) / volume
    density_rms = np.sqrt(np.maximum(density_squared_mean - density_mean * density_mean, 0.0))
    density_rms_fraction = density_rms / np.maximum(np.abs(density_mean), 1.0e-300)
    default_mach = max_velocity / max(float(metadata.get("iso_sound_speed", 1.0)), 1.0e-300)
    max_mach = history_column(names, data, "maxmach", default_mach)

    analysis_dir = case_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "energy_enstrophy_history.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "time,kinetic,enstrophy,velocity_rms,max_velocity,divergence_rms,"
            "forcing_power,density_mean,density_min,density_max,density_rms_fraction,max_mach\n"
        )
        for row in zip(
            time,
            kinetic / volume,
            enstrophy,
            velocity_rms,
            max_velocity,
            divergence_rms,
            forcing_power,
            density_mean,
            density_min,
            density_max,
            density_rms_fraction,
            max_mach,
        ):
            handle.write(",".join(f"{value:.16g}" for value in row) + "\n")
    return csv_path


def read_line(handle) -> str:
    line = handle.readline()
    if not line:
        return ""
    return line.decode("ascii").strip()


def consume_newline(handle) -> None:
    pos = handle.tell()
    char = handle.read(1)
    if char not in {b"\n", b"\r"}:
        handle.seek(pos)
        return
    if char == b"\r":
        pos = handle.tell()
        nxt = handle.read(1)
        if nxt != b"\n":
            handle.seek(pos)


def read_vtk_time(path: Path) -> float:
    with path.open("rb") as handle:
        for _ in range(3):
            line = read_line(handle)
            match = VTK_TIME_RE.search(line)
            if match:
                return float(match.group(1))
    return math.nan


def read_big_endian_floats(handle, count: int) -> np.ndarray:
    raw = handle.read(4 * count)
    if len(raw) != 4 * count:
        raise RuntimeError("Unexpected end of VTK binary payload")
    consume_newline(handle)
    return np.frombuffer(raw, dtype=">f4").astype(float)


def read_vtk(path: Path) -> dict:
    with path.open("rb") as handle:
        time = math.nan
        for _ in range(3):
            line = read_line(handle)
            match = VTK_TIME_RE.search(line)
            if match:
                time = float(match.group(1))
        dataset = read_line(handle)
        if dataset != "DATASET RECTILINEAR_GRID":
            raise RuntimeError(f"Unsupported VTK dataset in {path}: {dataset}")

        dims_line = read_line(handle)
        _, nx_coord, ny_coord, nz_coord = dims_line.split()
        nx_coord, ny_coord, nz_coord = int(nx_coord), int(ny_coord), int(nz_coord)
        coords = []
        for expected in ("X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"):
            parts = read_line(handle).split()
            if parts[0] != expected:
                raise RuntimeError(f"Expected {expected} in {path}, got {' '.join(parts)}")
            coords.append(read_big_endian_floats(handle, int(parts[1])))

        cell_data = read_line(handle).split()
        if cell_data[0] != "CELL_DATA":
            raise RuntimeError(f"Expected CELL_DATA in {path}")
        cell_count = int(cell_data[1])
        nx = nx_coord - 1 if nx_coord > 1 else 1
        ny = ny_coord - 1 if ny_coord > 1 else 1
        nz = nz_coord - 1 if nz_coord > 1 else 1
        fields = {}

        while True:
            line = read_line(handle)
            if not line:
                break
            parts = line.split()
            if parts[0] == "SCALARS":
                lookup = read_line(handle)
                if not lookup.startswith("LOOKUP_TABLE"):
                    raise RuntimeError(f"Expected LOOKUP_TABLE after SCALARS in {path}")
                values = read_big_endian_floats(handle, cell_count)
                fields[parts[1]] = values.reshape((nz, ny, nx))
            elif parts[0] == "VECTORS":
                values = read_big_endian_floats(handle, cell_count * 3)
                fields[parts[1]] = values.reshape((nz, ny, nx, 3))
            else:
                raise RuntimeError(f"Unsupported VTK field declaration in {path}: {line}")

    if "vel" not in fields:
        raise RuntimeError(f"VTK file does not contain velocity vector 'vel': {path}")
    vel = fields["vel"][0]
    return {
        "time": time,
        "x": coords[0],
        "y": coords[1],
        "vx": vel[:, :, 0],
        "vy": vel[:, :, 1],
    }


def snapshot_paths(case_dir: Path) -> list[Path]:
    paths = sorted((case_dir / "snapshots").glob("*.vtk"))
    if not paths:
        raise RuntimeError(f"No Athena++ VTK snapshots found in {case_dir / 'snapshots'}")
    return paths


def snapshot_time_records(paths: list[Path]) -> list[tuple[Path, float]]:
    records = []
    for fallback_index, path in enumerate(paths):
        time = read_vtk_time(path)
        if not math.isfinite(time):
            time = float(fallback_index)
        records.append((path, time))
    return sorted(records, key=lambda record: (record[1], str(record[0])))


def select_snapshot_paths(paths: list[Path], max_panels: int = SNAPSHOT_PANEL_COUNT) -> list[Path]:
    if max_panels <= 0 or not paths:
        return []

    records = snapshot_time_records(paths)
    count = min(max_panels, len(records))
    if count == len(records):
        return [path for path, _ in records]
    if count == 1:
        return [records[0][0]]

    times = [time for _, time in records]
    targets = np.linspace(times[0], times[-1], count)
    selected = {0, len(records) - 1}
    available = range(1, len(records) - 1)
    for target in targets[1:-1]:
        for index in sorted(available, key=lambda i: (abs(times[i] - target), i)):
            if index not in selected:
                selected.add(index)
                break

    return [records[index][0] for index in sorted(selected)]


def periodic_derivative_x(values: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * dx)


def periodic_derivative_y(values: np.ndarray, dy: float) -> np.ndarray:
    return (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2.0 * dy)


def vorticity(snapshot: dict) -> np.ndarray:
    x = snapshot["x"]
    y = snapshot["y"]
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    return periodic_derivative_x(snapshot["vy"], dx) - periodic_derivative_y(snapshot["vx"], dy)


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def subplot_grid(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    return 2, int(math.ceil(n / 2))


def hide_unused_axes(axes: np.ndarray, used: int) -> None:
    for axis in axes.ravel()[used:]:
        axis.set_visible(False)


def plot_field_snapshots(
    case_dir: Path,
    values_fn,
    cmap: str,
    colorbar_label: str,
    output_name: str,
    symmetric: bool = False,
) -> Path:
    plt = import_pyplot()
    paths = snapshot_paths(case_dir)
    selected = select_snapshot_paths(paths)
    snapshots = [read_vtk(path) for path in selected]
    fields = [values_fn(snapshot) for snapshot in snapshots]
    rows, cols = subplot_grid(len(snapshots))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows), squeeze=False)
    if symmetric:
        clim = max(float(np.max(np.abs(field))) for field in fields) or 1.0
        vmin, vmax = -clim, clim
    else:
        vmin = min(float(np.min(field)) for field in fields)
        vmax = max(float(np.max(field)) for field in fields)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

    image = None
    for axis, snapshot, field in zip(axes.ravel(), snapshots, fields):
        image = axis.imshow(
            field,
            origin="lower",
            extent=[snapshot["x"][0], snapshot["x"][-1], snapshot["y"][0], snapshot["y"][-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"t = {snapshot['time']:.3f}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    hide_unused_axes(axes, len(snapshots))
    fig.subplots_adjust(right=0.9, wspace=0.18, hspace=0.22)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(colorbar_label)
    output = case_dir / "figures" / output_name
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_vorticity_snapshots(case_dir: Path) -> Path:
    return plot_field_snapshots(
        case_dir,
        vorticity,
        "RdBu_r",
        r"$\omega_z$",
        "vorticity_snapshots.png",
        symmetric=True,
    )


def velocity_magnitude(snapshot: dict) -> np.ndarray:
    return np.sqrt(snapshot["vx"] ** 2 + snapshot["vy"] ** 2)


def velocity_x(snapshot: dict) -> np.ndarray:
    return snapshot["vx"]


def velocity_y(snapshot: dict) -> np.ndarray:
    return snapshot["vy"]


def plot_velocity_magnitude_snapshots(case_dir: Path) -> Path:
    return plot_field_snapshots(
        case_dir,
        velocity_magnitude,
        "viridis",
        "Velocity Magnitude",
        "velocity_magnitude_snapshots.png",
    )


def plot_velocity_x_snapshots(case_dir: Path) -> Path:
    return plot_field_snapshots(
        case_dir,
        velocity_x,
        "RdBu_r",
        r"$v_x$",
        "velocity_x_snapshots.png",
        symmetric=True,
    )


def plot_velocity_y_snapshots(case_dir: Path) -> Path:
    return plot_field_snapshots(
        case_dir,
        velocity_y,
        "RdBu_r",
        r"$v_y$",
        "velocity_y_snapshots.png",
        symmetric=True,
    )


def plot_velocity_phase_snapshots(case_dir: Path) -> Path:
    plt = import_pyplot()
    paths = snapshot_paths(case_dir)
    selected = select_snapshot_paths(paths)
    snapshots = [read_vtk(path) for path in selected]
    vmax = max(
        max(float(np.max(np.abs(snapshot["vx"]))), float(np.max(np.abs(snapshot["vy"]))))
        for snapshot in snapshots
    ) or 1.0
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots) + 0.6, 3.6), squeeze=False)
    for axis, snapshot in zip(axes.ravel(), snapshots):
        axis.scatter(snapshot["vx"].ravel(), snapshot["vy"].ravel(), s=2.0, color="#1f77b4", alpha=0.35, linewidths=0.0)
        axis.axhline(0.0, color="0.75", linewidth=0.8)
        axis.axvline(0.0, color="0.75", linewidth=0.8)
        axis.set_xlim(-vmax, vmax)
        axis.set_ylim(-vmax, vmax)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"t = {snapshot['time']:.3f}")
        axis.set_xlabel(r"$v_x$")
        axis.set_ylabel(r"$v_y$")
        axis.grid(alpha=0.18)
    fig.suptitle(r"Athena++ Kolmogorov HD $v_y$ vs $v_x$ snapshots", y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    output = case_dir / "figures" / "vy_vs_vx_snapshots.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def read_analysis_csv(path: Path) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.shape == ():
        raw = np.array([raw], dtype=raw.dtype)
    return {name: raw[name] for name in raw.dtype.names}


def plot_energy_enstrophy_history(case_dir: Path) -> Path:
    plt = import_pyplot()
    history = read_analysis_csv(case_dir / "analysis" / "energy_enstrophy_history.csv")
    fig, (energy_axis, enstrophy_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    energy_axis.plot(history["time"], history["kinetic"], color="#1f77b4", linewidth=2.0, label="kinetic energy")
    energy_axis.set_ylabel("kinetic energy")
    energy_axis.grid(alpha=0.25)
    energy_axis.legend(loc="best")
    enstrophy_axis.plot(history["time"], history["enstrophy"], color="#d62728", linewidth=2.0, label="enstrophy")
    enstrophy_axis.set_xlabel("time")
    enstrophy_axis.set_ylabel("enstrophy")
    enstrophy_axis.grid(alpha=0.25)
    enstrophy_axis.legend(loc="best")
    fig.suptitle("Athena++ Kolmogorov HD energy and enstrophy")
    fig.tight_layout()
    output = case_dir / "figures" / "energy_enstrophy_history.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def radial_energy_spectrum(snapshot: dict) -> tuple[np.ndarray, np.ndarray]:
    vx = snapshot["vx"]
    vy = snapshot["vy"]
    ny, nx = vx.shape
    vxh = np.fft.fft2(vx) / (nx * ny)
    vyh = np.fft.fft2(vy) / (nx * ny)
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    max_bin = int(math.floor(math.sqrt((nx / 2.0) ** 2 + (ny / 2.0) ** 2)))
    energy = np.zeros(max_bin + 1)
    for j, ky_value in enumerate(ky):
        for i, kx_value in enumerate(kx):
            bin_index = int(round(math.sqrt(kx_value * kx_value + ky_value * ky_value)))
            if 1 <= bin_index <= max_bin:
                energy[bin_index] += 0.5 * (abs(vxh[j, i]) ** 2 + abs(vyh[j, i]) ** 2)
    return np.arange(1, max_bin + 1), energy[1:]


def add_reference_slope(axis, k_values: np.ndarray, energy: np.ndarray, exponent: float, label: str) -> None:
    valid = np.where(np.isfinite(energy) & (energy > 0))[0]
    if len(valid) < 3:
        return
    anchor = valid[np.argmax(energy[valid])]
    k0 = k_values[anchor]
    e0 = energy[anchor]
    axis.loglog(k_values, e0 * (k_values / k0) ** exponent, color="0.35", linestyle="--", linewidth=1.0, label=label)


def plot_final_energy_spectrum(case_dir: Path, metadata: dict) -> Path:
    plt = import_pyplot()
    snapshot = read_vtk(snapshot_paths(case_dir)[-1])
    k_values, energy = radial_energy_spectrum(snapshot)
    positive = energy > 0
    fig, axis = plt.subplots(1, 1, figsize=(7.5, 5.5))
    axis.loglog(k_values[positive], energy[positive], color="black", marker="o", markersize=3.5, linewidth=1.5, label="simulation")
    add_reference_slope(axis, k_values, energy, -5 / 3, r"$k^{-5/3}$")
    add_reference_slope(axis, k_values, energy, -3, r"$k^{-3}$")
    axis.axvline(float(metadata.get("force_mode_y", 2)), color="#1f77b4", linestyle=":", linewidth=1.3, label="forcing mode")
    axis.set_xlabel("wavenumber k")
    axis.set_ylabel("energy spectrum E(k)")
    axis.set_title(f"Final energy spectrum at t = {snapshot['time']:.3f}")
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="best")
    fig.tight_layout()
    output = case_dir / "figures" / "energy_spectrum_final.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_athena_kolmogorov_hd(case_dir: Path) -> dict[str, Path]:
    metadata = parse_simple_toml(case_dir / "analysis" / "case_metadata.toml")
    (case_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_analysis_csv(case_dir, metadata)
    return {
        "vorticity": plot_vorticity_snapshots(case_dir),
        "velocity_magnitude": plot_velocity_magnitude_snapshots(case_dir),
        "velocity_x": plot_velocity_x_snapshots(case_dir),
        "velocity_y": plot_velocity_y_snapshots(case_dir),
        "velocity_phase": plot_velocity_phase_snapshots(case_dir),
        "history": plot_energy_enstrophy_history(case_dir),
        "spectrum": plot_final_energy_spectrum(case_dir, metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    paths = plot_athena_kolmogorov_hd(args.case_dir.resolve())
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
