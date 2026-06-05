is_plot_entrypoint = abspath(PROGRAM_FILE) == @__FILE__

if is_plot_entrypoint && !isempty(ARGS) && ARGS[1] in ["-h", "--help"]
    println("Usage:")
    println("  julia plot_kolmogorov_hd.jl [--config config.kolmogorov.example.toml] [case_dir]")
    println()
    println("When case_dir is omitted, the newest Kolmogorov HD case under the configured output_root is used.")
    exit(0)
end

import Pkg

if !isdefined(Main, :resolve_repo_path)
    include(joinpath(@__DIR__, "runner_config.jl"))
end

if is_plot_entrypoint
    config_path_arg, plot_positionals = split_config_args(ARGS)
    length(plot_positionals) <= 1 || error("Expected zero or one case directory. Run with --help for usage.")
    config_path = config_path_arg === nothing ? joinpath(@__DIR__, "config.kolmogorov.example.toml") : config_path_arg
    config_path, settings = load_config(config_path)
    Pkg.activate(mhdflows_project_path(settings))
end

using DelimitedFiles
using FFTW
using HDF5
using Printf
using Statistics
using TOML

if !haskey(ENV, "MPLBACKEND")
    ENV["MPLBACKEND"] = "Agg"
end

import PyPlot

function latest_kolmogorov_case_root(output_root::AbstractString)
    isdir(output_root) || error("No outputs directory exists yet: $(output_root)")
    cases = filter(isdir, readdir(output_root; join = true))
    cases = filter(path -> isfile(joinpath(path, "analysis", "energy_enstrophy_history.csv")), cases)
    isempty(cases) && error("No Kolmogorov HD cases found in $(output_root)")
    sort!(cases; by = path -> stat(path).mtime)
    return cases[end]
end

function read_kolmogorov_history(path::String)
    raw_data, raw_header = readdlm(path, ',', Float64; header = true)
    headers = String.(vec(raw_header))
    columns = Dict(name => index for (index, name) in enumerate(headers))
    required = ["time", "kinetic", "enstrophy", "velocity_rms", "divergence_rms", "forcing_power"]
    missing = filter(name -> !haskey(columns, name), required)
    isempty(missing) || error("Missing required CSV columns: $(join(missing, ", "))")

    data = ndims(raw_data) == 1 ? reshape(raw_data, 1, :) : raw_data
    size(data, 1) > 0 || error("Kolmogorov history is empty: $(path)")
    column(name) = Float64.(data[:, columns[name]])
    return (
        time = column("time"),
        kinetic = column("kinetic"),
        enstrophy = column("enstrophy"),
        velocity_rms = column("velocity_rms"),
        divergence_rms = column("divergence_rms"),
        forcing_power = column("forcing_power"),
    )
end

function snapshot_paths(case_dir::AbstractString)
    snapshot_dir = joinpath(case_dir, "snapshots")
    isdir(snapshot_dir) || error("Snapshot directory does not exist: $(snapshot_dir)")
    paths = filter(path -> endswith(lowercase(path), ".h5"), readdir(snapshot_dir; join = true))
    isempty(paths) && error("No HDF5 snapshots found in $(snapshot_dir)")
    sort!(paths)
    return paths
end

function read_velocity_snapshot(path::String)
    h5open(path, "r") do file
        return (
            ux = Array(read(file, "i_velocity")),
            uy = Array(read(file, "j_velocity")),
            uz = Array(read(file, "k_velocity")),
            time = Float64(read(file, "time")),
        )
    end
end

function centered_periodic_derivative_y(a, dy)
    nx, ny = size(a)
    out = similar(a, Float64)
    for j in 1:ny, i in 1:nx
        jm = j == 1 ? ny : j - 1
        jp = j == ny ? 1 : j + 1
        out[i, j] = (a[i, jp] - a[i, jm]) / (2 * dy)
    end
    return out
end

function centered_periodic_derivative_x(a, dx)
    nx, ny = size(a)
    out = similar(a, Float64)
    for j in 1:ny, i in 1:nx
        im = i == 1 ? nx : i - 1
        ip = i == nx ? 1 : i + 1
        out[i, j] = (a[ip, j] - a[im, j]) / (2 * dx)
    end
    return out
end

function mean_xy_component(a)
    ndims(a) == 2 && return Float64.(a)
    ndims(a) == 3 || error("Expected a 2D or 3D velocity array; got $(ndims(a)) dimensions")
    return Float64.(dropdims(mean(a; dims = 3); dims = 3))
end

function vorticity_z(snapshot, domain_length::Real)
    ux = mean_xy_component(snapshot.ux)
    uy = mean_xy_component(snapshot.uy)
    nx, ny = size(ux)
    dx = Float64(domain_length) / nx
    dy = Float64(domain_length) / ny
    return centered_periodic_derivative_x(uy, dx) .- centered_periodic_derivative_y(ux, dy)
end

function selected_snapshot_indices(n::Int, max_panels::Int = 4)
    count = min(max_panels, n)
    return unique(round.(Int, range(1, n; length = count)))
end

function plot_vorticity_snapshots(case_dir::AbstractString; max_panels::Int = 4)
    metadata = TOML.parsefile(joinpath(case_dir, "analysis", "case_metadata.toml"))
    domain_length = Float64(get(metadata, "domain_length", 1.0))
    paths = snapshot_paths(case_dir)
    selected = paths[selected_snapshot_indices(length(paths), max_panels)]
    snapshots = read_velocity_snapshot.(selected)
    vorticities = [vorticity_z(snapshot, domain_length) for snapshot in snapshots]
    clim = maximum(maximum(abs, omega) for omega in vorticities)
    clim = clim > 0 ? clim : 1.0

    figure_dir = joinpath(case_dir, "figures")
    mkpath(figure_dir)
    output_path = joinpath(figure_dir, "vorticity_snapshots.png")

    figure, axes = PyPlot.subplots(1, length(selected); figsize = (4 * length(selected) + 0.9, 3.6), squeeze = false)
    axes_vec = vec(axes)
    image = nothing
    for (axis, snapshot, omega) in zip(axes_vec, snapshots, vorticities)
        image = axis.imshow(transpose(omega);
            origin = "lower",
            extent = [0, domain_length, 0, domain_length],
            cmap = "RdBu_r",
            vmin = -clim,
            vmax = clim,
            interpolation = "nearest",
            aspect = "equal",
        )
        axis.set_title(@sprintf("t = %.3f", snapshot.time))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    end
    figure.subplots_adjust(right = 0.9, top = 0.78, wspace = 0.28)
    cax = figure.add_axes([0.925, 0.22, 0.014, 0.55])
    cbar = figure.colorbar(image, cax = cax)
    cbar.set_label(raw"$\omega_z$")
    figure.suptitle("Kolmogorov HD vorticity snapshots", y = 0.94)
    figure.savefig(output_path; dpi = 180, bbox_inches = "tight")
    PyPlot.close(figure)
    return output_path
end

function plot_energy_enstrophy_history(case_dir::AbstractString)
    csv_path = joinpath(case_dir, "analysis", "energy_enstrophy_history.csv")
    isfile(csv_path) || error("Kolmogorov history CSV does not exist: $(csv_path)")
    history = read_kolmogorov_history(csv_path)

    figure_dir = joinpath(case_dir, "figures")
    mkpath(figure_dir)
    output_path = joinpath(figure_dir, "energy_enstrophy_history.png")

    figure, axes = PyPlot.subplots(2, 1; figsize = (10, 8), sharex = true)
    energy_axis, enstrophy_axis = axes

    energy_axis.plot(history.time, history.kinetic; color = "#1f77b4", linewidth = 2.0, label = "kinetic energy")
    energy_axis.set_ylabel("kinetic energy")
    energy_axis.grid(; alpha = 0.25)
    energy_axis.legend(; loc = "best")

    enstrophy_axis.plot(history.time, history.enstrophy; color = "#d62728", linewidth = 2.0, label = "enstrophy")
    enstrophy_axis.set_xlabel("time")
    enstrophy_axis.set_ylabel("enstrophy")
    enstrophy_axis.grid(; alpha = 0.25)
    enstrophy_axis.legend(; loc = "best")

    figure.suptitle("Kolmogorov HD energy and enstrophy")
    figure.tight_layout()
    figure.savefig(output_path; dpi = 180, bbox_inches = "tight")
    PyPlot.close(figure)
    return output_path
end

function fft_mode_numbers(n::Int)
    return [i <= div(n, 2) + 1 ? i - 1 : i - 1 - n for i in 1:n]
end

function radial_energy_spectrum(snapshot)
    ux = mean_xy_component(snapshot.ux)
    uy = mean_xy_component(snapshot.uy)
    nx, ny = size(ux)
    uxh = fft(ux) ./ (nx * ny)
    uyh = fft(uy) ./ (nx * ny)
    kx = fft_mode_numbers(nx)
    ky = fft_mode_numbers(ny)
    max_bin = floor(Int, sqrt((nx / 2)^2 + (ny / 2)^2))
    energy = zeros(Float64, max_bin + 1)

    for j in 1:ny, i in 1:nx
        k = sqrt(kx[i]^2 + ky[j]^2)
        bin = round(Int, k)
        if 1 <= bin <= max_bin
            energy[bin + 1] += 0.5 * Float64(abs2(uxh[i, j]) + abs2(uyh[i, j]))
        end
    end

    k_values = collect(1:max_bin)
    return k_values, energy[2:end]
end

function add_reference_slope!(axis, k_values, energy, exponent, label)
    valid = findall(e -> isfinite(e) && e > 0, energy)
    length(valid) >= 3 || return nothing
    anchor = valid[argmax(energy[valid])]
    k0 = k_values[anchor]
    e0 = energy[anchor]
    ref = @. e0 * (k_values / k0)^exponent
    axis.loglog(k_values, ref; color = "0.35", linestyle = "--", linewidth = 1.0, label = label)
    return nothing
end

function plot_final_energy_spectrum(case_dir::AbstractString)
    metadata = TOML.parsefile(joinpath(case_dir, "analysis", "case_metadata.toml"))
    force_mode_y = Int(get(metadata, "force_mode_y", 2))
    paths = snapshot_paths(case_dir)
    snapshot = read_velocity_snapshot(paths[end])
    k_values, energy = radial_energy_spectrum(snapshot)

    figure_dir = joinpath(case_dir, "figures")
    mkpath(figure_dir)
    output_path = joinpath(figure_dir, "energy_spectrum_final.png")

    figure, axis = PyPlot.subplots(1, 1; figsize = (7.5, 5.5))
    positive = energy .> 0
    axis.loglog(k_values[positive], energy[positive]; color = "black", marker = "o", markersize = 3.5, linewidth = 1.5, label = "simulation")
    add_reference_slope!(axis, k_values, energy, -5 / 3, raw"$k^{-5/3}$")
    add_reference_slope!(axis, k_values, energy, -3, raw"$k^{-3}$")
    axis.axvline(force_mode_y; color = "#1f77b4", linestyle = ":", linewidth = 1.3, label = "forcing mode")
    axis.set_xlabel("wavenumber k")
    axis.set_ylabel("energy spectrum E(k)")
    axis.set_title(@sprintf("Final energy spectrum at t = %.3f", snapshot.time))
    axis.grid(; alpha = 0.25, which = "both")
    axis.legend(; loc = "best")
    figure.tight_layout()
    figure.savefig(output_path; dpi = 180, bbox_inches = "tight")
    PyPlot.close(figure)
    return output_path
end

function plot_kolmogorov_hd(case_dir::AbstractString)
    isfile(joinpath(case_dir, "analysis", "case_metadata.toml")) || error("Case metadata does not exist: $(joinpath(case_dir, "analysis", "case_metadata.toml"))")
    return (
        vorticity = plot_vorticity_snapshots(case_dir),
        history = plot_energy_enstrophy_history(case_dir),
        spectrum = plot_final_energy_spectrum(case_dir),
    )
end

if is_plot_entrypoint
    output_root = configured_output_root(settings)
    case_dir = isempty(plot_positionals) ? latest_kolmogorov_case_root(output_root) : resolve_repo_path(plot_positionals[1])
    paths = plot_kolmogorov_hd(case_dir)
    println("Config file: $(config_path)")
    println("Vorticity snapshots figure: $(paths.vorticity)")
    println("Energy/enstrophy figure: $(paths.history)")
    println("Final spectrum figure: $(paths.spectrum)")
end
