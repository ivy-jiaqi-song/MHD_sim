is_plot_entrypoint = abspath(PROGRAM_FILE) == @__FILE__

if is_plot_entrypoint && !isempty(ARGS) && ARGS[1] in ["-h", "--help"]
    println("Usage:")
    println("  julia scripts/plot_energy_history.jl [--config configs/config.local.toml] [case_dir]")
    println()
    println("When case_dir is omitted, the newest case under the configured output_root is used.")
    exit(0)
end

import Pkg

if !isdefined(Main, :resolve_repo_path)
    include(joinpath(@__DIR__, "runner_config.jl"))
end

if is_plot_entrypoint
    config_path_arg, plot_positionals = split_config_args(ARGS)
    length(plot_positionals) <= 1 || error("Expected zero or one case directory. Run with --help for usage.")
    config_path, settings = load_config(config_path_arg)
    Pkg.activate(mhdflows_project_path(settings))
end

using DelimitedFiles
using Printf
using Statistics
using TOML

if !haskey(ENV, "MPLBACKEND")
    ENV["MPLBACKEND"] = "Agg"
end

import PyPlot

function latest_case_root(output_root::AbstractString)
    isdir(output_root) || error("No outputs directory exists yet: $(output_root)")
    cases = filter(isdir, readdir(output_root; join = true))
    isempty(cases) && error("No simulation cases found in $(output_root)")
    sort!(cases; by = path -> stat(path).mtime)
    return cases[end]
end

function read_energy_history(path::String)
    raw_data, raw_header = readdlm(path, ',', Float64; header = true)
    headers = String.(vec(raw_header))
    columns = Dict(name => index for (index, name) in enumerate(headers))
    required = ["time", "kinetic", "magnetic_fluct", "fluct_total"]
    missing = filter(name -> !haskey(columns, name), required)
    isempty(missing) || error("Missing required CSV columns: $(join(missing, ", "))")

    data = ndims(raw_data) == 1 ? reshape(raw_data, 1, :) : raw_data
    size(data, 1) > 0 || error("Energy history is empty: $(path)")
    column(name) = Float64.(data[:, columns[name]])
    return (
        time = column("time"),
        kinetic = column("kinetic"),
        magnetic_fluct = column("magnetic_fluct"),
        fluct_total = column("fluct_total"),
    )
end

function late_window_indices(times::Vector{Float64}, fraction::Float64)
    0 < fraction <= 1 || error("late_window_fraction must be in the interval (0, 1]")
    start_time = times[1] + (1 - fraction) * (times[end] - times[1])
    indices = findall(t -> t >= start_time, times)
    isempty(indices) && error("No samples fall within the configured late-time window")
    return indices, times[indices[1]]
end

function normalized_to_window(values::Vector{Float64}, indices::Vector{Int})
    reference = mean(values[indices])
    reference != 0 || error("Cannot normalize an energy series with a zero late-window mean")
    return values ./ reference
end

function plot_energy_history(case_dir::AbstractString)
    csv_path = joinpath(case_dir, "analysis", "energy_history.csv")
    metadata_path = joinpath(case_dir, "analysis", "case_metadata.toml")
    isfile(csv_path) || error("Energy history CSV does not exist: $(csv_path)")
    isfile(metadata_path) || error("Case metadata does not exist: $(metadata_path)")

    history = read_energy_history(csv_path)
    metadata = TOML.parsefile(metadata_path)
    late_fraction = Float64(get(metadata, "late_window_fraction", 0.25))
    indices, late_start = late_window_indices(history.time, late_fraction)

    kinetic_normalized = normalized_to_window(history.kinetic, indices)
    magnetic_normalized = normalized_to_window(history.magnetic_fluct, indices)
    total_normalized = normalized_to_window(history.fluct_total, indices)

    figure_dir = joinpath(case_dir, "figures")
    mkpath(figure_dir)
    output_path = joinpath(figure_dir, "energy_history_support.png")

    figure, axes = PyPlot.subplots(2, 1; figsize = (12, 10), sharex = true)
    top, bottom = axes

    top.plot(history.time, history.fluct_total; color = "black", linewidth = 2.0, label = raw"$E_{\mathrm{tot,fluc}}$")
    top.plot(history.time, history.kinetic; color = "#1f77b4", linewidth = 1.8, label = raw"$E_k$")
    top.plot(history.time, history.magnetic_fluct; color = "#d62728", linestyle = "--", linewidth = 1.8, label = raw"$E_b$")
    top.axvline(late_start; color = "0.35", linestyle = "--", linewidth = 1.3, label = @sprintf("late window start = %.2f", late_start))
    top.set_ylabel("energy")
    top.legend(; loc = "upper left", ncol = 2)

    bottom.plot(history.time, total_normalized; color = "black", linewidth = 2.0, label = raw"$E_{\mathrm{tot,fluc}} / \langle E_{\mathrm{tot,fluc}} \rangle$")
    bottom.plot(history.time, kinetic_normalized; color = "#1f77b4", linewidth = 1.8, label = raw"$E_k / \langle E_k \rangle$")
    bottom.plot(history.time, magnetic_normalized; color = "#d62728", linestyle = "--", linewidth = 1.8, label = raw"$E_b / \langle E_b \rangle$")
    bottom.axvline(late_start; color = "0.35", linestyle = "--", linewidth = 1.3)
    bottom.axhline(1.0; color = "0.55", linestyle = ":", linewidth = 1.1)
    bottom.set_xlabel("time")
    bottom.set_ylabel("late-window normalized")
    bottom.legend(; loc = "upper left")

    for axis in axes
        axis.grid(; alpha = 0.25)
    end

    nx = Int(get(metadata, "nx", 0))
    top.set_title(@sprintf("Energy support | N=%d, late-window diagnostics for t >= %.2f", nx, late_start))
    figure.tight_layout()
    figure.savefig(output_path; dpi = 180, bbox_inches = "tight")
    PyPlot.close(figure)
    return output_path
end

if is_plot_entrypoint
    output_root = configured_output_root(settings)
    case_dir = isempty(plot_positionals) ? latest_case_root(output_root) : resolve_repo_path(plot_positionals[1])
    println("Config file: $(config_path)")
    println("Energy history figure: $(plot_energy_history(case_dir))")
end
