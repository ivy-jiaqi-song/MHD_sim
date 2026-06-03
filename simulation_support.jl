using MHDFlows
using CUDA
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

Base.@kwdef struct SimulationConfig
    name::String = "compressible_mhd_baseline"
    description::String = "Compressible MHD baseline with snapshots and energy-history diagnostics"
    nx::Int = 128
    box_size::Float64 = 2pi
    float_type::DataType = Float32
    sound_speed::Float64 = sqrt(2.0)
    viscosity::Float64 = 1.0e-2
    resistivity::Float64 = 1.0e-2
    mean_field::NTuple{3, Float64} = (1.0, 0.0, 0.0)
    forcing_wavenumber::Float64 = 2.0
    forcing_power::Float64 = 8.0e3
    forcing_width::Float64 = 1.0
    initial_velocity_power::Float64 = 5.0e-4
    fixed_dt::Float64 = 0.0
    end_time::Float64 = 60.0
    max_steps::Int = 80_000
    energy_sample_every::Int = 10
    snapshot_dt::Float64 = 5.0
    late_window_fraction::Float64 = 0.25
    stability_rel_band::Float64 = 0.12
    stability_abs_change::Float64 = 0.06
    prefer_gpu::Bool = true
    seed::Int = 1234
    tag_suffix::String = ""
end

mutable struct EnergyHistory
    sample_every::Int
    times::Vector{Float64}
    rho_mean::Vector{Float64}
    kinetic::Vector{Float64}
    magnetic_total::Vector{Float64}
    magnetic_fluct::Vector{Float64}
    total_resolved::Vector{Float64}
    fluct_total::Vector{Float64}
end

EnergyHistory(sample_every::Int) = EnergyHistory(
    sample_every,
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
)

workspace_root() = @__DIR__
outputs_root() = joinpath(workspace_root(), "outputs")

function float_tag(x::Real)
    tag = @sprintf("%.4g", float(x))
    return replace(tag, "-" => "m", "." => "p")
end

function case_tag(cfg::SimulationConfig)
    base = "$(cfg.name)_n$(cfg.nx)_cs$(float_tag(cfg.sound_speed))_B0$(float_tag(cfg.mean_field[1]))_nu$(float_tag(cfg.viscosity))_eta$(float_tag(cfg.resistivity))_P$(float_tag(cfg.forcing_power))_T$(float_tag(cfg.end_time))"
    return isempty(cfg.tag_suffix) ? base : "$(base)_$(cfg.tag_suffix)"
end

case_root(cfg::SimulationConfig) = joinpath(outputs_root(), case_tag(cfg))
analysis_root(cfg::SimulationConfig) = joinpath(case_root(cfg), "analysis")
figure_root(cfg::SimulationConfig) = joinpath(case_root(cfg), "figures")
snapshot_root(cfg::SimulationConfig) = joinpath(case_root(cfg), "snapshots")

function ensure_case_dirs!(cfg::SimulationConfig)
    mkpath(analysis_root(cfg))
    mkpath(figure_root(cfg))
    mkpath(snapshot_root(cfg))
    return nothing
end

function choose_device(cfg::SimulationConfig)
    if cfg.prefer_gpu && CUDA.functional()
        return GPU(), "GPU"
    end
    return CPU(), "CPU"
end

function seed_problem!(cfg::SimulationConfig, device_label::String)
    Random.seed!(cfg.seed)
    if device_label == "GPU"
        try
            CUDA.seed!(cfg.seed)
        catch
        end
    end
    return nothing
end

function build_problem(cfg::SimulationConfig; usr_func = [])
    dev, device_label = choose_device(cfg)
    seed_problem!(cfg, device_label)

    n = cfg.nx
    T = cfg.float_type
    usr_vars, calcF! = MHDFlows.GetA99LSvars_And_function(dev, n, n, n; T = T, C = true)

    prob = MHDFlows.Problem(dev;
        nx = n,
        ny = n,
        nz = n,
        Lx = T(cfg.box_size),
        Ly = T(cfg.box_size),
        Lz = T(cfg.box_size),
        cₛ = T(cfg.sound_speed),
        ν = T(cfg.viscosity),
        η = T(cfg.resistivity),
        B_field = true,
        Compressibility = true,
        calcF = calcF!,
        usr_vars = usr_vars,
        usr_func = usr_func,
        T = T,
    )

    MHDFlows.SetUpLSFk(prob;
        kf = T(cfg.forcing_wavenumber),
        P = T(cfg.forcing_power),
        σ² = T(cfg.forcing_width),
    )

    rho = ones(T, n, n, n)
    ux, uy, uz = MHDFlows.DivFreeSpectraMap(n, n, n;
        Lx = cfg.box_size,
        dev = CPU(),
        P = cfg.initial_velocity_power,
        T = T,
        k_peak = 1.0,
    )
    bx = fill(T(cfg.mean_field[1]), n, n, n)
    by = fill(T(cfg.mean_field[2]), n, n, n)
    bz = fill(T(cfg.mean_field[3]), n, n, n)

    MHDFlows.SetUpProblemIC!(prob; ρ = rho, ux = ux, uy = uy, uz = uz, bx = bx, by = by, bz = bz)
    return prob, device_label
end

function sync_real_state!(prob)
    prob.flag.c || error("Energy sampling requires a compressible problem")

    sol = prob.sol
    vars = prob.vars
    params = prob.params
    grid = prob.grid

    ldiv!(vars.ρ, grid.rfftplan, copy(@view sol[:, :, :, params.ρ_ind]))
    ldiv!(vars.ux, grid.rfftplan, copy(@view sol[:, :, :, params.ux_ind]))
    ldiv!(vars.uy, grid.rfftplan, copy(@view sol[:, :, :, params.uy_ind]))
    ldiv!(vars.uz, grid.rfftplan, copy(@view sol[:, :, :, params.uz_ind]))
    ldiv!(vars.bx, grid.rfftplan, copy(@view sol[:, :, :, params.bx_ind]))
    ldiv!(vars.by, grid.rfftplan, copy(@view sol[:, :, :, params.by_ind]))
    ldiv!(vars.bz, grid.rfftplan, copy(@view sol[:, :, :, params.bz_ind]))

    @. vars.ux = vars.ux / vars.ρ
    @. vars.uy = vars.uy / vars.ρ
    @. vars.uz = vars.uz / vars.ρ
    return nothing
end

function sample_energy!(history::EnergyHistory, prob; force::Bool = false)
    if !force && prob.clock.step % history.sample_every != 0
        return false
    end

    current_time = Float64(prob.clock.t)
    if !isempty(history.times) && isapprox(history.times[end], current_time; atol = 1.0e-12, rtol = 0.0)
        return false
    end

    sync_real_state!(prob)

    rho = Array(prob.vars.ρ)
    ux = Array(prob.vars.ux)
    uy = Array(prob.vars.uy)
    uz = Array(prob.vars.uz)
    bx = Array(prob.vars.bx)
    by = Array(prob.vars.by)
    bz = Array(prob.vars.bz)

    kinetic = 0.5 * mean(rho .* (ux .^ 2 .+ uy .^ 2 .+ uz .^ 2))
    magnetic_total = 0.5 * mean(bx .^ 2 .+ by .^ 2 .+ bz .^ 2)
    magnetic_fluct = 0.5 * mean(
        (bx .- mean(bx)) .^ 2 .+
        (by .- mean(by)) .^ 2 .+
        (bz .- mean(bz)) .^ 2
    )

    push!(history.times, current_time)
    push!(history.rho_mean, Float64(mean(rho)))
    push!(history.kinetic, Float64(kinetic))
    push!(history.magnetic_total, Float64(magnetic_total))
    push!(history.magnetic_fluct, Float64(magnetic_fluct))
    push!(history.total_resolved, Float64(kinetic + magnetic_total))
    push!(history.fluct_total, Float64(kinetic + magnetic_fluct))
    return true
end

function write_energy_csv(path::String, history::EnergyHistory)
    open(path, "w") do io
        println(io, "time,rho_mean,kinetic,magnetic_total,magnetic_fluct,total_resolved,fluct_total")
        for i in eachindex(history.times)
            println(io,
                "$(history.times[i]),$(history.rho_mean[i]),$(history.kinetic[i]),$(history.magnetic_total[i]),$(history.magnetic_fluct[i]),$(history.total_resolved[i]),$(history.fluct_total[i])")
        end
    end
    return path
end

function make_history_callback(history::EnergyHistory, csv_path::String)
    return function (prob)
        if sample_energy!(history, prob)
            write_energy_csv(csv_path, history)
        end
        return nothing
    end
end

function late_window_stats(history::EnergyHistory, cfg::SimulationConfig)
    length(history.times) < 3 && return nothing

    t_start = history.times[1] + (1 - cfg.late_window_fraction) * (history.times[end] - history.times[1])
    selection = findall(t -> t >= t_start, history.times)
    length(selection) < 3 && return nothing

    energies = history.fluct_total[selection]
    mean_energy = mean(energies)
    mean_energy <= 0 && return nothing

    return (
        t_start = history.times[selection[1]],
        band = (maximum(energies) - minimum(energies)) / mean_energy,
        signed_change = (energies[end] - energies[1]) / mean_energy,
    )
end

function stability_detected(history::EnergyHistory, cfg::SimulationConfig)
    stats = late_window_stats(history, cfg)
    stats === nothing && return false
    return stats.band <= cfg.stability_rel_band && abs(stats.signed_change) <= cfg.stability_abs_change
end

function write_case_metadata(path::String, cfg::SimulationConfig, device_label::String)
    metadata = Dict(
        "name" => cfg.name,
        "description" => cfg.description,
        "device" => device_label,
        "nx" => cfg.nx,
        "box_size" => cfg.box_size,
        "float_type" => string(cfg.float_type),
        "sound_speed" => cfg.sound_speed,
        "viscosity" => cfg.viscosity,
        "resistivity" => cfg.resistivity,
        "mean_field" => collect(cfg.mean_field),
        "forcing_wavenumber" => cfg.forcing_wavenumber,
        "forcing_power" => cfg.forcing_power,
        "forcing_width" => cfg.forcing_width,
        "initial_velocity_power" => cfg.initial_velocity_power,
        "fixed_dt" => cfg.fixed_dt,
        "end_time" => cfg.end_time,
        "max_steps" => cfg.max_steps,
        "energy_sample_every" => cfg.energy_sample_every,
        "snapshot_dt" => cfg.snapshot_dt,
        "late_window_fraction" => cfg.late_window_fraction,
        "stability_rel_band" => cfg.stability_rel_band,
        "stability_abs_change" => cfg.stability_abs_change,
        "seed" => cfg.seed,
    )
    open(path, "w") do io
        TOML.print(io, metadata; sorted = true)
    end
    return path
end

function config_from_args(args::Vector{String})
    length(args) <= 9 || error("Expected at most 9 positional arguments. Run with --help for usage.")
    defaults = SimulationConfig()
    return SimulationConfig(;
        nx = length(args) >= 1 ? parse(Int, args[1]) : defaults.nx,
        end_time = length(args) >= 2 ? parse(Float64, args[2]) : defaults.end_time,
        forcing_power = length(args) >= 3 ? parse(Float64, args[3]) : defaults.forcing_power,
        viscosity = length(args) >= 4 ? parse(Float64, args[4]) : defaults.viscosity,
        resistivity = length(args) >= 5 ? parse(Float64, args[5]) : defaults.resistivity,
        tag_suffix = length(args) >= 6 ? args[6] : defaults.tag_suffix,
        fixed_dt = length(args) >= 7 ? parse(Float64, args[7]) : defaults.fixed_dt,
        snapshot_dt = length(args) >= 8 ? parse(Float64, args[8]) : defaults.snapshot_dt,
        seed = length(args) >= 9 ? parse(Int, args[9]) : defaults.seed,
    )
end
