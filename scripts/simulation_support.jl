using MHDFlows
using CUDA
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

workspace_root() = isdefined(Main, :repo_root) ? repo_root() : normpath(joinpath(@__DIR__, ".."))

Base.@kwdef struct SimulationConfig
    output_root::String = joinpath(workspace_root(), "outputs")
    solver::String = "mhdflows"
    device::String = "auto"
    apply_cpu_compatibility_shim::Bool = true
    name::String = "compressible_mhd_baseline"
    description::String = "Compressible MHD baseline with snapshots and energy-history support metrics"
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

mutable struct SnapshotDiagnosticsHistory
    sound_speed::Float64
    snapshot_dt::Float64
    next_time::Float64
    next_file_number::Int
    file_numbers::Vector{Int}
    files::Vector{String}
    times::Vector{Float64}
    rho_mean::Vector{Float64}
    velocity_rms::Vector{Float64}
    velocity_fluct_rms::Vector{Float64}
    sonic_mach::Vector{Float64}
    sonic_mach_total::Vector{Float64}
    magnetic_mean_strength::Vector{Float64}
    magnetic_rms_total::Vector{Float64}
    magnetic_rms_fluct::Vector{Float64}
    alfven_speed_mean::Vector{Float64}
    alfven_speed_total::Vector{Float64}
    alfven_speed_fluct::Vector{Float64}
    alfven_mach_mean::Vector{Float64}
    alfven_mach_total::Vector{Float64}
    alfven_mach_fluct::Vector{Float64}
end

SnapshotDiagnosticsHistory(snapshot_dt::Real, sound_speed::Real) = SnapshotDiagnosticsHistory(
    Float64(sound_speed),
    Float64(snapshot_dt),
    Float64(snapshot_dt),
    1,
    Int[],
    String[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
)

function float_tag(x::Real)
    tag = @sprintf("%.4g", float(x))
    return replace(tag, "-" => "m", "." => "p")
end

function case_tag(cfg::SimulationConfig)
    base = "$(cfg.name)_n$(cfg.nx)_cs$(float_tag(cfg.sound_speed))_B0$(float_tag(cfg.mean_field[1]))_nu$(float_tag(cfg.viscosity))_eta$(float_tag(cfg.resistivity))_P$(float_tag(cfg.forcing_power))_T$(float_tag(cfg.end_time))"
    return isempty(cfg.tag_suffix) ? base : "$(base)_$(cfg.tag_suffix)"
end

case_root(cfg::SimulationConfig) = joinpath(cfg.output_root, cfg.solver, case_tag(cfg))
analysis_root(cfg::SimulationConfig) = joinpath(case_root(cfg), "analysis")
figure_root(cfg::SimulationConfig) = joinpath(case_root(cfg), "figures")
snapshot_root(cfg::SimulationConfig) = joinpath(case_root(cfg), "snapshots")

function ensure_case_dirs!(cfg::SimulationConfig)
    mkpath(analysis_root(cfg))
    mkpath(figure_root(cfg))
    mkpath(snapshot_root(cfg))
    return nothing
end

function install_cpu_compatibility_shim!()
    solver = MHDFlows.MHDSolver_compressible
    Core.eval(solver, quote
        using FourierFlows: CPU

        synchronize_if_gpu(grid) = grid.device == CPU() ? nothing : CUDA.synchronize()

        function MHDcalcN_advection!(N, sol, t, clock, vars, params, grid)
            @timeit_debug params.debugTimer "FFT Update" begin
                ldiv!(vars.ρ, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ρ_ind]))
                ldiv!(vars.ux, grid.rfftplan, deepcopy(@view sol[:, :, :, params.ux_ind]))
                ldiv!(vars.uy, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uy_ind]))
                ldiv!(vars.uz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.uz_ind]))
                ldiv!(vars.bx, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bx_ind]))
                ldiv!(vars.by, grid.rfftplan, deepcopy(@view sol[:, :, :, params.by_ind]))
                ldiv!(vars.bz, grid.rfftplan, deepcopy(@view sol[:, :, :, params.bz_ind]))

                @. vars.ux /= vars.ρ
                @. vars.uy /= vars.ρ
                @. vars.uz /= vars.ρ

                mul!(vars.uxh, grid.rfftplan, vars.ux)
                mul!(vars.uyh, grid.rfftplan, vars.uy)
                mul!(vars.uzh, grid.rfftplan, vars.uz)
                synchronize_if_gpu(grid)
            end

            @timeit_debug params.debugTimer "ρ Update" begin
                ρUpdate!(N, sol, t, clock, vars, params, grid)
                synchronize_if_gpu(grid)
            end

            @timeit_debug params.debugTimer "UᵢUpdate" begin
                UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction = "x")
                UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction = "y")
                UᵢUpdate!(N, sol, t, clock, vars, params, grid; direction = "z")
                synchronize_if_gpu(grid)
            end

            @timeit_debug params.debugTimer "BᵢUpdate" begin
                BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction = "x")
                BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction = "y")
                BᵢUpdate!(N, sol, t, clock, vars, params, grid; direction = "z")
                synchronize_if_gpu(grid)
            end
            return nothing
        end
    end)
    return nothing
end

function maybe_install_cpu_compatibility_shim!(cfg::SimulationConfig)
    cfg.apply_cpu_compatibility_shim || return nothing
    install_cpu_compatibility_shim!()
    println("Installed runtime CPU compatibility shim for MHDFlows compressible MHD solver")
    return nothing
end

function choose_device(cfg::SimulationConfig)
    mode = lowercase(strip(cfg.device))
    if mode == "cpu"
        return CPU(), "CPU"
    elseif mode == "gpu"
        CUDA.functional() || error("device = \"gpu\" was requested, but CUDA is not functional on this machine")
        return GPU(), "GPU"
    elseif mode == "auto"
        CUDA.functional() && return GPU(), "GPU"
        return CPU(), "CPU"
    end
    error("device must be \"auto\", \"cpu\", or \"gpu\"; got \"$(cfg.device)\"")
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
    maybe_install_cpu_compatibility_shim!(cfg)
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

safe_ratio(numerator::Real, denominator::Real) = isfinite(Float64(denominator)) && Float64(denominator) > 0 ? Float64(numerator) / Float64(denominator) : NaN

function mhd_energy_diagnostics(rho, ux, uy, uz, bx, by, bz)
    rho_mean = Float64(mean(rho))

    u2 = ux .^ 2 .+ uy .^ 2 .+ uz .^ 2
    kinetic_density = Float64(mean(rho .* u2))

    bx_mean = Float64(mean(bx))
    by_mean = Float64(mean(by))
    bz_mean = Float64(mean(bz))
    b2 = bx .^ 2 .+ by .^ 2 .+ bz .^ 2
    db2 = (bx .- bx_mean) .^ 2 .+ (by .- by_mean) .^ 2 .+ (bz .- bz_mean) .^ 2
    magnetic_total_density = Float64(mean(b2))
    magnetic_fluct_density = Float64(mean(db2))

    kinetic = 0.5 * kinetic_density
    magnetic_total = 0.5 * magnetic_total_density
    magnetic_fluct = 0.5 * magnetic_fluct_density

    return (
        rho_mean = rho_mean,
        kinetic = kinetic,
        magnetic_total = magnetic_total,
        magnetic_fluct = magnetic_fluct,
        total_resolved = kinetic + magnetic_total,
        fluct_total = kinetic + magnetic_fluct,
    )
end

function mhd_field_diagnostics(rho, ux, uy, uz, bx, by, bz, sound_speed::Real)
    rho_mean = Float64(mean(rho))
    sqrt_rho_mean = rho_mean > 0 ? sqrt(rho_mean) : NaN

    u2 = ux .^ 2 .+ uy .^ 2 .+ uz .^ 2
    kinetic_density = Float64(mean(rho .* u2))
    velocity_rms = rho_mean > 0 ? sqrt(max(0.0, kinetic_density / rho_mean)) : NaN

    ux_mean = safe_ratio(mean(rho .* ux), rho_mean)
    uy_mean = safe_ratio(mean(rho .* uy), rho_mean)
    uz_mean = safe_ratio(mean(rho .* uz), rho_mean)
    du2 = (ux .- ux_mean) .^ 2 .+ (uy .- uy_mean) .^ 2 .+ (uz .- uz_mean) .^ 2
    velocity_fluct_density = Float64(mean(rho .* du2))
    velocity_fluct_rms = rho_mean > 0 ? sqrt(max(0.0, velocity_fluct_density / rho_mean)) : NaN

    bx_mean = Float64(mean(bx))
    by_mean = Float64(mean(by))
    bz_mean = Float64(mean(bz))
    b2 = bx .^ 2 .+ by .^ 2 .+ bz .^ 2
    db2 = (bx .- bx_mean) .^ 2 .+ (by .- by_mean) .^ 2 .+ (bz .- bz_mean) .^ 2
    magnetic_total_density = Float64(mean(b2))
    magnetic_fluct_density = Float64(mean(db2))

    kinetic = 0.5 * kinetic_density
    magnetic_total = 0.5 * magnetic_total_density
    magnetic_fluct = 0.5 * magnetic_fluct_density
    magnetic_mean_strength = sqrt(max(0.0, bx_mean ^ 2 + by_mean ^ 2 + bz_mean ^ 2))
    magnetic_rms_total = sqrt(max(0.0, magnetic_total_density))
    magnetic_rms_fluct = sqrt(max(0.0, magnetic_fluct_density))

    alfven_speed_mean = safe_ratio(magnetic_mean_strength, sqrt_rho_mean)
    alfven_speed_total = safe_ratio(magnetic_rms_total, sqrt_rho_mean)
    alfven_speed_fluct = safe_ratio(magnetic_rms_fluct, sqrt_rho_mean)

    return (
        rho_mean = rho_mean,
        kinetic = kinetic,
        magnetic_total = magnetic_total,
        magnetic_fluct = magnetic_fluct,
        total_resolved = kinetic + magnetic_total,
        fluct_total = kinetic + magnetic_fluct,
        velocity_rms = velocity_rms,
        velocity_fluct_rms = velocity_fluct_rms,
        sonic_mach = safe_ratio(velocity_fluct_rms, sound_speed),
        sonic_mach_total = safe_ratio(velocity_rms, sound_speed),
        magnetic_mean_strength = magnetic_mean_strength,
        magnetic_rms_total = magnetic_rms_total,
        magnetic_rms_fluct = magnetic_rms_fluct,
        alfven_speed_mean = alfven_speed_mean,
        alfven_speed_total = alfven_speed_total,
        alfven_speed_fluct = alfven_speed_fluct,
        alfven_mach_mean = safe_ratio(velocity_fluct_rms, alfven_speed_mean),
        alfven_mach_total = safe_ratio(velocity_fluct_rms, alfven_speed_total),
        alfven_mach_fluct = safe_ratio(velocity_fluct_rms, alfven_speed_fluct),
    )
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

    diagnostics = mhd_energy_diagnostics(rho, ux, uy, uz, bx, by, bz)

    push!(history.times, current_time)
    push!(history.rho_mean, diagnostics.rho_mean)
    push!(history.kinetic, diagnostics.kinetic)
    push!(history.magnetic_total, diagnostics.magnetic_total)
    push!(history.magnetic_fluct, diagnostics.magnetic_fluct)
    push!(history.total_resolved, diagnostics.total_resolved)
    push!(history.fluct_total, diagnostics.fluct_total)
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

snapshot_relative_path(file_number::Integer) = "snapshots/state_t_$(lpad(string(file_number), 4, '0')).h5"

function record_snapshot_diagnostics!(history::SnapshotDiagnosticsHistory, prob, file_number::Integer)
    if !isempty(history.file_numbers) && history.file_numbers[end] == Int(file_number)
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

    diagnostics = mhd_field_diagnostics(rho, ux, uy, uz, bx, by, bz, history.sound_speed)

    push!(history.file_numbers, Int(file_number))
    push!(history.files, snapshot_relative_path(file_number))
    push!(history.times, Float64(prob.clock.t))
    push!(history.rho_mean, diagnostics.rho_mean)
    push!(history.velocity_rms, diagnostics.velocity_rms)
    push!(history.velocity_fluct_rms, diagnostics.velocity_fluct_rms)
    push!(history.sonic_mach, diagnostics.sonic_mach)
    push!(history.sonic_mach_total, diagnostics.sonic_mach_total)
    push!(history.magnetic_mean_strength, diagnostics.magnetic_mean_strength)
    push!(history.magnetic_rms_total, diagnostics.magnetic_rms_total)
    push!(history.magnetic_rms_fluct, diagnostics.magnetic_rms_fluct)
    push!(history.alfven_speed_mean, diagnostics.alfven_speed_mean)
    push!(history.alfven_speed_total, diagnostics.alfven_speed_total)
    push!(history.alfven_speed_fluct, diagnostics.alfven_speed_fluct)
    push!(history.alfven_mach_mean, diagnostics.alfven_mach_mean)
    push!(history.alfven_mach_total, diagnostics.alfven_mach_total)
    push!(history.alfven_mach_fluct, diagnostics.alfven_mach_fluct)
    return true
end

function initialize_snapshot_diagnostics!(history::SnapshotDiagnosticsHistory, prob)
    record_snapshot_diagnostics!(history, prob, 0)
    history.next_time = Float64(prob.clock.t) + history.snapshot_dt
    history.next_file_number = 1
    return nothing
end

function record_due_snapshot_diagnostics!(history::SnapshotDiagnosticsHistory, prob)
    if Float64(prob.clock.t) >= history.next_time
        recorded = record_snapshot_diagnostics!(history, prob, history.next_file_number)
        history.next_time += history.snapshot_dt
        history.next_file_number += 1
        return recorded
    end
    return false
end

function snapshot_diagnostics_rows(history::SnapshotDiagnosticsHistory)
    rows = Vector{Dict{String, Any}}()
    for i in eachindex(history.times)
        push!(rows, Dict{String, Any}(
            "file" => history.files[i],
            "time" => history.times[i],
            "rho_mean" => history.rho_mean[i],
            "velocity_fluct_rms" => history.velocity_fluct_rms[i],
            "sonic_mach" => history.sonic_mach[i],
            "magnetic_mean_strength" => history.magnetic_mean_strength[i],
            "magnetic_rms_fluct" => history.magnetic_rms_fluct[i],
            "alfven_mach_mean" => history.alfven_mach_mean[i],
        ))
    end
    return rows
end

function make_snapshot_metadata_callback(history::SnapshotDiagnosticsHistory, metadata_path::String, cfg::SimulationConfig, device_label_ref::Base.RefValue{String})
    return function (prob)
        if record_due_snapshot_diagnostics!(history, prob)
            write_case_metadata(metadata_path, cfg, device_label_ref[]; snapshot_diagnostics = history)
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

function write_case_metadata(path::String, cfg::SimulationConfig, device_label::String; snapshot_diagnostics = nothing)
    metadata = Dict(
        "name" => cfg.name,
        "solver" => cfg.solver,
        "description" => cfg.description,
        "device" => device_label,
        "device_request" => cfg.device,
        "output_root" => cfg.output_root,
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
        "apply_cpu_compatibility_shim" => cfg.apply_cpu_compatibility_shim,
        "seed" => cfg.seed,
    )
    if snapshot_diagnostics !== nothing
        metadata["snapshot_diagnostics"] = snapshot_diagnostics_rows(snapshot_diagnostics)
    end
    open(path, "w") do io
        TOML.print(io, metadata; sorted = true)
    end
    return path
end

as_string(value) = String(value)
as_int(value) = value isa Integer ? Int(value) : parse(Int, String(value))
as_float(value) = value isa Real ? Float64(value) : parse(Float64, String(value))
as_bool(value) = value isa Bool ? value : parse(Bool, lowercase(String(value)))

function as_datatype(value)
    text = String(value)
    text == "Float32" && return Float32
    text == "Float64" && return Float64
    error("float_type must be Float32 or Float64; got $(text)")
end

function as_tuple3(value)
    length(value) == 3 || error("mean_field must have exactly three values")
    values = Float64.(value)
    return (values[1], values[2], values[3])
end

function config_from_sources(settings, positionals::Vector{String})
    length(positionals) <= 9 || error("Expected at most 9 positional overrides. Run with --help for usage.")
    defaults = SimulationConfig()

    cfg = SimulationConfig(;
        output_root = configured_output_root(settings),
        solver = "mhdflows",
        device = as_string(get_config(settings, "device", defaults.device)),
        apply_cpu_compatibility_shim = as_bool(get_config(settings, "apply_cpu_compatibility_shim", defaults.apply_cpu_compatibility_shim)),
        name = as_string(get_config(settings, "name", defaults.name)),
        description = as_string(get_config(settings, "description", defaults.description)),
        nx = as_int(get_config(settings, "nx", defaults.nx)),
        box_size = as_float(get_config(settings, "box_size", defaults.box_size)),
        float_type = as_datatype(get_config(settings, "float_type", string(defaults.float_type))),
        sound_speed = as_float(get_config(settings, "sound_speed", defaults.sound_speed)),
        viscosity = as_float(get_config(settings, "viscosity", defaults.viscosity)),
        resistivity = as_float(get_config(settings, "resistivity", defaults.resistivity)),
        mean_field = as_tuple3(get_config(settings, "mean_field", collect(defaults.mean_field))),
        forcing_wavenumber = as_float(get_config(settings, "forcing_wavenumber", defaults.forcing_wavenumber)),
        forcing_power = as_float(get_config(settings, "forcing_power", defaults.forcing_power)),
        forcing_width = as_float(get_config(settings, "forcing_width", defaults.forcing_width)),
        initial_velocity_power = as_float(get_config(settings, "initial_velocity_power", defaults.initial_velocity_power)),
        fixed_dt = as_float(get_config(settings, "fixed_dt", defaults.fixed_dt)),
        end_time = as_float(get_config(settings, "end_time", defaults.end_time)),
        max_steps = as_int(get_config(settings, "max_steps", defaults.max_steps)),
        energy_sample_every = as_int(get_config(settings, "energy_sample_every", defaults.energy_sample_every)),
        snapshot_dt = as_float(get_config(settings, "snapshot_dt", defaults.snapshot_dt)),
        late_window_fraction = as_float(get_config(settings, "late_window_fraction", defaults.late_window_fraction)),
        stability_rel_band = as_float(get_config(settings, "stability_rel_band", defaults.stability_rel_band)),
        stability_abs_change = as_float(get_config(settings, "stability_abs_change", defaults.stability_abs_change)),
        seed = as_int(get_config(settings, "seed", defaults.seed)),
        tag_suffix = as_string(get_config(settings, "tag_suffix", defaults.tag_suffix)),
    )

    isempty(positionals) && return cfg
    return SimulationConfig(;
        output_root = cfg.output_root,
        solver = cfg.solver,
        device = cfg.device,
        apply_cpu_compatibility_shim = cfg.apply_cpu_compatibility_shim,
        name = cfg.name,
        description = cfg.description,
        nx = length(positionals) >= 1 ? parse(Int, positionals[1]) : cfg.nx,
        box_size = cfg.box_size,
        float_type = cfg.float_type,
        sound_speed = cfg.sound_speed,
        viscosity = length(positionals) >= 4 ? parse(Float64, positionals[4]) : cfg.viscosity,
        resistivity = length(positionals) >= 5 ? parse(Float64, positionals[5]) : cfg.resistivity,
        mean_field = cfg.mean_field,
        forcing_wavenumber = cfg.forcing_wavenumber,
        forcing_power = length(positionals) >= 3 ? parse(Float64, positionals[3]) : cfg.forcing_power,
        forcing_width = cfg.forcing_width,
        initial_velocity_power = cfg.initial_velocity_power,
        fixed_dt = length(positionals) >= 7 ? parse(Float64, positionals[7]) : cfg.fixed_dt,
        end_time = length(positionals) >= 2 ? parse(Float64, positionals[2]) : cfg.end_time,
        max_steps = cfg.max_steps,
        energy_sample_every = cfg.energy_sample_every,
        snapshot_dt = length(positionals) >= 8 ? parse(Float64, positionals[8]) : cfg.snapshot_dt,
        late_window_fraction = cfg.late_window_fraction,
        stability_rel_band = cfg.stability_rel_band,
        stability_abs_change = cfg.stability_abs_change,
        seed = length(positionals) >= 9 ? parse(Int, positionals[9]) : cfg.seed,
        tag_suffix = length(positionals) >= 6 ? positionals[6] : cfg.tag_suffix,
    )
end
