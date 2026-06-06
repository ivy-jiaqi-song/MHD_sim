using MHDFlows
using CUDA
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

Base.@kwdef struct KolmogorovHDConfig
    backend::String = "mhdflows"
    output_root::String = joinpath(@__DIR__, "outputs")
    device::String = "auto"
    name::String = "kolmogorov_hd"
    description::String = "2D incompressible Kolmogorov flow benchmark from arXiv:2507.08972v2"
    nx::Int = 128
    ny::Int = nx
    nz::Int = 2
    domain_length::Float64 = 1.0
    float_type::DataType = Float32
    reynolds_number::Float64 = 1.0e6
    viscosity::Float64 = 1.0e-6
    force_amplitude::Float64 = 0.1
    force_mode_y::Int = 2
    disable_package_dealiasing::Bool = true
    initial_condition::String = "fourier_divfree"
    initial_velocity_rms::Float64 = 1.0e-3
    grid_noise_velocity_amplitude::Float64 = 0.1
    initial_modes::Int = 4
    fixed_dt::Float64 = 1.0e-3
    end_time::Float64 = 5.0
    max_steps::Int = 100_000
    diagnostics_sample_every::Int = 10
    snapshot_dt::Float64 = 0.5
    plot_after_run::Bool = true
    reuse_existing_data::Bool = true
    seed::Int = 1234
    tag_suffix::String = ""
end

mutable struct KolmogorovHistory
    sample_every::Int
    times::Vector{Float64}
    kinetic::Vector{Float64}
    enstrophy::Vector{Float64}
    velocity_rms::Vector{Float64}
    max_velocity::Vector{Float64}
    divergence_rms::Vector{Float64}
    forcing_power::Vector{Float64}
end

KolmogorovHistory(sample_every::Int) = KolmogorovHistory(
    sample_every,
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
)

mutable struct SnapshotWriter
    prefix::String
    dt::Float64
    next_time::Float64
    count::Int
    last_time::Float64
end

function float_tag(x::Real)
    tag = @sprintf("%.4g", float(x))
    return replace(tag, "-" => "m", "." => "p")
end

effective_reynolds(cfg::KolmogorovHDConfig) = cfg.viscosity > 0 ? 1.0 / cfg.viscosity : Inf

function initial_condition_case_suffix(cfg::KolmogorovHDConfig)
    mode = lowercase(strip(cfg.initial_condition))
    if mode in ("fourier_divfree", "fourier-divfree", "divfree", "divergence_free")
        return ""
    elseif mode in ("grid_noise", "grid-noise", "noise")
        return "_ICgridnoise_U$(float_tag(cfg.grid_noise_velocity_amplitude))"
    end
    return "_IC$(replace(mode, r"[^a-z0-9]+" => ""))"
end

function warn_reynolds_viscosity_mismatch(cfg::KolmogorovHDConfig)
    if cfg.reynolds_number <= 0 || cfg.viscosity <= 0
        return nothing
    end

    effective_re = effective_reynolds(cfg)
    if !isapprox(effective_re, cfg.reynolds_number; rtol = 1.0e-8, atol = 0.0)
        @warn "Configured reynolds_number does not match viscosity; the HD solver uses viscosity, so effective Re is 1 / viscosity" configured_reynolds_number = cfg.reynolds_number viscosity = cfg.viscosity effective_reynolds = effective_re
    end
    return nothing
end

function kolmogorov_case_tag(cfg::KolmogorovHDConfig)
    base = "$(cfg.name)_n$(cfg.nx)x$(cfg.ny)_Re$(float_tag(effective_reynolds(cfg)))_A$(float_tag(cfg.force_amplitude))_k$(cfg.force_mode_y)_T$(float_tag(cfg.end_time))$(initial_condition_case_suffix(cfg))"
    return isempty(cfg.tag_suffix) ? base : "$(base)_$(cfg.tag_suffix)"
end

kolmogorov_case_root(cfg::KolmogorovHDConfig) = joinpath(cfg.output_root, kolmogorov_case_tag(cfg))
kolmogorov_analysis_root(cfg::KolmogorovHDConfig) = joinpath(kolmogorov_case_root(cfg), "analysis")
kolmogorov_snapshot_root(cfg::KolmogorovHDConfig) = joinpath(kolmogorov_case_root(cfg), "snapshots")

function kolmogorov_case_has_data(case_dir::AbstractString)
    csv_path = joinpath(case_dir, "analysis", "energy_enstrophy_history.csv")
    metadata_path = joinpath(case_dir, "analysis", "case_metadata.toml")
    snapshot_dir = joinpath(case_dir, "snapshots")
    isfile(csv_path) || return false
    isfile(metadata_path) || return false
    isdir(snapshot_dir) || return false
    return any(path -> endswith(lowercase(path), ".h5"), readdir(snapshot_dir; join = true))
end

function ensure_kolmogorov_case_dirs!(cfg::KolmogorovHDConfig)
    mkpath(kolmogorov_analysis_root(cfg))
    mkpath(kolmogorov_snapshot_root(cfg))
    return nothing
end

function choose_kolmogorov_device(cfg::KolmogorovHDConfig)
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

function seed_kolmogorov!(cfg::KolmogorovHDConfig, device_label::String)
    Random.seed!(cfg.seed)
    if device_label == "GPU"
        try
            CUDA.seed!(cfg.seed)
        catch
        end
    end
    return nothing
end

function make_kolmogorov_force!(cfg::KolmogorovHDConfig)
    force_cache = Ref{Any}(nothing)

    return function kolmogorov_force!(N, sol, t, clock, vars, params, grid)
        if force_cache[] === nothing
            force_cache[] = build_kolmogorov_force_cache(N, vars, params, grid, cfg)
        end

        Fxh = force_cache[]
        @views @. N[:, :, :, params.ux_ind] += Fxh
        return nothing
    end
end

function build_kolmogorov_force_cache(N, vars, params, grid, cfg::KolmogorovHDConfig)
    T = eltype(grid)
    fx = similar(vars.ux)
    fx_host = zeros(T, grid.nx, grid.ny, grid.nz)
    y = Float64.(Array(grid.y))
    Ly = Float64(grid.Ly)

    for k in 1:grid.nz, j in 1:grid.ny
        value = T(cfg.force_amplitude * sin(2 * pi * cfg.force_mode_y * y[j] / Ly))
        @views fx_host[:, j, k] .= value
    end

    copyto!(fx, fx_host)
    Fxh = similar(@view N[:, :, :, params.ux_ind])
    mul!(Fxh, grid.rfftplan, fx)
    return Fxh
end

function normalize_initial_velocity!(ux, uy, cfg::KolmogorovHDConfig)
    ux .-= eltype(ux)(mean(ux))
    uy .-= eltype(uy)(mean(uy))
    current_rms = sqrt(Float64(mean(ux .^ 2 .+ uy .^ 2)))
    if current_rms > 0
        scale = eltype(ux)(cfg.initial_velocity_rms / current_rms)
        ux .*= scale
        uy .*= scale
    end
    return ux, uy
end

function scale_grid_noise_velocity!(ux, uy, cfg::KolmogorovHDConfig)
    ux .-= eltype(ux)(mean(ux))
    uy .-= eltype(uy)(mean(uy))
    scale = eltype(ux)(cfg.grid_noise_velocity_amplitude)
    ux .*= scale
    uy .*= scale
    return ux, uy
end

function random_divfree_2d_initial_condition(cfg::KolmogorovHDConfig)
    T = cfg.float_type
    ux = zeros(T, cfg.nx, cfg.ny, cfg.nz)
    uy = zeros(T, cfg.nx, cfg.ny, cfg.nz)
    uz = zeros(T, cfg.nx, cfg.ny, cfg.nz)
    x = [(i - 1) * cfg.domain_length / cfg.nx for i in 1:cfg.nx]
    y = [(j - 1) * cfg.domain_length / cfg.ny for j in 1:cfg.ny]
    modes = max(1, cfg.initial_modes)

    for mx in 1:modes, my in 1:modes
        coeff = randn() / sqrt(mx^2 + my^2)
        phase = 2 * pi * rand()
        kx = 2 * pi * mx / cfg.domain_length
        ky = 2 * pi * my / cfg.domain_length

        for j in 1:cfg.ny, i in 1:cfg.nx
            arg = kx * x[i] + ky * y[j] + phase
            c = cos(arg)
            ux_value = coeff * ky * c
            uy_value = -coeff * kx * c
            for k in 1:cfg.nz
                ux[i, j, k] += T(ux_value)
                uy[i, j, k] += T(uy_value)
            end
        end
    end

    normalize_initial_velocity!(ux, uy, cfg)

    return ux, uy, uz
end

function random_grid_noise_initial_condition(cfg::KolmogorovHDConfig)
    T = cfg.float_type
    ux = zeros(T, cfg.nx, cfg.ny, cfg.nz)
    uy = zeros(T, cfg.nx, cfg.ny, cfg.nz)
    uz = zeros(T, cfg.nx, cfg.ny, cfg.nz)

    for j in 1:cfg.ny, i in 1:cfg.nx
        ux_value = T(centered_grid_noise(cfg.seed, i, j, 1))
        uy_value = T(centered_grid_noise(cfg.seed, i, j, 2))
        for k in 1:cfg.nz
            ux[i, j, k] = ux_value
            uy[i, j, k] = uy_value
        end
    end

    scale_grid_noise_velocity!(ux, uy, cfg)

    return ux, uy, uz
end

function unit_grid_phase(seed::Int, i::Int, j::Int, which::Int)
    x = sin(Float64(seed * 12 + i * 78 + j * 37 + which * 19) * 12.9898) * 43758.5453
    return x - floor(x)
end

centered_grid_noise(seed::Int, i::Int, j::Int, which::Int) = 2 * unit_grid_phase(seed, i, j, which) - 1

function kolmogorov_initial_condition(cfg::KolmogorovHDConfig)
    mode = lowercase(strip(cfg.initial_condition))
    if mode in ("fourier_divfree", "fourier-divfree", "divfree", "divergence_free")
        return random_divfree_2d_initial_condition(cfg)
    elseif mode in ("grid_noise", "grid-noise", "noise")
        return random_grid_noise_initial_condition(cfg)
    end
    error("initial_condition must be \"fourier_divfree\" or \"grid_noise\"; got \"$(cfg.initial_condition)\"")
end

function build_kolmogorov_problem(cfg::KolmogorovHDConfig; usr_func = [])
    maybe_install_hd_no_dealias_shim!(cfg)
    dev, device_label = choose_kolmogorov_device(cfg)
    seed_kolmogorov!(cfg, device_label)

    T = cfg.float_type
    calcF! = make_kolmogorov_force!(cfg)
    kwargs = Dict{Symbol, Any}(
        :nx => cfg.nx,
        :ny => cfg.ny,
        :nz => cfg.nz,
        :Lx => T(cfg.domain_length),
        :Ly => T(cfg.domain_length),
        :Lz => T(cfg.domain_length),
        Symbol("\u03bd") => T(cfg.viscosity),
        :B_field => false,
        :Compressibility => false,
        :calcF => calcF!,
        :usr_func => usr_func,
        :T => T,
    )

    prob = MHDFlows.Problem(dev; kwargs...)
    ux, uy, uz = kolmogorov_initial_condition(cfg)
    set_hd_velocity_ic!(prob; ux = ux, uy = uy, uz = uz)
    return prob, device_label
end

function install_hd_no_dealias_shim!()
    Core.eval(MHDFlows, quote
        function HDcalcN!(N, sol, t, clock, vars, params, grid)
            HDSolver.HDcalcN_advection!(N, sol, t, clock, vars, params, grid)
            addforcing!(N, sol, t, clock, vars, params, grid)
            return nothing
        end
    end)
    return nothing
end

function maybe_install_hd_no_dealias_shim!(cfg::KolmogorovHDConfig)
    cfg.disable_package_dealiasing || return nothing
    install_hd_no_dealias_shim!()
    println("Installed runtime no-dealias shim for incompressible HD Kolmogorov run")
    return nothing
end

function set_hd_velocity_ic!(prob; ux, uy, uz)
    vars = prob.vars
    params = prob.params
    grid = prob.grid

    for (u, prob_u, u_ind) in zip((ux, uy, uz), (vars.ux, vars.uy, vars.uz), (params.ux_ind, params.uy_ind, params.uz_ind))
        copyto!(prob_u, u)
        sol_component = @view prob.sol[:, :, :, u_ind]
        mul!(sol_component, grid.rfftplan, prob_u)
        ldiv!(prob_u, grid.rfftplan, copy(sol_component))
    end

    return nothing
end

function sync_hd_real_state!(prob)
    sol = prob.sol
    vars = prob.vars
    params = prob.params
    grid = prob.grid

    ldiv!(vars.ux, grid.rfftplan, copy(@view sol[:, :, :, params.ux_ind]))
    ldiv!(vars.uy, grid.rfftplan, copy(@view sol[:, :, :, params.uy_ind]))
    ldiv!(vars.uz, grid.rfftplan, copy(@view sol[:, :, :, params.uz_ind]))
    return nothing
end

function forcing_power(ux, cfg::KolmogorovHDConfig)
    nx, ny, nz = size(ux)
    total = 0.0

    for k in 1:nz, j in 1:ny
        y = (j - 1) * cfg.domain_length / ny
        force = cfg.force_amplitude * sin(2 * pi * cfg.force_mode_y * y / cfg.domain_length)
        for i in 1:nx
            total += Float64(ux[i, j, k]) * force
        end
    end

    return total / (nx * ny * nz)
end

function spectral_hd_diagnostics(prob)
    sol = prob.sol
    params = prob.params
    grid = prob.grid

    uxh = @view sol[:, :, :, params.ux_ind]
    uyh = @view sol[:, :, :, params.uy_ind]
    uzh = @view sol[:, :, :, params.uz_ind]
    omega_h = similar(uxh)
    div_h = similar(uxh)
    omega = similar(prob.vars.ux)
    div = similar(prob.vars.ux)

    @. omega_h = im * grid.kr * uyh - im * grid.l * uxh
    @. div_h = im * (grid.kr * uxh + grid.l * uyh + grid.m * uzh)
    ldiv!(omega, grid.rfftplan, omega_h)
    ldiv!(div, grid.rfftplan, div_h)

    omega_host = Array(omega)
    div_host = Array(div)
    return 0.5 * Float64(mean(omega_host .^ 2)), sqrt(Float64(mean(div_host .^ 2)))
end

function sample_kolmogorov!(history::KolmogorovHistory, prob, cfg::KolmogorovHDConfig; force::Bool = false)
    if !force && prob.clock.step % history.sample_every != 0
        return false
    end

    current_time = Float64(prob.clock.t)
    if !isempty(history.times) && isapprox(history.times[end], current_time; atol = 1.0e-12, rtol = 0.0)
        return false
    end

    sync_hd_real_state!(prob)
    ux = Array(prob.vars.ux)
    uy = Array(prob.vars.uy)
    uz = Array(prob.vars.uz)
    speed2 = ux .^ 2 .+ uy .^ 2 .+ uz .^ 2
    enstrophy, divergence_rms = spectral_hd_diagnostics(prob)

    push!(history.times, current_time)
    push!(history.kinetic, 0.5 * Float64(mean(speed2)))
    push!(history.enstrophy, enstrophy)
    push!(history.velocity_rms, sqrt(Float64(mean(speed2))))
    push!(history.max_velocity, sqrt(Float64(maximum(speed2))))
    push!(history.divergence_rms, divergence_rms)
    push!(history.forcing_power, forcing_power(ux, cfg))
    return true
end

function write_kolmogorov_csv(path::String, history::KolmogorovHistory)
    open(path, "w") do io
        println(io, "time,kinetic,enstrophy,velocity_rms,max_velocity,divergence_rms,forcing_power")
        for i in eachindex(history.times)
            println(io, "$(history.times[i]),$(history.kinetic[i]),$(history.enstrophy[i]),$(history.velocity_rms[i]),$(history.max_velocity[i]),$(history.divergence_rms[i]),$(history.forcing_power[i])")
        end
    end
    return path
end

function save_kolmogorov_snapshot!(writer::SnapshotWriter, prob; force::Bool = false)
    current_time = Float64(prob.clock.t)
    if !force && current_time + 1.0e-12 < writer.next_time
        return false
    end
    if isfinite(writer.last_time) && isapprox(writer.last_time, current_time; atol = 1.0e-12, rtol = 0.0)
        return false
    end

    sync_hd_real_state!(prob)
    MHDFlows.savefile(prob, writer.count; file_path_and_name = writer.prefix)
    writer.count += 1
    writer.last_time = current_time
    while writer.next_time <= current_time + 1.0e-12
        writer.next_time += writer.dt
    end
    return true
end

function make_kolmogorov_callback(history::KolmogorovHistory, csv_path::String, cfg::KolmogorovHDConfig, writer::SnapshotWriter)
    return function (prob)
        if sample_kolmogorov!(history, prob, cfg)
            write_kolmogorov_csv(csv_path, history)
        end
        save_kolmogorov_snapshot!(writer, prob)
        return nothing
    end
end

function write_kolmogorov_metadata(path::String, cfg::KolmogorovHDConfig, device_label::String)
    metadata = Dict(
        "name" => cfg.name,
        "backend" => cfg.backend,
        "package" => "MHDFlows",
        "description" => cfg.description,
        "device" => device_label,
        "device_request" => cfg.device,
        "output_root" => cfg.output_root,
        "nx" => cfg.nx,
        "ny" => cfg.ny,
        "nz" => cfg.nz,
        "domain_length" => cfg.domain_length,
        "float_type" => string(cfg.float_type),
        "reynolds_number_configured" => cfg.reynolds_number,
        "reynolds_number_effective" => effective_reynolds(cfg),
        "viscosity" => cfg.viscosity,
        "force_amplitude" => cfg.force_amplitude,
        "force_mode_y" => cfg.force_mode_y,
        "disable_package_dealiasing" => cfg.disable_package_dealiasing,
        "force_form" => "f_x = force_amplitude * sin(2*pi*force_mode_y*y/domain_length), f_y = 0, f_z = 0",
        "initial_condition" => cfg.initial_condition,
        "initial_velocity_rms" => cfg.initial_velocity_rms,
        "grid_noise_velocity_amplitude" => cfg.grid_noise_velocity_amplitude,
        "initial_modes" => cfg.initial_modes,
        "fixed_dt" => cfg.fixed_dt,
        "end_time" => cfg.end_time,
        "max_steps" => cfg.max_steps,
        "diagnostics_sample_every" => cfg.diagnostics_sample_every,
        "snapshot_dt" => cfg.snapshot_dt,
        "plot_after_run" => cfg.plot_after_run,
        "reuse_existing_data" => cfg.reuse_existing_data,
        "seed" => cfg.seed,
    )
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

function kolmogorov_config_from_sources(settings, positionals::Vector{String})
    length(positionals) <= 8 || error("Expected at most 8 positional overrides. Run with --help for usage.")
    defaults = KolmogorovHDConfig()
    re_configured = as_float(get_config(settings, "reynolds_number", defaults.reynolds_number))
    viscosity = as_float(get_config(settings, "viscosity", 1.0 / re_configured))
    force_amplitude = as_float(get_config(settings, "force_amplitude", defaults.force_amplitude))

    cfg = KolmogorovHDConfig(;
        backend = "mhdflows",
        output_root = configured_output_root(settings, "mhdflows"),
        device = as_string(get_config(settings, "device", defaults.device)),
        name = as_string(get_config(settings, "name", defaults.name)),
        description = as_string(get_config(settings, "description", defaults.description)),
        nx = as_int(get_config(settings, "nx", defaults.nx)),
        ny = as_int(get_config(settings, "ny", as_int(get_config(settings, "nx", defaults.ny)))),
        nz = as_int(get_config(settings, "nz", defaults.nz)),
        domain_length = as_float(get_config(settings, "domain_length", defaults.domain_length)),
        float_type = as_datatype(get_config(settings, "float_type", string(defaults.float_type))),
        reynolds_number = re_configured,
        viscosity = viscosity,
        force_amplitude = force_amplitude,
        force_mode_y = as_int(get_config(settings, "force_mode_y", defaults.force_mode_y)),
        disable_package_dealiasing = as_bool(get_config(settings, "disable_package_dealiasing", defaults.disable_package_dealiasing)),
        initial_condition = as_string(get_config(settings, "initial_condition", defaults.initial_condition)),
        initial_velocity_rms = as_float(get_config(settings, "initial_velocity_rms", defaults.initial_velocity_rms)),
        grid_noise_velocity_amplitude = as_float(get_config(settings, "grid_noise_velocity_amplitude", force_amplitude)),
        initial_modes = as_int(get_config(settings, "initial_modes", defaults.initial_modes)),
        fixed_dt = as_float(get_config(settings, "fixed_dt", defaults.fixed_dt)),
        end_time = as_float(get_config(settings, "end_time", defaults.end_time)),
        max_steps = as_int(get_config(settings, "max_steps", defaults.max_steps)),
        diagnostics_sample_every = as_int(get_config(settings, "diagnostics_sample_every", defaults.diagnostics_sample_every)),
        snapshot_dt = as_float(get_config(settings, "snapshot_dt", defaults.snapshot_dt)),
        plot_after_run = as_bool(get_config(settings, "plot_after_run", defaults.plot_after_run)),
        reuse_existing_data = as_bool(get_config(settings, "reuse_existing_data", defaults.reuse_existing_data)),
        seed = as_int(get_config(settings, "seed", defaults.seed)),
        tag_suffix = as_string(get_config(settings, "tag_suffix", defaults.tag_suffix)),
    )

    isempty(positionals) && return cfg
    positional_force_amplitude = length(positionals) >= 3 ? parse(Float64, positionals[3]) : cfg.force_amplitude
    positional_grid_noise_velocity_amplitude = length(positionals) >= 3 && !haskey(settings, "grid_noise_velocity_amplitude") ? positional_force_amplitude : cfg.grid_noise_velocity_amplitude
    return KolmogorovHDConfig(;
        output_root = cfg.output_root,
        backend = cfg.backend,
        device = cfg.device,
        name = cfg.name,
        description = cfg.description,
        nx = length(positionals) >= 1 ? parse(Int, positionals[1]) : cfg.nx,
        ny = length(positionals) >= 1 ? parse(Int, positionals[1]) : cfg.ny,
        nz = cfg.nz,
        domain_length = cfg.domain_length,
        float_type = cfg.float_type,
        reynolds_number = cfg.reynolds_number,
        viscosity = length(positionals) >= 4 ? parse(Float64, positionals[4]) : cfg.viscosity,
        force_amplitude = positional_force_amplitude,
        force_mode_y = cfg.force_mode_y,
        disable_package_dealiasing = cfg.disable_package_dealiasing,
        initial_condition = cfg.initial_condition,
        initial_velocity_rms = cfg.initial_velocity_rms,
        grid_noise_velocity_amplitude = positional_grid_noise_velocity_amplitude,
        initial_modes = cfg.initial_modes,
        fixed_dt = length(positionals) >= 6 ? parse(Float64, positionals[6]) : cfg.fixed_dt,
        end_time = length(positionals) >= 2 ? parse(Float64, positionals[2]) : cfg.end_time,
        max_steps = cfg.max_steps,
        diagnostics_sample_every = cfg.diagnostics_sample_every,
        snapshot_dt = length(positionals) >= 7 ? parse(Float64, positionals[7]) : cfg.snapshot_dt,
        plot_after_run = cfg.plot_after_run,
        reuse_existing_data = cfg.reuse_existing_data,
        seed = length(positionals) >= 8 ? parse(Int, positionals[8]) : cfg.seed,
        tag_suffix = length(positionals) >= 5 ? positionals[5] : cfg.tag_suffix,
    )
end
