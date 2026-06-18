using Printf
using Statistics
using TOML

Base.@kwdef mutable struct AthenaConfig
    output_root::String
    athena_project::String
    athena_executable::String = "bin/athena"
    name::String = "compressible_mhd_baseline"
    description::String = "Athena++ reference MHD sanity check"
    nx::Int = 128
    box_size::Float64 = 2pi
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
    snapshot_dt::Float64 = 5.0
    late_window_fraction::Float64 = 0.25
    stability_rel_band::Float64 = 0.12
    stability_abs_change::Float64 = 0.06
    seed::Int = 1234
    tag_suffix::String = ""
    athena_source_project::String = ""
    athena_problem::String = "mhdflows_turbulence"
    athena_input_template::String = ""
    athena_problem_id::String = ""
    athena_parse_only::Bool = false
    athena_dry_run::Bool = false
    athena_use_build_copy::Bool = true
    athena_build_copy::String = joinpath("build", "athena_mhdflows")
    athena_refresh_build_copy::Bool = false
    athena_pgen_source::String = joinpath("athena_pgen", "mhdflows_turbulence.cpp")
    athena_patch_fp16::Bool = true
    athena_configure::Bool = true
    athena_make::Bool = true
    athena_configure_args::Vector{String} = ["-b", "--prob=mhdflows_turbulence", "--eos=isothermal", "-hdf5", "-fft", "-omp"]
    athena_make_args::Vector{String} = String[]
    athena_run_args::Vector{String} = String[]
    athena_mpi_ranks::Int = 1
    athena_mpi_launcher::String = "mpirun"
    athena_mpi_args::Vector{String} = String[]
    athena_cfl_number::Float64 = 0.4
    athena_integrator::String = "vl2"
    athena_xorder::Int = 2
    athena_num_threads::Int = 4
    athena_nx1::Int = 128
    athena_nx2::Int = 128
    athena_nx3::Int = 128
    athena_meshblock_nx1::Int = 64
    athena_meshblock_nx2::Int = 64
    athena_meshblock_nx3::Int = 64
    athena_box_size::Float64 = 0.0
    athena_history_dt::Float64 = 0.1
    athena_hdf5_dt::Float64 = 5.0
    athena_output_variable::String = "prim"
    athena_hdf5_data_format::String = "float32"
    athena_turb_flag::Int = 3
    athena_tcorr::Float64 = 0.1
    athena_dtdrive::Float64 = 0.1
    athena_f_shear::Float64 = 1.0
end

athena_as_string(value) = String(value)
athena_as_int(value) = value isa Integer ? Int(value) : parse(Int, String(value))
athena_as_float(value) = value isa Real ? Float64(value) : parse(Float64, String(value))
athena_as_bool(value) = value isa Bool ? value : parse(Bool, lowercase(String(value)))

function athena_as_tuple3(value)
    length(value) == 3 || error("mean_field must have exactly three values")
    values = Float64.(value)
    return (values[1], values[2], values[3])
end

function athena_as_string_vector(value)
    value === nothing && return String[]
    if value isa AbstractVector
        return String.(value)
    end
    text = strip(String(value))
    isempty(text) && return String[]
    return split(text)
end

function athena_float_tag(x::Real)
    tag = @sprintf("%.4g", float(x))
    return replace(tag, "-" => "m", "." => "p")
end

sanitize_athena_id(text::AbstractString) = replace(String(text), r"[^A-Za-z0-9_.-]" => "_")

function athena_case_tag(cfg::AthenaConfig)
    base = "$(cfg.name)_n$(cfg.nx)_cs$(athena_float_tag(cfg.sound_speed))_B0$(athena_float_tag(cfg.mean_field[1]))_nu$(athena_float_tag(cfg.viscosity))_eta$(athena_float_tag(cfg.resistivity))_P$(athena_float_tag(cfg.forcing_power))_T$(athena_float_tag(cfg.end_time))"
    return isempty(cfg.tag_suffix) ? base : "$(base)_$(cfg.tag_suffix)"
end

athena_case_root(cfg::AthenaConfig) = joinpath(cfg.output_root, "athena", athena_case_tag(cfg))
athena_analysis_root(cfg::AthenaConfig) = joinpath(athena_case_root(cfg), "analysis")
athena_figure_root(cfg::AthenaConfig) = joinpath(athena_case_root(cfg), "figures")
athena_snapshot_root(cfg::AthenaConfig) = joinpath(athena_case_root(cfg), "snapshots")

function ensure_athena_case_dirs!(cfg::AthenaConfig)
    mkpath(athena_analysis_root(cfg))
    mkpath(athena_figure_root(cfg))
    mkpath(athena_snapshot_root(cfg))
    return nothing
end

function default_athena_history_dt(end_time::Real, snapshot_dt::Real)
    end_time > 0 || return max(float(snapshot_dt), eps(Float64))
    candidate = end_time / 200
    snapshot_dt > 0 && return min(float(snapshot_dt), max(candidate, eps(Float64)))
    return max(candidate, eps(Float64))
end

function default_athena_configure_args(problem::AbstractString)
    normalized = lowercase(strip(String(problem)))
    if normalized == "mhdflows_turbulence"
        return ["-b", "--prob=mhdflows_turbulence", "--eos=isothermal", "-hdf5", "-fft", "-omp"]
    elseif normalized == "turb"
        return ["-b", "--prob=turb", "--eos=isothermal", "-hdf5", "-fft", "-omp"]
    elseif normalized == "orszag_tang"
        return ["-b", "--prob=orszag_tang", "--eos=isothermal", "-hdf5", "-omp"]
    end
    return ["-b", "--prob=$(problem)", "--eos=isothermal", "-hdf5", "-omp"]
end

function configure_args_problem(args::Vector{String})
    for (index, arg) in enumerate(args)
        if startswith(arg, "--prob=")
            return split(arg, "=", limit = 2)[2]
        elseif arg == "--prob" && index < length(args)
            return args[index + 1]
        end
    end
    return nothing
end

function athena_config_from_sources(settings, positionals::Vector{String})
    length(positionals) <= 9 || error("Expected at most 9 positional overrides. Run with --help for usage.")

    nx = athena_as_int(get_config(settings, "nx", 128))
    end_time = athena_as_float(get_config(settings, "end_time", 60.0))
    forcing_power = athena_as_float(get_config(settings, "forcing_power", 8.0e3))
    viscosity = athena_as_float(get_config(settings, "viscosity", 1.0e-2))
    resistivity = athena_as_float(get_config(settings, "resistivity", 1.0e-2))
    tag_suffix = athena_as_string(get_config(settings, "tag_suffix", ""))
    fixed_dt = athena_as_float(get_config(settings, "fixed_dt", 0.0))
    snapshot_dt = athena_as_float(get_config(settings, "snapshot_dt", 5.0))
    seed = athena_as_int(get_config(settings, "seed", 1234))

    if !isempty(positionals)
        nx = length(positionals) >= 1 ? parse(Int, positionals[1]) : nx
        end_time = length(positionals) >= 2 ? parse(Float64, positionals[2]) : end_time
        forcing_power = length(positionals) >= 3 ? parse(Float64, positionals[3]) : forcing_power
        viscosity = length(positionals) >= 4 ? parse(Float64, positionals[4]) : viscosity
        resistivity = length(positionals) >= 5 ? parse(Float64, positionals[5]) : resistivity
        tag_suffix = length(positionals) >= 6 ? positionals[6] : tag_suffix
        fixed_dt = length(positionals) >= 7 ? parse(Float64, positionals[7]) : fixed_dt
        snapshot_dt = length(positionals) >= 8 ? parse(Float64, positionals[8]) : snapshot_dt
        seed = length(positionals) >= 9 ? parse(Int, positionals[9]) : seed
    end

    athena_nx1 = athena_as_int(get_config(settings, "athena_nx1", 0))
    athena_nx2 = athena_as_int(get_config(settings, "athena_nx2", 0))
    athena_nx3 = athena_as_int(get_config(settings, "athena_nx3", 0))
    athena_nx1 = athena_nx1 > 0 ? athena_nx1 : nx
    athena_nx2 = athena_nx2 > 0 ? athena_nx2 : nx
    athena_nx3 = athena_nx3 > 0 ? athena_nx3 : nx

    mb1 = athena_as_int(get_config(settings, "athena_meshblock_nx1", 0))
    mb2 = athena_as_int(get_config(settings, "athena_meshblock_nx2", 0))
    mb3 = athena_as_int(get_config(settings, "athena_meshblock_nx3", 0))
    mb1 = mb1 > 0 ? mb1 : athena_nx1
    mb2 = mb2 > 0 ? mb2 : athena_nx2
    mb3 = mb3 > 0 ? mb3 : athena_nx3

    history_dt = athena_as_float(get_config(settings, "athena_history_dt", 0.0))
    history_dt = history_dt > 0 ? history_dt : default_athena_history_dt(end_time, snapshot_dt)
    hdf5_dt = athena_as_float(get_config(settings, "athena_hdf5_dt", 0.0))
    hdf5_dt = hdf5_dt > 0 ? hdf5_dt : snapshot_dt

    template = athena_as_string(get_config(settings, "athena_input_template", ""))
    template_path = isempty(template) ? "" : resolve_repo_path(template)
    athena_problem = athena_as_string(get_config(settings, "athena_problem", "mhdflows_turbulence"))
    configure_args = haskey(settings, "athena_configure_args") ?
        athena_as_string_vector(settings["athena_configure_args"]) :
        default_athena_configure_args(athena_problem)
    source_project = athena_project_path(settings)
    athena_box_size_raw = athena_as_float(get_config(settings, "athena_box_size", 0.0))
    shared_box_size = athena_as_float(get_config(settings, "box_size", 2pi))
    athena_box_size = athena_box_size_raw > 0 ? athena_box_size_raw : shared_box_size
    tcorr_raw = athena_as_float(get_config(settings, "athena_tcorr", 0.0))
    dtdrive_raw = athena_as_float(get_config(settings, "athena_dtdrive", 0.0))

    cfg = AthenaConfig(;
        output_root = configured_output_root(settings),
        athena_project = source_project,
        athena_source_project = source_project,
        athena_executable = athena_as_string(get_config(settings, "athena_executable", "bin/athena")),
        name = athena_as_string(get_config(settings, "name", "compressible_mhd_baseline")),
        description = athena_as_string(get_config(settings, "description", "Athena++ reference MHD sanity check")),
        nx = nx,
        box_size = shared_box_size,
        sound_speed = athena_as_float(get_config(settings, "sound_speed", sqrt(2.0))),
        viscosity = viscosity,
        resistivity = resistivity,
        mean_field = athena_as_tuple3(get_config(settings, "mean_field", [1.0, 0.0, 0.0])),
        forcing_wavenumber = athena_as_float(get_config(settings, "forcing_wavenumber", 2.0)),
        forcing_power = forcing_power,
        forcing_width = athena_as_float(get_config(settings, "forcing_width", 1.0)),
        initial_velocity_power = athena_as_float(get_config(settings, "initial_velocity_power", 5.0e-4)),
        fixed_dt = fixed_dt,
        end_time = end_time,
        max_steps = athena_as_int(get_config(settings, "max_steps", 80_000)),
        snapshot_dt = snapshot_dt,
        late_window_fraction = athena_as_float(get_config(settings, "late_window_fraction", 0.25)),
        stability_rel_band = athena_as_float(get_config(settings, "stability_rel_band", 0.12)),
        stability_abs_change = athena_as_float(get_config(settings, "stability_abs_change", 0.06)),
        seed = seed,
        tag_suffix = tag_suffix,
        athena_problem = athena_problem,
        athena_input_template = template_path,
        athena_problem_id = athena_as_string(get_config(settings, "athena_problem_id", "")),
        athena_parse_only = athena_as_bool(get_config(settings, "athena_parse_only", false)),
        athena_dry_run = athena_as_bool(get_config(settings, "athena_dry_run", false)),
        athena_use_build_copy = athena_as_bool(get_config(settings, "athena_use_build_copy", true)),
        athena_build_copy = resolve_repo_path(athena_as_string(get_config(settings, "athena_build_copy", joinpath("build", "athena_mhdflows")))),
        athena_refresh_build_copy = athena_as_bool(get_config(settings, "athena_refresh_build_copy", false)),
        athena_pgen_source = resolve_repo_path(athena_as_string(get_config(settings, "athena_pgen_source", joinpath("athena_pgen", "mhdflows_turbulence.cpp")))),
        athena_patch_fp16 = athena_as_bool(get_config(settings, "athena_patch_fp16", true)),
        athena_configure = athena_as_bool(get_config(settings, "athena_configure", true)),
        athena_make = athena_as_bool(get_config(settings, "athena_make", true)),
        athena_configure_args = configure_args,
        athena_make_args = athena_as_string_vector(get_config(settings, "athena_make_args", String[])),
        athena_run_args = athena_as_string_vector(get_config(settings, "athena_run_args", String[])),
        athena_mpi_ranks = athena_as_int(get_config(settings, "athena_mpi_ranks", 1)),
        athena_mpi_launcher = athena_as_string(get_config(settings, "athena_mpi_launcher", "mpirun")),
        athena_mpi_args = athena_as_string_vector(get_config(settings, "athena_mpi_args", String[])),
        athena_cfl_number = athena_as_float(get_config(settings, "athena_cfl_number", 0.4)),
        athena_integrator = athena_as_string(get_config(settings, "athena_integrator", "vl2")),
        athena_xorder = athena_as_int(get_config(settings, "athena_xorder", 2)),
        athena_num_threads = athena_as_int(get_config(settings, "athena_num_threads", 4)),
        athena_nx1 = athena_nx1,
        athena_nx2 = athena_nx2,
        athena_nx3 = athena_nx3,
        athena_meshblock_nx1 = mb1,
        athena_meshblock_nx2 = mb2,
        athena_meshblock_nx3 = mb3,
        athena_box_size = athena_box_size,
        athena_history_dt = history_dt,
        athena_hdf5_dt = hdf5_dt,
        athena_output_variable = athena_as_string(get_config(settings, "athena_output_variable", "prim")),
        athena_hdf5_data_format = athena_as_string(get_config(settings, "athena_hdf5_data_format", "float32")),
        athena_turb_flag = athena_as_int(get_config(settings, "athena_turb_flag", 3)),
        athena_tcorr = tcorr_raw > 0 ? tcorr_raw : history_dt,
        athena_dtdrive = dtdrive_raw > 0 ? dtdrive_raw : history_dt,
        athena_f_shear = athena_as_float(get_config(settings, "athena_f_shear", 1.0)),
    )

    cfg.athena_problem_id = isempty(cfg.athena_problem_id) ? sanitize_athena_id("athena_$(athena_case_tag(cfg))") : sanitize_athena_id(cfg.athena_problem_id)
    configured_problem = configure_args_problem(cfg.athena_configure_args)
    if cfg.athena_configure && configured_problem !== nothing && configured_problem != cfg.athena_problem
        error("athena_configure_args selects --prob=$(configured_problem), but athena_problem=$(cfg.athena_problem). Keep them aligned for a meaningful run.")
    end
    return cfg
end

function write_athena_generated_input(path::String, cfg::AthenaConfig)
    if !isempty(cfg.athena_input_template)
        isfile(cfg.athena_input_template) || error("Athena input template does not exist: $(cfg.athena_input_template)")
        cp(cfg.athena_input_template, path; force = true)
        return path
    end

    problem = lowercase(strip(cfg.athena_problem))
    problem in ("mhdflows_turbulence", "orszag_tang", "turb") || error("No built-in generated Athena input for athena_problem=$(cfg.athena_problem). Set athena_input_template for custom problems.")

    half_box = cfg.athena_box_size / 2
    open(path, "w") do io
        println(io, "<comment>")
        println(io, "problem   = Generated Athena++ reference input")
        println(io, "reference = Generated by scripts/run_athena.jl")
        if problem == "mhdflows_turbulence"
            println(io, "note      = Uses repo-owned mhdflows_turbulence pgen plus Athena's turbulence driver")
        else
            println(io, "note      = Stock Athena problem; not a one-to-one clone of the MHDFlows forcing model")
        end
        println(io)
        println(io, "<job>")
        println(io, "problem_id = $(cfg.athena_problem_id)")
        println(io)
        println(io, "<output1>")
        println(io, "file_type = hst")
        println(io, "dt        = $(cfg.athena_history_dt)")
        println(io, "data_format = %.16e")
        println(io)
        println(io, "<output2>")
        println(io, "file_type   = hdf5")
        println(io, "variable    = $(cfg.athena_output_variable)")
        println(io, "dt          = $(cfg.athena_hdf5_dt)")
        if !isempty(strip(cfg.athena_hdf5_data_format))
            println(io, "data_format = $(cfg.athena_hdf5_data_format)")
        end
        println(io)
        println(io, "<time>")
        println(io, "cfl_number = $(cfg.athena_cfl_number)")
        println(io, "nlim       = $(cfg.max_steps)")
        println(io, "tlim       = $(cfg.end_time)")
        println(io, "integrator = $(cfg.athena_integrator)")
        println(io, "xorder     = $(cfg.athena_xorder)")
        println(io, "ncycle_out = 100")
        println(io)
        println(io, "<mesh>")
        println(io, "nx1        = $(cfg.athena_nx1)")
        println(io, "x1min      = $(-half_box)")
        println(io, "x1max      = $(half_box)")
        println(io, "ix1_bc     = periodic")
        println(io, "ox1_bc     = periodic")
        println(io)
        println(io, "nx2        = $(cfg.athena_nx2)")
        println(io, "x2min      = $(-half_box)")
        println(io, "x2max      = $(half_box)")
        println(io, "ix2_bc     = periodic")
        println(io, "ox2_bc     = periodic")
        println(io)
        println(io, "nx3        = $(cfg.athena_nx3)")
        println(io, "x3min      = $(-half_box)")
        println(io, "x3max      = $(half_box)")
        println(io, "ix3_bc     = periodic")
        println(io, "ox3_bc     = periodic")
        println(io)
        println(io, "num_threads = $(cfg.athena_num_threads)")
        println(io, "refinement  = none")
        println(io)
        println(io, "<meshblock>")
        println(io, "nx1 = $(cfg.athena_meshblock_nx1)")
        println(io, "nx2 = $(cfg.athena_meshblock_nx2)")
        println(io, "nx3 = $(cfg.athena_meshblock_nx3)")
        println(io)
        println(io, "<hydro>")
        println(io, "iso_sound_speed = $(cfg.sound_speed)")
        println(io, "gamma           = 1.666666666666667")
        println(io)
        println(io, "<problem>")
        if problem == "mhdflows_turbulence"
            println(io, "rho0                       = 1.0")
            println(io, "sound_speed                = $(cfg.sound_speed)")
            println(io, "pressure                   = $(cfg.sound_speed^2)")
            println(io, "mean_field_x               = $(cfg.mean_field[1])")
            println(io, "mean_field_y               = $(cfg.mean_field[2])")
            println(io, "mean_field_z               = $(cfg.mean_field[3])")
            println(io, "initial_velocity_power     = $(cfg.initial_velocity_power)")
            println(io, "initial_velocity_wavenumber = 1.0")
        end
        println(io, "nu_iso  = $(cfg.viscosity)")
        println(io, "eta_ohm = $(cfg.resistivity)")

        if problem in ("mhdflows_turbulence", "turb")
            nlow = max(0, floor(Int, cfg.forcing_wavenumber - cfg.forcing_width))
            nhigh = max(nlow + 1, ceil(Int, cfg.forcing_wavenumber + cfg.forcing_width))
            println(io)
            println(io, "<turbulence>")
            println(io, "turb_flag = $(cfg.athena_turb_flag)")
            println(io, "dedt      = $(cfg.forcing_power)")
            println(io, "nlow      = $(nlow)")
            println(io, "nhigh     = $(nhigh)")
            println(io, "expo      = 2.0")
            println(io, "tcorr     = $(max(cfg.athena_tcorr, eps(Float64)))")
            println(io, "dtdrive   = $(max(cfg.athena_dtdrive, eps(Float64)))")
            println(io, "f_shear   = $(cfg.athena_f_shear)")
            println(io, "rseed     = $(cfg.seed)")
        end
    end
    return path
end

function resolve_athena_executable(cfg::AthenaConfig; require_exists::Bool = true)
    configured = expanduser(cfg.athena_executable)
    path = isabspath(configured) ? normpath(configured) : normpath(joinpath(cfg.athena_project, configured))
    if Sys.iswindows() && !isfile(path) && isfile(path * ".exe")
        path *= ".exe"
    end
    if require_exists
        isfile(path) || error("Athena executable not found at $(path). Build Athena or set athena_executable.")
    end
    return path
end

function athena_status(message::AbstractString; log_path::AbstractString = "")
    println(message)
    flush(stdout)
    if !isempty(log_path)
        mkpath(dirname(log_path))
        open(log_path, "a") do io
            println(io, message)
        end
    end
    return nothing
end

function path_is_within(path::AbstractString, parent::AbstractString)
    relative = relpath(abspath(path), abspath(parent))
    parts = splitpath(relative)
    return relative == "." || (isempty(parts) || parts[1] != "..")
end

function patch_athena_fp16_detection!(athena_project::String; log_path::AbstractString = "")
    header_path = joinpath(athena_project, "src", "athena.hpp")
    isfile(header_path) || error("Athena header not found for fp16 patch: $(header_path)")
    text = read(header_path, String)
    old = """
#ifndef __INTEL_LLVM_COMPILER
#if defined(__fp16) || defined(__FLT16_MAX__) || defined(__ARM_FP16_FORMAT_IEEE)
#define fp16_t __fp16
#elif defined(_Float16)
#define fp16_t _Float16
#endif
#else
#define fp16_t_not_supported
#endif // __INTEL_LLVM_COMPILER
"""
    new = """
#ifndef __INTEL_LLVM_COMPILER
#if defined(__ARM_FP16_FORMAT_IEEE)
#define fp16_t __fp16
#elif defined(__FLT16_MAX__)
#define fp16_t _Float16
#elif defined(_Float16)
#define fp16_t _Float16
#endif
#else
#define fp16_t_not_supported
#endif // __INTEL_LLVM_COMPILER
"""
    if occursin(new, text)
        athena_status("Athena fp16 patch already present: $(header_path)"; log_path = log_path)
        return nothing
    end
    if !occursin(old, text)
        athena_status("Athena fp16 patch skipped; detection block was not recognized in $(header_path)"; log_path = log_path)
        return nothing
    end
    write(header_path, replace(text, old => new))
    athena_status("Applied Athena fp16 patch: $(header_path)"; log_path = log_path)
    return nothing
end

function prepare_athena_build_copy!(cfg::AthenaConfig; log_path::AbstractString = "")
    cfg.athena_source_project = isempty(cfg.athena_source_project) ? cfg.athena_project : cfg.athena_source_project
    cfg.athena_use_build_copy || return cfg.athena_project

    build_root = resolve_repo_path("build")
    build_project = normpath(cfg.athena_build_copy)
    path_is_within(build_project, build_root) || error("athena_build_copy must stay inside $(build_root); got $(build_project)")
    abspath(build_project) != abspath(cfg.athena_source_project) || error("athena_build_copy must not be the source Athena checkout")

    if cfg.athena_refresh_build_copy && isdir(build_project)
        athena_status("Removing existing Athena build copy: $(build_project)"; log_path = log_path)
        rm(build_project; recursive = true, force = true)
    end

    if !isdir(build_project)
        athena_status("Copying Athena source from $(cfg.athena_source_project) to $(build_project)"; log_path = log_path)
        mkpath(dirname(build_project))
        cp(cfg.athena_source_project, build_project)
        athena_status("Finished copying Athena build tree: $(build_project)"; log_path = log_path)
    else
        athena_status("Using existing Athena build copy: $(build_project)"; log_path = log_path)
    end

    if !isempty(strip(cfg.athena_pgen_source))
        isfile(cfg.athena_pgen_source) || error("Athena pgen source does not exist: $(cfg.athena_pgen_source)")
        target = joinpath(build_project, "src", "pgen", "$(cfg.athena_problem).cpp")
        mkpath(dirname(target))
        cp(cfg.athena_pgen_source, target; force = true)
        athena_status("Installed Athena problem generator: $(target)"; log_path = log_path)
    end

    if cfg.athena_patch_fp16
        patch_athena_fp16_detection!(build_project; log_path = log_path)
    end

    cfg.athena_project = build_project
    return cfg.athena_project
end

command_text(parts::Vector{String}) = join(map(part -> occursin(r"\s", part) ? "\"$(replace(part, "\"" => "\\\""))\"" : part, parts), " ")

function log_tail(path::String, max_lines::Int = 40)
    isfile(path) || return ["<log file does not exist>"]
    lines = readlines(path)
    isempty(lines) && return ["<log file is empty>"]
    start = max(1, length(lines) - max_lines + 1)
    return lines[start:end]
end

function print_log_tail(label::String, path::String)
    println(stderr, "--- $(label): $(path) ---")
    for line in log_tail(path)
        println(stderr, line)
    end
    return nothing
end

function run_logged(command_parts::Vector{String}, working_dir::String, stdout_path::String, stderr_path::String)
    try
        open(stdout_path, "w") do out
            open(stderr_path, "w") do err
                cd(working_dir) do
                    run(pipeline(Cmd(command_parts); stdout = out, stderr = err))
                end
            end
        end
    catch err
        println(stderr, "Command failed in $(working_dir): $(command_text(command_parts))")
        println(stderr, "Exception: $(err)")
        print_log_tail("stdout tail", stdout_path)
        print_log_tail("stderr tail", stderr_path)
        error("Command failed. See logs above or inspect $(stdout_path) and $(stderr_path).")
    end
    return nothing
end

function maybe_configure_and_make_athena!(cfg::AthenaConfig)
    analysis_dir = athena_analysis_root(cfg)
    if cfg.athena_configure
        stdout_path = joinpath(analysis_dir, "athena_configure.out.log")
        stderr_path = joinpath(analysis_dir, "athena_configure.err.log")
        python = get(ENV, "PYTHON", "python")
        command = [python, "configure.py"]
        append!(command, cfg.athena_configure_args)
        println("Configuring Athena: $(command_text(command))")
        run_logged(command, cfg.athena_project, stdout_path, stderr_path)
    end
    if cfg.athena_make
        stdout_path = joinpath(analysis_dir, "athena_make.out.log")
        stderr_path = joinpath(analysis_dir, "athena_make.err.log")
        command = ["make"]
        append!(command, cfg.athena_make_args)
        println("Building Athena: $(command_text(command))")
        run_logged(command, cfg.athena_project, stdout_path, stderr_path)
    end
    return nothing
end

function athena_run_command(cfg::AthenaConfig, input_path::String, case_dir::String; require_executable::Bool = true)
    executable = resolve_athena_executable(cfg; require_exists = require_executable)
    command = String[]
    if cfg.athena_mpi_ranks > 1
        push!(command, cfg.athena_mpi_launcher)
        append!(command, cfg.athena_mpi_args)
        append!(command, ["-n", string(cfg.athena_mpi_ranks)])
    end
    append!(command, [executable, "-i", input_path, "-d", case_dir])
    cfg.athena_parse_only && push!(command, "-n")
    append!(command, cfg.athena_run_args)
    return command
end

function validate_athena_parallelism(cfg::AthenaConfig)
    cfg.athena_num_threads >= 1 || error("athena_num_threads must be at least 1")
    cfg.athena_mpi_ranks >= 1 || error("athena_mpi_ranks must be at least 1")
    isempty(strip(cfg.athena_mpi_launcher)) && cfg.athena_mpi_ranks > 1 &&
        error("athena_mpi_launcher must be set when athena_mpi_ranks > 1")

    if cfg.athena_num_threads > 1 && !("-omp" in cfg.athena_configure_args)
        error("athena_num_threads > 1 requires -omp in athena_configure_args")
    end
    if cfg.athena_mpi_ranks > 1 && !("-mpi" in cfg.athena_configure_args)
        error("athena_mpi_ranks > 1 requires -mpi in athena_configure_args")
    end

    mesh = (cfg.athena_nx1, cfg.athena_nx2, cfg.athena_nx3)
    blocks = (cfg.athena_meshblock_nx1, cfg.athena_meshblock_nx2, cfg.athena_meshblock_nx3)
    all(>(0), mesh) || error("Athena mesh dimensions must be positive")
    all(>(0), blocks) || error("Athena meshblock dimensions must be positive")
    all(mesh[i] % blocks[i] == 0 for i in 1:3) ||
        error("Each Athena mesh dimension must be divisible by its meshblock dimension")
    num_meshblocks = prod(mesh[i] ÷ blocks[i] for i in 1:3)
    required_meshblocks = cfg.athena_mpi_ranks * cfg.athena_num_threads
    num_meshblocks >= required_meshblocks ||
        error("$(cfg.athena_mpi_ranks) MPI rank(s) x $(cfg.athena_num_threads) OpenMP thread(s) require at least $(required_meshblocks) meshblocks; configured $(num_meshblocks)")
    return num_meshblocks
end

function check_athena_run_logs(stdout_path::String, stderr_path::String)
    fatal_markers = ("FATAL ERROR", "terminate called", "Segmentation fault")
    for path in (stdout_path, stderr_path)
        isfile(path) || continue
        contents = read(path, String)
        if any(marker -> occursin(marker, contents), fatal_markers)
            print_log_tail("Athena fatal log", path)
            error("Athena reported a fatal runtime error in $(path)")
        end
    end
    return nothing
end

function parse_athena_hst_header(lines::Vector{String})
    header_line = ""
    for line in lines
        if startswith(strip(line), "#") && occursin("[1]=time", line)
            header_line = line
            break
        end
    end
    isempty(header_line) && error("Could not find Athena .hst column header")
    columns = Dict{String, Int}()
    for match in eachmatch(r"\[(\d+)\]=([^\s]+)", header_line)
        columns[match.captures[2]] = parse(Int, match.captures[1])
    end
    return columns
end

function read_athena_hst(path::String)
    lines = readlines(path)
    columns = parse_athena_hst_header(lines)
    rows = Vector{Vector{Float64}}()
    for line in lines
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue
        push!(rows, parse.(Float64, split(stripped)))
    end
    isempty(rows) && error("Athena history file has no data rows: $(path)")
    return columns, rows
end

function athena_column(rows::Vector{Vector{Float64}}, index::Int)
    return [row[index] for row in rows]
end

function athena_optional_column(rows::Vector{Vector{Float64}}, columns::Dict{String, Int}, name::String)
    return haskey(columns, name) ? athena_column(rows, columns[name]) : zeros(Float64, length(rows))
end

function convert_athena_history_to_csv(hst_path::String, csv_path::String, cfg::AthenaConfig)
    columns, rows = read_athena_hst(hst_path)
    haskey(columns, "time") || error("Athena history file is missing time")
    haskey(columns, "mass") || error("Athena history file is missing mass")

    volume = cfg.athena_box_size ^ 3
    time = athena_column(rows, columns["time"])
    mass = athena_column(rows, columns["mass"])
    rho_mean = mass ./ volume
    momentum = (
        athena_optional_column(rows, columns, "1-mom"),
        athena_optional_column(rows, columns, "2-mom"),
        athena_optional_column(rows, columns, "3-mom"),
    )
    kinetic = (
        athena_optional_column(rows, columns, "1-KE") .+
        athena_optional_column(rows, columns, "2-KE") .+
        athena_optional_column(rows, columns, "3-KE")
    ) ./ volume
    magnetic_total = (
        athena_optional_column(rows, columns, "1-ME") .+
        athena_optional_column(rows, columns, "2-ME") .+
        athena_optional_column(rows, columns, "3-ME")
    ) ./ volume
    guide_field_strength = sqrt(sum(abs2, cfg.mean_field))
    guide_field_energy = 0.5 * guide_field_strength ^ 2
    magnetic_fluct = max.(0.0, magnetic_total .- guide_field_energy)
    total_resolved = kinetic .+ magnetic_total
    fluct_total = kinetic .+ magnetic_fluct

    mean_velocity_squared = zeros(Float64, length(time))
    for component in momentum
        mean_velocity_squared .+= (component ./ mass) .^ 2
    end
    velocity_fluct_rms = sqrt.(max.(0.0, 2 .* kinetic ./ rho_mean .- mean_velocity_squared))
    sonic_mach = velocity_fluct_rms ./ cfg.sound_speed
    alfven_speed_mean = guide_field_strength > 0 ? guide_field_strength ./ sqrt.(rho_mean) : fill(NaN, length(time))
    alfven_mach_velocity = guide_field_strength > 0 ? velocity_fluct_rms ./ alfven_speed_mean : fill(NaN, length(time))
    magnetic_rms_fluct = sqrt.(2 .* magnetic_fluct)
    alfven_mach_magnetic = guide_field_strength > 0 ? magnetic_rms_fluct ./ guide_field_strength : fill(NaN, length(time))

    open(csv_path, "w") do io
        println(io, "time,rho_mean,kinetic,magnetic_total,magnetic_fluct,total_resolved,fluct_total")
        for i in eachindex(time)
            println(io, "$(time[i]),$(rho_mean[i]),$(kinetic[i]),$(magnetic_total[i]),$(magnetic_fluct[i]),$(total_resolved[i]),$(fluct_total[i])")
        end
    end

    return (
        time = time,
        rho_mean = rho_mean,
        kinetic = kinetic,
        magnetic_total = magnetic_total,
        magnetic_fluct = magnetic_fluct,
        total_resolved = total_resolved,
        fluct_total = fluct_total,
        velocity_fluct_rms = velocity_fluct_rms,
        sonic_mach = sonic_mach,
        magnetic_mean_strength = fill(guide_field_strength, length(time)),
        magnetic_rms_fluct = magnetic_rms_fluct,
        alfven_mach_velocity = alfven_mach_velocity,
        alfven_mach_magnetic = alfven_mach_magnetic,
    )
end

function athena_late_window_stats(history, cfg::AthenaConfig)
    length(history.time) < 3 && return nothing
    t_start = history.time[1] + (1 - cfg.late_window_fraction) * (history.time[end] - history.time[1])
    selection = findall(t -> t >= t_start, history.time)
    length(selection) < 3 && return nothing
    energies = history.fluct_total[selection]
    mean_energy = mean(energies)
    mean_energy <= 0 && return nothing
    return (
        t_start = history.time[selection[1]],
        band = (maximum(energies) - minimum(energies)) / mean_energy,
        signed_change = (energies[end] - energies[1]) / mean_energy,
    )
end

function collect_athena_snapshots!(cfg::AthenaConfig, case_dir::String)
    snapshot_dir = athena_snapshot_root(cfg)
    candidates = filter(path -> isfile(path) && (endswith(lowercase(path), ".athdf") || endswith(lowercase(path), ".h5")), readdir(case_dir; join = true))
    sort!(candidates)
    rows = Vector{Dict{String, Any}}()
    for (idx, source) in enumerate(candidates)
        target_name = "athena_state_t_$(lpad(string(idx - 1), 4, '0')).h5"
        target = joinpath(snapshot_dir, target_name)
        cp(source, target; force = true)
        push!(rows, Dict{String, Any}(
            "file" => joinpath("snapshots", target_name),
            "source_file" => basename(source),
            "time_estimate" => (idx - 1) * cfg.athena_hdf5_dt,
        ))
    end
    return rows
end

function attach_athena_snapshot_diagnostics!(rows, history)
    for row in rows
        snapshot_time = Float64(row["time_estimate"])
        history_index = argmin(abs.(history.time .- snapshot_time))
        row["time"] = snapshot_time
        row["diagnostic_time"] = history.time[history_index]
        row["rho_mean"] = history.rho_mean[history_index]
        row["velocity_fluct_rms"] = history.velocity_fluct_rms[history_index]
        row["sonic_mach"] = history.sonic_mach[history_index]
        row["magnetic_mean_strength"] = history.magnetic_mean_strength[history_index]
        row["magnetic_rms_fluct"] = history.magnetic_rms_fluct[history_index]
        row["alfven_mach_mean"] = history.alfven_mach_velocity[history_index]
        row["alfven_mach_velocity"] = history.alfven_mach_velocity[history_index]
        row["alfven_mach_magnetic"] = history.alfven_mach_magnetic[history_index]
    end
    return rows
end

function find_athena_hst(cfg::AthenaConfig, case_dir::String)
    expected = joinpath(case_dir, "$(cfg.athena_problem_id).hst")
    isfile(expected) && return expected
    candidates = filter(path -> isfile(path) && endswith(lowercase(path), ".hst"), readdir(case_dir; join = true))
    isempty(candidates) && return nothing
    sort!(candidates; by = path -> stat(path).mtime)
    return candidates[end]
end

function existing_athena_outputs(case_dir::String)
    isdir(case_dir) || return String[]
    extensions = (".hst", ".athdf", ".xdmf", ".rst")
    return sort(filter(path -> isfile(path) && any(ext -> endswith(lowercase(path), ext), extensions), readdir(case_dir; join = true)))
end

function require_fresh_athena_case(case_dir::String)
    existing = existing_athena_outputs(case_dir)
    isempty(existing) && return nothing
    preview = join(basename.(existing[1:min(end, 4)]), ", ")
    error("Athena case already contains raw solver outputs ($(preview)). Use a new tag_suffix to avoid appending to or mixing simulation data.")
end

function write_athena_metadata(path::String, cfg::AthenaConfig; input_path::String, command::Vector{String}, history = nothing, snapshot_rows = nothing, status::String = "prepared")
    metadata = Dict{String, Any}(
        "name" => cfg.name,
        "solver" => "athena",
        "description" => cfg.description,
        "status" => status,
        "output_root" => cfg.output_root,
        "athena_source_project" => cfg.athena_source_project,
        "athena_project" => cfg.athena_project,
        "athena_executable" => cfg.athena_executable,
        "athena_problem" => cfg.athena_problem,
        "athena_problem_id" => cfg.athena_problem_id,
        "athena_input" => input_path,
        "athena_command" => command,
        "athena_parse_only" => cfg.athena_parse_only,
        "athena_dry_run" => cfg.athena_dry_run,
        "athena_use_build_copy" => cfg.athena_use_build_copy,
        "athena_build_copy" => cfg.athena_build_copy,
        "athena_refresh_build_copy" => cfg.athena_refresh_build_copy,
        "athena_pgen_source" => cfg.athena_pgen_source,
        "athena_patch_fp16" => cfg.athena_patch_fp16,
        "athena_configure" => cfg.athena_configure,
        "athena_make" => cfg.athena_make,
        "athena_configure_args" => cfg.athena_configure_args,
        "athena_make_args" => cfg.athena_make_args,
        "athena_run_args" => cfg.athena_run_args,
        "athena_mpi_ranks" => cfg.athena_mpi_ranks,
        "athena_mpi_launcher" => cfg.athena_mpi_launcher,
        "athena_mpi_args" => cfg.athena_mpi_args,
        "athena_num_threads" => cfg.athena_num_threads,
        "athena_total_cpu_threads" => cfg.athena_mpi_ranks * cfg.athena_num_threads,
        "athena_reference_note" => "mhdflows_turbulence maps the shared config into an Athena problem generator and Athena's native turbulence driver; it is intended for controlled comparison, but solver algorithms and forcing implementation are still Athena-specific.",
        "athena_history_note" => "magnetic_fluct subtracts the configured uniform guide-field energy from Athena's volume-averaged magnetic energy.",
        "nx" => cfg.nx,
        "athena_nx1" => cfg.athena_nx1,
        "athena_nx2" => cfg.athena_nx2,
        "athena_nx3" => cfg.athena_nx3,
        "athena_meshblock_nx1" => cfg.athena_meshblock_nx1,
        "athena_meshblock_nx2" => cfg.athena_meshblock_nx2,
        "athena_meshblock_nx3" => cfg.athena_meshblock_nx3,
        "box_size" => cfg.box_size,
        "athena_box_size" => cfg.athena_box_size,
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
        "snapshot_dt" => cfg.snapshot_dt,
        "athena_history_dt" => cfg.athena_history_dt,
        "athena_hdf5_dt" => cfg.athena_hdf5_dt,
        "athena_turb_flag" => cfg.athena_turb_flag,
        "athena_tcorr" => cfg.athena_tcorr,
        "athena_dtdrive" => cfg.athena_dtdrive,
        "athena_f_shear" => cfg.athena_f_shear,
        "late_window_fraction" => cfg.late_window_fraction,
        "stability_rel_band" => cfg.stability_rel_band,
        "stability_abs_change" => cfg.stability_abs_change,
        "seed" => cfg.seed,
    )
    if history !== nothing
        metadata["final_sonic_mach"] = history.sonic_mach[end]
        metadata["final_alfven_mach_velocity"] = history.alfven_mach_velocity[end]
        metadata["final_alfven_mach_magnetic"] = history.alfven_mach_magnetic[end]
        stats = athena_late_window_stats(history, cfg)
        if stats !== nothing
            metadata["late_window_start_time"] = stats.t_start
            metadata["late_window_relative_band"] = stats.band
            metadata["late_window_signed_change"] = stats.signed_change
            metadata["stability_detected"] = stats.band <= cfg.stability_rel_band && abs(stats.signed_change) <= cfg.stability_abs_change
        end
    end
    if snapshot_rows !== nothing
        metadata["snapshot_diagnostics"] = snapshot_rows
    end
    open(path, "w") do io
        TOML.print(io, metadata; sorted = true)
    end
    return path
end

function run_athena_simulation(config_path::String, settings, positionals::Vector{String})
    cfg = athena_config_from_sources(settings, positionals)
    cfg.snapshot_dt > 0 || error("snapshot_dt must be positive")
    cfg.athena_history_dt > 0 || error("athena_history_dt must be positive")
    cfg.athena_hdf5_dt > 0 || error("athena_hdf5_dt must be positive")
    num_meshblocks = validate_athena_parallelism(cfg)

    ensure_athena_case_dirs!(cfg)
    case_dir = athena_case_root(cfg)
    require_fresh_athena_case(case_dir)
    analysis_dir = athena_analysis_root(cfg)
    preflight_log = joinpath(analysis_dir, "athena_preflight.log")
    open(preflight_log, "w") do io
        println(io, "Athena preflight log")
    end

    athena_status("Running Athena++ reference simulation"; log_path = preflight_log)
    athena_status("Config file: $(config_path)"; log_path = preflight_log)
    athena_status("Case directory: $(case_dir)"; log_path = preflight_log)
    athena_status("Athena source project: $(cfg.athena_project)"; log_path = preflight_log)
    athena_status("CPU parallelism: $(cfg.athena_mpi_ranks) MPI rank(s) x $(cfg.athena_num_threads) OpenMP thread(s), $(num_meshblocks) meshblock(s)"; log_path = preflight_log)

    prepare_athena_build_copy!(cfg; log_path = preflight_log)

    input_path = joinpath(analysis_dir, "athinput.generated")
    csv_path = joinpath(analysis_dir, "energy_history.csv")
    metadata_path = joinpath(analysis_dir, "case_metadata.toml")

    write_athena_generated_input(input_path, cfg)
    command = athena_run_command(cfg, input_path, case_dir; require_executable = false)
    write_athena_metadata(metadata_path, cfg; input_path = input_path, command = command, status = "prepared")

    athena_status("Athena project: $(cfg.athena_project)"; log_path = preflight_log)
    athena_status("Athena input: $(input_path)"; log_path = preflight_log)
    athena_status("Athena command: $(command_text(command))"; log_path = preflight_log)

    if cfg.athena_dry_run
        athena_status("Athena dry run requested; generated input and metadata only."; log_path = preflight_log)
        return nothing
    end

    maybe_configure_and_make_athena!(cfg)
    command = athena_run_command(cfg, input_path, case_dir; require_executable = true)
    write_athena_metadata(metadata_path, cfg; input_path = input_path, command = command, status = "built")

    stdout_path = joinpath(analysis_dir, "athena_run.out.log")
    stderr_path = joinpath(analysis_dir, "athena_run.err.log")
    run_logged(command, case_dir, stdout_path, stderr_path)
    check_athena_run_logs(stdout_path, stderr_path)

    snapshot_rows = collect_athena_snapshots!(cfg, case_dir)
    if isempty(snapshot_rows)
        println("No Athena HDF5 snapshots were found in $(case_dir)")
    else
        println("Copied $(length(snapshot_rows)) Athena HDF5 snapshot(s) into $(athena_snapshot_root(cfg))")
    end

    hst_path = find_athena_hst(cfg, case_dir)
    if hst_path === nothing
        println("No Athena .hst file was found; energy CSV and figure were not generated.")
        write_athena_metadata(metadata_path, cfg; input_path = input_path, command = command, snapshot_rows = snapshot_rows, status = "completed_without_history")
        return nothing
    end

    history = convert_athena_history_to_csv(hst_path, csv_path, cfg)
    attach_athena_snapshot_diagnostics!(snapshot_rows, history)
    write_athena_metadata(metadata_path, cfg; input_path = input_path, command = command, history = history, snapshot_rows = snapshot_rows, status = "completed")

    println("Athena history file: $(hst_path)")
    println("Energy history CSV: $(csv_path)")
    println("Case metadata TOML: $(metadata_path)")
    stats = athena_late_window_stats(history, cfg)
    if stats === nothing
        println("Stability heuristic: insufficient samples")
    else
        println("Late-window start time: $(round(stats.t_start, digits = 4))")
        println("Late-window relative band: $(round(stats.band, digits = 4))")
        println("Late-window signed change: $(round(stats.signed_change, digits = 4))")
        println("Stability heuristic: $((stats.band <= cfg.stability_rel_band && abs(stats.signed_change) <= cfg.stability_abs_change) ? "stable" : "not stable")")
    end

    try
        include(joinpath(@__DIR__, "plot_energy_history.jl"))
        figure_path = Base.invokelatest(() -> getfield(@__MODULE__, :plot_energy_history)(case_dir))
        println("Energy history figure: $(figure_path)")
    catch err
        @warn "Athena run completed, but the energy-history figure could not be generated. Run scripts/plot_energy_history.jl after fixing the plotting environment." exception = (err, catch_backtrace())
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "runner_config.jl"))
    config_path_arg, positionals = split_config_args(ARGS)
    config_path, settings = load_config(config_path_arg)
    run_athena_simulation(config_path, settings, positionals)
end
