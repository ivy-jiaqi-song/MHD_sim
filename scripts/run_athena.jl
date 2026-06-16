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
    fixed_dt::Float64 = 0.0
    end_time::Float64 = 60.0
    max_steps::Int = 80_000
    snapshot_dt::Float64 = 5.0
    late_window_fraction::Float64 = 0.25
    stability_rel_band::Float64 = 0.12
    stability_abs_change::Float64 = 0.06
    seed::Int = 1234
    tag_suffix::String = ""
    athena_problem::String = "orszag_tang"
    athena_input_template::String = ""
    athena_problem_id::String = ""
    athena_parse_only::Bool = false
    athena_dry_run::Bool = false
    athena_configure::Bool = false
    athena_make::Bool = false
    athena_configure_args::Vector{String} = ["-b", "--prob=orszag_tang", "--eos=isothermal", "-hdf5"]
    athena_make_args::Vector{String} = String[]
    athena_run_args::Vector{String} = String[]
    athena_cfl_number::Float64 = 0.4
    athena_integrator::String = "vl2"
    athena_xorder::Int = 2
    athena_num_threads::Int = 1
    athena_nx1::Int = 128
    athena_nx2::Int = 128
    athena_nx3::Int = 128
    athena_meshblock_nx1::Int = 128
    athena_meshblock_nx2::Int = 128
    athena_meshblock_nx3::Int = 128
    athena_box_size::Float64 = 1.0
    athena_history_dt::Float64 = 0.1
    athena_hdf5_dt::Float64 = 5.0
    athena_output_variable::String = "prim"
    athena_hdf5_data_format::String = "float32"
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

    cfg = AthenaConfig(;
        output_root = configured_output_root(settings),
        athena_project = athena_project_path(settings),
        athena_executable = athena_as_string(get_config(settings, "athena_executable", "bin/athena")),
        name = athena_as_string(get_config(settings, "name", "compressible_mhd_baseline")),
        description = athena_as_string(get_config(settings, "description", "Athena++ reference MHD sanity check")),
        nx = nx,
        box_size = athena_as_float(get_config(settings, "box_size", 2pi)),
        sound_speed = athena_as_float(get_config(settings, "sound_speed", sqrt(2.0))),
        viscosity = viscosity,
        resistivity = resistivity,
        mean_field = athena_as_tuple3(get_config(settings, "mean_field", [1.0, 0.0, 0.0])),
        forcing_wavenumber = athena_as_float(get_config(settings, "forcing_wavenumber", 2.0)),
        forcing_power = forcing_power,
        forcing_width = athena_as_float(get_config(settings, "forcing_width", 1.0)),
        fixed_dt = fixed_dt,
        end_time = end_time,
        max_steps = athena_as_int(get_config(settings, "max_steps", 80_000)),
        snapshot_dt = snapshot_dt,
        late_window_fraction = athena_as_float(get_config(settings, "late_window_fraction", 0.25)),
        stability_rel_band = athena_as_float(get_config(settings, "stability_rel_band", 0.12)),
        stability_abs_change = athena_as_float(get_config(settings, "stability_abs_change", 0.06)),
        seed = seed,
        tag_suffix = tag_suffix,
        athena_problem = athena_as_string(get_config(settings, "athena_problem", "orszag_tang")),
        athena_input_template = template_path,
        athena_problem_id = athena_as_string(get_config(settings, "athena_problem_id", "")),
        athena_parse_only = athena_as_bool(get_config(settings, "athena_parse_only", false)),
        athena_dry_run = athena_as_bool(get_config(settings, "athena_dry_run", false)),
        athena_configure = athena_as_bool(get_config(settings, "athena_configure", false)),
        athena_make = athena_as_bool(get_config(settings, "athena_make", false)),
        athena_configure_args = athena_as_string_vector(get_config(settings, "athena_configure_args", ["-b", "--prob=orszag_tang", "--eos=isothermal", "-hdf5"])),
        athena_make_args = athena_as_string_vector(get_config(settings, "athena_make_args", String[])),
        athena_run_args = athena_as_string_vector(get_config(settings, "athena_run_args", String[])),
        athena_cfl_number = athena_as_float(get_config(settings, "athena_cfl_number", 0.4)),
        athena_integrator = athena_as_string(get_config(settings, "athena_integrator", "vl2")),
        athena_xorder = athena_as_int(get_config(settings, "athena_xorder", 2)),
        athena_num_threads = athena_as_int(get_config(settings, "athena_num_threads", 1)),
        athena_nx1 = athena_nx1,
        athena_nx2 = athena_nx2,
        athena_nx3 = athena_nx3,
        athena_meshblock_nx1 = mb1,
        athena_meshblock_nx2 = mb2,
        athena_meshblock_nx3 = mb3,
        athena_box_size = athena_as_float(get_config(settings, "athena_box_size", 1.0)),
        athena_history_dt = history_dt,
        athena_hdf5_dt = hdf5_dt,
        athena_output_variable = athena_as_string(get_config(settings, "athena_output_variable", "prim")),
        athena_hdf5_data_format = athena_as_string(get_config(settings, "athena_hdf5_data_format", "float32")),
    )

    cfg.athena_problem_id = isempty(cfg.athena_problem_id) ? sanitize_athena_id("athena_$(athena_case_tag(cfg))") : sanitize_athena_id(cfg.athena_problem_id)
    return cfg
end

function write_athena_generated_input(path::String, cfg::AthenaConfig)
    if !isempty(cfg.athena_input_template)
        isfile(cfg.athena_input_template) || error("Athena input template does not exist: $(cfg.athena_input_template)")
        cp(cfg.athena_input_template, path; force = true)
        return path
    end

    problem = lowercase(strip(cfg.athena_problem))
    problem in ("orszag_tang", "turb") || error("No built-in generated Athena input for athena_problem=$(cfg.athena_problem). Set athena_input_template for custom problems.")

    half_box = cfg.athena_box_size / 2
    open(path, "w") do io
        println(io, "<comment>")
        println(io, "problem   = Generated Athena++ reference input")
        println(io, "reference = Generated by scripts/run_athena.jl")
        println(io, "note      = Stock Athena problem; not a one-to-one clone of the MHDFlows forcing model")
        println(io)
        println(io, "<job>")
        println(io, "problem_id = $(cfg.athena_problem_id)")
        println(io)
        println(io, "<output1>")
        println(io, "file_type = hst")
        println(io, "dt        = $(cfg.athena_history_dt)")
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
        println(io, "nu_iso  = $(cfg.viscosity)")
        println(io, "eta_ohm = $(cfg.resistivity)")

        if problem == "turb"
            nlow = max(0, floor(Int, cfg.forcing_wavenumber - cfg.forcing_width))
            nhigh = max(nlow + 1, ceil(Int, cfg.forcing_wavenumber + cfg.forcing_width))
            println(io)
            println(io, "<turbulence>")
            println(io, "turb_flag = 3")
            println(io, "dedt      = $(cfg.forcing_power)")
            println(io, "nlow      = $(nlow)")
            println(io, "nhigh     = $(nhigh)")
            println(io, "expo      = 2.0")
            println(io, "tcorr     = $(max(cfg.athena_history_dt, eps(Float64)))")
            println(io, "dtdrive   = $(max(cfg.athena_history_dt, eps(Float64)))")
            println(io, "f_shear   = 0.5")
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

command_text(parts::Vector{String}) = join(map(part -> occursin(r"\s", part) ? "\"$(replace(part, "\"" => "\\\""))\"" : part, parts), " ")

function run_logged(command_parts::Vector{String}, working_dir::String, stdout_path::String, stderr_path::String)
    cmd = Cmd(command_parts; dir = working_dir)
    open(stdout_path, "a") do out
        open(stderr_path, "a") do err
            run(pipeline(cmd; stdout = out, stderr = err))
        end
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
    command = [executable, "-i", input_path, "-d", case_dir]
    cfg.athena_parse_only && push!(command, "-n")
    append!(command, cfg.athena_run_args)
    return command
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
    rho_mean = athena_column(rows, columns["mass"]) ./ volume
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
    magnetic_fluct = copy(magnetic_total)
    total_resolved = kinetic .+ magnetic_total
    fluct_total = kinetic .+ magnetic_fluct

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

function find_athena_hst(cfg::AthenaConfig, case_dir::String)
    expected = joinpath(case_dir, "$(cfg.athena_problem_id).hst")
    isfile(expected) && return expected
    candidates = filter(path -> isfile(path) && endswith(lowercase(path), ".hst"), readdir(case_dir; join = true))
    isempty(candidates) && return nothing
    sort!(candidates; by = path -> stat(path).mtime)
    return candidates[end]
end

function write_athena_metadata(path::String, cfg::AthenaConfig; input_path::String, command::Vector{String}, history = nothing, snapshot_rows = nothing, status::String = "prepared")
    metadata = Dict{String, Any}(
        "name" => cfg.name,
        "solver" => "athena",
        "description" => cfg.description,
        "status" => status,
        "output_root" => cfg.output_root,
        "athena_project" => cfg.athena_project,
        "athena_executable" => cfg.athena_executable,
        "athena_problem" => cfg.athena_problem,
        "athena_problem_id" => cfg.athena_problem_id,
        "athena_input" => input_path,
        "athena_command" => command,
        "athena_parse_only" => cfg.athena_parse_only,
        "athena_dry_run" => cfg.athena_dry_run,
        "athena_configure" => cfg.athena_configure,
        "athena_make" => cfg.athena_make,
        "athena_configure_args" => cfg.athena_configure_args,
        "athena_make_args" => cfg.athena_make_args,
        "athena_run_args" => cfg.athena_run_args,
        "athena_reference_note" => "Default generated Athena input is a stock reference problem and is not an exact clone of the MHDFlows driven turbulence setup.",
        "athena_history_note" => "magnetic_fluct mirrors magnetic_total because Athena .hst output does not separate guide-field and fluctuating magnetic energy.",
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
        "fixed_dt" => cfg.fixed_dt,
        "end_time" => cfg.end_time,
        "max_steps" => cfg.max_steps,
        "snapshot_dt" => cfg.snapshot_dt,
        "athena_history_dt" => cfg.athena_history_dt,
        "athena_hdf5_dt" => cfg.athena_hdf5_dt,
        "late_window_fraction" => cfg.late_window_fraction,
        "stability_rel_band" => cfg.stability_rel_band,
        "stability_abs_change" => cfg.stability_abs_change,
        "seed" => cfg.seed,
    )
    if history !== nothing
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

    ensure_athena_case_dirs!(cfg)
    case_dir = athena_case_root(cfg)
    analysis_dir = athena_analysis_root(cfg)
    input_path = joinpath(analysis_dir, "athinput.generated")
    csv_path = joinpath(analysis_dir, "energy_history.csv")
    metadata_path = joinpath(analysis_dir, "case_metadata.toml")

    write_athena_generated_input(input_path, cfg)
    command = athena_run_command(cfg, input_path, case_dir; require_executable = !cfg.athena_dry_run)
    write_athena_metadata(metadata_path, cfg; input_path = input_path, command = command, status = "prepared")

    println("Running Athena++ reference simulation")
    println("Config file: $(config_path)")
    println("Case directory: $(case_dir)")
    println("Athena project: $(cfg.athena_project)")
    println("Athena input: $(input_path)")
    println("Athena command: $(command_text(command))")

    if cfg.athena_dry_run
        println("Athena dry run requested; generated input and metadata only.")
        return nothing
    end

    maybe_configure_and_make_athena!(cfg)

    stdout_path = joinpath(analysis_dir, "athena_run.out.log")
    stderr_path = joinpath(analysis_dir, "athena_run.err.log")
    run_logged(command, case_dir, stdout_path, stderr_path)

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
        figure_path = plot_energy_history(case_dir)
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
