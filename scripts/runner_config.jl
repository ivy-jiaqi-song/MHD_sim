using TOML

repo_root() = normpath(joinpath(@__DIR__, ".."))
config_root() = joinpath(repo_root(), "configs")

function resolve_repo_path(path::AbstractString)
    expanded = expanduser(String(path))
    return isabspath(expanded) ? normpath(expanded) : normpath(joinpath(repo_root(), expanded))
end

function default_config_path()
    local_path = joinpath(config_root(), "config.local.toml")
    legacy_local_path = joinpath(repo_root(), "config.local.toml")
    example_path = joinpath(config_root(), "config.example.toml")
    if isfile(local_path)
        return local_path
    elseif isfile(legacy_local_path)
        return legacy_local_path
    end
    return example_path
end

function load_config(path::Union{Nothing, AbstractString})
    config_path = path === nothing ? default_config_path() : resolve_repo_path(path)
    isfile(config_path) || error("Config file does not exist: $(config_path)")
    return config_path, TOML.parsefile(config_path)
end

function get_config(settings, key::String, default)
    return haskey(settings, key) ? settings[key] : default
end

function mhdflows_project_path(settings)
    configured = String(get_config(settings, "mhdflows_project", "MHDFlows_dev-main"))
    path = get(ENV, "MHDFLOWS_PROJECT", configured)
    resolved = resolve_repo_path(path)
    isdir(resolved) || error("MHDFlows project not found at $(resolved). Set mhdflows_project in configs/config.local.toml or set MHDFLOWS_PROJECT.")
    isfile(joinpath(resolved, "Project.toml")) || error("MHDFlows project is missing Project.toml: $(resolved)")
    return resolved
end

function athena_project_path(settings)
    configured = String(get_config(settings, "athena_project", "athena"))
    path = get(ENV, "ATHENA_PROJECT", configured)
    resolved = resolve_repo_path(path)
    isdir(resolved) || error("Athena project not found at $(resolved). Set athena_project in configs/config.local.toml or set ATHENA_PROJECT.")
    isfile(joinpath(resolved, "configure.py")) || error("Athena project is missing configure.py: $(resolved)")
    return resolved
end

function configured_output_root(settings)
    return resolve_repo_path(String(get_config(settings, "output_root", "outputs")))
end

function configured_solver(settings, override::Union{Nothing, AbstractString} = nothing)
    raw = override === nothing ? get_config(settings, "solver", "mhdflows") : override
    solver = lowercase(strip(String(raw)))
    solver in ("mhdflows", "athena") || error("solver must be \"mhdflows\" or \"athena\"; got \"$(raw)\"")
    return solver
end

function split_config_args(args::Vector{String})
    config_path = nothing
    positionals = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--config"
            i < length(args) || error("--config requires a file path")
            config_path = args[i + 1]
            i += 2
        elseif startswith(arg, "--config=")
            config_path = split(arg, "=", limit = 2)[2]
            i += 1
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        else
            push!(positionals, arg)
            i += 1
        end
    end
    return config_path, positionals
end

function split_runner_args(args::Vector{String})
    config_path = nothing
    solver = nothing
    positionals = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--config"
            i < length(args) || error("--config requires a file path")
            config_path = args[i + 1]
            i += 2
        elseif startswith(arg, "--config=")
            config_path = split(arg, "=", limit = 2)[2]
            i += 1
        elseif arg == "--solver"
            i < length(args) || error("--solver requires a backend name")
            solver = args[i + 1]
            i += 2
        elseif startswith(arg, "--solver=")
            solver = split(arg, "=", limit = 2)[2]
            i += 1
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        else
            push!(positionals, arg)
            i += 1
        end
    end
    return config_path, solver, positionals
end
