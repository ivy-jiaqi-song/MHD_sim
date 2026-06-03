using TOML

repo_root() = @__DIR__

function resolve_repo_path(path::AbstractString)
    expanded = expanduser(String(path))
    return isabspath(expanded) ? normpath(expanded) : normpath(joinpath(repo_root(), expanded))
end

function default_config_path()
    local_path = joinpath(repo_root(), "config.local.toml")
    example_path = joinpath(repo_root(), "config.example.toml")
    return isfile(local_path) ? local_path : example_path
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
    isdir(resolved) || error("MHDFlows project not found at $(resolved). Set mhdflows_project in config.local.toml or set MHDFLOWS_PROJECT.")
    isfile(joinpath(resolved, "Project.toml")) || error("MHDFlows project is missing Project.toml: $(resolved)")
    return resolved
end

function configured_output_root(settings)
    return resolve_repo_path(String(get_config(settings, "output_root", "outputs")))
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
