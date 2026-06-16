if !isempty(ARGS) && ARGS[1] in ["-h", "--help"]
    println("Usage:")
    println("  julia scripts/run_simulation.jl [--config configs/config.local.toml] [--solver mhdflows|athena]")
    println("  julia scripts/run_simulation.jl [--config configs/config.local.toml] [--solver mhdflows|athena] [nx] [end_time] [forcing_power] [viscosity] [resistivity] [tag_suffix] [fixed_dt] [snapshot_dt] [seed]")
    println()
    println("Default config order: configs/config.local.toml, legacy config.local.toml, then configs/config.example.toml.")
    println("The solver defaults to the config value `solver`, or `mhdflows` when omitted.")
    println("Set fixed_dt = 0 to use the solver's adaptive CFL timestep where supported.")
    exit(0)
end

include(joinpath(@__DIR__, "runner_config.jl"))

println("MHD_sim launcher started")
println("Working directory: $(pwd())")
println("Entry script: $(abspath(PROGRAM_FILE))")
println("Arguments: $(isempty(ARGS) ? "<none>" : join(ARGS, " "))")
flush(stdout)

config_path_arg, solver_arg, positionals = split_runner_args(ARGS)
config_path, settings = load_config(config_path_arg)
solver = configured_solver(settings, solver_arg)

println("Loaded config: $(config_path)")
println("Selected solver: $(solver)")
flush(stdout)

if solver == "mhdflows"
    include(joinpath(@__DIR__, "run_mhdflows.jl"))
    run_mhdflows_simulation(config_path, settings, positionals)
elseif solver == "athena"
    include(joinpath(@__DIR__, "run_athena.jl"))
    run_athena_simulation(config_path, settings, positionals)
else
    error("Unsupported solver: $(solver)")
end
