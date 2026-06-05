if !isempty(ARGS) && ARGS[1] in ["-h", "--help"]
    println("Usage:")
    println("  julia scripts/run_simulation.jl [--config configs/config.local.toml]")
    println("  julia scripts/run_simulation.jl [--config configs/config.local.toml] [nx] [end_time] [forcing_power] [viscosity] [resistivity] [tag_suffix] [fixed_dt] [snapshot_dt] [seed]")
    println()
    println("Default config order: configs/config.local.toml, legacy config.local.toml, then configs/config.example.toml.")
    println("Set fixed_dt = 0 to use the solver's adaptive CFL timestep.")
    exit(0)
end

import Pkg

include(joinpath(@__DIR__, "runner_config.jl"))

config_path_arg, positionals = split_config_args(ARGS)
config_path, settings = load_config(config_path_arg)

Pkg.activate(mhdflows_project_path(settings))
include(joinpath(@__DIR__, "simulation_support.jl"))

cfg = config_from_sources(settings, positionals)
cfg.snapshot_dt > 0 || error("snapshot_dt must be positive")
cfg.energy_sample_every > 0 || error("energy_sample_every must be positive")

ensure_case_dirs!(cfg)
case_dir = case_root(cfg)
csv_path = joinpath(analysis_root(cfg), "energy_history.csv")
metadata_path = joinpath(analysis_root(cfg), "case_metadata.toml")

println("Running compressible MHD simulation")
println("Config file: $(config_path)")
println("Case directory: $(case_dir)")

history = EnergyHistory(cfg.energy_sample_every, cfg.sound_speed)
callback_energy = make_history_callback(history, csv_path)
prob, device_label = build_problem(cfg; usr_func = [callback_energy])
write_case_metadata(metadata_path, cfg, device_label)

println("Device: $(device_label)")
println("Resolution: $(cfg.nx)^3")
println("Sound speed: $(cfg.sound_speed)")
println("Target end time: $(cfg.end_time)")
println("Forcing power parameter: $(cfg.forcing_power)")
println("Viscosity / resistivity: $(cfg.viscosity), $(cfg.resistivity)")
println("Snapshot interval: $(cfg.snapshot_dt)")
println("Random seed: $(cfg.seed)")

sample_energy!(history, prob; force = true)
write_energy_csv(csv_path, history)

MHDFlows.TimeIntegrator!(prob, cfg.end_time, cfg.max_steps;
    usr_dt = cfg.fixed_dt,
    dynamic_dashboard = false,
    loop_number = 200,
    save = true,
    save_loc = snapshot_root(cfg) * "/",
    filename = "state",
    dump_dt = cfg.snapshot_dt,
)

sample_energy!(history, prob; force = true)
write_energy_csv(csv_path, history)

stats = late_window_stats(history, cfg)
println("Energy history CSV: $(csv_path)")
println("Snapshot directory: $(snapshot_root(cfg))")
if !isempty(history.times)
    last_sample = lastindex(history.times)
    println("Final sonic Mach: $(round(history.sonic_mach[last_sample], digits = 4))")
    println("Final Alfven Mach mean/total/fluct: $(round(history.alfven_mach_mean[last_sample], digits = 4)), $(round(history.alfven_mach_total[last_sample], digits = 4)), $(round(history.alfven_mach_fluct[last_sample], digits = 4))")
end
if stats === nothing
    println("Stability heuristic: insufficient samples")
else
    println("Late-window start time: $(round(stats.t_start, digits = 4))")
    println("Late-window relative band: $(round(stats.band, digits = 4))")
    println("Late-window signed change: $(round(stats.signed_change, digits = 4))")
    println("Stability heuristic: $(stability_detected(history, cfg) ? "stable" : "not stable")")
end

try
    include(joinpath(@__DIR__, "plot_energy_history.jl"))
    figure_path = plot_energy_history(case_dir)
    println("Energy history figure: $(figure_path)")
catch err
    @warn "Simulation completed, but the energy-history figure could not be generated. Run scripts/plot_energy_history.jl after fixing the plotting environment." exception = (err, catch_backtrace())
end
