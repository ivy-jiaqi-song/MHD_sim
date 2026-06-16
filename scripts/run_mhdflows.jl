function run_mhdflows_simulation(config_path::String, settings, positionals::Vector{String})
    import Pkg

    Pkg.activate(mhdflows_project_path(settings))
    include(joinpath(@__DIR__, "simulation_support.jl"))

    cfg = config_from_sources(settings, positionals)
    cfg.snapshot_dt > 0 || error("snapshot_dt must be positive")
    cfg.energy_sample_every > 0 || error("energy_sample_every must be positive")

    ensure_case_dirs!(cfg)
    case_dir = case_root(cfg)
    csv_path = joinpath(analysis_root(cfg), "energy_history.csv")
    metadata_path = joinpath(analysis_root(cfg), "case_metadata.toml")

    println("Running compressible MHD simulation with MHDFlows")
    println("Config file: $(config_path)")
    println("Case directory: $(case_dir)")

    history = EnergyHistory(cfg.energy_sample_every)
    snapshot_diagnostics = SnapshotDiagnosticsHistory(cfg.snapshot_dt, cfg.sound_speed)
    device_label_ref = Ref("")
    callback_energy = make_history_callback(history, csv_path)
    callback_snapshots = make_snapshot_metadata_callback(snapshot_diagnostics, metadata_path, cfg, device_label_ref)
    prob, device_label = build_problem(cfg; usr_func = [callback_energy, callback_snapshots])
    device_label_ref[] = device_label
    initialize_snapshot_diagnostics!(snapshot_diagnostics, prob)
    write_case_metadata(metadata_path, cfg, device_label; snapshot_diagnostics = snapshot_diagnostics)

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
    write_case_metadata(metadata_path, cfg, device_label; snapshot_diagnostics = snapshot_diagnostics)

    stats = late_window_stats(history, cfg)
    println("Energy history CSV: $(csv_path)")
    println("Case metadata TOML: $(metadata_path)")
    println("Snapshot directory: $(snapshot_root(cfg))")
    if !isempty(snapshot_diagnostics.times)
        last_snapshot = lastindex(snapshot_diagnostics.times)
        println("Last snapshot sonic Mach: $(round(snapshot_diagnostics.sonic_mach[last_snapshot], digits = 4))")
        println("Last snapshot Alfven Mach mean/total/fluct: $(round(snapshot_diagnostics.alfven_mach_mean[last_snapshot], digits = 4)), $(round(snapshot_diagnostics.alfven_mach_total[last_snapshot], digits = 4)), $(round(snapshot_diagnostics.alfven_mach_fluct[last_snapshot], digits = 4))")
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
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "runner_config.jl"))
    config_path_arg, positionals = split_config_args(ARGS)
    config_path, settings = load_config(config_path_arg)
    run_mhdflows_simulation(config_path, settings, positionals)
end
