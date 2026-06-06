if !isempty(ARGS) && ARGS[1] in ["-h", "--help"]
    println("Usage:")
    println("  julia run_kolmogorov_hd.jl [--config config.kolmogorov.example.toml]")
    println("  julia run_kolmogorov_hd.jl [--config config.kolmogorov.example.toml] [nx] [end_time] [force_amplitude] [viscosity] [tag_suffix] [fixed_dt] [snapshot_dt] [seed]")
    println()
    println("Default config: config.kolmogorov.example.toml.")
    println("For the paper benchmark use domain_length = 1, force_amplitude = 0.1, force_mode_y = 2, viscosity = 1e-6.")
    exit(0)
end

import Pkg

include(joinpath(@__DIR__, "runner_config.jl"))

config_path_arg, positionals = split_config_args(ARGS)
config_path = config_path_arg === nothing ? joinpath(@__DIR__, "config.kolmogorov.example.toml") : config_path_arg
config_path, settings = load_config(config_path)

Pkg.activate(mhdflows_project_path(settings))
include(joinpath(@__DIR__, "kolmogorov_hd_support.jl"))

cfg = kolmogorov_config_from_sources(settings, positionals)
cfg.snapshot_dt > 0 || error("snapshot_dt must be positive")
cfg.diagnostics_sample_every > 0 || error("diagnostics_sample_every must be positive")
cfg.nz >= 2 && iseven(cfg.nz) || error("MHDFlows/FourierFlows requires even nz; use nz = 2 for a degenerate 2D run")
cfg.fixed_dt > 0 || error("Use a positive fixed_dt for this high-Reynolds-number Kolmogorov run")
warn_reynolds_viscosity_mismatch(cfg)

case_dir = kolmogorov_case_root(cfg)
csv_path = joinpath(kolmogorov_analysis_root(cfg), "energy_enstrophy_history.csv")
metadata_path = joinpath(kolmogorov_analysis_root(cfg), "case_metadata.toml")
snapshot_prefix = joinpath(kolmogorov_snapshot_root(cfg), "state")

println("Running 2D incompressible Kolmogorov flow")
println("Config file: $(config_path)")
println("Case directory: $(case_dir)")

if cfg.reuse_existing_data && kolmogorov_case_has_data(case_dir)
    println("Existing Kolmogorov HD data found; skipping simulation")
    if cfg.plot_after_run
        include(joinpath(@__DIR__, "plot_kolmogorov_hd.jl"))
        paths = plot_kolmogorov_hd(case_dir)
        println("Vorticity snapshots figure: $(paths.vorticity)")
        println("v_y vs v_x snapshots figure: $(paths.velocity_phase)")
        println("Energy/enstrophy figure: $(paths.history)")
        println("Final spectrum figure: $(paths.spectrum)")
    else
        println("Automatic plotting is disabled")
    end
    exit(0)
end

ensure_kolmogorov_case_dirs!(cfg)

history = KolmogorovHistory(cfg.diagnostics_sample_every)
snapshot_writer = SnapshotWriter(snapshot_prefix, cfg.snapshot_dt, 0.0, 0, NaN)
callback = make_kolmogorov_callback(history, csv_path, cfg, snapshot_writer)
prob, device_label = build_kolmogorov_problem(cfg; usr_func = [callback])
write_kolmogorov_metadata(metadata_path, cfg, device_label)

println("Device: $(device_label)")
println("Resolution: $(cfg.nx) x $(cfg.ny) x $(cfg.nz)")
println("Domain: [0, $(cfg.domain_length)]^2")
println("2D embedding: fields and forcing are z-independent; nz = $(cfg.nz) only satisfies the package grid constraint")
println("Viscosity: $(cfg.viscosity) (effective Re = $(effective_reynolds(cfg)))")
println("Force: f_x = $(cfg.force_amplitude) * sin(2*pi*$(cfg.force_mode_y)*y/L), f_y = 0, f_z = 0")
println("Initial condition: $(cfg.initial_condition)")
println("Initial velocity RMS for fourier_divfree: $(cfg.initial_velocity_rms)")
println("Grid-noise component amplitude: $(cfg.grid_noise_velocity_amplitude)")
println("Runtime no-dealias shim: $(cfg.disable_package_dealiasing)")
println("Fixed dt: $(cfg.fixed_dt)")
println("Target end time: $(cfg.end_time)")
println("Snapshot interval: $(cfg.snapshot_dt)")
println("Automatic plotting: $(cfg.plot_after_run)")
println("Reuse existing data: $(cfg.reuse_existing_data)")
println("Random seed: $(cfg.seed)")

sample_kolmogorov!(history, prob, cfg; force = true)
write_kolmogorov_csv(csv_path, history)
save_kolmogorov_snapshot!(snapshot_writer, prob; force = true)

MHDFlows.TimeIntegrator!(prob, cfg.end_time, cfg.max_steps;
    usr_dt = cfg.fixed_dt,
    dynamic_dashboard = false,
    loop_number = typemax(Int),
    save = false,
)

sample_kolmogorov!(history, prob, cfg; force = true)
write_kolmogorov_csv(csv_path, history)
save_kolmogorov_snapshot!(snapshot_writer, prob; force = true)

last_sample = lastindex(history.times)
println("Diagnostics CSV: $(csv_path)")
println("Snapshot directory: $(kolmogorov_snapshot_root(cfg))")
println("Final time: $(round(history.times[last_sample], digits = 6))")
println("Final kinetic energy: $(history.kinetic[last_sample])")
println("Final enstrophy: $(history.enstrophy[last_sample])")
println("Final velocity RMS: $(history.velocity_rms[last_sample])")
println("Final divergence RMS: $(history.divergence_rms[last_sample])")

if cfg.plot_after_run
    include(joinpath(@__DIR__, "plot_kolmogorov_hd.jl"))
    paths = plot_kolmogorov_hd(case_dir)
    println("Vorticity snapshots figure: $(paths.vorticity)")
    println("v_y vs v_x snapshots figure: $(paths.velocity_phase)")
    println("Energy/enstrophy figure: $(paths.history)")
    println("Final spectrum figure: $(paths.spectrum)")
end
