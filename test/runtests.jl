using Test

include(joinpath(@__DIR__, "..", "scripts", "runner_config.jl"))
include(joinpath(@__DIR__, "..", "scripts", "run_athena.jl"))

@testset "Athena history diagnostics" begin
    mktempdir() do temp_dir
        cfg = AthenaConfig(
            output_root = temp_dir,
            athena_project = temp_dir,
            athena_box_size = 2.0,
            sound_speed = 2.0,
            mean_field = (1.0, 0.0, 0.0),
        )
        hst_path = joinpath(temp_dir, "fixture.hst")
        csv_path = joinpath(temp_dir, "energy_history.csv")
        write(hst_path, """
# Athena++ history data
# [1]=time [2]=dt [3]=mass [4]=1-mom [5]=2-mom [6]=3-mom [7]=1-KE [8]=2-KE [9]=3-KE [10]=1-ME [11]=2-ME [12]=3-ME
0.0 0.1 8.0 0.0 0.0 0.0 4.0 0.0 0.0 5.0 0.0 0.0
0.1 0.1 8.0 0.0 0.0 0.0 4.0 0.0 0.0 5.0 0.0 0.0
""")

        history = convert_athena_history_to_csv(hst_path, csv_path, cfg)
        @test history.rho_mean == [1.0, 1.0]
        @test history.magnetic_fluct == [0.125, 0.125]
        @test history.velocity_fluct_rms == [1.0, 1.0]
        @test history.sonic_mach == [0.5, 0.5]
        @test history.alfven_mach_velocity == [1.0, 1.0]
        @test history.alfven_mach_magnetic == [0.5, 0.5]

        rows = [Dict{String, Any}("time_estimate" => 0.1)]
        attach_athena_snapshot_diagnostics!(rows, history)
        @test rows[1]["alfven_mach_velocity"] == 1.0
        @test rows[1]["alfven_mach_magnetic"] == 0.5
        @test rows[1]["sonic_mach"] == 0.5
    end
end

@testset "Athena CPU parallelism" begin
    cfg = AthenaConfig(
        output_root = tempdir(),
        athena_project = tempdir(),
        athena_executable = "bin/athena",
        athena_configure_args = ["-b", "-omp", "-mpi"],
        athena_mpi_ranks = 2,
        athena_mpi_launcher = "mpirun",
        athena_mpi_args = ["--bind-to", "core"],
        athena_num_threads = 2,
        athena_nx1 = 8,
        athena_nx2 = 8,
        athena_nx3 = 8,
        athena_meshblock_nx1 = 4,
        athena_meshblock_nx2 = 4,
        athena_meshblock_nx3 = 4,
    )
    @test validate_athena_parallelism(cfg) == 8
    command = athena_run_command(cfg, "/tmp/athinput", "/tmp/case"; require_executable = false)
    @test command[1:6] == ["mpirun", "--bind-to", "core", "-n", "2", joinpath(tempdir(), "bin", "athena")]

    cfg.athena_configure_args = ["-b", "-mpi"]
    @test_throws ErrorException validate_athena_parallelism(cfg)

    mktempdir() do temp_dir
        stdout_path = joinpath(temp_dir, "stdout.log")
        stderr_path = joinpath(temp_dir, "stderr.log")
        write(stdout_path, "### FATAL ERROR in Athena\n")
        write(stderr_path, "")
        @test_throws ErrorException check_athena_run_logs(stdout_path, stderr_path)

        @test isnothing(require_fresh_athena_case(temp_dir))
        write(joinpath(temp_dir, "existing.hst"), "old run")
        @test_throws ErrorException require_fresh_athena_case(temp_dir)
    end
end
