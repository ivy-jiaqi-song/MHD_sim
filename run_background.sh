#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_background.sh [--config config.local.toml] [--log-root logs] [simulation positional overrides...]

Examples:
  ./run_background.sh
  ./run_background.sh --config config.local.toml
  ./run_background.sh --config config.local.toml 8 0.001 10 0.01 0.01 smoke 0.001 0.01 1234
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config=""
log_root="logs"
run_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a file path" >&2; exit 2; }
      config="$2"
      shift 2
      ;;
    --config=*)
      config="${1#--config=}"
      shift
      ;;
    --log-root)
      [[ $# -ge 2 ]] || { echo "--log-root requires a directory" >&2; exit 2; }
      log_root="$2"
      shift 2
      ;;
    --log-root=*)
      log_root="${1#--log-root=}"
      shift
      ;;
    --)
      shift
      run_args+=("$@")
      break
      ;;
    *)
      run_args+=("$1")
      shift
      ;;
  esac
done

if [[ "$log_root" != /* ]]; then
  log_root="$repo_root/$log_root"
fi
mkdir -p "$log_root"

stamp="$(date +%Y%m%d_%H%M%S)"
log_file="$log_root/simulation_$stamp.log"
pid_file="$log_root/simulation_$stamp.pid"

cmd=(julia "$repo_root/run_simulation.jl")
if [[ -n "$config" ]]; then
  cmd+=(--config "$config")
fi
cmd+=("${run_args[@]}")

nohup "${cmd[@]}" >"$log_file" 2>&1 &
pid="$!"
printf '%s\n' "$pid" >"$pid_file"

echo "Started background simulation."
echo "PID: $pid"
echo "log: $log_file"
echo "pid file: $pid_file"
