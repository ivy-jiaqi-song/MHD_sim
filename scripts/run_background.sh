#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_background.sh [--config configs/config.local.toml] [--solver mhdflows|athena] [--log-root logs] [simulation positional overrides...]

Examples:
  ./scripts/run_background.sh
  ./scripts/run_background.sh --config configs/config.local.toml
  ./scripts/run_background.sh --config configs/config.local.toml --solver athena
  ./scripts/run_background.sh --config configs/config.local.toml 8 0.001 10 0.01 0.01 smoke 0.001 0.01 1234
EOF
}

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_root/.." && pwd)"
config=""
solver=""
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
    --solver)
      [[ $# -ge 2 ]] || { echo "--solver requires a backend name" >&2; exit 2; }
      solver="$2"
      shift 2
      ;;
    --solver=*)
      solver="${1#--solver=}"
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
git_commit="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || true)"

cmd=(julia "$repo_root/scripts/run_simulation.jl")
if [[ -n "$config" ]]; then
  cmd+=(--config "$config")
fi
if [[ -n "$solver" ]]; then
  cmd+=(--solver "$solver")
fi
cmd+=("${run_args[@]}")

{
  printf 'started: %s\n' "$(date -Is)"
  printf 'repo: %s\n' "$repo_root"
  if [[ -n "$git_commit" ]]; then
    printf 'git commit: %s\n' "$git_commit"
  fi
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n\n'
} | tee "$log_file"

nohup "${cmd[@]}" >>"$log_file" 2>&1 &
pid="$!"
printf '%s\n' "$pid" >"$pid_file"

{
  printf 'background pid: %s\n' "$pid"
  printf 'pid file: %s\n' "$pid_file"
  printf '\n'
} | tee -a "$log_file"

echo "Started background simulation."
echo "PID: $pid"
echo "log: $log_file"
echo "pid file: $pid_file"
