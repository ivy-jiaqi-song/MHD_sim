#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path=""
backend="mhdflows"
backend_explicit=false
positionals=()

usage() {
  cat <<'EOF'
Usage:
  ./run.sh [--backend mhdflows|athena] [--config config.kolmogorov.example.toml] [runner args...]

Examples:
  ./run.sh --backend mhdflows --config config.kolmogorov.local.toml
  ATHENA_PROJECT=/path/to/athena ./run.sh --backend athena --config config.kolmogorov.local.toml

Positional runner args match the Julia runner:
  [nx] [end_time] [force_amplitude] [viscosity] [tag_suffix] [fixed_dt] [snapshot_dt] [seed]
EOF
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --backend)
      [[ $# -ge 2 ]] || { echo "--backend requires a value" >&2; exit 2; }
      backend="$2"
      backend_explicit=true
      shift 2
      ;;
    --backend=*)
      backend="${1#*=}"
      backend_explicit=true
      shift
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a file path" >&2; exit 2; }
      config_path="$2"
      shift 2
      ;;
    --config=*)
      config_path="${1#*=}"
      shift
      ;;
    --)
      shift
      while (($#)); do
        positionals+=("$1")
        shift
      done
      ;;
    *)
      positionals+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$config_path" ]]; then
  if [[ -f "$script_dir/config.kolmogorov.local.toml" ]]; then
    config_path="$script_dir/config.kolmogorov.local.toml"
  else
    config_path="$script_dir/config.kolmogorov.example.toml"
  fi
fi

if [[ "$backend_explicit" == false && -f "$config_path" ]]; then
  config_backend="$(
    awk -F= '
      /^[[:space:]]*backend[[:space:]]*=/ {
        value=$2
        sub(/#.*/, "", value)
        gsub(/[[:space:]"]/, "", value)
        print value
        exit
      }
    ' "$config_path" 2>/dev/null || true
  )"
  [[ -n "$config_backend" ]] && backend="$config_backend"
fi

case "$backend" in
  mhdflows|MHDFlows)
    exec julia "$script_dir/run_kolmogorov_hd.jl" --config "$config_path" "${positionals[@]}"
    ;;
  athena|Athena)
    exec python3 "$script_dir/backends/athena/run_athena_kolmogorov_hd.py" --config "$config_path" "${positionals[@]}"
    ;;
  *)
    echo "Unknown backend: $backend" >&2
    usage >&2
    exit 2
    ;;
esac
