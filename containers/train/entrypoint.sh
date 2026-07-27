#!/usr/bin/env bash
set -euo pipefail

export GRADLAB_PROJECT_ROOT="${GRADLAB_PROJECT_ROOT:-/root/gradlab}"
export PYTHONPATH="${GRADLAB_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${GRADLAB_PROJECT_ROOT}/runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/.wandb-data}"

mkdir -p "$MPLCONFIGDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

if [ "$#" -eq 0 ]; then
  exec gradlab-container-smoke
fi

exec "$@"
