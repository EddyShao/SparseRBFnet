#!/bin/bash

# List of YAML config files (without path)
YAMLS=(
  gaussian_full.yaml
  gaussian_full_smooth.yaml
  gaussian_mix.yaml
  gaussian_mix_smooth.yaml
  gaussian_outer.yaml
  gaussian_outer_smooth.yaml
  matern_full.yaml
  matern_full_smooth.yaml
  matern_mix.yaml
  matern_mix_smooth.yaml
  matern_outer.yaml
  matern_outer_smooth.yaml
)

# Base directories
EXP_DIR="exps/exp_lap"
LOG_DIR="logs/exp_lap"

# Loop over seeds
for SEED in {200..204}; do
  for CFG in "${YAMLS[@]}"; do

    # Remove .yaml to get name
    BASE="${CFG%.yaml}"

    # Save directory used by solver
    SAVE_DIR="${EXP_DIR}/${BASE}_results"

    # Config file path
    CONFIG_PATH="${EXP_DIR}/${CFG}"

    # Log directory and file
    LOG_SAVE_DIR="${LOG_DIR}/${BASE}_results"
    LOG_FILE="${LOG_SAVE_DIR}/${SEED}.log"

    # Create directories if not existing
    mkdir -p "${SAVE_DIR}"
    mkdir -p "${LOG_SAVE_DIR}"

    echo "Running: $CONFIG_PATH | SEED=$SEED"
    echo "Log → $LOG_FILE"

    # Run the command and redirect output to log
    python scripts/solve_pde.py \
        --config "$CONFIG_PATH" \
        --seed "$SEED" \
        --save_dir "$SAVE_DIR" \
        > "$LOG_FILE" 2>&1

  done
done