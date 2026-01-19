#!/bin/bash

# Base directories
EXP_DIR="exps/exp_bi"
LOG_DIR="logs/exp_bi"

# Collect ALL yaml files matching bilap_*.yaml EXCEPT base.yaml
YAMLS=($(ls ${EXP_DIR}/bilap_*.yaml | grep -v "base.yaml"))

# Loop over seeds
for SEED in {200..209}; do
  for CONFIG_PATH in "${YAMLS[@]}"; do

    # Extract filename only
    CFG=$(basename "$CONFIG_PATH")

    # Remove .yaml → base name
    BASE="${CFG%.yaml}"

    # Save directory for solver output
    SAVE_DIR="${EXP_DIR}/${BASE}_results"

    # Log directory + file
    LOG_SAVE_DIR="${LOG_DIR}/${BASE}_results"
    LOG_FILE="${LOG_SAVE_DIR}/${SEED}.log"

    # Create directories if not existing
    mkdir -p "${SAVE_DIR}"
    mkdir -p "${LOG_SAVE_DIR}"

    if ls "${SAVE_DIR}"/*"${SEED}"*.pkl >/dev/null 2>&1; then
        echo "SKIP: Results exist for ${BASE}, seed ${SEED}"
        continue
    fi

    echo "--------------------------------------------------"
    echo "Running config: $CONFIG_PATH"

    echo "Seed: $SEED"
    echo "Results → $SAVE_DIR"
    echo "Log → $LOG_FILE"
    echo "--------------------------------------------------"

    # Execute solver and write log
    python scripts/solve_pde_aux.py \
        --config "$CONFIG_PATH" \
        --seed "$SEED" \
        --save_dir "$SAVE_DIR" \
        > "$LOG_FILE" 2>&1

  done
done