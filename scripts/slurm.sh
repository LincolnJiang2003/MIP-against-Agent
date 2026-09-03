#!/usr/bin/env bash
#SBATCH --job-name=mip-experiment
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -u

CONFIG_PATH=${CONFIG_PATH:-configs/desktop.yaml}
RUN_ID=${RUN_ID:-${SLURM_JOB_ID}}
TARGET_FILE=${TARGET_FILE:-}

args=(mip-attack run --config "$CONFIG_PATH" --run-id "$RUN_ID")
if [[ -n "$TARGET_FILE" ]]; then
  args+=(--target-file "$TARGET_FILE")
fi
srun "${args[@]}"
status=$?
if [[ $status -eq 75 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
