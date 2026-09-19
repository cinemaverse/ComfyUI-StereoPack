#!/usr/bin/env bash
# Train ONE noise stage of a same-instant stereo LoRA (Wan 2.2 I2V A14B).
#
#   ./train.sh <low|high> <dataset_toml> [epochs] [output_name] [extra musubi args...]
#
# low  -> timesteps 0..875     high -> timesteps 875..1000
# The two stages MUST be separate runs: --offload_inactive_dit needs ~96 GB of system RAM.
#
# This is stage 2 of the recipe in ../docs/TRAIN_YOUR_OWN.md -- the continuation that
# fixes pacing. Set INIT to a stage-1 checkpoint (epoch 10-20, not the last one) and use
# a stride-3 dataset. All paths below are examples: edit them.
set -euo pipefail

# ----------------------------------------------------------------------------- edit me
PY="${PY:-python}"                       # a python with musubi-tuner's deps installed
MUSUBI="${MUSUBI:-$HOME/musubi-tuner}"   # musubi-tuner checkout
VAE="${VAE:-wan_2.1_vae.safetensors}"    # ComfyUI/models/vae/
T5="${T5:-models_t5_umt5-xxl-enc-bf16.pth}"
DIT_LOW="${DIT_LOW:-wan2.2_i2v_low_noise_14B_fp16.safetensors}"
DIT_HIGH="${DIT_HIGH:-wan2.2_i2v_high_noise_14B_fp16.safetensors}"
OUT_DIR="${OUT_DIR:-./out}"
# INIT: a checkpoint to continue from. Empty = train from the base model.
#   stage 1: your base stereo LoRA (or empty)
#   stage 2: <OUT_DIR>/sameinstant_low-000020.safetensors
INIT="${INIT:-}"
# -------------------------------------------------------------------------------------

# Half the usual rate: the delta keeps growing ~1.1x per 10 epochs and the freeze grows
# with it, so a full-rate continuation inflates intrusion while fixing pacing.
LR="${LR:-1.5e-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"

WHICH="${1:?usage: train.sh <low|high> <dataset_toml> [epochs] [output_name] [extra...]}"
CFG="${2:?usage: train.sh <low|high> <dataset_toml> [epochs] [output_name] [extra...]}"
EPOCHS="${3:-20}"
NAME="${4:-sameinstant_${WHICH}}"
shift $(( $# < 4 ? $# : 4 ))
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"   # musubi cds away: make absolute

if [ "$WHICH" = "low" ]; then
  DIT="$DIT_LOW";  MIN=0;   MAX=875;  SEED=42
elif [ "$WHICH" = "high" ]; then
  DIT="$DIT_HIGH"; MIN=875; MAX=1000; SEED=41
else
  echo "first arg must be low or high"; exit 2
fi

WEIGHTS=()
if [ -n "$INIT" ]; then
  [ -f "$INIT" ] || { echo "INIT not found: $INIT"; exit 3; }
  WEIGHTS=(--network_weights "$INIT")
fi

echo "== $WHICH  epochs=$EPOCHS  lr=$LR  init=${INIT:-<base model>}"
"$PY" "$MUSUBI/src/musubi_tuner/wan_train_network.py" \
  --task i2v-A14B \
  --dit "$DIT" "${WEIGHTS[@]}" \
  --vae "$VAE" --t5 "$T5" \
  --dataset_config "$CFG" \
  --sdpa --mixed_precision fp16 --fp8_base \
  --blocks_to_swap 20 --use_pinned_memory_for_block_swap \
  --optimizer_type adamw --learning_rate "$LR" --gradient_checkpointing \
  --network_module networks.lora_wan --network_dim 16 --network_alpha 16 \
  --timestep_sampling shift --discrete_flow_shift 5.0 \
  --min_timestep "$MIN" --max_timestep "$MAX" --preserve_distribution_shape \
  --max_train_epochs "$EPOCHS" --save_every_n_epochs "$SAVE_EVERY" --seed "$SEED" \
  --max_data_loader_n_workers 1 --persistent_data_loader_workers \
  --output_dir "$OUT_DIR" --output_name "$NAME" "$@"
