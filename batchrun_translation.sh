#!/usr/bin/env bash
# scripts/batchrun.sh
set -euo pipefail

# ---------- Required (passed to Python) ----------
DATA_DIR=""         # ABS path containing amazon_reviews_multi/
OUTPUT_ROOT=""      # ABS path for translated output mirror
MODEL_DIR=""        # ABS path to store/load NLLB snapshot
HF_TOKEN=""         # Hugging Face token (required)
HF_REPO_ID="facebook/nllb-200-3.3B"

# ---------- Slurm defaults ----------
ACCOUNT="${ACCOUNT:-ds_ga_1011_001-2025fa}"
PARTITION="${PARTITION:-c12m85-a100-1}"
TIME="${TIME:-12:00:00}"
GRES="${GRES:-gpu:1}"
JOB_NAME="${JOB_NAME:-translate}"

# ---------- Singularity / Conda ----------
OVERLAY="${OVERLAY:-/scratch/$USER/nlp/overlay-50G-10M-fixed.ext3}"
SIF="${SIF:-/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
CONDA_INIT="${CONDA_INIT:-/ext3/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-nlp}"

# ---------- Python entrypoint ----------
PYTHON_FILE="${PYTHON_FILE:-/scratch/$USER/nlp/translation.py}"

# ---------- Translator knobs ----------
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BEAM_SIZE="${BEAM_SIZE:-3}"
DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LANGS="${LANGS:-ja fr zh}"
SPLITS="${SPLITS:-train validation test}"
BEGIN="${BEGIN:-0}"
RESUME="${RESUME:-true}"

usage() {
  echo "Usage: $0 --data_dir ABS --output_root ABS --model_dir ABS --hf_token TOKEN \
[--python_file PATH] [--hf_repo_id REPO] [--beam_size N] \
[--account ACC] [--partition PART] [--time HH:MM:SS] [--gres SPEC] [--job_name NAME] \
[--overlay PATH] [--sif PATH] [--conda_init PATH] [--conda_env NAME] \
[--batch_size N] [--max_new_tokens N] [--dtype DTYPE] \
[--device_map MAP] [--langs \"ja fr zh\"] [--splits \"train validation test\"] \
[--begin N] [--resume true|false]" >&2
  exit 2
}

# ---------- CLI ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2;;
    --output_root) OUTPUT_ROOT="$2"; shift 2;;
    --model_dir) MODEL_DIR="$2"; shift 2;;
    --hf_token) HF_TOKEN="$2"; shift 2;;
    --python_file) PYTHON_FILE="$2"; shift 2;;
    --hf_repo_id) HF_REPO_ID="$2"; shift 2;;
    --beam_size) BEAM_SIZE="$2"; shift 2;;
    --account) ACCOUNT="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --gres) GRES="$2"; shift 2;;
    --job_name) JOB_NAME="$2"; shift 2;;
    --overlay) OVERLAY="$2"; shift 2;;
    --sif) SIF="$2"; shift 2;;
    --conda_init) CONDA_INIT="$2"; shift 2;;
    --conda_env) CONDA_ENV="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --device_map) DEVICE_MAP="$2"; shift 2;;
    --langs) LANGS="$2"; shift 2;;
    --splits) SPLITS="$2"; shift 2;;
    --begin) BEGIN="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    *) usage;;
  esac
done

# ---------- Validate ----------
for var in DATA_DIR OUTPUT_ROOT MODEL_DIR PYTHON_FILE; do
  val="${!var}"
  [[ -z "$val" ]] && echo "Missing required: --$(echo "$var" | tr '[:upper:]' '[:lower:]')" >&2 && usage
  [[ "${val:0:1}" != "/" ]] && echo "Path for $var must be absolute: $val" >&2 && exit 2
done
[[ -z "$HF_TOKEN" ]] && { echo "Missing required: --hf_token"; exit 2; }

# ---------- Submit ----------
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=${TIME}
#SBATCH --gres=${GRES}
#SBATCH --requeue

set -euo pipefail

echo "[DEBUG] Node: \$(hostname)"
echo "[DEBUG] Working dir: \$(pwd)"

export SINGULARITYENV_HF_TOKEN="${HF_TOKEN}"
export SINGULARITYENV_TRANSFORMERS_VERBOSITY="error"

singularity exec --bind /scratch --nv --overlay ${OVERLAY}:ro ${SIF} /bin/bash -lc "
  set -euo pipefail
  source ${CONDA_INIT}
  conda activate ${CONDA_ENV}

  echo '[INFO] Starting translation job with BEGIN=${BEGIN}, RESUME=${RESUME}'

  python ${PYTHON_FILE} \\
    --data_dir ${DATA_DIR} \\
    --output_root ${OUTPUT_ROOT} \\
    --model_dir ${MODEL_DIR} \\
    --hf_repo_id ${HF_REPO_ID} \\
    --hf_token \"\$HF_TOKEN\" \\
    --languages \"${LANGS}\" \\
    --splits \"${SPLITS}\" \\
    --begin ${BEGIN} \\
    --batch_size ${BATCH_SIZE} \\
    --max_new_tokens ${MAX_NEW_TOKENS} \\
    --beam_size ${BEAM_SIZE} \\
    --dtype ${DTYPE} \\
    --device_map ${DEVICE_MAP} \\
    \$([[ '${RESUME}' == 'true' ]] && echo --resume)
"
EOF
