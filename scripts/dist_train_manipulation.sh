MASTER_ADDR=127.0.0.1
MASTER_PORT=3456
NNODES=$1
NODE_RANK=$2
GPUS=$3

ROOT_DIR="$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}

torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
--nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS}" \
"${ROOT_DIR}/trainer/train_manipulation.py" \
--config_path "${ROOT_DIR}/config/celebahq_manipulation.yml" \
--run_path "${ROOT_DIR}/diffusion_output/celebahq_manipulation"

# torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
# --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS}" \
# "${ROOT_DIR}/trainer/"train_manipulation.py \
# --config_path "${ROOT_DIR}/diffusion_output/celebahq_manipulation/config.yml" \
# --run_path "${ROOT_DIR}/diffusion_output/celebahq_manipulation" \
# --resume "${ROOT_DIR}/diffusion_output/celebahq_manipulation/checkpoints/latest.pt"