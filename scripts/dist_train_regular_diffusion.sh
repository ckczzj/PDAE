MASTER_ADDR=127.0.0.1
MASTER_PORT=3456
NNODES=$1
NODE_RANK=$2
GPUS=$3

ROOT_DIR="$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}

torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
--nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS}" \
"${ROOT_DIR}/trainer/train_regular_diffusion.py" \
--config_path "${ROOT_DIR}/config/mnist_regular.yml" \
--run_path "${ROOT_DIR}/diffusion_output/mnist_regular"

# torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
# --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS}" \
# "${ROOT_DIR}/trainer/"train_regular_diffusion.py \
# --config_path "${ROOT_DIR}/diffusion_output/mnist_regular/config.yml" \
# --run_path "${ROOT_DIR}/diffusion_output/mnist_regular" \
# --resume "${ROOT_DIR}/diffusion_output/mnist_regular/checkpoints/latest.pt"