MASTER_ADDR=127.0.0.1
MASTER_PORT=3456
NNODES=$1
NODE_RANK=$2
GPUS=$3

ROOT_DIR="$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}

torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
--nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS}" \
"${ROOT_DIR}/trainer/train_latent_diffusion.py" \
--config_path "${ROOT_DIR}/config/ffhq_latent.yml" \
--run_path "${ROOT_DIR}/diffusion_output/ffhq_latent"

# torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
# --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS}" \
# "${ROOT_DIR}/trainer/"train_latent_diffusion.py \
# --config_path "${ROOT_DIR}/diffusion_output/ffhq_latent/config.yml" \
# --run_path "${ROOT_DIR}/diffusion_output/ffhq_latent" \
# --resume "${ROOT_DIR}/diffusion_output/ffhq_latent/checkpoints/latest.pt"