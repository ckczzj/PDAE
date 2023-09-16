MASTER_ADDR=127.0.0.1
MASTER_PORT=3456
NNODES=$1
NODE_RANK=$2
GPUS=$3

ROOT_DIR="$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}

torchrun --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
--nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS}" \
"${ROOT_DIR}/sampler/autoencoding_eval.py"

# "${ROOT_DIR}/sampler/autoencoding_eval.py"
# "${ROOT_DIR}/sampler/gap_measure.py"
# "${ROOT_DIR}/sampler/unconditional_sample.py"