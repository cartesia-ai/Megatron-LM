#!/bin/bash
# Megatron-Core FSDP benchmark launcher for SLURM (Together cluster)
# Runs inside Apptainer via gypsum's sbatch.sh -> run.sh pipeline.
#
# Usage (via gypsum sbatch.sh):
#   cd $GYPSUM_DIR && GYPSUM_DIR=$GYPSUM_DIR scripts/sbatch.sh \
#     --nodes 2 --devices 8 --custom-script \
#     /home/dwromero/projects/fsdp-bench/megatron-lm/run_megatron_benchmark.sh \
#     --model-size 8b --strategy optim_grads_params
#
# Arguments:
#   --model-size   : 1b, 3b, 8b, or 8b-moe (LLaMA / MoE architecture)
#   --strategy     : ddp, optim, optim_grads, optim_grads_params, hsdp
#   --tp-size N    : tensor model parallel size (default: 1)
#   --cp-size N    : context parallel size (default: 1)
#   --ep-size N    : expert model parallel size (default: 1, requires MoE model)

set -e

MEGATRON_DIR="/home/dwromero/projects/fsdp-bench/megatron-lm"

# ============================================================================
# Parse arguments
# ============================================================================
MODEL_SIZE=""
STRATEGY=""
TP_SIZE=1
CP_SIZE=1
EP_SIZE=1
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --cp-size)
            CP_SIZE="$2"
            shift 2
            ;;
        --ep-size)
            EP_SIZE="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODEL_SIZE" ]] || [[ -z "$STRATEGY" ]]; then
    echo "ERROR: --model-size and --strategy are required"
    echo "Usage: run_megatron_benchmark.sh --model-size {1b,3b,8b,8b-moe} --strategy {ddp,optim,optim_grads,optim_grads_params,hsdp} [--tp-size N] [--cp-size N] [--ep-size N]"
    exit 1
fi

# ============================================================================
# Sentinel-based coordination: only one torchrun per node
# ============================================================================
SENTINEL="/tmp/megatron_done_${SLURM_JOB_ID}_${SLURM_NODEID}"
LOCAL_ID=${SLURM_LOCALID:-0}
if [[ "$LOCAL_ID" != "0" ]]; then
    echo "Task $LOCAL_ID: Waiting for torchrun to finish..."
    while [[ ! -f "$SENTINEL" ]]; do
        sleep 2
    done
    echo "Task $LOCAL_ID: Torchrun finished, exiting"
    exit 0
fi
rm -f "$SENTINEL"

# ============================================================================
# Determine if TransformerEngine is needed
# ============================================================================
# TP, CP, and EP in Megatron-Core require TransformerEngine (TEDotProductAttention
# for CP, te_general_gemm for MoE routing, TE layers for TP + FSDP composition).
# Pure FSDP benchmarks (TP=CP=EP=1) use --transformer-impl local to avoid the dep.
USE_TE=0
if [[ "$TP_SIZE" -gt 1 || "$CP_SIZE" -gt 1 || "$EP_SIZE" -gt 1 ]]; then
    USE_TE=1
fi

# ============================================================================
# Install Megatron-Core (and remove Apex to avoid FusedLayerNorm issues)
# ============================================================================
echo "=== Installing Megatron-Core ==="
cd "$MEGATRON_DIR"

# Remove Apex if present: its FusedLayerNorm doesn't support RMSNorm.
# Without Apex/TE, Megatron falls back to WrappedTorchNorm which works fine.
pip uninstall -y apex 2>/dev/null || true

pip install --quiet -e . 2>&1 | tail -5

# ============================================================================
# Install TransformerEngine if needed (for TP/CP/EP)
# ============================================================================
if [[ "$USE_TE" -eq 1 ]]; then
    echo "=== Installing TransformerEngine (required for TP/CP/EP) ==="
    # 1. Meta package + core CUDA library (prebuilt wheel, ~288 MB)
    pip install --quiet transformer-engine==2.11.0 transformer-engine-cu12==2.11.0 2>&1 | tail -3
    # 2. Missing dependency for TE 2.11
    pip install --quiet onnxscript 2>&1 | tail -3
    # 3. PyTorch bindings (needs compilation; cuDNN headers from pip aren't on default include path)
    #    Find cuDNN include dir: check nvidia.cudnn pip package, then common system paths
    CUDNN_INCLUDE=""
    for candidate in \
        "$(python -c 'import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "include"))' 2>/dev/null)" \
        "$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' 2>/dev/null)/include" \
        "/usr/local/cuda/include" \
        "/usr/include"; do
        if [[ -f "$candidate/cudnn.h" ]]; then
            CUDNN_INCLUDE="$candidate"
            break
        fi
    done
    if [[ -z "$CUDNN_INCLUDE" ]]; then
        # Fallback: search site-packages for nvidia/cudnn/include
        CUDNN_INCLUDE=$(find "$(python -c 'import site; print(site.getsitepackages()[0])')" -path "*/nvidia/cudnn/include/cudnn.h" -printf "%h" -quit 2>/dev/null || echo "")
    fi
    if [[ -z "$CUDNN_INCLUDE" ]]; then
        echo "ERROR: Could not find cudnn.h for TransformerEngine compilation"
        exit 1
    fi
    echo "  cuDNN include path: $CUDNN_INCLUDE"
    CPLUS_INCLUDE_PATH="$CUDNN_INCLUDE:${CPLUS_INCLUDE_PATH:-}" \
    C_INCLUDE_PATH="$CUDNN_INCLUDE:${C_INCLUDE_PATH:-}" \
    pip install --quiet --no-build-isolation transformer-engine-torch==2.11.0 2>&1 | tail -3
    # Verify
    python -c "from transformer_engine.pytorch import TransformerLayer; print('TransformerEngine import OK')"
fi

python -c "import megatron; print('Megatron-Core import OK')" || \
    python -c "from megatron.training import pretrain; print('Megatron pretrain import OK')"

echo "=== Environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
if [[ "$USE_TE" -eq 1 ]]; then
    python -c "import transformer_engine; print(f'TransformerEngine {transformer_engine.__version__}')"
fi
echo "Model size: $MODEL_SIZE"
echo "Strategy: $STRATEGY"
echo "TP=$TP_SIZE, CP=$CP_SIZE, EP=$EP_SIZE"

# ============================================================================
# SLURM / distributed setup
# ============================================================================
echo "SLURM env:"
echo "  SLURM_NODELIST=$SLURM_NODELIST"
echo "  SLURM_NNODES=$SLURM_NNODES"
echo "  SLURM_NODEID=$SLURM_NODEID"

# Parse master address from SLURM
if [[ -n "$SLURM_NODELIST" ]]; then
    if [[ "$SLURM_NODELIST" =~ ^([^[]+)\[([0-9]+) ]]; then
        MASTER_ADDR="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    elif [[ "$SLURM_NODELIST" =~ ^([^,]+), ]]; then
        MASTER_ADDR="${BASH_REMATCH[1]}"
    else
        MASTER_ADDR="$SLURM_NODELIST"
    fi
    export MASTER_ADDR
    echo "Parsed MASTER_ADDR=$MASTER_ADDR"
fi

export MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${SLURM_NNODES:-1}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
NODE_RANK=${SLURM_NODEID:-0}

echo "NNODES=$NNODES, NPROC=$NPROC_PER_NODE, NODE_RANK=$NODE_RANK"

# ============================================================================
# Model configuration (LLaMA architecture)
# ============================================================================
# Common LLaMA flags
# When TP/CP/EP > 1, we use TransformerEngine (--transformer-impl transformer_engine)
# because Megatron's TP, CP, and EP require TE's fused kernels.
# For pure FSDP benchmarks, we use --transformer-impl local to keep things simple.
if [[ "$USE_TE" -eq 1 ]]; then
    LLAMA_ARGS=(
        --transformer-impl transformer_engine
        --position-embedding-type rope
        --rotary-base 500000
        --rotary-percent 1.0
        --swiglu
        --normalization RMSNorm
        --group-query-attention
        --num-query-groups 8
        --untie-embeddings-and-output-weights
        --disable-bias-linear
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --no-position-embedding
        --no-masked-softmax-fusion
        --attention-softmax-in-fp32
    )
else
    LLAMA_ARGS=(
        --transformer-impl local
        --position-embedding-type rope
        --rotary-base 500000
        --rotary-percent 1.0
        --no-rope-fusion
        --swiglu
        --normalization RMSNorm
        --group-query-attention
        --num-query-groups 8
        --untie-embeddings-and-output-weights
        --disable-bias-linear
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --no-position-embedding
        --no-masked-softmax-fusion
        --attention-softmax-in-fp32
        --no-persist-layer-norm
    )
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
MOE_ARGS=()

case "$MODEL_SIZE" in
    1b)
        # LLaMA 3.2 1B configuration
        MODEL_ARGS=(
            --num-layers 16
            --hidden-size 2048
            --ffn-hidden-size 8192
            --num-attention-heads 32
            --kv-channels 64
            --seq-length 2048
            --max-position-embeddings 2048
        )
        MICRO_BATCH_SIZE=2
        ;;
    3b)
        # LLaMA 3.2 3B configuration
        # micro_batch_size=1 so DDP fits (at mbs=2, DDP and optim OOM at 1N)
        MODEL_ARGS=(
            --num-layers 28
            --hidden-size 3072
            --ffn-hidden-size 8192
            --num-attention-heads 24
            --kv-channels 128
            --seq-length 2048
            --max-position-embeddings 2048
        )
        MICRO_BATCH_SIZE=1
        ;;
    8b)
        # LLaMA 3.1 8B configuration
        MODEL_ARGS=(
            --num-layers 32
            --hidden-size 4096
            --ffn-hidden-size 14336
            --num-attention-heads 32
            --kv-channels 128
            --seq-length 2048
            --max-position-embeddings 2048
        )
        MICRO_BATCH_SIZE=1
        ;;
    8b-moe)
        # Mixtral-style MoE: LLaMA 8B backbone + 8 experts, top-k=2
        # Dense params ~8B, total with experts much larger
        # Uses --num-experts 8 and MoE-specific args
        MODEL_ARGS=(
            --num-layers 32
            --hidden-size 4096
            --ffn-hidden-size 14336
            --num-attention-heads 32
            --kv-channels 128
            --seq-length 2048
            --max-position-embeddings 2048
        )
        MOE_ARGS=(
            --num-experts 8
            --moe-router-topk 2
            --moe-router-load-balancing-type aux_loss
            --moe-aux-loss-coeff 1e-2
            --moe-token-dispatcher-type alltoall
        )
        MICRO_BATCH_SIZE=1
        ;;
    *)
        echo "ERROR: Unknown model size '$MODEL_SIZE'. Use '1b', '3b', '8b', or '8b-moe'."
        exit 1
        ;;
esac

# ============================================================================
# TP / CP / EP parallelism args
# ============================================================================
PARALLEL_ARGS=()

if [[ "$TP_SIZE" -gt 1 ]]; then
    PARALLEL_ARGS+=(--tensor-model-parallel-size "$TP_SIZE")
fi

if [[ "$CP_SIZE" -gt 1 ]]; then
    PARALLEL_ARGS+=(--context-parallel-size "$CP_SIZE")
fi

if [[ "$EP_SIZE" -gt 1 ]]; then
    PARALLEL_ARGS+=(--expert-model-parallel-size "$EP_SIZE")
fi

# Compute DP size: WORLD_SIZE / (TP * CP)
# EP is carved from the FSDP dimension, not from DP directly
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * CP_SIZE)))
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * DP_SIZE))

# ============================================================================
# Sharding strategy configuration
# ============================================================================
STRATEGY_ARGS=()

case "$STRATEGY" in
    ddp)
        # Pure DDP: no distributed optimizer, no FSDP
        # Standard all-reduce on gradients, replicated optimizer states
        STRATEGY_ARGS=(
            --overlap-grad-reduce
            --no-gradient-accumulation-fusion
        )
        # DDP needs CUDA_DEVICE_MAX_CONNECTIONS=1 for optimal overlap
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        ;;
    optim)
        # Megatron FSDP: shard optimizer states only (ZeRO-1 equivalent)
        STRATEGY_ARGS=(
            --use-distributed-optimizer
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim
            --no-gradient-accumulation-fusion
            --ckpt-format fsdp_dtensor
            --overlap-grad-reduce
            --overlap-param-gather
            --calculate-per-token-loss
        )
        # FSDP requires CUDA_DEVICE_MAX_CONNECTIONS != 1
        unset CUDA_DEVICE_MAX_CONNECTIONS
        ;;
    optim_grads)
        # Megatron FSDP: shard optimizer + gradients (ZeRO-2 equivalent)
        STRATEGY_ARGS=(
            --use-distributed-optimizer
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads
            --no-gradient-accumulation-fusion
            --ckpt-format fsdp_dtensor
            --overlap-grad-reduce
            --overlap-param-gather
            --calculate-per-token-loss
        )
        unset CUDA_DEVICE_MAX_CONNECTIONS
        ;;
    optim_grads_params)
        # Megatron FSDP: shard everything (ZeRO-3 equivalent)
        STRATEGY_ARGS=(
            --use-distributed-optimizer
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --no-gradient-accumulation-fusion
            --ckpt-format fsdp_dtensor
            --overlap-grad-reduce
            --overlap-param-gather
            --calculate-per-token-loss
        )
        unset CUDA_DEVICE_MAX_CONNECTIONS
        ;;
    hsdp)
        # Hybrid Sharded Data Parallel: ZeRO-3 intra-node, replicate inter-node
        # --num-distributed-optimizer-instances $NNODES creates one shard group per node
        # At 1N this is identical to optim_grads_params; use for 2N+ only
        STRATEGY_ARGS=(
            --use-distributed-optimizer
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --num-distributed-optimizer-instances $NNODES
            --no-gradient-accumulation-fusion
            --ckpt-format fsdp_dtensor
            --overlap-grad-reduce
            --overlap-param-gather
            --calculate-per-token-loss
        )
        unset CUDA_DEVICE_MAX_CONNECTIONS
        ;;
    *)
        echo "ERROR: Unknown strategy '$STRATEGY'. Use 'ddp', 'optim', 'optim_grads', 'optim_grads_params', or 'hsdp'."
        exit 1
        ;;
esac

# ============================================================================
# CUDA_DEVICE_MAX_CONNECTIONS handling for TP/CP + FSDP
# ============================================================================
# TP and CP need CUDA_DEVICE_MAX_CONNECTIONS=1 for sequence parallelism overlap.
# FSDP asserts CUDA_DEVICE_MAX_CONNECTIONS != 1.
# On pre-Blackwell (H100), this is a known conflict. FSDP's assert is hard,
# so we must unset it when FSDP is active. Megatron prints a warning but proceeds.
if [[ ("$TP_SIZE" -gt 1 || "$CP_SIZE" -gt 1) && "$STRATEGY" != "ddp" ]]; then
    echo "WARNING: TP=$TP_SIZE / CP=$CP_SIZE with FSDP strategy '$STRATEGY'."
    echo "  CUDA_DEVICE_MAX_CONNECTIONS must be unset for FSDP (hard assert)."
    echo "  TP/CP sequence parallelism overlap may be suboptimal on H100."
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi

# ============================================================================
# Training configuration
# ============================================================================
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 500
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-iters 500
    --lr-warmup-iters 20
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --log-interval 1
    --log-throughput
    --eval-interval 1000
    --eval-iters 0
    --distributed-timeout-minutes 30
)

# ============================================================================
# Data configuration: MOCK data (no preprocessing needed!)
# ============================================================================
DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 128256
    --split 99,1,0
)

# ============================================================================
# Launch
# ============================================================================
echo "=== Launch Configuration ==="
echo "  Model: $MODEL_SIZE"
echo "  Strategy: $STRATEGY"
echo "  TP=$TP_SIZE, CP=$CP_SIZE, EP=$EP_SIZE"
echo "  World size: $WORLD_SIZE (DP=$DP_SIZE)"
echo "  Micro batch size: $MICRO_BATCH_SIZE"
echo "  Global batch size: $GLOBAL_BATCH_SIZE"
echo "  CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-<unset>}"
echo ""

cd "$MEGATRON_DIR"

EXIT_CODE=0
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    "${LLAMA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${STRATEGY_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${REMAINING_ARGS[@]}" || EXIT_CODE=$?

touch "$SENTINEL"
echo "Task 0: Created sentinel $SENTINEL, exiting with code $EXIT_CODE"
exit $EXIT_CODE
