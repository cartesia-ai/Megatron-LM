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
#   --model-size   : 1b, 3b, 8b, 8b-moe, deepseek-16b-moe, nemotron-30b-moe, nemotron-h-4b
#   --strategy     : ddp, optim, optim_grads, optim_grads_params, hsdp
#   --tp-size N    : tensor model parallel size (default: 1)
#   --cp-size N    : context parallel size (default: 1)
#   --ep-size N    : expert model parallel size (default: 1, requires MoE model)
#   --pp-size N    : pipeline model parallel size (default: 1)
#   --seq-len N    : override default seq_len for the model
#   --train-iters N: override default train iterations (default: 650)

set -e

MEGATRON_DIR="${MEGATRON_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# ============================================================================
# Parse arguments
# ============================================================================
MODEL_SIZE=""
STRATEGY=""
TP_SIZE=1
CP_SIZE=1
EP_SIZE=1
PP_SIZE=1
SEQ_LEN_OVERRIDE=""
TRAIN_ITERS=650
FULL_AC=${FULL_AC:-0}
NO_AC=${NO_AC:-0}
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
        --pp-size)
            PP_SIZE="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN_OVERRIDE="$2"
            shift 2
            ;;
        --train-iters)
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --full-ac)
            FULL_AC=1
            shift
            ;;
        --no-ac)
            NO_AC=1
            shift
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODEL_SIZE" ]] || [[ -z "$STRATEGY" ]]; then
    echo "ERROR: --model-size and --strategy are required"
    echo "Usage: run_megatron_benchmark.sh --model-size {1b,3b,8b,8b-moe,deepseek-16b-moe,nemotron-30b-moe,nemotron-h-3b,nemotron-h-4b} --strategy {ddp,optim,...} [--tp-size N] [--cp-size N] [--ep-size N] [--pp-size N] [--seq-len N] [--train-iters N]"
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
# TransformerEngine: always enabled in V2
# ============================================================================
# V2 uses --transformer-impl transformer_engine for ALL experiments (pure DP included)
# for representative real-world Megatron performance. V1 used --transformer-impl local
# for pure DP runs; V2 standardizes on TE across the board.

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
# Install TransformerEngine (always, for V2)
# ============================================================================
echo "=== Installing TransformerEngine ==="
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

python -c "import megatron; print('Megatron-Core import OK')" || \
    python -c "from megatron.training import pretrain; print('Megatron pretrain import OK')"

# ============================================================================
# Install Mamba dependencies (only for hybrid models)
# ============================================================================
if [[ "$MODEL_SIZE" == "nemotron-h-3b" || "$MODEL_SIZE" == "nemotron-h-4b" || "$MODEL_SIZE" == "nemotron-30b-moe" ]]; then
    echo "=== Installing Mamba dependencies (mamba-ssm, causal-conv1d) ==="
    pip install --quiet mamba-ssm causal-conv1d 2>&1 | tail -5
    python -c "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; print('mamba-ssm import OK')"
    python -c "from causal_conv1d import causal_conv1d_fn; print('causal-conv1d import OK')"
fi

echo "=== Environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import transformer_engine; print(f'TransformerEngine {transformer_engine.__version__}')"
echo "Model size: $MODEL_SIZE"
echo "Strategy: $STRATEGY"
echo "TP=$TP_SIZE, CP=$CP_SIZE, EP=$EP_SIZE, PP=$PP_SIZE"

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
# Common LLaMA flags (V2: always TransformerEngine)
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

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
MOE_ARGS=()
USE_PRETRAIN_MAMBA=0

case "$MODEL_SIZE" in
    1b)
        # LLaMA 3.2 1B configuration (V2: seq_len=4096, confirmed from S1–S3)
        MODEL_ARGS=(
            --num-layers 16
            --hidden-size 2048
            --ffn-hidden-size 8192
            --num-attention-heads 32
            --kv-channels 64
            --seq-length 4096
            --max-position-embeddings 4096
        )
        MICRO_BATCH_SIZE=2
        ;;
    3b)
        # LLaMA 3.2 3B configuration (V2: seq_len=2048, DDP OOM at 4096)
        # micro_batch_size=1 so DDP fits (at mbs=2, DDP and optim OOM at 1N)
        # For CP=2 experiments, use --seq-len 16384 override
        MODEL_ARGS=(
            --num-layers 28
            --hidden-size 3072
            --ffn-hidden-size 8192
            --num-attention-heads 24
            --kv-channels 128
            --seq-length 2048
            --max-position-embeddings 131072
        )
        MICRO_BATCH_SIZE=1
        ;;
    8b)
        # LLaMA 3.1 8B configuration (V2: seq_len=4096, mbs=1; used with TP/PP only)
        # mbs=1 (not 2) so DDP strategy fits in 80GB with selective AC
        MODEL_ARGS=(
            --num-layers 32
            --hidden-size 4096
            --ffn-hidden-size 14336
            --num-attention-heads 32
            --kv-channels 128
            --seq-length 4096
            --max-position-embeddings 4096
        )
        MICRO_BATCH_SIZE=1
        ;;
    8b-moe)
        # Mixtral-style MoE: LLaMA 8B backbone + 8 experts, top-k=2
        # Dense params ~8B, total with experts much larger
        # V2: seq_len=2048 (reduced from 4096 so DDP fits), selective AC
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
            --recompute-activations
            --recompute-granularity selective
        )
        MICRO_BATCH_SIZE=1
        ;;
    deepseek-16b-moe)
        # DeepSeek-V3 16B MoE (S5.5): 64 routed + 2 shared experts, top-6
        # hidden=2048, layers=27, heads=16 (MHA), ffn=10944, moe_ffn=1408
        # ~16B total params. EP=2 recommended. seq=2048 (4096 OOMs with DDP+EP=2)
        MODEL_ARGS=(
            --num-layers 27
            --hidden-size 2048
            --ffn-hidden-size 10944
            --num-attention-heads 16
            --kv-channels 128
            --seq-length 2048
            --max-position-embeddings 4096
        )
        # DeepSeek uses MHA (16 heads, no GQA) — remove GQA from LLAMA_ARGS
        LLAMA_ARGS=(
            --transformer-impl transformer_engine
            --position-embedding-type rope
            --rotary-base 1000000
            --rotary-percent 1.0
            --swiglu
            --normalization RMSNorm
            --untie-embeddings-and-output-weights
            --disable-bias-linear
            --attention-dropout 0.0
            --hidden-dropout 0.0
            --no-masked-softmax-fusion
            --attention-softmax-in-fp32
        )
        MOE_ARGS=(
            --num-experts 64
            --moe-router-topk 6
            --moe-ffn-hidden-size 1408
            --moe-shared-expert-intermediate-size 2816
            --moe-router-load-balancing-type seq_aux_loss
            --moe-aux-loss-coeff 1e-3
            --moe-router-score-function sigmoid
            --moe-token-dispatcher-type alltoall
            --moe-grouped-gemm
            --recompute-activations
            --recompute-granularity selective
        )
        MICRO_BATCH_SIZE=1
        ;;
    nemotron-h-3b)
        # Nemotron-H ~3B: Reduced version of Nemotron-H-4B for DDP feasibility test.
        # Same architecture (hidden=3072, heads=32, kv_heads=8, ffn=12288) but 26 layers
        # instead of 52. Pattern auto-generated to keep ~8% attention, ~46% MLP ratio.
        # ~3B params (half the layer params + same embedding overhead).
        # Uses pretrain_mamba.py (not pretrain_gpt.py)
        USE_PRETRAIN_MAMBA=1
        LLAMA_ARGS=()
        MODEL_ARGS=(
            --transformer-impl transformer_engine
            --use-rotary-position-embeddings
            --rotary-percent 0.5
            --rotary-base 10000
            --no-rope-fusion
            --no-position-embedding
            --squared-relu
            --normalization RMSNorm
            --group-query-attention
            --num-query-groups 8
            --untie-embeddings-and-output-weights
            --disable-bias-linear
            --no-masked-softmax-fusion
            --attention-softmax-in-fp32
            --is-hybrid-model
            --hybrid-override-pattern "M-M-M-M-*M-M-M-M-*M-M-M-M-"
            --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
            --mamba-head-dim 64
            --mamba-num-heads 112
            --mamba-num-groups 8
            --mamba-state-dim 128
            --use-mcore-models
            --num-layers 26
            --hidden-size 3072
            --ffn-hidden-size 12288
            --num-attention-heads 32
            --kv-channels 128
            --seq-length 2048
            --max-position-embeddings 8192
        )
        MICRO_BATCH_SIZE=1
        ;;
    nemotron-h-4b)
        # Nemotron-H-4B-Instruct (S5.8): Dense hybrid Mamba-2 + GQA
        # hidden=3072, layers=52, heads=32, kv_heads=8, ffn=12288
        # Pattern: mostly Mamba-2 with attention every ~5 layers. ~4B params.
        # seq=2048 (4096 OOMs). No AC needed with ZeRO-1+ (DDP infeasible).
        # Uses pretrain_mamba.py (not pretrain_gpt.py)
        USE_PRETRAIN_MAMBA=1
        LLAMA_ARGS=()
        MODEL_ARGS=(
            --transformer-impl transformer_engine
            --use-rotary-position-embeddings
            --rotary-percent 0.5
            --rotary-base 10000
            --no-rope-fusion
            --no-position-embedding
            --squared-relu
            --normalization RMSNorm
            --group-query-attention
            --num-query-groups 8
            --untie-embeddings-and-output-weights
            --disable-bias-linear
            --no-masked-softmax-fusion
            --attention-softmax-in-fp32
            --is-hybrid-model
            --hybrid-override-pattern "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
            --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
            --mamba-head-dim 64
            --mamba-num-heads 112
            --mamba-num-groups 8
            --mamba-state-dim 128
            --use-mcore-models
            --num-layers 52
            --hidden-size 3072
            --ffn-hidden-size 12288
            --num-attention-heads 32
            --kv-channels 128
            --seq-length 2048
            --max-position-embeddings 8192
        )
        MICRO_BATCH_SIZE=1
        ;;
    nemotron-30b-moe)
        # Nemotron-3-Nano-30B-A3B (S5.7): Hybrid Mamba-2 + GQA + MoE
        # hidden=2688, layers=52, heads=32, kv_heads=2, ffn=1856
        # 128 experts, top-6; 31.6B total, 3.2B active per token.
        # Uses pretrain_mamba.py (not pretrain_gpt.py)
        USE_PRETRAIN_MAMBA=1
        LLAMA_ARGS=()
        MODEL_ARGS=(
            --transformer-impl transformer_engine
            --position-embedding-type none
            --squared-relu
            --normalization RMSNorm
            --group-query-attention
            --num-query-groups 2
            --untie-embeddings-and-output-weights
            --disable-bias-linear
            --no-masked-softmax-fusion
            --attention-softmax-in-fp32
            --is-hybrid-model
            --hybrid-override-pattern "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
            --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
            --mamba-num-heads 64
            --mamba-head-dim 64
            --use-mcore-models
            --num-layers 52
            --hidden-size 2688
            --num-attention-heads 32
            --ffn-hidden-size 1856
            --kv-channels 128
            --seq-length 4096
            --max-position-embeddings 8192
        )
        MOE_ARGS=(
            --num-experts 128
            --moe-router-topk 6
            --moe-aux-loss-coeff 1e-4
            --moe-router-topk-scaling-factor 2.5
            --moe-router-enable-expert-bias
            --moe-router-dtype fp32
            --moe-router-load-balancing-type seq_aux_loss
            --moe-router-score-function sigmoid
            --moe-shared-expert-intermediate-size 3712
            --moe-token-dispatcher-type allgather
            --moe-grouped-gemm
            --recompute-activations
            --recompute-granularity selective
        )
        MICRO_BATCH_SIZE=1
        ;;
    *)
        echo "ERROR: Unknown model size '$MODEL_SIZE'. Use '1b', '3b', '8b', '8b-moe', 'deepseek-16b-moe', 'nemotron-30b-moe', 'nemotron-h-3b', or 'nemotron-h-4b'."
        exit 1
        ;;
esac

# ============================================================================
# Apply seq-len override (must come after model config, before GBS calculation)
# ============================================================================
if [[ -n "$SEQ_LEN_OVERRIDE" ]]; then
    echo "Overriding seq_len to $SEQ_LEN_OVERRIDE"
    # Replace --seq-length in MODEL_ARGS
    NEW_MODEL_ARGS=()
    skip_next=0
    for arg in "${MODEL_ARGS[@]}"; do
        if [[ $skip_next -eq 1 ]]; then
            skip_next=0
            continue
        fi
        if [[ "$arg" == "--seq-length" ]]; then
            NEW_MODEL_ARGS+=("--seq-length" "$SEQ_LEN_OVERRIDE")
            skip_next=1
        else
            NEW_MODEL_ARGS+=("$arg")
        fi
    done
    MODEL_ARGS=("${NEW_MODEL_ARGS[@]}")
fi

# ============================================================================
# TP / CP / EP / PP parallelism args
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

if [[ "$PP_SIZE" -gt 1 ]]; then
    PARALLEL_ARGS+=(--pipeline-model-parallel-size "$PP_SIZE")
fi

# ============================================================================
# Activation checkpointing for multi-dimensional parallelism
# ============================================================================
# When TP/CP/PP > 1, selective AC is needed so DDP (no sharding) can fit in memory.
# Without AC, 8B DDP+TP OOMs at 75.9/79 GiB. With selective AC, activations are
# recomputed during backward, freeing ~20-30% memory.
# For MoE, AC is already set in MOE_ARGS above.
# For pure DP (5.1, 5.2), no AC needed — models are small enough.
# For nemotron-h-4b, AC is set in the model case (52 layers too deep without it).
if [[ ${#RECOMPUTE_ARGS[@]} -eq 0 ]]; then
    RECOMPUTE_ARGS=()
    if [[ "${NO_AC:-0}" == "1" ]]; then
        echo "Activation checkpointing DISABLED (--no-ac)"
    elif [[ "${FULL_AC:-0}" == "1" ]]; then
        echo "Enabling FULL activation checkpointing (uniform, recompute all layers)"
        RECOMPUTE_ARGS=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
    elif [[ "$TP_SIZE" -gt 1 || "$CP_SIZE" -gt 1 || "$PP_SIZE" -gt 1 ]]; then
        echo "Enabling selective activation checkpointing (TP=$TP_SIZE, CP=$CP_SIZE, PP=$PP_SIZE)"
        RECOMPUTE_ARGS=(--recompute-activations --recompute-granularity selective)
    fi
fi

# Compute DP size: WORLD_SIZE / (TP * CP * PP)
# EP is carved from the FSDP dimension, not from DP directly
DP_SIZE=$((WORLD_SIZE / (TP_SIZE * CP_SIZE * PP_SIZE)))
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

# Disable Gloo process groups: Gloo binds to localhost inside Apptainer containers,
# which breaks multi-node runs. NCCL handles all GPU communication fine without Gloo.
STRATEGY_ARGS+=( --disable-gloo-process-groups )

# ============================================================================
# CUDA_DEVICE_MAX_CONNECTIONS handling for TP/CP + FSDP
# ============================================================================
# TP and CP need CUDA_DEVICE_MAX_CONNECTIONS=1 for sequence parallelism overlap.
# FSDP asserts CUDA_DEVICE_MAX_CONNECTIONS != 1.
# On pre-Blackwell (H100), this is a known conflict. FSDP's assert is hard,
# so we must unset it when FSDP is active. Megatron prints a warning but proceeds.
if [[ ("$TP_SIZE" -gt 1 || "$CP_SIZE" -gt 1 || "$PP_SIZE" -gt 1) && "$STRATEGY" != "ddp" ]]; then
    echo "WARNING: TP=$TP_SIZE / CP=$CP_SIZE / PP=$PP_SIZE with FSDP strategy '$STRATEGY'."
    echo "  CUDA_DEVICE_MAX_CONNECTIONS must be unset for FSDP (hard assert)."
    echo "  TP/CP sequence parallelism overlap may be suboptimal on H100."
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi

# ============================================================================
# Training configuration
# ============================================================================
# Cap warmup iters at train_iters - 1 (Megatron asserts warmup < decay)
LR_WARMUP_ITERS=20
if [[ $TRAIN_ITERS -le $LR_WARMUP_ITERS ]]; then
    LR_WARMUP_ITERS=$((TRAIN_ITERS - 1))
fi

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-iters $TRAIN_ITERS
    --lr-warmup-iters $LR_WARMUP_ITERS
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
echo "  TP=$TP_SIZE, CP=$CP_SIZE, EP=$EP_SIZE, PP=$PP_SIZE"
echo "  World size: $WORLD_SIZE (DP=$DP_SIZE)"
echo "  Train iters: $TRAIN_ITERS"
echo "  Micro batch size: $MICRO_BATCH_SIZE"
echo "  Global batch size: $GLOBAL_BATCH_SIZE"
echo "  CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-<unset>}"
echo ""

cd "$MEGATRON_DIR"

# Record launch timestamp for init_time calculation
# init_time = time from launch to first training iteration
LAUNCH_TS=$(date +%s.%N)
echo "MEGATRON_LAUNCH_TIMESTAMP=$LAUNCH_TS"

ENTRY_POINT="pretrain_gpt.py"
if [[ "$USE_PRETRAIN_MAMBA" -eq 1 ]]; then
    ENTRY_POINT="pretrain_mamba.py"
    echo "Using Mamba entry point: $ENTRY_POINT"
fi

EXIT_CODE=0
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $ENTRY_POINT \
    "${LLAMA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${STRATEGY_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${REMAINING_ARGS[@]}" || EXIT_CODE=$?

FINISH_TS=$(date +%s.%N)
TOTAL_WALL=$(python3 -c "print(f'{$FINISH_TS - $LAUNCH_TS:.1f}')")
echo "MEGATRON_FINISH_TIMESTAMP=$FINISH_TS"
echo "MEGATRON_TOTAL_WALL_SECONDS=$TOTAL_WALL"

touch "$SENTINEL"
echo "Task 0: Created sentinel $SENTINEL, exiting with code $EXIT_CODE"
exit $EXIT_CODE
