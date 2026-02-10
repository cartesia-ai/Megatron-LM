#!/bin/bash
# Submit Megatron-Core FSDP benchmark jobs via gypsum's sbatch.sh
#
# Usage:
#   ./submit_megatron_benchmarks.sh [--dry-run] [--smoke-test] [--models 1b,8b] [--strategies ddp,optim,...] [--nodes 1,2,4]
#   ./submit_megatron_benchmarks.sh --tp    [--dry-run] [--nodes 1,2,4]   # TP=2 experiments (LLaMA 8B)
#   ./submit_megatron_benchmarks.sh --cp    [--dry-run] [--nodes 1,2,4]   # CP=2 experiments (LLaMA 8B)
#   ./submit_megatron_benchmarks.sh --ep    [--dry-run] [--nodes 1,2,4]   # EP=2 experiments (8B MoE)
#
# Options:
#   --dry-run       Print what would be submitted without actually submitting
#   --smoke-test    Submit only 1-node jobs
#   --models        Comma-separated model sizes (default: 1b,3b,8b)
#   --strategies    Comma-separated strategies (default: ddp,optim,optim_grads,optim_grads_params,hsdp)
#   --nodes         Comma-separated node counts (default: 1,2,4)
#   --tp            Run TP=2 experiments (LLaMA 8B, 5 FSDP strategies x 3 scales)
#   --cp            Run CP=2 experiments (LLaMA 8B, 5 FSDP strategies x 3 scales)
#   --ep            Run EP=2 experiments (8B MoE, 5 FSDP strategies x 3 scales)
#
# Smart skipping:
#   - 8B + DDP/optim are skipped (confirmed OOM on H100 80GB)
#   - HSDP at 1N is skipped (identical to optim_grads_params at 1N)

set -euo pipefail

GYPSUM_DIR="${GYPSUM_DIR:-$HOME/projects/gypsum}"
BENCHMARK_SCRIPT="$HOME/projects/fsdp-bench/megatron-lm/run_megatron_benchmark.sh"

# Defaults
DRY_RUN=0
MODELS="1b,3b,8b"
STRATEGIES="ddp,optim,optim_grads,optim_grads_params,hsdp"
NODE_COUNTS="1,2,4"
EXPERIMENT_MODE=""  # "", "tp", "cp", or "ep"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --smoke-test)
            NODE_COUNTS="1"
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --nodes)
            NODE_COUNTS="$2"
            shift 2
            ;;
        --tp)
            EXPERIMENT_MODE="tp"
            shift
            ;;
        --cp)
            EXPERIMENT_MODE="cp"
            shift
            ;;
        --ep)
            EXPERIMENT_MODE="ep"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Override defaults for TP/CP/EP experiment modes
EXTRA_ARGS=""
if [[ "$EXPERIMENT_MODE" == "tp" ]]; then
    MODELS="8b"
    STRATEGIES="ddp,optim,optim_grads,optim_grads_params,hsdp"
    EXTRA_ARGS="--tp-size 2"
    echo "=== TP=2 Experiment Mode (LLaMA 8B) ==="
elif [[ "$EXPERIMENT_MODE" == "cp" ]]; then
    MODELS="8b"
    STRATEGIES="ddp,optim,optim_grads,optim_grads_params,hsdp"
    EXTRA_ARGS="--cp-size 2"
    echo "=== CP=2 Experiment Mode (LLaMA 8B) ==="
elif [[ "$EXPERIMENT_MODE" == "ep" ]]; then
    MODELS="8b-moe"
    STRATEGIES="ddp,optim,optim_grads,optim_grads_params,hsdp"
    EXTRA_ARGS="--ep-size 2"
    echo "=== EP=2 Experiment Mode (8B MoE, 8 experts) ==="
fi

# Convert comma-separated to arrays
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra STRATEGY_ARR <<< "$STRATEGIES"
IFS=',' read -ra NODE_ARR <<< "$NODE_COUNTS"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== DRY RUN (no jobs will be submitted) ==="
    echo ""
fi

echo "Benchmark matrix:"
echo "  Models:     ${MODEL_ARR[*]}"
echo "  Strategies: ${STRATEGY_ARR[*]}"
echo "  Nodes:      ${NODE_ARR[*]}"
[[ -n "$EXTRA_ARGS" ]] && echo "  Extra args: $EXTRA_ARGS"
echo ""

submitted=0
skipped=0
for model in "${MODEL_ARR[@]}"; do
    for strategy in "${STRATEGY_ARR[@]}"; do
        for nodes in "${NODE_ARR[@]}"; do
            # Skip known-OOM combos: 8B with DDP or optim (ZeRO-1) -- only for pure DP (no TP/CP)
            if [[ "$model" == "8b" && "$EXPERIMENT_MODE" == "" && ("$strategy" == "ddp" || "$strategy" == "optim") ]]; then
                echo "SKIP: megatron_${model}_${strategy}_${nodes}N (confirmed OOM)"
                skipped=$((skipped + 1))
                continue
            fi

            # Skip HSDP at 1N (identical to optim_grads_params)
            if [[ "$strategy" == "hsdp" && "$nodes" == "1" ]]; then
                echo "SKIP: megatron_${model}_hsdp_1N (same as optim_grads_params at 1N)"
                skipped=$((skipped + 1))
                continue
            fi

            # Build job name with experiment prefix
            if [[ -n "$EXPERIMENT_MODE" ]]; then
                job_name="megatron_${EXPERIMENT_MODE}2_${model}_${strategy}_${nodes}N"
            else
                job_name="megatron_${model}_${strategy}_${nodes}N"
            fi

            echo "Submitting: $job_name (model=$model, strategy=$strategy, nodes=$nodes${EXTRA_ARGS:+, $EXTRA_ARGS})"

            if [[ $DRY_RUN -eq 0 ]]; then
                cd "$GYPSUM_DIR"
                GYPSUM_DIR="$GYPSUM_DIR" scripts/sbatch.sh \
                    --nodes "$nodes" \
                    --devices 8 \
                    --custom-script \
                    --n-retries 0 \
                    -J "$job_name" \
                    "$BENCHMARK_SCRIPT" \
                    --model-size "$model" \
                    --strategy "$strategy" \
                    $EXTRA_ARGS
                echo "  -> Submitted"
                # Small delay to avoid overloading the scheduler
                sleep 2
            fi

            submitted=$((submitted + 1))
        done
    done
done

echo ""
echo "=== Total: $submitted jobs submitted, $skipped skipped ==="
if [[ $DRY_RUN -eq 1 ]]; then
    echo "(dry run -- no jobs were actually submitted)"
fi
