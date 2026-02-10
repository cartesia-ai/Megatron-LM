#!/bin/bash
# Submit Megatron-Core FSDP benchmark jobs via gypsum's sbatch.sh
#
# Usage:
#   ./submit_megatron_benchmarks.sh [--dry-run] [--smoke-test] [--models 1b,8b] [--strategies ddp,optim,...] [--nodes 1,2,4]
#
# Options:
#   --dry-run       Print what would be submitted without actually submitting
#   --smoke-test    Submit only 1-node jobs
#   --models        Comma-separated model sizes (default: 1b,8b)
#   --strategies    Comma-separated strategies (default: ddp,optim,optim_grads,optim_grads_params,hsdp)
#   --nodes         Comma-separated node counts (default: 1,2,4)
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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

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
echo ""

submitted=0
skipped=0
for model in "${MODEL_ARR[@]}"; do
    for strategy in "${STRATEGY_ARR[@]}"; do
        for nodes in "${NODE_ARR[@]}"; do
            # Skip known-OOM combos: 8B with DDP or optim (ZeRO-1)
            if [[ "$model" == "8b" && ("$strategy" == "ddp" || "$strategy" == "optim") ]]; then
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

            job_name="megatron_${model}_${strategy}_${nodes}N"

            echo "Submitting: $job_name (model=$model, strategy=$strategy, nodes=$nodes)"

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
                    --strategy "$strategy"
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
