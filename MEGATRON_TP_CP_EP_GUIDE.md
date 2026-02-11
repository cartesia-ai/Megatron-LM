# Megatron-Core TP/CP/EP Experiment Launch Guide

How to reproduce the Megatron-Core TP=2, CP=2, and EP=2 benchmark experiments.

## Prerequisites

1. **gypsum** repo cloned at `$HOME/projects/gypsum` (provides `sbatch.sh` for job submission)
2. **fsdp-bench** repo cloned at `$HOME/projects/fsdp-bench` (contains `megatron-lm/` submodule)
3. SLURM cluster with H100 80GB GPUs, 8 GPUs/node
4. Apptainer/Singularity container image at `/shared/images/gypsum_refactor_26.01.22.sif` (see `gypsum/scripts/container-config.sh`)

## Quick Start: Submit All 42 Jobs

From the **head node** (no GPU needed — jobs run inside Apptainer containers on compute nodes):

```bash
cd ~/projects/fsdp-bench/megatron-lm

# TP=2 experiments: LLaMA 8B, 5 FSDP strategies × {1N, 2N, 4N} minus HSDP@1N = 14 jobs
./submit_megatron_benchmarks.sh --tp

# CP=2 experiments: LLaMA 8B, 5 strategies × 3 scales = 14 jobs
./submit_megatron_benchmarks.sh --cp

# EP=2 experiments: 8B-MoE (8 experts), 5 strategies × 3 scales = 14 jobs
./submit_megatron_benchmarks.sh --ep
```

**Dry run** (prints what would be submitted without actually submitting):
```bash
./submit_megatron_benchmarks.sh --tp --dry-run
./submit_megatron_benchmarks.sh --cp --dry-run
./submit_megatron_benchmarks.sh --ep --dry-run
```

## What Each Mode Does

### `--tp` (Tensor Parallelism = 2)
- **Model**: LLaMA 8B (`--model-size 8b`)
- **Extra arg**: `--tp-size 2` → `--tensor-model-parallel-size 2`
- **Effect**: DP_SIZE = WORLD_SIZE / TP_SIZE (e.g., 4 DP × 2 TP on 1 node)
- **Requires**: TransformerEngine (`--transformer-impl transformer_engine`)

### `--cp` (Context Parallelism = 2)
- **Model**: LLaMA 8B (`--model-size 8b`)
- **Extra arg**: `--cp-size 2` → `--context-parallel-size 2`
- **Effect**: DP_SIZE = WORLD_SIZE / CP_SIZE (e.g., 4 DP × 2 CP on 1 node)
- **Requires**: TransformerEngine (`--transformer-impl transformer_engine`)

### `--ep` (Expert Parallelism = 2)
- **Model**: 8B-MoE (`--model-size 8b-moe`) — LLaMA 8B backbone + 8 experts, top-k=2
- **Extra arg**: `--ep-size 2` → `--expert-model-parallel-size 2`
- **Effect**: EP is carved from the FSDP dimension (EP=2 means experts are split across 2 GPUs)
- **Requires**: TransformerEngine (`--transformer-impl transformer_engine`)

## Batch Size Computation

Batch sizes are **computed dynamically** by `run_megatron_benchmark.sh` based on the model and parallelism dimensions. This is critical — getting it wrong causes Megatron to error or silently produce wrong results.

### Formula

```
DP_SIZE = WORLD_SIZE / (TP_SIZE × CP_SIZE)
GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE × DP_SIZE
```

- `MICRO_BATCH_SIZE` is set per model: **1** for 8B and 8B-MoE, **2** for 1B, **1** for 3B
- EP does **not** reduce DP_SIZE — it's carved from the FSDP sharding dimension, not from DP
- HSDP does **not** change DP_SIZE either — it only changes how the DP group is sharded internally

### Concrete Values for TP/CP/EP Experiments

All TP/CP/EP experiments use 8B or 8B-MoE, so `MICRO_BATCH_SIZE = 1` in all cases.

**TP=2 (LLaMA 8B, micro_batch=1):**

| Nodes | GPUs | DP_SIZE = GPUs/(TP×CP) | GLOBAL_BATCH_SIZE |
|-------|------|------------------------|-------------------|
| 1     | 8    | 8 / 2 = **4**          | 1 × 4 = **4**    |
| 2     | 16   | 16 / 2 = **8**         | 1 × 8 = **8**    |
| 4     | 32   | 32 / 2 = **16**        | 1 × 16 = **16**  |

**CP=2 (LLaMA 8B, micro_batch=1):**

| Nodes | GPUs | DP_SIZE = GPUs/(TP×CP) | GLOBAL_BATCH_SIZE |
|-------|------|------------------------|-------------------|
| 1     | 8    | 8 / 2 = **4**          | 1 × 4 = **4**    |
| 2     | 16   | 16 / 2 = **8**         | 1 × 8 = **8**    |
| 4     | 32   | 32 / 2 = **16**        | 1 × 16 = **16**  |

**EP=2 (8B-MoE, micro_batch=1):**

| Nodes | GPUs | DP_SIZE = GPUs/(TP×CP) | GLOBAL_BATCH_SIZE |
|-------|------|------------------------|-------------------|
| 1     | 8    | 8 / 1 = **8**          | 1 × 8 = **8**    |
| 2     | 16   | 16 / 1 = **16**        | 1 × 16 = **16**  |
| 4     | 32   | 32 / 1 = **32**        | 1 × 32 = **32**  |

> **Note**: EP=2 has the same DP_SIZE as pure FSDP (no TP/CP), so the global batch size is larger than TP=2 or CP=2 at the same node count. EP splits the *experts* across 2 GPUs but all GPUs still participate in data parallelism.

### Memory Implications

- **TP=2 reduces per-GPU parameter memory** — each GPU holds half the model. This is why TP=2+DDP fits (75.2 GiB) while pure DDP OOMs.
- **CP=2 does NOT reduce per-GPU parameter memory** — it splits the sequence, not the model. With DP_SIZE halved (4 instead of 8 on 1N), FSDP shards across fewer GPUs → *more* memory per GPU. This is why CP=2 at 1N is tight.
- **EP=2 reduces expert memory** — each GPU holds 4 of 8 experts instead of all 8. But the dense backbone (attention, embeddings) is still fully replicated in DDP or sharded in FSDP.

## Exact Invocation Chain

### 1. `submit_megatron_benchmarks.sh` calls `gypsum/scripts/sbatch.sh`

For each (model, strategy, nodes) combination, submit_megatron_benchmarks.sh runs:

```bash
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
    $EXTRA_ARGS   # e.g., --tp-size 2
```

**Environment variables required**:
- `GYPSUM_DIR` — path to gypsum repo (default: `$HOME/projects/gypsum`)

### 2. `sbatch.sh` submits a SLURM job

`sbatch.sh` does the following:
1. Takes a git snapshot of the current gypsum commit (creates worktree under `$GYPSUM_DIR/sbatch/$COMMIT_HASH/`)
2. Generates an `sbatch` command with `--nodes`, `--ntasks-per-node=8`, etc.
3. The SLURM job runs `scripts/run.sh` which:
   - Starts an **Apptainer container** from the SIF image (`/shared/images/gypsum_refactor_26.01.22.sif`)
   - Mounts `/scratch`, `/home`, `/shared` (and optionally `/data_nfs`, `/data_vast`, `/fsx`)
   - Uses `--writable-tmpfs` (allows pip installs inside the container)
   - Since `--custom-script` is set, it runs `run_megatron_benchmark.sh` directly instead of `python train.py`

### 3. `run_megatron_benchmark.sh` runs inside the Apptainer container

This is the actual benchmark script. For TP/CP/EP experiments, it:

#### a) Installs Megatron-Core
```bash
cd /home/dwromero/projects/fsdp-bench/megatron-lm
pip uninstall -y apex 2>/dev/null || true   # Remove Apex (conflicts with RMSNorm)
pip install --quiet -e .
```

#### b) Installs TransformerEngine (only when TP/CP/EP > 1)
```bash
# 1. Meta package + core CUDA library (prebuilt wheel)
pip install --quiet transformer-engine==2.11.0 transformer-engine-cu12==2.11.0

# 2. Missing dependency for TE 2.11
pip install --quiet onnxscript

# 3. PyTorch bindings (needs compilation — cuDNN headers must be found)
#    The script searches for cudnn.h in:
#      - nvidia.cudnn pip package (site-packages/nvidia/cudnn/include/)
#      - /usr/local/cuda/include
#      - /usr/include
CUDNN_INCLUDE="<detected path to directory containing cudnn.h>"
CPLUS_INCLUDE_PATH="$CUDNN_INCLUDE:${CPLUS_INCLUDE_PATH:-}" \
C_INCLUDE_PATH="$CUDNN_INCLUDE:${C_INCLUDE_PATH:-}" \
pip install --quiet --no-build-isolation transformer-engine-torch==2.11.0

# 4. Verify
python -c "from transformer_engine.pytorch import TransformerLayer; print('TransformerEngine import OK')"
```

**Common failure point**: If `cudnn.h` is not found, the `transformer-engine-torch` compilation fails with `fatal error: cudnn.h: No such file or directory`. The fix is ensuring the `nvidia-cudnn-cu12` pip package is installed (it comes with the container's PyTorch) and that the script can locate its `include/` directory.

#### c) Launches torchrun

```bash
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    pretrain_gpt.py \
    <LLAMA_ARGS> <MODEL_ARGS> <MOE_ARGS> <PARALLEL_ARGS> <TRAINING_ARGS> <STRATEGY_ARGS> <DATA_ARGS>
```

## Concrete Example: TP=2, ZeRO-3, 2 Nodes

The full `torchrun` command for `megatron_tp2_8b_optim_grads_params_2N`:

```bash
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rotary-percent 1.0 \
    --swiglu \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --tensor-model-parallel-size 2 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 500 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --lr-decay-iters 500 \
    --lr-warmup-iters 20 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 1000 \
    --eval-iters 0 \
    --distributed-timeout-minutes 30 \
    --use-distributed-optimizer \
    --use-megatron-fsdp \
    --data-parallel-sharding-strategy optim_grads_params \
    --no-gradient-accumulation-fusion \
    --ckpt-format fsdp_dtensor \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --calculate-per-token-loss \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --split 99,1,0
```

**Note**: `CUDA_DEVICE_MAX_CONNECTIONS` is **unset** when FSDP is active (Megatron has a hard assert against it). For DDP + TP, it's set to `1`. For FSDP + TP/CP, it's unset with a warning about suboptimal TP/CP overlap on H100.

## Concrete Example: EP=2, HSDP, 4 Nodes

The 8B-MoE with EP=2 and HSDP on 4 nodes:

```bash
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rotary-percent 1.0 \
    --swiglu \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --kv-channels 128 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --moe-token-dispatcher-type alltoall \
    --expert-model-parallel-size 2 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --train-iters 500 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --lr-decay-iters 500 \
    --lr-warmup-iters 20 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 1000 \
    --eval-iters 0 \
    --distributed-timeout-minutes 30 \
    --use-distributed-optimizer \
    --use-megatron-fsdp \
    --data-parallel-sharding-strategy optim_grads_params \
    --num-distributed-optimizer-instances 4 \
    --no-gradient-accumulation-fusion \
    --ckpt-format fsdp_dtensor \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --calculate-per-token-loss \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 128256 \
    --split 99,1,0
```

## Smart Skipping Rules

The submission script automatically skips known-bad combinations:
- **8B + DDP** at any node count (OOM on H100 80GB) — but only for pure DP (TP/CP modes still try DDP+TP/CP)
- **HSDP at 1 node** (identical to ZeRO-3 at 1N, since intra-node shard = full shard)

## Interactive Debugging

If jobs fail, use `gpu-session.sh` and `gpu-exec.sh` for interactive debugging inside the same container environment:

```bash
# Start an interactive GPU session (allocates 1 GPU node via SLURM)
GYPSUM_DIR=$HOME/projects/gypsum $HOME/projects/gypsum/scripts/agents/gpu-session.sh start --gpus 1

# Run commands inside the container (same Apptainer image as batch jobs)
GYPSUM_DIR=$HOME/projects/gypsum $HOME/projects/gypsum/scripts/agents/gpu-exec.sh --timeout 300 '
  cd /home/dwromero/projects/fsdp-bench/megatron-lm
  pip install -e .
  pip install transformer-engine==2.11.0 transformer-engine-cu12==2.11.0
  pip install onnxscript
  python -c "from transformer_engine.pytorch import TransformerLayer; print(\"TE OK\")"
'

# When done
GYPSUM_DIR=$HOME/projects/gypsum $HOME/projects/gypsum/scripts/agents/gpu-session.sh stop
```

## Key Differences: TE vs Non-TE Runs

| Aspect | Pure FSDP (TP=CP=EP=1) | TP/CP/EP experiments |
|--------|------------------------|----------------------|
| `--transformer-impl` | `local` | `transformer_engine` |
| TransformerEngine | Not installed | Required (2.11.0) |
| `--no-rope-fusion` | Yes | No (TE handles RoPE) |
| `--no-persist-layer-norm` | Yes | No (TE handles norms) |
| CUDA_DEVICE_MAX_CONNECTIONS | Unset for FSDP, =1 for DDP | Unset when FSDP active |
| Apex | Uninstalled (conflicts) | Uninstalled (conflicts) |

## Troubleshooting

### `fatal error: cudnn.h: No such file or directory`
TransformerEngine's PyTorch bindings (`transformer-engine-torch`) need cuDNN headers at compile time. The script auto-detects them but may fail if:
- The `nvidia-cudnn-cu12` pip package is missing
- The container's include paths differ

**Fix**: Manually find `cudnn.h` and set `CPLUS_INCLUDE_PATH`:
```bash
find / -name "cudnn.h" 2>/dev/null
export CPLUS_INCLUDE_PATH=/path/to/dir/containing/cudnn.h
export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
pip install --no-build-isolation transformer-engine-torch==2.11.0
```

### `RuntimeError: Found empty transformer-engine meta package`
Only the meta package was installed, not the CUDA extensions. Install explicitly:
```bash
pip install transformer-engine==2.11.0 transformer-engine-cu12==2.11.0
pip install --no-build-isolation transformer-engine-torch==2.11.0
```

### `ModuleNotFoundError: No module named 'onnxscript'`
Missing dependency for TE 2.11:
```bash
pip install onnxscript
```

### `AssertionError` about `CUDA_DEVICE_MAX_CONNECTIONS`
Megatron FSDP asserts `CUDA_DEVICE_MAX_CONNECTIONS != 1`. Make sure it's unset:
```bash
unset CUDA_DEVICE_MAX_CONNECTIONS
```

### OOM for 8B-MoE or 8B TP=2 at 1 Node
Some 1N configs OOM because TransformerEngine uses more intermediate memory than `--transformer-impl local`. Multi-node runs (2N, 4N) provide more aggregate memory and should succeed.

## Files

| File | Purpose |
|------|---------|
| `submit_megatron_benchmarks.sh` | Job submission orchestrator — iterates over models × strategies × nodes |
| `run_megatron_benchmark.sh` | Benchmark launcher — runs inside Apptainer container on compute nodes |
| `gypsum/scripts/sbatch.sh` | SLURM job submission wrapper (creates git worktree, generates sbatch) |
| `gypsum/scripts/run.sh` | SLURM job entry point (starts Apptainer, runs custom script) |
| `gypsum/scripts/container-config.sh` | Container image path and mount points |
| `gypsum/scripts/agents/gpu-session.sh` | Interactive GPU session manager |
| `gypsum/scripts/agents/gpu-exec.sh` | Execute commands in interactive GPU container |
