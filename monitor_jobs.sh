#!/bin/bash
# Monitor running Megatron benchmark jobs.
# Detects stuck jobs (running > STUCK_THRESHOLD_MIN without training iterations)
# and OOM/failure as soon as they happen.
#
# Usage: ./monitor_jobs.sh [--interval SECONDS] [--kill-stuck]
#
# Options:
#   --interval N     Check every N seconds (default: 120)
#   --kill-stuck     Automatically cancel jobs detected as stuck

set -euo pipefail

INTERVAL=120
KILL_STUCK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) INTERVAL="$2"; shift 2 ;;
        --kill-stuck) KILL_STUCK=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

STUCK_THRESHOLD_MIN=12

echo "=== Megatron Job Monitor ==="
echo "  Check interval: ${INTERVAL}s"
echo "  Stuck threshold: ${STUCK_THRESHOLD_MIN} min"
echo "  Auto-kill stuck: $( [[ $KILL_STUCK -eq 1 ]] && echo YES || echo NO )"
echo ""

while true; do
    timestamp=$(date '+%H:%M:%S')
    running_jobs=$(squeue -u david.romero --states=RUNNING --noheader --format="%.10i %.40j %.10M %.6D" 2>/dev/null || true)
    pending_count=$(squeue -u david.romero --states=PENDING --noheader 2>/dev/null | wc -l || echo 0)
    running_count=$(echo "$running_jobs" | grep -c '[0-9]' || echo 0)

    echo "[$timestamp] Running: $running_count | Pending: $pending_count"

    if [[ $running_count -eq 0 && $pending_count -eq 0 ]]; then
        echo "[$timestamp] All jobs finished!"
        break
    fi

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        jid=$(echo "$line" | awk '{print $1}')
        jname=$(echo "$line" | awk '{print $2}')
        elapsed=$(echo "$line" | awk '{print $3}')

        # Parse elapsed time MM:SS or H:MM:SS to minutes
        if [[ "$elapsed" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
            mins=$(( ${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]} ))
        elif [[ "$elapsed" =~ ^([0-9]+):([0-9]+)$ ]]; then
            mins=${BASH_REMATCH[1]}
        else
            mins=0
        fi

        # Only check jobs running longer than threshold
        if [[ $mins -ge $STUCK_THRESHOLD_MIN ]]; then
            f="/shared/slurm-outputs/slurm-${jid}.out"
            if [[ -f "$f" ]]; then
                # Check for OOM
                if rg -q "CUDA out of memory|OutOfMemoryError" "$f" 2>/dev/null; then
                    echo "  !! OOM detected: $jid ($jname) — still running, cancelling"
                    scancel "$jid" 2>/dev/null || true
                    continue
                fi

                # Check for fatal errors (stale NFS, etc)
                if rg -q "Stale file handle|MockGPTDataset failed" "$f" 2>/dev/null; then
                    echo "  !! NFS/Dataset error: $jid ($jname) at ${elapsed} — cancelling"
                    scancel "$jid" 2>/dev/null || true
                    continue
                fi

                # Check if training iterations are happening
                has_iters=$(rg -c "elapsed time per iteration" "$f" 2>/dev/null || echo 0)
                if [[ "$has_iters" -eq 0 ]]; then
                    echo "  ** STUCK: $jid ($jname) running ${elapsed} with NO iterations"
                    if [[ $KILL_STUCK -eq 1 ]]; then
                        echo "     -> Cancelling stuck job"
                        scancel "$jid" 2>/dev/null || true
                    fi
                fi
            fi
        fi
    done <<< "$running_jobs"

    # Check recently completed jobs for failures
    for jid in $(squeue -u david.romero --states=COMPLETED,FAILED --noheader --format="%.10i" 2>/dev/null || true); do
        f="/shared/slurm-outputs/slurm-${jid}.out"
        [[ ! -f "$f" ]] && continue
        if rg -q "CUDA out of memory" "$f" 2>/dev/null; then
            echo "  !! Completed OOM: $jid"
        fi
    done

    sleep "$INTERVAL"
done
