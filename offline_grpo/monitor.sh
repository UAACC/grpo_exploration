#!/bin/bash
#
# Monitor SLURM jobs for the offline GRPO pipeline.
#
# Usage:
#   bash monitor.sh                  # show all your running/pending jobs + latest logs
#   bash monitor.sh <job_id>         # monitor a specific job
#   bash monitor.sh watch            # auto-refresh every 30s
#   bash monitor.sh tail <job_id>    # tail the output of a specific job
#

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"
cd "${WORK_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

separator() {
    echo -e "${BLUE}$(printf '=%.0s' {1..70})${NC}"
}

show_jobs() {
    separator
    echo -e "${BLUE}SLURM Jobs${NC}"
    separator
    local jobs=$(squeue -u "$USER" --format="%.10i %.20j %.2t %.10M %.6D %.4C %.10m %.20R" 2>/dev/null)
    if [ -z "$jobs" ] || [ "$(echo "$jobs" | wc -l)" -le 1 ]; then
        echo -e "  ${YELLOW}No running or pending jobs${NC}"
    else
        echo "$jobs"
    fi
    echo ""
}

show_recent_completed() {
    separator
    echo -e "${BLUE}Recently Completed Jobs (last 2 hours)${NC}"
    separator
    sacct -u "$USER" --starttime=$(date -d '2 hours ago' '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -v-2H '+%Y-%m-%dT%H:%M:%S' 2>/dev/null) \
        --format="JobID%10,JobName%20,State%12,Elapsed%10,ExitCode%8,NodeList%15" \
        --noheader 2>/dev/null | grep -v "\.batch" | grep -v "\.extern" | head -10
    echo ""
}

show_output_files() {
    separator
    echo -e "${BLUE}Latest Output Files${NC}"
    separator
    local files=$(ls -t ${WORK_DIR}/*.out 2>/dev/null | head -5)
    if [ -z "$files" ]; then
        echo "  No .out files found"
    else
        for f in $files; do
            local size=$(du -h "$f" 2>/dev/null | cut -f1)
            local modified=$(stat -c '%Y' "$f" 2>/dev/null)
            local age=""
            if [ -n "$modified" ]; then
                local now=$(date +%s)
                local diff=$((now - modified))
                if [ $diff -lt 60 ]; then
                    age="${diff}s ago"
                elif [ $diff -lt 3600 ]; then
                    age="$((diff/60))m ago"
                else
                    age="$((diff/3600))h ago"
                fi
            fi
            local lines=$(wc -l < "$f" 2>/dev/null)
            local last_line=$(tail -1 "$f" 2>/dev/null | head -c 100)
            echo -e "  ${GREEN}$(basename $f)${NC} (${size}, ${lines} lines, ${age})"
            echo "    last: ${last_line}"
        done
    fi
    echo ""
}

show_rollout_progress() {
    separator
    echo -e "${BLUE}Rollout Progress${NC}"
    separator

    local shard_dir="${SCRATCH}/rollouts/math_teacher"
    local merged="${shard_dir}/rollouts_full.jsonl"

    if [ -f "$merged" ]; then
        local count=$(wc -l < "$merged")
        echo -e "  ${GREEN}Merged file exists: ${count} records${NC}"
        return
    fi

    local any_shard=false
    for i in 0 1 2 3; do
        local shard="${shard_dir}/rollouts_shard_${i}.jsonl"
        if [ -f "$shard" ]; then
            any_shard=true
            local count=$(wc -l < "$shard" 2>/dev/null)
            local size=$(du -h "$shard" 2>/dev/null | cut -f1)
            local expected=3000
            local pct=$((count * 100 / expected))
            echo -e "  Shard ${i}: ${count}/${expected} problems (${pct}%) [${size}]"
        fi
    done

    if [ "$any_shard" = false ]; then
        echo -e "  ${YELLOW}No shard files yet — rollouts may not have started generating${NC}"
    fi
    echo ""
}

show_training_progress() {
    separator
    echo -e "${BLUE}Training Progress${NC}"
    separator

    local output_dir="${SCRATCH}/checkpoints/offline_grpo_math"
    if [ -d "$output_dir" ]; then
        local checkpoints=$(ls -d ${output_dir}/checkpoint-* 2>/dev/null | wc -l)
        echo "  Output dir: ${output_dir}"
        echo "  Checkpoints saved: ${checkpoints}"
        if [ $checkpoints -gt 0 ]; then
            ls -dt ${output_dir}/checkpoint-* 2>/dev/null | head -3 | while read ckpt; do
                echo "    $(basename $ckpt) ($(du -sh "$ckpt" 2>/dev/null | cut -f1))"
            done
        fi
    else
        echo -e "  ${YELLOW}No training output yet${NC}"
    fi
    echo ""
}

show_job_detail() {
    local job_id=$1
    separator
    echo -e "${BLUE}Job ${job_id} Detail${NC}"
    separator

    # Job info
    scontrol show job "$job_id" 2>/dev/null | grep -E "JobId|JobName|JobState|RunTime|TimeLimit|NumNodes|NumCPUs|Gres|WorkDir|StdOut|StdErr|ExitCode" | sed 's/^/  /'

    echo ""

    # Find output file
    local outfile=$(scontrol show job "$job_id" 2>/dev/null | grep "StdOut=" | sed 's/.*StdOut=//')
    if [ -n "$outfile" ] && [ -f "$outfile" ]; then
        echo -e "${BLUE}Last 20 lines of output:${NC}"
        tail -20 "$outfile"
    else
        # Try common patterns
        for pattern in *-${job_id}.out; do
            if [ -f "$pattern" ]; then
                echo -e "${BLUE}Last 20 lines of ${pattern}:${NC}"
                tail -20 "$pattern"
                break
            fi
        done
    fi
}

tail_job() {
    local job_id=$1
    local outfile=""

    # Find output file
    outfile=$(scontrol show job "$job_id" 2>/dev/null | grep "StdOut=" | sed 's/.*StdOut=//')
    if [ -z "$outfile" ] || [ ! -f "$outfile" ]; then
        for pattern in *-${job_id}.out; do
            if [ -f "$pattern" ]; then
                outfile="$pattern"
                break
            fi
        done
    fi

    if [ -n "$outfile" ] && [ -f "$outfile" ]; then
        echo -e "${BLUE}Tailing: ${outfile}${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        tail -f "$outfile"
    else
        echo -e "${RED}Cannot find output file for job ${job_id}${NC}"
    fi
}

# ── Main ──────────────────────────────────────────────────────

case "${1}" in

""|status)
    echo ""
    echo -e "${BLUE}Pipeline Monitor — $(date)${NC}"
    show_jobs
    show_recent_completed
    show_output_files
    show_rollout_progress
    show_training_progress
    ;;

watch)
    while true; do
        clear
        echo -e "${BLUE}Pipeline Monitor — $(date) (refreshing every 30s, Ctrl+C to stop)${NC}"
        show_jobs
        show_recent_completed
        show_output_files
        show_rollout_progress
        show_training_progress
        sleep 30
    done
    ;;

tail)
    if [ -z "$2" ]; then
        echo "Usage: bash monitor.sh tail <job_id>"
        exit 1
    fi
    tail_job "$2"
    ;;

[0-9]*)
    show_job_detail "$1"
    ;;

*)
    echo "Usage: bash monitor.sh [command]"
    echo ""
    echo "  (no args)       Show full pipeline status"
    echo "  watch           Auto-refresh every 30s"
    echo "  <job_id>        Show detail for a specific job"
    echo "  tail <job_id>   Tail the output of a specific job"
    ;;

esac
