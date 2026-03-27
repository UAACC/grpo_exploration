#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:59:00
#SBATCH --mail-user=mrli@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
cd /project/aip-szepesva/mrli/backup_dongheng
module load cuda/12.6
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
export WANDB_MODE=offline
source .venv/bin/activate
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="/scratch/mrli/models"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
run_eval() {
  local label="$1"
  local model="$2"
  log "START EVALUATING $label"
  CUDA_VISIBLE_DEVICES=0 python -u eval_math_boxed.py --model_name "$model" --runs=10 --max_model_len=3072 --temperature=0.0 --top_p=1.0 --top_k=0 --max_tokens=2048
  # CUDA_VISIBLE_DEVICES=0 python -u eval_gsm_boxed.py --model_name "$model" --runs=10 --max_model_len=1024 --temperature=0.0 --top_p=1.0 --top_k=0 --max_tokens=786
  log "FINISH EVALUATING $label"
}
# base_path_is="/scratch/mrli/outputs/Qwen2.5-0.5B-GRPO-baseline-math-20250902-000715-math"
# run_eval "baseline 0.5B, step 3000, constant with warmup" "$base_path_is"
# base_path_is_8="/scratch/mrli/outputs/Qwen2.5-1.5B-Instruct-RPG-bs32_lr8e-06_max_grad_norm0.1_beta0.0_bsize2_run0_sched_constant_with_warmup-20250914-035740-math"
# run_eval "baseline math 1.5B, lr 8e-6, epoch 3000, RPG" "$base_path_is_8/checkpoint-3000"

# base_path_is_9="/scratch/mrli/outputs/Qwen2.5-1.5B-Instruct-RPG-bs32_lr9e-06_max_grad_norm0.1_beta0.0_bsize2_run0_sched_constant_with_warmup-20250914-040632-math"
# run_eval "baseline math 1.5B, lr 9e-6, epoch 3000, RPG" "$base_path_is_9/checkpoint-3000"
# run_eval "baseline math 1.5B, epoch 1500" "$base_path_is/checkpoint-1500"
# run_eval "baseline math 1.5B, epoch 2000" "$base_path_is/checkpoint-2000"
# run_eval "baseline 1.5B, step 3000, constant with warmup" "$base_path_is"
# Qwen2.5-0.5B-GRPO_bs256_ISUB1.75_ISLB0.25_beta0.0_bsize4_run0_sched_constant_with_warmup-20250826-194337-math
# for checkpoint in {500..2500..500}; do
#     run_eval "baseline 1.5B, step $checkpoint, constant with warmup, RPG" "$base_path_is_8/checkpoint-$checkpoint"
# done

# for checkpoint in {500..2500..500}; do
#     run_eval "baseline 1.5B, step $checkpoint, constant with warmup, RPG" "$base_path_is/checkpoint-$checkpoint"
#     run_eval "baseline, step $checkpoint, constant with warmup" "$base_path_bl/checkpoint-$checkpoint"
# done
# base_path_is_9="/scratch/mrli/outputs/Qwen2.5-1.5B-Instruct-RPG-bs32_lr5e-06_max_grad_norm0.1_beta0.0_bsize2_run0_sched_constant_with_warmup-20250913-223103-math"
# run_eval "baseline math 1.5B, lr 5e-6, epoch 3000, RPG" "$base_path_is_9"
base_path_is_9="Qwen/Qwen2.5-1.5B-Instruct"
run_eval "Qwen2.5-1.5B-Instruct" "$base_path_is_9"

base_path_is_9="/scratch/mrli/outputs/Qwen2.5-1.5B-Instruct-baseline-bs32_lr5e-07_sched_constant_with_warmup_max_grad_norm1.0_beta0.4_bsize1_run0-ref_steps16-warm_ratio0.5-20251209-163516-math"
run_eval "Qwen2.5-1.5B-5e-7-1.0-baseline-GRPO" "$base_path_is_9"

base_path_is_9="/scratch/mrli/outputs/Qwen2.5-1.5B-Instruct-baseline-bs32_lr1e-06_sched_constant_with_warmup_max_grad_norm0.1_beta0.4_bsize1_run0-ref_steps16-warm_ratio0.5-20251210-131357-math"
run_eval "Qwen2.5-1.5B-lr1e-06-0.1-baseline" "$base_path_is_9"
# base_path_is_9 = "/scratch/mrli/outputs/Qwen2.5-0.5B-Instruct-RPG-detached-bs32_lr5e-06_sched_constant_with_warmup-curw0.3-pastw0.7_max_grad_norm0.1_beta0.2_bsize2_run0-ref_steps32-warm_ratio0.1-20251124-140753-math"
# run_eval "Qwen 0.5B maxnorm 0.1, beta0.2 lr5e-06 ref_steps32 past0.3 warm_ratio0.1" "$base_path_is_9"
# run_eval "baseline math 1.5B, epoch 1500" "$base_path_is/checkpoint-1500"
# run_eval "baseline math 1.5B, epoch 2000" "$base_path_is/checkpoint-2000"
# run_eval "baseline 1.5B, step 3000, constant with warmup" "$base_path_is"
# Qwen2.5-0.5B-GRPO_bs256_ISUB1.75_ISLB0.25_beta0.0_bsize4_run0_sched_constant_with_warmup-20250826-194337-math




# python eval.py
# run_eval "buffer 256, full buffer, run1" "/scratch/mrli/outputs/Qwen2.5-1.5B-GRPO-base-Qwen2.5-1.5B-GRPO_bs256_clip10.0_beta0.0_bsize2_no-snis_20250817-222542_run1-gsm8k/checkpoint-final"
# run_eval "buffer 256, full buffer, run2" "/scratch/mrli/outputs/Qwen2.5-1.5B-GRPO-base-Qwen2.5-1.5B-GRPO_bs256_clip10.0_beta0.0_bsize2_no-snis_20250817-222542_run2-gsm8k/checkpoint-final"
# run_eval "buffer 256, full buffer, run3" "/scratch/mrli/outputs/Qwen2.5-1.5B-GRPO-base-Qwen2.5-1.5B-GRPO_bs256_clip10.0_beta0.0_bsize2_no-snis_20250817-222542_run3-gsm8k/checkpoint-final"
# run_eval "buffer 256, full buffer, run4, checkpoint-500" "/scratch/mrli/outputs/Qwen2.5-1.5B-GRPO-base-Qwen2.5-1.5B-GRPO_bs256_clip10.0_beta0.0_bsize2_no-snis_20250819-005117_run4-gsm8k/checkpoint-500"
# run_eval "baseline, checkpoint-467" "/project/aip-szepesva/mrli/backup_dongheng/outputs/Qwen-1.5B-GRPO-vanilla-baseline/checkpoint-1869"

# run_eval "1.5, step 10, constant with warmup" "Qwen2.5-1.5B-GRPO_bs256_ISUB1.25_ISLB0.75_beta0.0_bsize2_run0_sched_constant_with_warmuptruncated400-20250821-040447-gsm8k/checkpoint-10"
# run_eval "1.5, step 1000, constant with warmup" "/scratch/mrli/outputs/Qwen2.5-1.5B-GRPO_bs256_ISUB1.5_ISLB0.5_beta0.0_bsize2_run1_sched_constant_with_warmup-gsm8k/checkpoint-1000"
# run_eval "1.5, step 1500, constant with warmup" "/scratch/mrli/outputs/Qwen2.5-1.5B-GRPO_bs256_ISUB1.5_ISLB0.5_beta0.0_bsize2_run1_sched_constant_with_warmup-gsm8k/checkpoint-1500"
# run_eval "1.5, step final, constant with warmup" "/scratch/mrli/outputsQwen2.5-1.5B-GRPO_bs256_ISUB1.25_ISLB0.75_beta0.0_bsize4_run0_sched_constant_with_warmup-20250825-203454-gsm8k"                                                                                                                                           1,1           Top
