# Vulcan Cluster Operations Guide

A comprehensive guide for running ML experiments safely on the Digital Research Alliance of Canada's **Vulcan** cluster.

---

## 1. Cluster Overview

### Vulcan Hardware
| Resource | Spec |
|----------|------|
| GPU | NVIDIA L40s (48 GB VRAM each) |
| CPUs per GPU | 16 |
| Nodes | 252 GPU nodes |
| Max walltime | 7 days |
| Interconnect | High-speed InfiniBand |

### GPU Request Syntax
```bash
--gpus-per-node=l40s:1    # 1 L40s GPU
--gpus-per-node=l40s:4    # 4 L40s GPUs (multi-GPU)
```
Each GPU comes with 16 CPUs and proportional memory by default.

---

## 2. Storage Systems

| Filesystem | Env Var | Quota | Backup | Purge Policy | Use For |
|------------|---------|-------|--------|-------------|---------|
| Home | `$HOME` | ~100 GB | Yes | Never | Code, configs, virtualenvs |
| Project | `$PROJECT` | Group-shared, larger | Yes | Never | Shared code, datasets |
| Scratch | `$SCRATCH` | ~25 TB | **No** | **60-day purge** (unused files deleted) | Large datasets, model checkpoints, rollouts |

### Critical Rules
- **NEVER store irreplaceable data on `$SCRATCH`** — it is purged automatically.
- Copy final results/checkpoints to `$PROJECT` or `$HOME` after experiments.
- Use `$SCRATCH` for intermediate outputs, caches, and large temporary files.
- Check quota: `diskusage_report`
- Touch files on scratch periodically to prevent purge: `find $SCRATCH/important_dir -exec touch {} +`

### Our Current Layout
```
$HOME/projects/aip-szepesva/shuai14/
├── backup_dongheng/offline_grpo/    ← code (backed up via $PROJECT)
├── verifiers/.venv/                 ← virtualenv (backed up)

$SCRATCH = /home/shuai14/scratch/
├── huggingface_cache/hub/           ← downloaded models (NOT backed up)
│   ├── models--Qwen--Qwen2.5-0.5B-Instruct/
│   ├── models--Qwen--Qwen2.5-Math-7B-Instruct/
│   ├── models--Qwen--Qwen2.5-1.5B-Instruct/
│   └── models--Qwen--Qwen3-8B/
├── datasets/MATH/                   ← MATH dataset cache
└── rollouts_*.jsonl                 ← pre-generated rollouts
```

---

## 3. Module System

Modules provide system-level software (compilers, interpreters, CUDA). They must be loaded **before** activating virtualenvs.

### Essential Commands
```bash
module avail python         # List available Python versions
module avail cuda           # List available CUDA versions
module load python/3.11     # Load Python 3.11 interpreter
module load cuda/12.6       # Load CUDA toolkit (needed for DeepSpeed, Flash Attention)
module list                 # Show currently loaded modules
module purge                # Unload all modules (use with care)
```

### Load Order Matters
```bash
# CORRECT order:
module load python/3.11 cuda/12.6
source /path/to/.venv/bin/activate

# WRONG — venv won't see module-provided Python packages:
source /path/to/.venv/bin/activate
module load python/3.11
```

---

## 4. Python Environment (uv + venv)

Our environment uses a three-layer system:
1. **Module** (`module load python/3.11`) → provides the Python interpreter
2. **Virtualenv** (`.venv/`) → provides installed packages
3. **uv** → fast pip alternative used to install packages

### Activation
```bash
module load python/3.11 cuda/12.6
source /home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate
```

### Package Management
```bash
# List installed packages (pip won't show uv-installed packages!)
uv pip freeze              # ← correct way
uv pip list                # ← also works

# Install from PyPI (login node only — compute nodes have no internet)
uv pip install <package>
# OR
pip install <package>

# Install from Alliance pre-built wheels (faster, optimized)
pip install --no-index <package>
avail_wheels <package>     # Check if Alliance wheel exists
```

### Key Packages in Our Env
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.8.0 | PyTorch |
| vllm | 0.11.0 | Fast LLM inference (rollout generation) |
| transformers | 4.56.1 | Model loading, tokenizers |
| trl | 0.21.0 | GRPO trainer |
| peft | 0.17.1 | LoRA adapters |
| flash_attn | 2.8.3 | Flash Attention 2 |
| accelerate | 1.10.1 | Distributed training |
| datasets | 4.1.0 | HuggingFace datasets |
| math_verify | 0.8.0 | Math answer verification |

---

## 5. Job Submission with SLURM

### Key Concepts
- **Login node**: For editing code, submitting jobs, light file operations. **Never run GPU workloads here.**
- **Compute node**: Where actual computation happens. **No internet access.**
- **srun**: Submit a non-interactive job (runs a command, returns output).
- **salloc**: Get an interactive shell on a compute node.
- **sbatch**: Submit a batch job script (runs in background).

### srun — Non-Interactive (Recommended for Testing)
```bash
srun --account=aip-szepesva \
     --time=0:30:00 \
     --gpus-per-node=l40s:1 \
     --cpus-per-task=16 \
     --mem=64G \
     bash -c 'module load python/3.11 cuda/12.6 && \
              source /home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate && \
              cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo && \
              bash run_test.sh train'
```

### salloc — Interactive Session
```bash
# Request resources (blocks until allocated)
salloc --account=aip-szepesva \
       --time=1:00:00 \
       --gpus-per-node=l40s:1 \
       --cpus-per-task=16 \
       --mem=64G

# Once allocated, you're on a compute node with a shell
module load python/3.11 cuda/12.6
source /home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo

# Run commands interactively
python train.py --rollout_path rollouts_test.jsonl ...

# Exit when done (releases resources immediately)
exit
```

### sbatch — Batch Job Script
```bash
#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --job-name=offline_grpo

module load python/3.11 cuda/12.6
source /home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo

python train.py \
    --rollout_path rollouts_test.jsonl \
    --target_model /path/to/student \
    --output_dir outputs/experiment_name
```

Submit with: `sbatch job_script.sh`

### Resource Guidelines for Our Workloads

| Task | GPUs | Time | Memory | Notes |
|------|------|------|--------|-------|
| Rollout generation (7B teacher) | 1 L40s | 30 min (test) / 2-4 hr (full) | 48-64 GB | vLLM uses most of GPU VRAM |
| Training (0.5B student, LoRA) | 1 L40s | 10 min (test) / 1-2 hr (full) | 48-64 GB | Light GPU usage with LoRA |
| Training (7B+ student) | 1-4 L40s | 1-4 hr | 64-128 GB | May need DeepSpeed ZeRO |
| Evaluation | 1 L40s | 15-30 min | 48 GB | vLLM inference |
| Diagnostics | 1 L40s | 10-30 min | 48 GB | Forward pass only |

---

## 6. Job Monitoring

```bash
# Check job status
squeue -u $USER                    # Your running/pending jobs
squeue -u $USER --start            # Estimated start time for pending jobs
sq                                 # Short alias (if available)

# Job details
scontrol show job <JOBID>          # Full job details
sacct -j <JOBID> --format=JobID,JobName,MaxRSS,Elapsed,State,ExitCode
                                   # Resource usage after completion

# Cancel a job
scancel <JOBID>                    # Cancel specific job
scancel -u $USER                   # Cancel ALL your jobs (use with care)

# Check cluster utilization
sinfo -p gpu                       # GPU partition status
```

### Monitor GPU Usage (on compute node)
```bash
nvidia-smi                         # Snapshot of GPU usage
watch -n 2 nvidia-smi              # Live monitoring (every 2 sec)
```

---

## 7. Safety Checklist

### Before Submitting a Job
- [ ] Test your script logic locally or with a minimal `srun` first
- [ ] Verify all file paths exist (models, datasets, rollouts)
- [ ] Confirm modules load correctly: `module load python/3.11 cuda/12.6`
- [ ] Confirm venv activates and has needed packages: `uv pip list | grep torch`
- [ ] Set reasonable `--time` (don't request 7 days for a 30-min job)
- [ ] Set `--output` and `--error` to capture logs
- [ ] Check disk quota: `diskusage_report`

### Things That Can Go Wrong
| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: torch` | Module not loaded before venv | Load `python/3.11` first |
| `MissingCUDAException` | CUDA module not loaded | Add `module load cuda/12.6` |
| `No space left on device` | Quota exceeded | Check `diskusage_report`, clean $SCRATCH |
| `OOM Killed` | Insufficient memory requested | Increase `--mem` |
| `CUDA out of memory` | Model too large for GPU | Reduce batch size, use DeepSpeed, or request more GPUs |
| Job pending forever | Requesting too many resources | Reduce `--time`, `--mem`, or GPU count |
| Internet error on compute node | Compute nodes have no internet | Download models/data on login node first |
| Files disappeared from scratch | 60-day purge policy | Use `$PROJECT` for important files |

### Resource Etiquette
- **Don't hog the login node** — no GPU work, no heavy computation.
- **Don't over-request resources** — wastes allocation and delays other users.
- **Cancel jobs you don't need** — `scancel <JOBID>` releases resources immediately.
- **Clean up scratch** — remove old checkpoints and rollouts you no longer need.
- **Use `--test-only`** to verify job scripts without actually submitting: `sbatch --test-only job.sh`

---

## 8. Multi-GPU and Distributed Training

### DeepSpeed (ZeRO)
For large models that don't fit on a single GPU:
```bash
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

# In training script, use accelerate + DeepSpeed:
accelerate launch --config_file ds_config.yaml train.py ...
```

### Key Environment Variables
```bash
export HF_HOME=/home/shuai14/scratch/huggingface_cache
export HF_DATASETS_CACHE=/home/shuai14/scratch/datasets/MATH
export TRANSFORMERS_OFFLINE=1          # Prevent accidental downloads on compute
export HF_DATASETS_OFFLINE=1           # Prevent accidental downloads on compute
export TOKENIZERS_PARALLELISM=false     # Avoid deadlocks with DataLoader workers
export NCCL_DEBUG=INFO                  # Debug multi-GPU communication issues
```

Setting `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` on compute nodes is a safety measure — it ensures the job fails loudly if a model/dataset isn't pre-downloaded, rather than hanging on a network timeout.

---

## 9. Our Experiment Pipeline

### Step 1: Generate Rollouts (Teacher)
```bash
srun --account=aip-szepesva --time=0:30:00 --gpus-per-node=l40s:1 \
     --cpus-per-task=16 --mem=64G \
     bash -c '... python generate_rollouts.py \
         --model_name /path/to/teacher \
         --output_path rollouts.jsonl \
         --test_mode'
```
Output: `rollouts.jsonl` (JSONL with completions, per-token logprobs, correctness)

### Step 2: Train Student (Offline GRPO)
```bash
srun --account=aip-szepesva --time=1:00:00 --gpus-per-node=l40s:1 \
     --cpus-per-task=16 --mem=64G \
     bash -c '... python train.py \
         --rollout_path rollouts.jsonl \
         --target_model /path/to/student \
         --output_dir outputs/experiment_name'
```
Output: LoRA checkpoint in `outputs/experiment_name/`

### Step 3: Evaluate
```bash
srun --account=aip-szepesva --time=0:30:00 --gpus-per-node=l40s:1 \
     --cpus-per-task=16 --mem=64G \
     bash -c '... python evaluate.py \
         --model_path outputs/experiment_name \
         --base_model /path/to/student'
```

### Step 4: Diagnose (Optional)
```bash
srun --account=aip-szepesva --time=0:30:00 --gpus-per-node=l40s:1 \
     --cpus-per-task=6 --mem=48G \
     bash -c '... python diagnose.py \
         --rollout_path rollouts.jsonl \
         --target_model /path/to/student'
```

---

## 10. Quick Reference

### Common Command Patterns
```bash
# Full activation sequence (use in every job)
module load python/3.11 cuda/12.6
source /home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME=/home/shuai14/scratch/huggingface_cache
export HF_DATASETS_CACHE=/home/shuai14/scratch/datasets/MATH
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo

# Check what's running
squeue -u $USER

# Quick interactive session (1 GPU, 1 hour)
salloc --account=aip-szepesva --time=1:00:00 --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G

# Check disk usage
diskusage_report

# Check GPU on compute node
nvidia-smi
```

### Model Paths (Pre-Downloaded)
```
Student (0.5B): /home/shuai14/scratch/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
Teacher (7B):   /home/shuai14/scratch/huggingface_cache/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d
```

---

*Last updated: 2026-03-04*