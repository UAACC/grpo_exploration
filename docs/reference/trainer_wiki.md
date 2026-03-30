# HuggingFace Trainer Wiki

Comprehensive reference for `transformers.Trainer` and `transformers.TrainingArguments`.
Based on the installed version in our environment (`transformers` with `trl`).

---

## Table of Contents

1. [TrainingArguments — All Fields](#trainingarguments--all-fields)
2. [Trainer.__init__ — Constructor Parameters](#trainer__init__--constructor-parameters)
3. [Trainer — Public Methods](#trainer--public-methods)

---

## TrainingArguments — All Fields

TrainingArguments is a dataclass. Pass these as keyword arguments to `TrainingArguments(...)` or any subclass (e.g., `GRPOConfig`).

### Output & Saving

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `output_dir` | `Optional[str]` | `None` | Directory for model checkpoints and predictions. Required for most use cases. |
| `overwrite_output_dir` | `bool` | `False` | If `True`, overwrite contents of `output_dir`. Use to continue training from `output_dir` with `save_total_limit`. |
| `save_strategy` | `str \| SaveStrategy` | `"steps"` | When to save checkpoints: `"no"`, `"steps"`, `"epoch"`, `"best"`. |
| `save_steps` | `float` | `500` | Number of update steps between saves (when `save_strategy="steps"`). Can be float < 1 for fraction of total steps. |
| `save_total_limit` | `Optional[int]` | `None` | Max number of checkpoints to keep. Deletes older ones. If `load_best_model_at_end`, keeps the best + latest. |
| `save_safetensors` | `Optional[bool]` | `True` | Use `safetensors` format instead of `pickle`. |
| `save_on_each_node` | `bool` | `False` | Save checkpoints on each node in multi-node training (instead of only on main node). |
| `save_only_model` | `bool` | `False` | Only save model weights (skip optimizer/scheduler/rng states). Reduces checkpoint size but can't resume training. |
| `restore_callback_states_from_checkpoint` | `bool` | `False` | Restore callback states from checkpoint. Needed if callbacks carry state between steps (e.g., EarlyStoppingCallback). |

### Training Core

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `do_train` | `bool` | `False` | Whether to run training. Used by scripts that use `HfArgumentParser` to decide which phase to run. Not needed when calling `trainer.train()` directly. |
| `do_eval` | `bool` | `False` | Whether to run evaluation. Same as above. |
| `do_predict` | `bool` | `False` | Whether to run prediction on test set. Same as above. |
| `num_train_epochs` | `float` | `3.0` | Total number of training epochs. Can be fractional (e.g., `0.5` for half an epoch). |
| `max_steps` | `int` | `-1` | If > 0, overrides `num_train_epochs`. Total number of training steps. |
| `seed` | `int` | `42` | Random seed for initialization (Python, NumPy, PyTorch). |
| `data_seed` | `Optional[int]` | `None` | Random seed for data sampling. If `None`, uses `seed`. |
| `full_determinism` | `bool` | `False` | Enable full determinism (may impact performance). Sets `CUBLAS_WORKSPACE_CONFIG=:16:8`. |

### Batch Size & Accumulation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `per_device_train_batch_size` | `int` | `8` | Training batch size per GPU/TPU/CPU. |
| `per_device_eval_batch_size` | `int` | `8` | Evaluation batch size per GPU/TPU/CPU. |
| `gradient_accumulation_steps` | `int` | `1` | Number of steps to accumulate gradients before an optimizer step. Effective batch = `per_device * grad_accum * num_gpus`. |
| `eval_accumulation_steps` | `Optional[int]` | `None` | Accumulate predictions on CPU every N steps during eval (saves GPU memory for large eval sets). |
| `auto_find_batch_size` | `bool` | `False` | Automatically reduce batch size by 2x on OOM. Requires `accelerate`. |
| `per_gpu_train_batch_size` | `Optional[int]` | `None` | **Deprecated.** Use `per_device_train_batch_size`. |
| `per_gpu_eval_batch_size` | `Optional[int]` | `None` | **Deprecated.** Use `per_device_eval_batch_size`. |

### Optimizer

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `learning_rate` | `float` | `5e-5` | Initial learning rate for AdamW. |
| `weight_decay` | `float` | `0.0` | Weight decay (L2 regularization) applied to all parameters except bias and LayerNorm. |
| `adam_beta1` | `float` | `0.9` | Beta1 for AdamW. |
| `adam_beta2` | `float` | `0.999` | Beta2 for AdamW. |
| `adam_epsilon` | `float` | `1e-8` | Epsilon for AdamW numerical stability. |
| `max_grad_norm` | `float` | `1.0` | Max gradient norm for clipping. Set to `0` to disable. |
| `optim` | `str` | `"adamw_torch_fused"` | Optimizer name. Options: `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_hf"`, `"sgd"`, `"adafactor"`, `"adagrad"`, `"adamw_bnb_8bit"`, `"paged_adamw_8bit"`, `"lion"`, etc. |
| `optim_args` | `Optional[str]` | `None` | Extra optimizer arguments as `"key1=val1, key2=val2"` string. |
| `optim_target_modules` | `Optional[str \| list[str]]` | `None` | Target modules for optimizer (used with layerwise optimizers). |
| `adafactor` | `bool` | `False` | **Deprecated.** Use `optim="adafactor"`. |

### Learning Rate Schedule

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `lr_scheduler_type` | `str \| SchedulerType` | `"linear"` | Type of LR scheduler: `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, `"constant"`, `"constant_with_warmup"`, `"inverse_sqrt"`, `"reduce_lr_on_plateau"`. |
| `lr_scheduler_kwargs` | `Optional[dict]` | `{}` | Extra keyword args for the LR scheduler (e.g., `num_cycles` for cosine_with_restarts). |
| `warmup_ratio` | `float` | `0.0` | Fraction of total steps for linear warmup. Mutually exclusive with `warmup_steps`. |
| `warmup_steps` | `int` | `0` | Number of warmup steps. Overrides `warmup_ratio` if > 0. |

### Precision

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `bf16` | `bool` | `False` | Use bfloat16 mixed precision (requires Ampere+ GPU or TPU). |
| `fp16` | `bool` | `False` | Use float16 mixed precision. |
| `fp16_opt_level` | `str` | `"O1"` | Apex fp16 optimization level: `"O0"`, `"O1"`, `"O2"`, `"O3"`. |
| `half_precision_backend` | `str` | `"auto"` | Backend for mixed precision: `"auto"`, `"apex"`, `"cpu_amp"`. |
| `bf16_full_eval` | `bool` | `False` | Use bf16 for full eval (not mixed precision — entire model in bf16). Faster but may affect metrics. |
| `fp16_full_eval` | `bool` | `False` | Use fp16 for full eval. |
| `tf32` | `Optional[bool]` | `None` | Enable TF32 mode on Ampere+ GPUs. `None` means use PyTorch default. |

### Logging

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `logging_dir` | `Optional[str]` | `None` | TensorBoard log directory. Defaults to `output_dir/runs/DATETIME`. |
| `logging_strategy` | `str` | `"steps"` | When to log: `"no"`, `"steps"`, `"epoch"`. |
| `logging_steps` | `float` | `500` | Log every N update steps (when `logging_strategy="steps"`). Can be float < 1 for fraction. |
| `logging_first_step` | `bool` | `False` | Log metrics at step 0 (before any training). |
| `logging_nan_inf_filter` | `bool` | `True` | Filter NaN/Inf losses from logs. If `True`, averages non-NaN steps. |
| `report_to` | `str \| list[str] \| None` | `None` | Logging integrations: `"wandb"`, `"tensorboard"`, `"mlflow"`, `"comet_ml"`, `"clearml"`, `"none"`, `"all"`. `None` auto-detects installed loggers. |
| `run_name` | `Optional[str]` | `None` | Name for the experiment run (used by wandb, mlflow, etc.). Defaults to `output_dir`. |
| `log_level` | `str` | `"passive"` | Logger level for main process: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`, `"passive"` (keeps current level). |
| `log_level_replica` | `str` | `"warning"` | Logger level for non-main processes. |
| `log_on_each_node` | `bool` | `True` | Log on each node in multi-node training. If `False`, only log on main node. |
| `include_tokens_per_second` | `Optional[bool]` | `False` | Log tokens/second in training metrics. |
| `include_num_input_tokens_seen` | `Optional[bool]` | `False` | Track and log the total number of input tokens seen. |

### Evaluation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `eval_strategy` | `str \| IntervalStrategy` | `"no"` | When to evaluate: `"no"`, `"steps"`, `"epoch"`. |
| `eval_steps` | `Optional[float]` | `None` | Evaluate every N steps (when `eval_strategy="steps"`). Defaults to `logging_steps`. |
| `eval_delay` | `Optional[float]` | `0` | Number of epochs/steps to wait before first evaluation. |
| `eval_on_start` | `bool` | `False` | Run evaluation before training starts. |
| `prediction_loss_only` | `bool` | `False` | Only return loss during eval (skip prediction generation). |
| `eval_do_concat_batches` | `bool` | `True` | Concatenate all eval batch predictions. Set `False` for variable-length outputs. |
| `eval_use_gather_object` | `Optional[bool]` | `False` | Use `gather_object` instead of `gather` for eval predictions. Slower but handles variable-length tensors. |
| `batch_eval_metrics` | `bool` | `False` | Compute metrics per-batch during eval and average at the end (reduces memory for large eval sets). |
| `load_best_model_at_end` | `Optional[bool]` | `False` | Load the best checkpoint at training end (requires `eval_strategy` and `save_strategy` to match). |
| `metric_for_best_model` | `Optional[str]` | `None` | Metric to determine best model. Defaults to `"loss"`. Must be logged during eval. |
| `greater_is_better` | `Optional[bool]` | `None` | Whether higher `metric_for_best_model` is better. Auto-detected for common metrics. |
| `include_inputs_for_metrics` | `bool` | `False` | **Deprecated.** Use `include_for_metrics=["inputs"]`. |
| `include_for_metrics` | `list[str]` | `[]` | Extra data to pass to `compute_metrics`: `"inputs"`, `"losses"`. |

### Distributed Training

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `local_rank` | `int` | `-1` | Local process rank for distributed training. Set automatically by `torchrun`/`accelerate`. |
| `ddp_backend` | `Optional[str]` | `None` | DDP backend: `"nccl"`, `"gloo"`, `"mpi"`. Auto-detected if `None`. |
| `ddp_find_unused_parameters` | `Optional[bool]` | `None` | DDP `find_unused_parameters` flag. Default depends on `gradient_checkpointing`. |
| `ddp_bucket_cap_mb` | `Optional[int]` | `None` | DDP bucket size in MB for gradient allreduce. |
| `ddp_broadcast_buffers` | `Optional[bool]` | `None` | DDP buffer synchronization. |
| `ddp_timeout` | `int` | `1800` | Timeout (seconds) for DDP operations. |
| `parallelism_config` | `Optional[ParallelismConfig]` | `None` | Configuration for tensor/pipeline parallelism. |

### FSDP (Fully Sharded Data Parallel)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `fsdp` | `str \| list[FSDPOption]` | `""` | FSDP options: `"full_shard"`, `"shard_grad_op"`, `"offload"`, `"auto_wrap"`. Combine with space separator. |
| `fsdp_min_num_params` | `int` | `0` | Minimum number of parameters for FSDP auto-wrapping. |
| `fsdp_config` | `Optional[dict \| str]` | `None` | FSDP config dict or path to JSON/YAML file. |
| `fsdp_transformer_layer_cls_to_wrap` | `Optional[str]` | `None` | Transformer layer class name(s) to wrap with FSDP (e.g., `"BertLayer"`). |

### DeepSpeed

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `deepspeed` | `dict \| str \| None` | `None` | DeepSpeed config dict or path to JSON file. Enables DeepSpeed training. |

### Accelerate

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `accelerator_config` | `dict \| str \| None` | `None` | Config for `accelerate.Accelerator`. Can set dispatch batches, gradient sync, etc. |

### Data Loading

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataloader_drop_last` | `bool` | `False` | Drop the last incomplete batch. |
| `dataloader_num_workers` | `int` | `0` | Number of subprocesses for data loading. `0` = main process only. |
| `dataloader_prefetch_factor` | `Optional[int]` | `None` | Number of batches to prefetch per worker. |
| `dataloader_pin_memory` | `bool` | `True` | Pin memory in DataLoader for faster GPU transfer. |
| `dataloader_persistent_workers` | `bool` | `False` | Keep DataLoader workers alive between epochs. |
| `remove_unused_columns` | `Optional[bool]` | `True` | Remove dataset columns not used by the model. Set `False` if your `compute_loss` needs extra columns. |
| `group_by_length` | `bool` | `False` | Group similar-length samples in batches (reduces padding). |
| `length_column_name` | `Optional[str]` | `"length"` | Column name for pre-computed lengths when `group_by_length=True`. |
| `label_names` | `Optional[list[str]]` | `None` | Column names to treat as labels. Auto-detected from model signature if `None`. |
| `ignore_data_skip` | `bool` | `False` | Skip data fast-forwarding when resuming from checkpoint. Set `True` if data order doesn't matter. |

### Hub (HuggingFace Hub)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `push_to_hub` | `bool` | `False` | Push model to HF Hub after training. |
| `hub_model_id` | `Optional[str]` | `None` | Hub repo name. Defaults to `output_dir` name. |
| `hub_strategy` | `str \| HubStrategy` | `"every_save"` | When to push: `"end"`, `"every_save"`, `"checkpoint"`, `"all_checkpoints"`. |
| `hub_token` | `Optional[str]` | `None` | Hub authentication token. |
| `hub_private_repo` | `Optional[bool]` | `None` | Create private repo on Hub. |
| `hub_always_push` | `bool` | `False` | Push even if the previous push is not finished. |
| `hub_revision` | `Optional[str]` | `None` | Hub branch to push to. |
| `push_to_hub_model_id` | `Optional[str]` | `None` | **Deprecated.** Use `hub_model_id`. |
| `push_to_hub_organization` | `Optional[str]` | `None` | **Deprecated.** Use `hub_model_id`. |
| `push_to_hub_token` | `Optional[str]` | `None` | **Deprecated.** Use `hub_token`. |

### Device & Hardware

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `no_cuda` | `bool` | `False` | **Deprecated.** Use `use_cpu`. |
| `use_cpu` | `bool` | `False` | Force CPU training even if GPU is available. |
| `use_mps_device` | `bool` | `False` | **Deprecated.** MPS is auto-detected. |
| `tpu_num_cores` | `Optional[int]` | `None` | Number of TPU cores (set automatically). |
| `tpu_metrics_debug` | `bool` | `False` | **Deprecated.** Use `debug="tpu_metrics_debug"`. |

### Gradient Checkpointing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `gradient_checkpointing` | `bool` | `False` | Recompute activations during backward pass to save memory. Trades compute for memory (~30% more time, ~60% less memory). |
| `gradient_checkpointing_kwargs` | `Optional[dict]` | `None` | Kwargs for `torch.utils.checkpoint.checkpoint()` (e.g., `{"use_reentrant": False}`). |

### Miscellaneous

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `resume_from_checkpoint` | `Optional[str]` | `None` | Path to checkpoint directory to resume from. When passed to `trainer.train()`, also accepts `True` to auto-find the latest checkpoint in `output_dir`. |
| `label_smoothing_factor` | `float` | `0.0` | Label smoothing factor. `0.0` = no smoothing. |
| `debug` | `str \| list[DebugOption]` | `""` | Debug options: `"underflow_overflow"` (detect fp16 underflows). |
| `past_index` | `int` | `-1` | Index of past key/values in model output (for models with `past` output). |
| `disable_tqdm` | `Optional[bool]` | `None` | Disable tqdm progress bars. `None` = disable in non-interactive environments. |
| `skip_memory_metrics` | `bool` | `True` | Skip memory profiling metrics (they slow training). |
| `use_legacy_prediction_loop` | `bool` | `False` | Use legacy prediction loop instead of `evaluation_loop`. |
| `jit_mode_eval` | `bool` | `False` | Enable PyTorch JIT tracing for eval. |
| `use_ipex` | `bool` | `False` | Use Intel IPEX optimizations. |
| `torch_compile` | `bool` | `False` | Compile model with `torch.compile()`. Requires PyTorch 2.0+. |
| `torch_compile_backend` | `Optional[str]` | `None` | Backend for `torch.compile()`: `"inductor"`, `"eager"`, etc. |
| `torch_compile_mode` | `Optional[str]` | `None` | Mode for `torch.compile()`: `"default"`, `"reduce-overhead"`, `"max-autotune"`. |
| `torchdynamo` | `Optional[str]` | `None` | **Deprecated.** Use `torch_compile_*`. |
| `ray_scope` | `Optional[str]` | `"last"` | Scope for Ray Tune hyperparameter search. |
| `neftune_noise_alpha` | `Optional[float]` | `None` | NEFTune noise alpha for embedding augmentation. Helpful for instruction tuning. Typical values: 5-15. |
| `use_liger_kernel` | `Optional[bool]` | `False` | Use Liger kernel for fused operations (SwiGLU, CrossEntropy, RMSNorm, ROPE, etc.). Improves throughput. |
| `liger_kernel_config` | `Optional[dict[str, bool]]` | `None` | Per-kernel config for Liger (e.g., `{"rope": True, "swiglu": False}`). |
| `torch_empty_cache_steps` | `Optional[int]` | `None` | Call `torch.cuda.empty_cache()` every N steps. Reduces fragmentation but slows training. |
| `average_tokens_across_devices` | `Optional[bool]` | `True` | Average token counts across devices for loss normalization in distributed training. |
| `mp_parameters` | `str` | `""` | SageMaker model parallel parameters. |
| `fp16_backend` | `str` | `"auto"` | **Deprecated.** Use `half_precision_backend`. |

---

## Trainer.__init__ — Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `PreTrainedModel \| nn.Module \| None` | `None` | The model to train. If `None`, must provide `model_init`. |
| `args` | `TrainingArguments` | `None` | Training configuration. If `None`, uses default `TrainingArguments`. |
| `data_collator` | `Optional[DataCollator]` | `None` | Collates batch samples. Defaults to `default_data_collator` or `DataCollatorWithPadding` if tokenizer is provided. |
| `train_dataset` | `Dataset \| IterableDataset \| None` | `None` | Training dataset. Must implement `__len__` (for non-iterable) for epoch-based training. Columns not accepted by the model's `forward` are auto-removed (unless `remove_unused_columns=False`). |
| `eval_dataset` | `Dataset \| dict[str, Dataset] \| None` | `None` | Evaluation dataset. If dict, evaluates each dataset separately with metrics prefixed by the key. |
| `processing_class` | `PreTrainedTokenizerBase \| BaseImageProcessor \| FeatureExtractionMixin \| ProcessorMixin \| None` | `None` | The tokenizer/processor. Used for padding, saving alongside model, and `processing_class` attribute. In older code this was called `tokenizer`. |
| `model_init` | `Optional[Callable[[], PreTrainedModel]]` | `None` | Function that returns a fresh model instance. Required for hyperparameter search. |
| `compute_loss_func` | `Optional[Callable]` | `None` | Custom loss function. Receives `(model_output, labels)` or `(model_output, labels, num_items_in_batch)`. Overrides default loss computation. |
| `compute_metrics` | `Optional[Callable[[EvalPrediction], dict]]` | `None` | Computes metrics from predictions. Receives `EvalPrediction` (named tuple with `predictions` and `label_ids`). Returns dict of metric names to values. |
| `callbacks` | `Optional[list[TrainerCallback]]` | `None` | Extra callbacks to add. Default callbacks always included: `DefaultFlowCallback`, `PrinterCallback` or `ProgressCallback`, and integration callbacks (wandb, tensorboard, etc.). |
| `optimizers` | `tuple[Optimizer, LRScheduler]` | `(None, None)` | Provide custom optimizer and scheduler. If `None`, Trainer creates them from `args`. |
| `optimizer_cls_and_kwargs` | `Optional[tuple[type[Optimizer], dict]]` | `None` | Custom optimizer class and its kwargs. Alternative to passing a pre-built optimizer. |
| `preprocess_logits_for_metrics` | `Optional[Callable[[Tensor, Tensor], Tensor]]` | `None` | Preprocess logits before caching for metrics (e.g., argmax to save memory). Receives `(logits, labels)`. |

---

## Trainer — Public Methods

### Core Training & Evaluation

#### `train(resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs)`
Main training entry point.

- `resume_from_checkpoint`:
  - `str`: Path to a specific checkpoint directory (e.g., `"output/checkpoint-500"`)
  - `True`: Auto-find the latest checkpoint in `output_dir`
  - `None`: Train from scratch
- `trial`: For hyperparameter search (optuna/Ray Tune)
- `ignore_keys_for_eval`: Model output keys to ignore during eval

**Important**: The string `"latest"` is NOT valid — use `True` for auto-detection.

#### `evaluate(eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> dict`
Run evaluation loop and return metrics.

- `eval_dataset`: Override the eval dataset from `__init__`. Can be a dict for multi-dataset eval.
- `metric_key_prefix`: Prefix for metric keys in returned dict (default `"eval"` → `"eval_loss"`, `"eval_accuracy"`, etc.)

#### `predict(test_dataset, ignore_keys=None, metric_key_prefix="test") -> PredictionOutput`
Run prediction on test set. Returns `PredictionOutput` with `.predictions`, `.label_ids`, `.metrics`.

#### `training_step(model, inputs, num_items_in_batch=None) -> Tensor`
Perform a single training step. Override this to customize the training loop (e.g., custom forward pass, multiple loss terms).

#### `prediction_step(model, inputs, prediction_loss_only, ignore_keys=None) -> tuple`
Perform a single evaluation step. Returns `(loss, logits, labels)`. Override for custom eval logic.

#### `compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None)`
Compute loss from model inputs. Override for custom loss computation. Default: calls `model(**inputs)` and extracts loss from output.

### Model Saving & Loading

#### `save_model(output_dir=None, _internal_call=False)`
Save the model so it can be reloaded with `from_pretrained()`. For PEFT/LoRA models, saves only the adapter weights.

#### `save_state()`
Save the full trainer state (optimizer, scheduler, RNG states, trainer_state.json). Called automatically during checkpointing.

#### `push_to_hub(commit_message="End of training", blocking=True, token=None, revision=None) -> str`
Upload model and tokenizer to HuggingFace Hub. Returns the commit URL.

#### `create_model_card(language=None, license=None, tags=None, ...)`
Generate a draft model card with training metadata.

### Optimizer & Scheduler

#### `create_optimizer()`
Create the optimizer from `args.optim` and `args.learning_rate`. Called automatically. Override to use a custom optimizer.

#### `create_scheduler(num_training_steps, optimizer=None)`
Create the LR scheduler from `args.lr_scheduler_type`. Called automatically.

#### `create_optimizer_and_scheduler(num_training_steps)`
Convenience method that calls `create_optimizer()` then `create_scheduler()`.

#### `get_optimizer_cls_and_kwargs(args, model=None) -> tuple` [static]
Returns `(optimizer_class, kwargs)` based on `args.optim`. Useful for inspecting what optimizer would be created.

#### `get_optimizer_group(param=None)`
Get optimizer parameter group for a specific parameter, or all groups.

#### `get_learning_rates()`
Returns the current learning rate for each parameter group.

#### `get_decay_parameter_names(model) -> list[str]`
Get parameter names that will have weight decay applied (excludes bias and LayerNorm).

### Data

#### `get_train_dataloader() -> DataLoader`
Build the training DataLoader. Override to customize batching, sampling, etc.

#### `get_eval_dataloader(eval_dataset=None) -> DataLoader`
Build the evaluation DataLoader.

#### `get_test_dataloader(test_dataset) -> DataLoader`
Build the test DataLoader.

#### `get_batch_samples(epoch_iterator, num_batches, device) -> tuple`
Collect `num_batches` batches from the iterator. Used internally for gradient accumulation.

### Callbacks

#### `add_callback(callback)`
Add a `TrainerCallback` instance or class to the callback list.

#### `remove_callback(callback)`
Remove a callback by instance or class. Does not return it.

#### `pop_callback(callback)`
Remove a callback and return it. Useful for modifying and re-adding.

### Metrics & Logging

#### `log(logs, start_time=None)`
Send metrics to all registered loggers (wandb, tensorboard, etc.).

#### `log_metrics(split, metrics)`
Pretty-print metrics in a formatted table. `split` is a label like `"train"`, `"eval"`.

#### `save_metrics(split, metrics, combined=True)`
Save metrics to `{split}_results.json`. If `combined=True`, also appends to `all_results.json`.

#### `metrics_format(metrics) -> dict`
Format metric values for human-readable display (e.g., round floats, convert seconds to HH:MM:SS).

### Info & Utilities

#### `get_num_trainable_parameters()`
Number of trainable parameters in the model.

#### `get_total_train_batch_size(args) -> int`
Calculates `per_device_batch * gradient_accumulation * dp_world_size`.

#### `get_tp_size() -> int`
Get tensor parallel size (from model config or DeepSpeed).

#### `num_examples(dataloader) -> int`
Number of samples in a DataLoader.

#### `num_tokens(train_dl, max_steps=None) -> int` [static]
Number of tokens in a DataLoader (for token-based training).

#### `is_local_process_zero() -> bool`
Whether this is the local main process (rank 0 on this machine).

#### `is_world_process_zero() -> bool`
Whether this is the global main process (rank 0 across all machines).

#### `floating_point_ops(inputs) -> int`
Estimated FLOPs for the given inputs (if model implements `floating_point_ops`).

#### `store_flos()`
Accumulate floating point operations counter.

### Advanced

#### `hyperparameter_search(hp_space=None, compute_objective=None, n_trials=20, direction="minimize", backend=None, ...) -> BestRun`
Run hyperparameter search with optuna, Ray Tune, or SigOpt.

#### `create_accelerator_and_postprocess()`
Create the `Accelerator` object and apply post-processing. Called during `__init__`.

#### `propagate_args_to_deepspeed(auto_find_batch_size=False)`
Sync Trainer args to DeepSpeed config (batch size, precision, etc.).

#### `set_initial_training_values(args, dataloader, total_train_batch_size)`
Calculate training steps, warmup steps, and other values derived from the data.

#### `compare_trainer_and_checkpoint_args(training_args, trainer_state)`
Warn if current args differ from the checkpoint's args (e.g., different learning rate on resume).

#### `autocast_smart_context_manager(cache_enabled=True)`
Context manager for mixed precision autocast.

#### `compute_loss_context_manager()`
Context manager wrapping the forward pass (used for autocast, etc.).

#### `call_model_init(trial=None)`
Call `model_init` to get a fresh model (used in hyperparameter search).

#### `torch_jit_model_eval(model, dataloader, training=False)`
Apply JIT tracing to model for faster eval.

#### `init_hf_repo(token=None)`
Initialize a git repo for Hub pushing.

#### `evaluation_loop(dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval") -> EvalLoopOutput`
The inner evaluation loop. Called by `evaluate()` and `predict()`. Override for fine-grained control.

#### `prediction_loop(...)` [legacy]
Legacy evaluation loop. Use `evaluation_loop` instead.

---

## Key Patterns for Our Offline GRPO Setup

### What we use in `train.py`

```python
training_args = GRPOConfig(
    output_dir=...,                        # checkpoint directory
    learning_rate=5e-6,                    # lower than default 5e-5
    adam_beta1=0.9, adam_beta2=0.99,        # slightly different beta2
    beta=0.1,                              # GRPO-specific: KL penalty
    weight_decay=0.1,                      # regularization
    warmup_ratio=0.1,                      # 10% warmup
    lr_scheduler_type="cosine",            # cosine annealing
    bf16=True,                             # bfloat16 mixed precision
    per_device_train_batch_size=2,         # small due to long sequences
    gradient_accumulation_steps=8,         # effective batch = 16
    num_generations=4,                     # GRPO-specific: completions per prompt
    max_prompt_length=256,                 # GRPO-specific
    max_completion_length=786,             # GRPO-specific
    num_train_epochs=1,                    # single pass
    save_steps=500,                        # checkpoint every 500 steps
    max_grad_norm=0.1,                     # aggressive clipping (default 1.0)
    report_to="wandb",                     # logging
)
```

### Resuming from checkpoint

```python
# In train.py:
trainer.train(resume_from_checkpoint=True)   # auto-find latest in output_dir
trainer.train(resume_from_checkpoint="path/to/checkpoint-3500")  # specific checkpoint

# DO NOT pass the string "latest" — it's treated as a directory path
```

### What `save_model()` saves
- With LoRA/PEFT: Only adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
- Without LoRA: Full model weights
- Always: `training_args.bin`, tokenizer files
