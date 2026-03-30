# HuggingFace Transformers TrainingArguments Reference

> Extracted from the official HuggingFace Transformers Trainer documentation.
> `TrainingArguments` centralizes all hyperparameters, optimization settings, logging preferences, and infrastructure choices needed for training with the `Trainer` class.
>
> Source: [transformers/training_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L179)

---

## Table of Contents

1. [Output and Directory](#1-output-and-directory)
2. [Training Basics](#2-training-basics)
3. [Optimizer](#3-optimizer)
4. [Learning Rate Scheduler](#4-learning-rate-scheduler)
5. [Logging](#5-logging)
6. [Evaluation](#6-evaluation)
7. [Checkpointing / Saving](#7-checkpointing--saving)
8. [DataLoader](#8-dataloader)
9. [Push to Hub](#9-push-to-hub)
10. [Testing / Prediction](#10-testing--prediction)

---

## 1. Output and Directory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `"trainer_output"` | The output directory where the model predictions and checkpoints will be written. |

---

## 2. Training Basics

These parameters are grouped under `set_training()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_train` | `bool` | `False` | Whether to run training. Automatically set to `True` when `set_training()` is called. |
| `learning_rate` | `float` | `5e-5` | The initial learning rate for the optimizer. |
| `per_device_train_batch_size` | `int` | `8` | The batch size per device (GPU/TPU core/CPU) used for training. Set via `batch_size` in `set_training()`. |
| `weight_decay` | `float` | `0` | The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the optimizer. |
| `num_train_epochs` | `float` | `3.0` | Total number of training epochs to perform. If not an integer, will perform the decimal part percents of the last epoch before stopping training. |
| `max_steps` | `int` | `-1` | If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`. For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until `max_steps` is reached. |
| `gradient_accumulation_steps` | `int` | `1` | Number of updates steps to accumulate the gradients for, before performing a backward/update pass. When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples. |
| `seed` | `int` | `42` | Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the `model_init` function to instantiate the model if it has some randomly initialized parameters. |
| `gradient_checkpointing` | `bool` | `False` | If `True`, use gradient checkpointing to save memory at the expense of slower backward pass. |

---

## 3. Optimizer

These parameters are grouped under `set_optimizer()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optim` | `str` or `OptimizerNames` | `"adamw_torch"` | The optimizer to use. Options include: `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`, `"adamw_anyprecision"`, `"adafactor"`. Set via `name` in `set_optimizer()`. |
| `learning_rate` | `float` | `5e-5` | The initial learning rate. (Also listed under Training Basics.) |
| `weight_decay` | `float` | `0` | The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights. |
| `adam_beta1` | `float` | `0.9` | The beta1 hyperparameter for the Adam optimizer or its variants. Set via `beta1` in `set_optimizer()`. |
| `adam_beta2` | `float` | `0.999` | The beta2 hyperparameter for the Adam optimizer or its variants. Set via `beta2` in `set_optimizer()`. |
| `adam_epsilon` | `float` | `1e-8` | The epsilon hyperparameter for the Adam optimizer or its variants. Set via `epsilon` in `set_optimizer()`. |
| `optim_args` | `str` | `None` | Optional arguments that are supplied to AnyPrecisionAdamW (only useful when `optim="adamw_anyprecision"`). Set via `args` in `set_optimizer()`. |

---

## 4. Learning Rate Scheduler

These parameters are grouped under `set_lr_scheduler()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_scheduler_type` | `str` or `SchedulerType` | `"linear"` | The scheduler type to use. See the documentation of `SchedulerType` for all possible values (e.g., `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, `"constant"`, `"constant_with_warmup"`, `"inverse_sqrt"`, `"reduce_on_plateau"`). Set via `name` in `set_lr_scheduler()`. |
| `num_train_epochs` | `float` | `3.0` | Total number of training epochs to perform. (Also listed under Training Basics.) Set via `num_epochs` in `set_lr_scheduler()`. |
| `max_steps` | `int` | `-1` | If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`. (Also listed under Training Basics.) |
| `warmup_steps` | `float` | `0` | Number of steps used for a linear warmup from 0 to `learning_rate`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps used for warmup. |

---

## 5. Logging

These parameters are grouped under `set_logging()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logging_strategy` | `str` or `IntervalStrategy` | `"steps"` | The logging strategy to adopt during training. Possible values: `"no"` (no logging during training), `"epoch"` (logging at the end of each epoch), `"steps"` (logging every `logging_steps`). Set via `strategy` in `set_logging()`. |
| `logging_steps` | `int` | `500` | Number of update steps between two logs if `logging_strategy="steps"`. Set via `steps` in `set_logging()`. |
| `log_level` | `str` | `"passive"` | Logger log level to use on the main process. Possible choices: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`, plus `"passive"` which doesn't set anything and lets the application set the level. Set via `level` in `set_logging()`. |
| `report_to` | `str` or `list[str]` | `"none"` | The list of integrations to report results and logs to. Supported platforms: `"azure_ml"`, `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"swanlab"`, `"tensorboard"`, `"trackio"`, `"wandb"`. Use `"all"` to report to all installed integrations, `"none"` for none. |
| `logging_first_step` | `bool` | `False` | Whether to log and evaluate the first `global_step` or not. Set via `first_step` in `set_logging()`. |
| `logging_nan_inf_filter` | `bool` | `True` | Whether to filter `nan` and `inf` losses for logging. If set to `True`, the loss of every step that is `nan` or `inf` is filtered and the average loss of the current logging window is taken instead. This only influences the logging of loss values, it does not change the behavior of how the gradient is computed or applied to the model. Set via `nan_inf_filter` in `set_logging()`. |
| `log_on_each_node` | `bool` | `True` | In multinode distributed training, whether to log using `log_level` once per node, or only on the main node. Set via `on_each_node` in `set_logging()`. |
| `log_level_replica` | `str` | `"passive"` | Logger log level to use on replicas. Same choices as `log_level`. Set via `replica_level` in `set_logging()`. |

---

## 6. Evaluation

These parameters are grouped under `set_evaluate()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_eval` | `bool` | `False` | Whether to run evaluation. Automatically set to `True` when `set_evaluate()` is called with a strategy different from `"no"`. |
| `eval_strategy` | `str` or `IntervalStrategy` | `"no"` | The evaluation strategy to adopt during training. Possible values: `"no"` (no evaluation during training), `"steps"` (evaluation done and logged every `eval_steps`), `"epoch"` (evaluation done at the end of each epoch). Set via `strategy` in `set_evaluate()`. |
| `eval_steps` | `int` | `500` | Number of update steps between two evaluations if `eval_strategy="steps"`. Set via `steps` in `set_evaluate()`. |
| `per_device_eval_batch_size` | `int` | `8` | The batch size per device (GPU/TPU core/CPU) used for evaluation. Set via `batch_size` in `set_evaluate()`. |
| `eval_accumulation_steps` | `int` | `None` | Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but requires more memory). Set via `accumulation_steps` in `set_evaluate()`. |
| `eval_delay` | `float` | `None` | Number of epochs or steps to wait for before the first evaluation can be performed, depending on the `eval_strategy`. Set via `delay` in `set_evaluate()`. |
| `prediction_loss_only` | `bool` | `False` | Ignores all outputs except the loss during evaluation. Set via `loss_only` in `set_evaluate()`. |

---

## 7. Checkpointing / Saving

These parameters are grouped under `set_save()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_strategy` | `str` or `IntervalStrategy` | `"steps"` | The checkpoint save strategy to adopt during training. Possible values: `"no"` (no save during training), `"epoch"` (save at the end of each epoch), `"steps"` (save every `save_steps`). Set via `strategy` in `set_save()`. |
| `save_steps` | `int` | `500` | Number of update steps before two checkpoint saves if `save_strategy="steps"`. Set via `steps` in `set_save()`. |
| `save_total_limit` | `int` | `None` | If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`. Set via `total_limit` in `set_save()`. |
| `save_on_each_node` | `bool` | `False` | When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one. This should not be activated when the different nodes use the same storage as the files will be saved with the same names for each node. Set via `on_each_node` in `set_save()`. |

---

## 8. DataLoader

These parameters are grouped under `set_dataloader()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataloader_drop_last` | `bool` | `False` | Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size) or not. Set via `drop_last` in `set_dataloader()`. |
| `dataloader_num_workers` | `int` | `0` | Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process. Set via `num_workers` in `set_dataloader()`. |
| `dataloader_pin_memory` | `bool` | `True` | Whether to pin memory in data loaders or not. Set via `pin_memory` in `set_dataloader()`. |
| `dataloader_persistent_workers` | `bool` | `False` | If `True`, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows maintaining the workers' Dataset instances alive. Can potentially speed up training, but will increase RAM usage. Set via `persistent_workers` in `set_dataloader()`. |
| `dataloader_prefetch_factor` | `int` | `None` | Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches prefetched across all workers. Set via `prefetch_factor` in `set_dataloader()`. |
| `auto_find_batch_size` | `bool` | `False` | Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`). |
| `ignore_data_skip` | `bool` | `False` | When resuming training, whether or not to skip the epochs and batches to get the data loading at the same stage as in the previous training. If set to `True`, the training will begin faster (as that skipping step can take a long time) but will not yield the same results as the interrupted training would have. |
| `data_seed` | `int` | `None` | Random seed to be used with data samplers. If not set, random generators for data sampling will use the same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model seed. Set via `sampler_seed` in `set_dataloader()`. |

---

## 9. Push to Hub

These parameters are grouped under `set_push_to_hub()`. Calling `set_push_to_hub()` automatically sets `push_to_hub` to `True`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `push_to_hub` | `bool` | `False` | Whether to push the model to the Hub every time the model is saved. If activated, `output_dir` will begin a git directory synced with the repo (determined by `hub_model_id`) and the content will be pushed each time a save is triggered. |
| `hub_model_id` | `str` | `None` | The name of the repository to keep in sync with the local `output_dir`. Can be a simple model ID (pushed to your namespace) or a full repository name like `"user_name/model"` or `"organization_name/model"`. Set via `model_id` in `set_push_to_hub()`. |
| `hub_strategy` | `str` or `HubStrategy` | `"every_save"` | Defines the scope of what is pushed to the Hub and when. Possible values: `"end"` (push model, config, tokenizer, and model card draft when `save_model()` is called), `"every_save"` (push on every model save, asynchronously), `"checkpoint"` (like `"every_save"` but also pushes latest checkpoint in a `last-checkpoint` subfolder), `"all_checkpoints"` (like `"checkpoint"` but pushes all checkpoints). Set via `strategy` in `set_push_to_hub()`. |
| `hub_token` | `str` | `None` | The token to use to push the model to the Hub. Defaults to the token in the cache folder obtained with `hf auth login`. Set via `token` in `set_push_to_hub()`. |
| `hub_private_repo` | `bool` | `False` | Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists. Set via `private_repo` in `set_push_to_hub()`. |
| `hub_always_push` | `bool` | `False` | Unless this is `True`, the Trainer will skip pushing a checkpoint when the previous push is not finished. Set via `always_push` in `set_push_to_hub()`. |
| `push_to_hub_revision` | `str` | `None` | The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash. Set via `revision` in `set_push_to_hub()`. |

---

## 10. Testing / Prediction

These parameters are grouped under `set_testing()`. Calling `set_testing()` automatically sets `do_predict` to `True`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_predict` | `bool` | `False` | Whether to run predictions on the test set. Automatically set to `True` when `set_testing()` is called. |
| `per_device_eval_batch_size` | `int` | `8` | The batch size per device (GPU/TPU core/CPU) used for testing. (Shared with evaluation.) Set via `batch_size` in `set_testing()`. |
| `prediction_loss_only` | `bool` | `False` | Ignores all outputs except the loss. (Shared with evaluation.) Set via `loss_only` in `set_testing()`. |

---

## Additional Notes

### Helper Methods on TrainingArguments

The `TrainingArguments` class provides convenience setter methods that group related parameters:

| Method | Description |
|--------|-------------|
| `set_training(...)` | Set training-related args (learning rate, batch size, epochs, etc.) |
| `set_evaluate(...)` | Set evaluation-related args (strategy, steps, batch size, etc.) |
| `set_testing(...)` | Set testing/prediction args (batch size, loss_only) |
| `set_save(...)` | Set checkpoint saving args (strategy, steps, total limit, etc.) |
| `set_logging(...)` | Set logging args (strategy, steps, log level, report destinations, etc.) |
| `set_optimizer(...)` | Set optimizer args (optimizer name, learning rate, weight decay, Adam params, etc.) |
| `set_lr_scheduler(...)` | Set learning rate scheduler args (scheduler type, warmup steps, epochs, etc.) |
| `set_push_to_hub(...)` | Set Hub synchronization args (model ID, strategy, token, etc.) |
| `set_dataloader(...)` | Set data loading args (batch size, num workers, pin memory, etc.) |

### Serialization Methods

| Method | Description |
|--------|-------------|
| `to_dict()` | Serializes the instance, replacing `Enum` values by their string values. Obfuscates token values. |
| `to_json_string()` | Serializes the instance to a JSON string. |
| `to_sanitized_dict()` | Sanitized serialization for use with TensorBoard's hparams. |

### Utility Methods

| Method | Description |
|--------|-------------|
| `get_process_log_level()` | Returns the log level depending on whether the current process is the main process of node 0, main process of another node, or a non-main process. For the main process, defaults to the logging level set (WARNING if unchanged) unless overridden by `log_level`. For replicas, defaults to WARNING unless overridden by `log_level_replica`. |
| `get_warmup_steps(num_training_steps)` | Get the number of steps used for a linear warmup. |
| `main_process_first(local=True, desc="work")` | Context manager for distributed environments where the main process performs an operation while blocking replicas. `local=True` means process of rank 0 on each node; `local=False` means only rank 0 of node 0 (useful with shared filesystems). |

### Seq2SeqTrainingArguments

`Seq2SeqTrainingArguments` inherits from `TrainingArguments` and adds parameters specific to sequence-to-sequence tasks (e.g., summarization, translation). It shares the same `output_dir` default of `"trainer_output"` and includes an overridden `to_dict()` method that also serializes `GenerationConfig` as dictionaries.

---

*This reference was extracted from the official HuggingFace Transformers Trainer documentation page.*
