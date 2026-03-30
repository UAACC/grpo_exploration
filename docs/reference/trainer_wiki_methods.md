# HuggingFace Transformers Trainer -- Complete Method Reference

Extracted from the official HuggingFace Transformers documentation.

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Trainer.__init__ Parameters](#2-trainer__init__-parameters)
- [3. Trainer Class Methods](#3-trainer-class-methods)
  - [add_callback](#add_callback)
  - [autocast_smart_context_manager](#autocast_smart_context_manager)
  - [call_model_init](#call_model_init)
  - [compute_loss](#compute_loss)
  - [compute_loss_context_manager](#compute_loss_context_manager)
  - [create_accelerator_and_postprocess](#create_accelerator_and_postprocess)
  - [create_model_card](#create_model_card)
  - [create_optimizer](#create_optimizer)
  - [create_optimizer_and_scheduler](#create_optimizer_and_scheduler)
  - [create_scheduler](#create_scheduler)
  - [evaluate](#evaluate)
  - [evaluation_loop](#evaluation_loop)
  - [floating_point_ops](#floating_point_ops)
  - [get_batch_samples](#get_batch_samples)
  - [get_cp_size](#get_cp_size)
  - [get_decay_parameter_names](#get_decay_parameter_names)
  - [get_eval_dataloader](#get_eval_dataloader)
  - [get_learning_rates](#get_learning_rates)
  - [get_num_trainable_parameters](#get_num_trainable_parameters)
  - [get_optimizer_cls_and_kwargs](#get_optimizer_cls_and_kwargs)
  - [get_optimizer_group](#get_optimizer_group)
  - [get_sp_size](#get_sp_size)
  - [get_test_dataloader](#get_test_dataloader)
  - [get_total_train_batch_size](#get_total_train_batch_size)
  - [get_tp_size](#get_tp_size)
  - [get_train_dataloader](#get_train_dataloader)
  - [hyperparameter_search](#hyperparameter_search)
  - [init_hf_repo](#init_hf_repo)
  - [is_local_process_zero](#is_local_process_zero)
  - [is_world_process_zero](#is_world_process_zero)
  - [log](#log)
  - [log_metrics](#log_metrics)
  - [metrics_format](#metrics_format)
  - [num_examples](#num_examples)
  - [pop_callback](#pop_callback)
  - [predict](#predict)
  - [prediction_step](#prediction_step)
  - [push_to_hub](#push_to_hub)
  - [remove_callback](#remove_callback)
  - [save_metrics](#save_metrics)
  - [save_model](#save_model)
  - [save_state](#save_state)
  - [set_initial_training_values](#set_initial_training_values)
  - [store_flos](#store_flos)
  - [train](#train)
  - [training_step](#training_step)
- [4. Seq2SeqTrainer Differences](#4-seq2seqtrainer-differences)
  - [Seq2SeqTrainer.evaluate](#seq2seqtrainerevaluate)
  - [Seq2SeqTrainer.predict](#seq2seqtrainerpredict)
- [5. TrainingArguments Methods](#5-trainingarguments-methods)
  - [get_process_log_level](#get_process_log_level)
  - [get_warmup_steps](#get_warmup_steps)
  - [main_process_first](#main_process_first)
  - [set_dataloader](#set_dataloader)
  - [set_evaluate](#set_evaluate)
  - [set_logging](#set_logging)
  - [set_lr_scheduler](#set_lr_scheduler)
  - [set_optimizer](#set_optimizer)
  - [set_push_to_hub](#set_push_to_hub)
  - [set_save](#set_save)
  - [set_testing](#set_testing)
  - [set_training](#set_training)
  - [to_dict](#to_dict)
  - [to_json_string](#to_json_string)
  - [to_sanitized_dict](#to_sanitized_dict)
- [6. Seq2SeqTrainingArguments Methods](#6-seq2seqtrainingarguments-methods)
  - [Seq2SeqTrainingArguments.to_dict](#seq2seqtrainingargumentsto_dict)
- [7. Callbacks and Hooks](#7-callbacks-and-hooks)
- [8. Important Attributes](#8-important-attributes)

---

## 1. Overview

The `Trainer` class provides a feature-complete training and evaluation loop for PyTorch, optimized for HuggingFace Transformers. It supports:

- Distributed training on multiple GPUs/TPUs
- Mixed precision (NVIDIA GPUs via apex, AMD GPUs via ROCm, and `torch.amp`)
- Works with `TrainingArguments` for full configuration

Requirements for custom models:
- Must always return tuples or subclasses of `ModelOutput`
- Must compute loss if a `labels` argument is provided (returned as first element of tuple)
- Can accept multiple label arguments via `label_names` in `TrainingArguments`, but none should be named `"label"`

---

## 2. Trainer.__init__ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `PreTrainedModel` or `torch.nn.Module` | *optional* | The model to train/evaluate/predict. If not provided, `model_init` must be passed. Optimized for `PreTrainedModel` but works with any `torch.nn.Module` that follows the same conventions. |
| `args` | `TrainingArguments` | *optional* | Training arguments. Defaults to a basic `TrainingArguments` with `output_dir` set to `tmp_trainer`. |
| `data_collator` | `DataCollator` | *optional* | Function to form a batch from list of elements. Defaults to `default_data_collator` if no `processing_class`, otherwise `DataCollatorWithPadding`. |
| `train_dataset` | `torch.utils.data.Dataset` / `IterableDataset` / `datasets.Dataset` | *optional* | Training dataset. Columns not accepted by `model.forward()` are auto-removed. For `IterableDataset` with randomization in distributed training, must use internal `generator` attribute or `set_epoch()` method. |
| `eval_dataset` | `torch.utils.data.Dataset` / `dict[str, Dataset]` / `datasets.Dataset` | *optional* | Evaluation dataset. If a dictionary, evaluates on each dataset prepending the key to metric names. |
| `processing_class` | `PreTrainedTokenizerBase` / `BaseImageProcessor` / `FeatureExtractionMixin` / `ProcessorMixin` | *optional* | Processing class for automatic input processing. Saved alongside the model. |
| `model_init` | `Callable[[], PreTrainedModel]` | *optional* | Function that instantiates a fresh model for each `train()` call. Can accept zero args or one arg (optuna/Ray Tune trial object). |
| `compute_loss_func` | `Callable` | *optional* | Custom loss function accepting raw model outputs, labels, and number of items in accumulated batch (`batch_size * gradient_accumulation_steps`). |
| `compute_metrics` | `Callable[[EvalPrediction], dict]` | *optional* | Function to compute metrics at evaluation. When `batch_eval_metrics=True` in TrainingArgs, must accept a `compute_result` boolean argument. |
| `callbacks` | `list[TrainerCallback]` | *optional* | List of callbacks to customize the training loop. Added to default callbacks. Use `remove_callback()` to remove defaults. |
| `optimizers` | `tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]` | `(None, None)` | Optimizer and scheduler tuple. Defaults to `AdamW` + `get_linear_schedule_with_warmup()`. |
| `optimizer_cls_and_kwargs` | `tuple[Type[torch.optim.Optimizer], dict[str, Any]]` | *optional* | Optimizer class and kwargs. Overrides `optim`/`optim_args` in args. Incompatible with `optimizers`. Avoids needing to place model params on correct devices before init. |
| `preprocess_logits_for_metrics` | `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]` | *optional* | Preprocesses logits before caching at each eval step. Takes (logits, labels) and returns processed logits. Labels may be `None` if dataset lacks them. |

---

## 3. Trainer Class Methods

---

### add_callback

**Source:** `trainer.py#L4337`

```python
add_callback(callback: type[TrainerCallback] | TrainerCallback)
```

**Description:** Add a callback to the current list of `TrainerCallback`. If a class is passed (not an instance), it will be instantiated.

**Parameters:**
- `callback` (`type` or `TrainerCallback`): A `TrainerCallback` class or instance.

**Return type:** None

---

### autocast_smart_context_manager

**Source:** `trainer.py#L2034`

```python
autocast_smart_context_manager()
```

**Description:** A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired arguments. Relies on Accelerate for autocast, so it effectively does nothing itself.

**Return type:** Context manager

---

### call_model_init

**Source:** `trainer.py#L4228`

```python
call_model_init(trial=None)
```

**Description:** Invoke `model_init` to get a fresh model instance, optionally conditioned on a hyperparameter trial.

**Parameters:**
- `trial`: Optional hyperparameter trial object.

**Return type:** `PreTrainedModel`

---

### compute_loss

**Source:** `trainer.py#L1938`

```python
compute_loss(
    model: nn.Module,
    inputs: dict[str, torch.Tensor | Any],
    return_outputs: bool = False,
    num_items_in_batch: Optional[torch.Tensor] = None
)
```

**Description:** Computes the loss for a given model and inputs. By default, all models return the loss in the first element. Subclass and override for custom behavior.

**Parameters:**
- `model` (`nn.Module`): The model to compute loss for.
- `inputs` (`dict[str, torch.Tensor | Any]`): Input data for the model.
- `return_outputs` (`bool`, default `False`): Whether to return model outputs along with the loss.
- `num_items_in_batch` (`Optional[torch.Tensor]`): Number of items in the batch. If not passed, loss uses default batch size reduction.

**Return type:** `torch.Tensor` (loss) or `tuple[torch.Tensor, ModelOutput]` if `return_outputs=True`

**Important notes:**
- If you are not using `num_items_in_batch` in your custom loss, set `self.model_accepts_loss_kwargs` to `False`. Otherwise loss calculation may be slightly inaccurate during gradient accumulation.

---

### compute_loss_context_manager

**Source:** `trainer.py#L2022`

```python
compute_loss_context_manager()
```

**Description:** A helper wrapper to group together context managers used during loss computation.

**Return type:** Context manager

---

### create_accelerator_and_postprocess

**Source:** `trainer.py#L750`

```python
create_accelerator_and_postprocess()
```

**Description:** Create the Accelerate accelerator and perform post-creation setup (FSDP, DeepSpeed, etc.).

**Return type:** None

---

### create_model_card

**Source:** `trainer.py#L3912`

```python
create_model_card(
    language: str = None,
    license: str = None,
    tags: str | list[str] = None,
    model_name: str = None,
    finetuned_from: str = None,
    tasks: str | list[str] = None,
    dataset_tags: str | list[str] = None,
    dataset: str | list[str] = None,
    dataset_args: str | list[str] = None
)
```

**Description:** Creates a draft model card using information available to the Trainer.

**Parameters:**
- `language` (`str`, optional): Language of the model.
- `license` (`str`, optional): License. Defaults to the pretrained model's license if from Hub.
- `tags` (`str` or `list[str]`, optional): Tags for metadata.
- `model_name` (`str`, optional): Name of the model.
- `finetuned_from` (`str`, optional): Name of base model. Defaults to Hub repo name if applicable.
- `tasks` (`str` or `list[str]`, optional): Task identifiers for metadata.
- `dataset_tags` (`str` or `list[str]`, optional): Dataset tags for metadata.
- `dataset` (`str` or `list[str]`, optional): Dataset identifiers for metadata.
- `dataset_args` (`str` or `list[str]`, optional): Dataset arguments for metadata.

**Return type:** None

---

### create_optimizer

**Source:** `trainer.py#L1143`

```python
create_optimizer()
```

**Description:** Sets up the optimizer. Provides a reasonable default. Can be overridden by passing a tuple via `optimizers` in `__init__`, or by subclassing and overriding this method.

**Return type:** `torch.optim.Optimizer`

---

### create_optimizer_and_scheduler

**Source:** `trainer.py#L1132`

```python
create_optimizer_and_scheduler(num_training_steps: int)
```

**Description:** Sets up both the optimizer and learning rate scheduler. Provides a reasonable default. Override by passing `optimizers` tuple, or subclass and override this method (or `create_optimizer`/`create_scheduler` individually).

**Parameters:**
- `num_training_steps` (`int`): The number of training steps.

**Return type:** None

---

### create_scheduler

**Source:** `trainer.py#L1219`

```python
create_scheduler(num_training_steps: int, optimizer: torch.optim.Optimizer = None)
```

**Description:** Sets up the learning rate scheduler. The optimizer must have been set up before this method is called, or passed as an argument.

**Parameters:**
- `num_training_steps` (`int`): The number of training steps to do.

**Return type:** `torch.optim.lr_scheduler.LRScheduler`

---

### evaluate

**Source:** `trainer.py#L2508`

```python
evaluate(
    eval_dataset: Dataset | dict[str, Dataset] = None,
    ignore_keys: list[str] = None,
    metric_key_prefix: str = "eval"
)
```

**Description:** Run evaluation and return metrics. The calling script must provide `compute_metrics` (passed at init). Subclass and override for custom behavior.

**Parameters:**
- `eval_dataset` (`Dataset` or `dict[str, Dataset]`, optional): Override `self.eval_dataset`. If a dict, evaluates on each dataset, prepending the key to metric names. Must implement `__len__`.
- `ignore_keys` (`list[str]`, optional): Keys in model output dict to ignore when gathering predictions.
- `metric_key_prefix` (`str`, default `"eval"`): Prefix for metric keys (e.g., `"eval_bleu"`).

**Return type:** `dict[str, float]` -- Contains evaluation loss and computed metrics plus epoch number from training state.

**Important notes:**
- When using `load_best_model_at_end` with dict datasets, `metric_for_best_model` must reference exactly one dataset (e.g., `"eval_data1_loss"`).

---

### evaluation_loop

**Source:** `trainer.py#L2608`

```python
evaluation_loop(
    dataloader,
    description,
    prediction_loss_only=None,
    ignore_keys=None,
    metric_key_prefix="eval"
)
```

**Description:** Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`. Works both with and without labels.

**Return type:** `EvalLoopOutput` (namedtuple)

---

### floating_point_ops

**Source:** `trainer.py#L3873`

```python
floating_point_ops(inputs: dict[str, torch.Tensor | Any])
```

**Description:** Computes the number of floating point operations for every backward + forward pass. For `PreTrainedModel` subclasses, uses the model's built-in method. For other models, implement the method in the model or subclass and override.

**Parameters:**
- `inputs` (`dict[str, torch.Tensor | Any]`): The inputs and targets of the model.

**Return type:** `int`

---

### get_batch_samples

**Source:** `trainer.py#L2092`

```python
get_batch_samples(epoch_iterator, num_batches)
```

**Description:** Collects a specified number of batches from the epoch iterator and optionally counts the number of items in the batches to properly scale the loss.

**Return type:** `tuple` (batch samples, optional item count)

---

### get_cp_size

**Source:** `trainer.py#L2383`

```python
get_cp_size()
```

**Description:** Get the context parallel size.

**Return type:** `int`

---

### get_decay_parameter_names

**Source:** `trainer.py#L1280`

```python
get_decay_parameter_names(model: nn.Module)
```

**Description:** Get all parameter names that weight decay will be applied to. Filters out parameters in two ways: (1) by layer type (instances of layers in `ALL_LAYERNORM_LAYERS`), and (2) by parameter name patterns (containing `'bias'` or variations of `'norm'`).

**Return type:** `list[str]`

---

### get_eval_dataloader

**Source:** `trainer.py#L882`

```python
get_eval_dataloader(eval_dataset: str | torch.utils.data.Dataset = None)
```

**Description:** Returns the evaluation DataLoader. Subclass and override for custom behavior.

**Parameters:**
- `eval_dataset` (`str` or `Dataset`, optional): If `str`, uses `self.eval_dataset[eval_dataset]`. If `Dataset`, overrides `self.eval_dataset` and must implement `__len__`. Columns not accepted by `model.forward()` are auto-removed.

**Return type:** `torch.utils.data.DataLoader`

---

### get_learning_rates

**Source:** `trainer_pt_utils.py#L982`

```python
get_learning_rates()
```

**Description:** Returns the learning rate of each parameter from `self.optimizer`.

**Return type:** `list[float]`

---

### get_num_trainable_parameters

**Source:** `trainer_pt_utils.py#L974`

```python
get_num_trainable_parameters()
```

**Description:** Get the number of trainable parameters.

**Return type:** `int`

---

### get_optimizer_cls_and_kwargs

**Source:** `trainer.py#L1249`

```python
get_optimizer_cls_and_kwargs(
    args: TrainingArguments,
    model: PreTrainedModel = None
) -> tuple[type[torch.optim.Optimizer], dict[str, Any]]
```

**Description:** Returns the optimizer class and optimizer parameters based on training arguments. This is a static/class method.

**Parameters:**
- `args` (`TrainingArguments`): The training arguments.
- `model` (`PreTrainedModel`, optional): The model being trained. Required for some optimizers (GaLore, Apollo, LOMO).

**Return type:** `tuple[type[Optimizer], dict[str, Any]]`

---

### get_optimizer_group

**Source:** `trainer_pt_utils.py#L992`

```python
get_optimizer_group(param: str | torch.nn.parameter.Parameter = None)
```

**Description:** Returns optimizer group for a specific parameter if given, otherwise returns all optimizer groups for params.

**Parameters:**
- `param` (`str` or `torch.nn.parameter.Parameter`, optional): The parameter for which the optimizer group is needed.

**Return type:** `dict` or `list[dict]`

---

### get_sp_size

**Source:** `trainer.py#L2375`

```python
get_sp_size()
```

**Description:** Get the sequence parallel size.

**Return type:** `int`

---

### get_test_dataloader

**Source:** `trainer.py#L921`

```python
get_test_dataloader(test_dataset: torch.utils.data.Dataset)
```

**Description:** Returns the test DataLoader. Subclass and override for custom behavior.

**Parameters:**
- `test_dataset` (`Dataset`, optional): Must implement `__len__`. Columns not accepted by `model.forward()` are auto-removed.

**Return type:** `torch.utils.data.DataLoader`

---

### get_total_train_batch_size

**Source:** `trainer.py#L2357`

```python
get_total_train_batch_size()
```

**Description:** Calculates total batch size (`micro_batch * grad_accum * dp_world_size`). Accounts for all parallelism dimensions: TP (Tensor Parallelism), CP (Context Parallelism via Ring Attention/FSDP2), and SP (Sequence Parallelism via ALST/Ulysses/DeepSpeed). Formula: `dp_world_size = world_size // (tp_size * cp_size * sp_size)`.

**Return type:** `int`

---

### get_tp_size

**Source:** `trainer.py#L2391`

```python
get_tp_size()
```

**Description:** Get the tensor parallel size from either the model or DeepSpeed config.

**Return type:** `int`

---

### get_train_dataloader

**Source:** `trainer.py#L862`

```python
get_train_dataloader()
```

**Description:** Returns the training DataLoader. Uses no sampler if `train_dataset` does not implement `__len__`, otherwise uses a random sampler (adapted for distributed training if necessary). Subclass and override for custom behavior.

**Return type:** `torch.utils.data.DataLoader`

---

### hyperparameter_search

**Source:** `trainer.py#L4147`

```python
hyperparameter_search(
    hp_space: Callable[["optuna.Trial"], dict[str, float]] = None,
    compute_objective: Callable[[dict[str, float]], float] = None,
    n_trials: int = 100,
    direction: str | list[str] = "minimize",
    backend: str | HPSearchBackend = None,
    hp_name: Callable[["optuna.Trial"], str] = None,
    **kwargs
)
```

**Description:** Launch hyperparameter search using `optuna` or `Ray Tune`. The optimized quantity is determined by `compute_objective` (defaults to eval loss if no metric, sum of all metrics otherwise).

**Parameters:**
- `hp_space` (callable, optional): Defines HP search space. Defaults to `default_hp_space_optuna()` or `default_hp_space_ray()`.
- `compute_objective` (callable, optional): Computes objective from metrics dict. Defaults to `default_compute_objective()`.
- `n_trials` (`int`, default `100`): Number of trial runs.
- `direction` (`str` or `list[str]`, default `"minimize"`): `"minimize"` or `"maximize"`. For multi-objective, pass a list.
- `backend` (`str` or `HPSearchBackend`, optional): Defaults to optuna or Ray Tune based on installation. If both installed, defaults to optuna.
- `hp_name` (callable, optional): Defines trial/run name.
- `**kwargs`: Backend-specific kwargs (optuna: `create_study` params + `timeout`, `n_jobs`, `gc_after_trial`; ray: `tune.run` params).

**Return type:** `BestRun` or `list[BestRun]` (for multi-objective)

**Important notes:**
- Requires `model_init` to be provided at Trainer initialization.
- Incompatible with `optimizers` argument. Subclass and override `create_optimizer_and_scheduler` for custom optimizer/scheduler.

---

### init_hf_repo

**Source:** `trainer.py#L3894`

```python
init_hf_repo()
```

**Description:** Initializes a git repo in `self.args.hub_model_id`.

**Return type:** None

---

### is_local_process_zero

**Source:** `trainer.py#L4377`

```python
is_local_process_zero()
```

**Description:** Whether or not this process is the local (e.g., on one machine) main process when training in a distributed fashion on several machines.

**Return type:** `bool`

---

### is_world_process_zero

**Source:** `trainer.py#L4384`

```python
is_world_process_zero()
```

**Description:** Whether or not this process is the global main process (only `True` for one process across all machines in distributed training).

**Return type:** `bool`

---

### log

**Source:** `trainer.py#L3838`

```python
log(logs: dict[str, float], start_time: Optional[float] = None)
```

**Description:** Log `logs` on the various objects watching training. Subclass and override for custom behavior.

**Parameters:**
- `logs` (`dict[str, float]`): The values to log.
- `start_time` (`Optional[float]`): The start of training.

**Return type:** None

---

### log_metrics

**Source:** `trainer_pt_utils.py#L830`

```python
log_metrics(split: str, metrics: dict[str, float])
```

**Description:** Log metrics in a specially formatted way. Only runs on rank 0 in distributed environments. Includes detailed memory usage reports (CPU RSS and GPU allocated/peak) if `psutil` is installed.

**Parameters:**
- `split` (`str`): Mode/split name: one of `train`, `eval`, `test`.
- `metrics` (`dict[str, float]`): The metrics dict.

**Return type:** None

**Important notes:**
- Memory reports include `*_alloc_delta` (net memory change) and `*_peaked_delta` (extra memory consumed then freed).
- GPU memory tracking uses `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` -- only tracks pytorch-specific allocations.
- CPU peak memory uses a sampling thread (may miss peaks due to GIL).
- Nested evaluation calls during training may cause inaccurate GPU peak memory stats due to `torch.cuda.max_memory_allocated` being a single counter.

---

### metrics_format

**Source:** `trainer_pt_utils.py#L803`

```python
metrics_format(metrics: dict[str, float])
```

**Description:** Reformat Trainer metrics values to a human-readable format.

**Parameters:**
- `metrics` (`dict[str, float]`): The metrics from train/evaluate/predict.

**Return type:** `dict[str, float]`

---

### num_examples

**Source:** `trainer.py#L939`

```python
num_examples(dataloader: torch.utils.data.DataLoader)
```

**Description:** Helper to get number of samples in a DataLoader by accessing its dataset. When `dataloader.dataset` does not exist or has no length, estimates as best it can.

**Return type:** `int`

---

### pop_callback

**Source:** `trainer.py#L4348`

```python
pop_callback(callback: type[TrainerCallback] | TrainerCallback)
```

**Description:** Remove a callback from the current list of `TrainerCallback` and return it. If not found, returns `None` (no error raised).

**Parameters:**
- `callback` (`type` or `TrainerCallback`): A class or instance. If a class, pops the first member of that class found.

**Return type:** `TrainerCallback` or `None`

---

### predict

**Source:** `trainer.py#L2815`

```python
predict(
    test_dataset: Dataset,
    ignore_keys: list[str] = None,
    metric_key_prefix: str = "test"
)
```

**Description:** Run prediction and return predictions and potential metrics. If the test dataset contains labels, also returns metrics (like `evaluate()`).

**Parameters:**
- `test_dataset` (`Dataset`): Dataset to run predictions on. Columns not accepted by `model.forward()` are auto-removed. Must implement `__len__`.
- `ignore_keys` (`list[str]`, optional): Keys in model output dict to ignore.
- `metric_key_prefix` (`str`, default `"test"`): Prefix for metric keys.

**Return type:** `PredictionOutput` (NamedTuple) with keys:
- `predictions` (`np.ndarray`): Predictions on test dataset.
- `label_ids` (`np.ndarray`, optional): Labels if dataset contained them.
- `metrics` (`dict[str, float]`, optional): Metrics if labels were present.

**Important notes:**
- If predictions or labels have different sequence lengths (e.g., dynamic padding), predictions are padded on the right with index -100.

---

### prediction_step

**Source:** `trainer.py#L2876`

```python
prediction_step(
    model: nn.Module,
    inputs: dict[str, torch.Tensor | Any],
    prediction_loss_only: bool,
    ignore_keys: list[str] = None
)
```

**Description:** Perform an evaluation step on `model` using `inputs`. Subclass and override for custom behavior.

**Parameters:**
- `model` (`nn.Module`): The model to evaluate.
- `inputs` (`dict[str, torch.Tensor | Any]`): Inputs and targets. Dictionary is unpacked before feeding to model. Most models expect targets under `labels`.
- `prediction_loss_only` (`bool`): Whether to return loss only.
- `ignore_keys` (`list[str]`, optional): Keys to ignore in model output dict.

**Return type:** `tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]` -- (loss, logits, labels), each optional.

---

### push_to_hub

**Source:** `trainer.py#L3986`

```python
push_to_hub(
    commit_message: str = "End of training",
    blocking: bool = True,
    token: str = None,
    revision: str = None,
    **kwargs
)
```

**Description:** Upload `self.model` and `self.processing_class` to the HuggingFace model hub on repo `self.args.hub_model_id`.

**Parameters:**
- `commit_message` (`str`, default `"End of training"`): Commit message.
- `blocking` (`bool`, default `True`): Whether to wait for `git push` to finish.
- `token` (`str`, optional): Write-permission token. Overrides Trainer's original args.
- `revision` (`str`, optional): Git revision to commit from. Defaults to head of `"main"`.
- `**kwargs`: Additional kwargs passed to `create_model_card()`.

**Return type:** URL of the repository (`str`) if `blocking=True`, or a `Future` object if `blocking=False`.

---

### remove_callback

**Source:** `trainer.py#L4364`

```python
remove_callback(callback: type[TrainerCallback] | TrainerCallback)
```

**Description:** Remove a callback from the current list of `TrainerCallback`.

**Parameters:**
- `callback` (`type` or `TrainerCallback`): A class or instance. If a class, removes the first member of that class found.

**Return type:** None

---

### save_metrics

**Source:** `trainer_pt_utils.py#L921`

```python
save_metrics(
    split: str,
    metrics: dict[str, float],
    combined: bool = True
)
```

**Description:** Save metrics into a JSON file for the given split (e.g., `train_results.json`). Only runs on rank 0 in distributed environments. Saves raw unformatted numbers (unlike `log_metrics` which formats them).

**Parameters:**
- `split` (`str`): Mode/split name: one of `train`, `eval`, `test`, `all`.
- `metrics` (`dict[str, float]`): The metrics dict.
- `combined` (`bool`, default `True`): If `True`, also updates `all_results.json` with these metrics.

**Return type:** None

---

### save_model

**Source:** `trainer.py#L3739`

```python
save_model(output_dir: str = None, _internal_call: bool = False)
```

**Description:** Save the model so it can be reloaded with `from_pretrained()`. Only saves from the main process.

**Return type:** None

---

### save_state

**Source:** `trainer_pt_utils.py#L960`

```python
save_state()
```

**Description:** Saves the Trainer state (since `save_model` only saves the tokenizer with the model). Only runs on rank 0 in distributed environments.

**Return type:** None

---

### set_initial_training_values

**Source:** `trainer.py#L2287`

```python
set_initial_training_values(args, train_dataset, data_collator, train_dataloader)
```

**Description:** Calculates and returns training setup values:
- `num_train_epochs`
- `num_update_steps_per_epoch`
- `num_examples`
- `num_train_samples`
- `total_train_batch_size`
- `steps_in_epoch` (total batches per epoch)
- `max_steps`

**Return type:** Tuple of computed values

---

### store_flos

**Source:** `trainer.py#L3862`

```python
store_flos()
```

**Description:** Store the number of floating-point operations that went into the model.

**Return type:** None

---

### train

**Source:** `trainer.py#L1322`

```python
train(
    resume_from_checkpoint: str | bool = None,
    trial: "optuna.Trial" | dict[str, Any] = None,
    ignore_keys_for_eval: list[str] = None,
    **kwargs
)
```

**Description:** Main training entry point.

**Parameters:**
- `resume_from_checkpoint` (`str` or `bool`, optional): If `str`, local path to a saved checkpoint. If `True`, loads last checkpoint in `args.output_dir`. Training resumes from loaded model/optimizer/scheduler states.
- `trial` (`optuna.Trial` or `dict[str, Any]`, optional): Trial run or HP dictionary for HP search.
- `ignore_keys_for_eval` (`list[str]`, optional): Keys in model output to ignore when gathering predictions for evaluation during training.

**Return type:** `TrainOutput` -- Object containing global step count, training loss, and metrics.

---

### training_step

**Source:** `trainer.py#L1867`

```python
training_step(
    model: nn.Module,
    inputs: dict[str, torch.Tensor | Any]
)
```

**Description:** Perform a training step on a batch of inputs. Subclass and override for custom behavior.

**Parameters:**
- `model` (`nn.Module`): The model to train.
- `inputs` (`dict[str, torch.Tensor | Any]`): Inputs and targets. Dictionary is unpacked before feeding to model. Most models expect targets under `labels`.

**Return type:** `torch.Tensor` -- The tensor with training loss on this batch.

---

## 4. Seq2SeqTrainer Differences

`Seq2SeqTrainer` inherits from `Trainer` and is adapted for sequence-to-sequence tasks (summarization, translation). It overrides `evaluate` and `predict` to add generation-specific parameters.

**Source:** `trainer_seq2seq.py#L53`

---

### Seq2SeqTrainer.evaluate

**Source:** `trainer_seq2seq.py#L137`

```python
evaluate(
    eval_dataset: Dataset = None,
    ignore_keys: list[str] = None,
    metric_key_prefix: str = "eval",
    **gen_kwargs
)
```

**Description:** Same as `Trainer.evaluate()` but with additional generation-specific parameters.

**Additional parameters vs. Trainer.evaluate:**
- `max_length` (`int`, optional): Maximum target length for `generate` method.
- `num_beams` (`int`, optional): Number of beams for beam search in `generate`. 1 = no beam search.
- `**gen_kwargs`: Additional `generate`-specific keyword arguments.

**Return type:** `dict[str, float]` -- Evaluation loss and computed metrics plus epoch number.

---

### Seq2SeqTrainer.predict

**Source:** `trainer_seq2seq.py#L193`

```python
predict(
    test_dataset: Dataset,
    ignore_keys: list[str] = None,
    metric_key_prefix: str = "eval",
    **gen_kwargs
)
```

**Description:** Same as `Trainer.predict()` but with additional generation-specific parameters.

**Additional parameters vs. Trainer.predict:**
- `max_length` (`int`, optional): Maximum target length for `generate` method.
- `num_beams` (`int`, optional): Number of beams for beam search in `generate`. 1 = no beam search.
- `**gen_kwargs`: Additional `generate`-specific keyword arguments.

**Return type:** `PredictionOutput` (NamedTuple) with keys:
- `predictions` (`np.ndarray`): Predictions on test dataset.
- `label_ids` (`np.ndarray`, optional): Labels if present.
- `metrics` (`dict[str, float]`, optional): Metrics if labels were present.

**Important notes:**
- Predictions padded on the right with -100 if sequence lengths differ.

---

## 5. TrainingArguments Methods

---

### get_process_log_level

**Source:** `training_args.py#L1967`

```python
get_process_log_level()
```

**Description:** Returns the log level depending on whether this process is the main process of node 0, main process of non-0 node, or a non-main process. Main process defaults to the set logging level (or `logging.WARNING`), overridden by `log_level`. Replicas default to `logging.WARNING`, overridden by `log_level_replica`.

**Return type:** `int` (logging level)

---

### get_warmup_steps

**Source:** `training_args.py#L2056`

```python
get_warmup_steps(num_training_steps: int)
```

**Description:** Get number of steps used for a linear warmup.

**Return type:** `int`

---

### main_process_first

**Source:** `training_args.py#L2005`

```python
main_process_first(local: bool = True, desc: str = "work")
```

**Description:** Context manager for torch distributed environment. Runs something on the main process while blocking replicas, then releases replicas on completion. Useful for `datasets`' `map` feature.

**Parameters:**
- `local` (`bool`, default `True`): If `True`, "first" means rank 0 of each node. If `False`, means rank 0 of node rank 0. Use `local=False` for shared filesystems in multi-node setups.
- `desc` (`str`, default `"work"`): Description for debug logs.

**Return type:** Context manager

---

### set_dataloader

**Source:** `training_args.py#L2590`

```python
set_dataloader(
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = None,
    auto_find_batch_size: bool = False,
    ignore_data_skip: bool = False,
    sampler_seed: int = None
)
```

**Description:** Regroups all arguments linked to dataloader creation.

**Parameters:**
- `drop_last` (`bool`, default `False`): Drop last incomplete batch.
- `num_workers` (`int`, default `0`): Subprocesses for data loading. 0 = main process.
- `pin_memory` (`bool`, default `True`): Pin memory in data loaders.
- `persistent_workers` (`bool`, default `False`): Keep worker processes alive between epochs. Faster but uses more RAM.
- `prefetch_factor` (`int`, optional): Batches loaded in advance per worker.
- `auto_find_batch_size` (`bool`, default `False`): Auto-find batch size via exponential decay. Requires `accelerate`.
- `ignore_data_skip` (`bool`, default `False`): Skip epoch/batch fast-forwarding when resuming. Faster start but different results.
- `sampler_seed` (`int`, optional): Seed for data samplers. Defaults to `self.seed`.

**Return type:** `TrainingArguments` (self)

---

### set_evaluate

**Source:** `training_args.py#L2202`

```python
set_evaluate(
    strategy: str | IntervalStrategy = "no",
    steps: int = 500,
    batch_size: int = 8,
    accumulation_steps: int = None,
    delay: float = None,
    loss_only: bool = False
)
```

**Description:** Regroups all arguments linked to evaluation. Setting strategy != `"no"` sets `self.do_eval` to `True`.

**Parameters:**
- `strategy` (`str` or `IntervalStrategy`, default `"no"`): `"no"`, `"steps"`, or `"epoch"`.
- `steps` (`int`, default `500`): Steps between evaluations if `strategy="steps"`.
- `batch_size` (`int`, default `8`): Per-device eval batch size.
- `accumulation_steps` (`int`, optional): Prediction steps to accumulate before moving to CPU.
- `delay` (`float`, optional): Epochs or steps to wait before first evaluation.
- `loss_only` (`bool`, default `False`): Ignore all outputs except loss.

**Return type:** `TrainingArguments` (self)

---

### set_logging

**Source:** `training_args.py#L2344`

```python
set_logging(
    strategy: str | IntervalStrategy = "steps",
    steps: int = 500,
    level: str = "passive",
    report_to: str | list[str] = "none",
    first_step: bool = False,
    nan_inf_filter: bool = True,
    on_each_node: bool = True,
    replica_level: str = "passive"
)
```

**Description:** Regroups all arguments linked to logging.

**Parameters:**
- `strategy` (`str` or `IntervalStrategy`, default `"steps"`): `"no"`, `"epoch"`, or `"steps"`.
- `steps` (`int`, default `500`): Steps between logs if `strategy="steps"`.
- `level` (`str`, default `"passive"`): Log level: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`, `"passive"`.
- `report_to` (`str` or `list[str]`, default `"none"`): Integrations: `"azure_ml"`, `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"swanlab"`, `"tensorboard"`, `"trackio"`, `"wandb"`, `"all"`, `"none"`.
- `first_step` (`bool`, default `False`): Log/evaluate at first `global_step`.
- `nan_inf_filter` (`bool`, default `True`): Filter `nan`/`inf` losses for logging (does not affect gradient computation).
- `on_each_node` (`bool`, default `True`): In multinode training, log on each node vs. only main node.
- `replica_level` (`str`, default `"passive"`): Log level for replica processes.

**Return type:** `TrainingArguments` (self)

---

### set_lr_scheduler

**Source:** `training_args.py#L2544`

```python
set_lr_scheduler(
    name: str | SchedulerType = "linear",
    num_epochs: float = 3.0,
    max_steps: int = -1,
    warmup_steps: float = 0
)
```

**Description:** Regroups all arguments linked to the learning rate scheduler.

**Parameters:**
- `name` (`str` or `SchedulerType`, default `"linear"`): Scheduler type. See `SchedulerType` for all options.
- `num_epochs` (`float`, default `3.0`): Total training epochs. Decimal = partial last epoch.
- `max_steps` (`int`, default `-1`): If positive, overrides `num_train_epochs`.
- `warmup_steps` (`float`, default `0`): Steps for linear warmup. If `< 1`, interpreted as ratio.

**Return type:** `TrainingArguments` (self)

---

### set_optimizer

**Source:** `training_args.py#L2493`

```python
set_optimizer(
    name: str | OptimizerNames = "adamw_torch",
    learning_rate: float = 5e-5,
    weight_decay: float = 0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    args: str = None
)
```

**Description:** Regroups all arguments linked to the optimizer.

**Parameters:**
- `name` (`str` or `OptimizerNames`, default `"adamw_torch"`): Optimizer: `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`, `"adamw_anyprecision"`, `"adafactor"`.
- `learning_rate` (`float`, default `5e-5`): Initial learning rate.
- `weight_decay` (`float`, default `0`): Applied to all layers except bias and LayerNorm weights.
- `beta1` (`float`, default `0.9`): Adam beta1.
- `beta2` (`float`, default `0.999`): Adam beta2.
- `epsilon` (`float`, default `1e-8`): Adam epsilon.
- `args` (`str`, optional): Optional arguments for `AnyPrecisionAdamW`.

**Return type:** `TrainingArguments` (self)

---

### set_push_to_hub

**Source:** `training_args.py#L2419`

```python
set_push_to_hub(
    model_id: str,
    strategy: str | HubStrategy = "every_save",
    token: str = None,
    private_repo: bool = False,
    always_push: bool = False,
    revision: str = None
)
```

**Description:** Regroups all arguments linked to Hub synchronization. Calling sets `self.push_to_hub` to `True`.

**Parameters:**
- `model_id` (`str`): Repository name. Simple ID pushes to your namespace; full name (e.g., `"org/model"`) pushes to org.
- `strategy` (`str` or `HubStrategy`, default `"every_save"`): `"end"`, `"every_save"`, `"checkpoint"`, `"all_checkpoints"`.
- `token` (`str`, optional): Hub token. Defaults to cached token from `hf auth login`.
- `private_repo` (`bool`, default `False`): Make repo private. Ignored if repo exists.
- `always_push` (`bool`, default `False`): If `False`, skips pushing when previous push is unfinished.
- `revision` (`str`, optional): Git revision for pushing.

**Return type:** `TrainingArguments` (self)

---

### set_save

**Source:** `training_args.py#L2295`

```python
set_save(
    strategy: str | IntervalStrategy = "steps",
    steps: int = 500,
    total_limit: int = None,
    on_each_node: bool = False
)
```

**Description:** Regroups all arguments linked to checkpoint saving.

**Parameters:**
- `strategy` (`str` or `IntervalStrategy`, default `"steps"`): `"no"`, `"epoch"`, or `"steps"`.
- `steps` (`int`, default `500`): Steps between saves if `strategy="steps"`.
- `total_limit` (`int`, optional): Max checkpoints to keep. Deletes older ones.
- `on_each_node` (`bool`, default `False`): Save on each node in multi-node training. Do not activate with shared storage.

**Return type:** `TrainingArguments` (self)

---

### set_testing

**Source:** `training_args.py#L2259`

```python
set_testing(
    batch_size: int = 8,
    loss_only: bool = False
)
```

**Description:** Regroups basic arguments linked to testing on a held-out dataset. Calling sets `self.do_predict` to `True`.

**Parameters:**
- `batch_size` (`int`, default `8`): Per-device test batch size.
- `loss_only` (`bool`, default `False`): Ignore all outputs except loss.

**Return type:** `TrainingArguments` (self)

---

### set_training

**Source:** `training_args.py#L2127`

```python
set_training(
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    weight_decay: float = 0,
    num_train_epochs: float = 3.0,
    max_steps: int = -1,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    gradient_checkpointing: bool = False
)
```

**Description:** Regroups basic training arguments. Calling sets `self.do_train` to `True`.

**Parameters:**
- `learning_rate` (`float`, default `5e-5`): Initial learning rate.
- `batch_size` (`int`, default `8`): Per-device train batch size.
- `weight_decay` (`float`, default `0`): Weight decay for non-bias/LayerNorm layers.
- `num_train_epochs` (`float`, default `3.0`): Total epochs. Decimal = partial last epoch.
- `max_steps` (`int`, default `-1`): If positive, overrides `num_train_epochs`.
- `gradient_accumulation_steps` (`int`, default `1`): Steps to accumulate before backward/update. Logging/eval/save happen every `gradient_accumulation_steps * xxx_step` examples.
- `seed` (`int`, default `42`): Random seed. Use `model_init` for reproducibility with random model params.
- `gradient_checkpointing` (`bool`, default `False`): Trade compute for memory.

**Return type:** `TrainingArguments` (self)

---

### to_dict

**Source:** `training_args.py#L2077`

```python
to_dict()
```

**Description:** Serializes this instance, replacing `Enum` values with their string representations (for JSON support). Obfuscates token values by removing them.

**Return type:** `dict`

---

### to_json_string

**Source:** `training_args.py#L2107`

```python
to_json_string()
```

**Description:** Serializes this instance to a JSON string.

**Return type:** `str`

---

### to_sanitized_dict

**Source:** `training_args.py#L2113`

```python
to_sanitized_dict()
```

**Description:** Sanitized serialization for use with TensorBoard's hparams.

**Return type:** `dict`

---

## 6. Seq2SeqTrainingArguments Methods

`Seq2SeqTrainingArguments` inherits from `TrainingArguments` and adds sequence-to-sequence specific configuration.

**Source:** `training_args_seq2seq.py#L29`

---

### Seq2SeqTrainingArguments.to_dict

**Source:** `training_args_seq2seq.py#L84`

```python
to_dict()
```

**Description:** Serializes this instance, replacing `Enum` values and `GenerationConfig` objects with dictionaries (for JSON support). Obfuscates token values.

**Return type:** `dict`

---

## 7. Callbacks and Hooks

The Trainer supports a callback system via `TrainerCallback` for customizing the training loop:

### Callback Management Methods

| Method | Description |
|--------|-------------|
| `add_callback(callback)` | Add a `TrainerCallback` class or instance to the callback list |
| `remove_callback(callback)` | Remove a callback (by class or instance); removes first match |
| `pop_callback(callback)` | Remove and return a callback; returns `None` if not found |

### How Callbacks Work

- Pass callbacks at `__init__` via the `callbacks` parameter
- Callbacks are added to a list of **default callbacks** (see HuggingFace callback documentation)
- To remove a default callback, use `Trainer.remove_callback()`
- Callbacks can be either a `TrainerCallback` class (auto-instantiated) or an instance

### Key Hook Points (via TrainerCallback)

Callbacks can hook into the following events during training:
- Training start/end
- Epoch start/end
- Step start/end
- Evaluation
- Logging
- Saving
- Prediction

The `compute_metrics` function (passed at init) serves as a hook for custom metric computation during evaluation. When `batch_eval_metrics=True` in `TrainingArguments`, the function receives a `compute_result` boolean argument triggered after the last eval batch.

The `preprocess_logits_for_metrics` function (passed at init) is a hook for preprocessing logits before they are cached at each evaluation step.

---

## 8. Important Attributes

| Attribute | Description |
|-----------|-------------|
| `model` | Always points to the core model. For transformers models, a `PreTrainedModel` subclass. |
| `model_wrapped` | Points to the most external model (e.g., after DeepSpeed + DDP wrapping). Use for forward pass. Same as `model` if no wrapping. |
| `is_model_parallel` | Whether the model is in model parallel mode (layers split across GPUs, different from data parallelism). |
| `place_model_on_device` | Whether to auto-place model on device. `True` by default unless model parallel, DeepSpeed, FSDP, full fp16/bf16 eval, or SageMaker MP is active. |
| `is_in_train` | Whether the model is currently running `train` (e.g., when `evaluate` is called during `train`). |
