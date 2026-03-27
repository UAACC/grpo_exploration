import torch
import torch.nn.functional as F
from collections import deque
from typing import Dict, Optional, Any, List, Iterator, Tuple
import copy
import wandb
import numpy as np
import random
from trl import GRPOTrainer
import math
import pdb
from typing import Union
from contextlib import nullcontext
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available
import re
from trl.trainer.grpo_trainer import truncate_with_protected_tokens
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.trainer.utils import (
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from transformers import Trainer
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
from math_verify import parse, verify

import json

def nan_debug(tensor: torch.Tensor, tensor_name: str, rank: int):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"[NaN debug] {tensor_name} has NaN/Inf on rank {rank}")

def extract_boxed_answer(text: str) -> str | None:
    """
    Finds the last \\boxed{...} in the text and extracts its content,
    handling nested braces correctly.
    """
    if "\\boxed{" not in text:
        return None

    # We generally want the *last* boxed answer in the response, 
    # as models often use the end of the chain-of-thought for the final result.
    idx = text.rfind("\\boxed{")
    
    if idx == -1:
        return None

    # Start looking after \boxed{
    i = idx + 7 # len("\\boxed{") is 7
    brace_count = 1
    content_start = i

    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        
        if brace_count == 0:
            return text[content_start:i]
        
        i += 1
            
    return None

def correctness_reward_func_original(completion, answer) -> list[float]:
    extracted_response = extract_boxed_answer(completion)
    return 2.0 if verify(parse(extracted_response),parse(answer)) == True else 0.0
    # Return a larger reward for correctness to make its signal stronger
    # return [2.0 if verify(parse(r),parse(a)) == True else 0.0 for r, a in zip(extracted_responses, answer)]

def load_jsonl(path: str) -> Dict[Tuple[int, int], dict]:
    """
    Build a dict with keys (question_id, run_id) and values containing
    original_problem, system_prompt, and response.
    """
    result: Dict[Tuple[int, int], dict] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            qid = item["question_id"]
            original_problem = item["original_problem"]
            answer = item["ground_truth_answer"]
            system_prompt = item["system_prompt"]
            for run in item["runs"]:
                rid = run["run_id"]
                logprobs = run["logprobs"]
                result[(qid, rid)] = {
                    "original_problem": original_problem,
                    "system_prompt": system_prompt,
                    "response": run["response"],
                    "logprobs": logprobs,
                    "answer": answer,
                    "completion_ids": run["completion_ids"],
                }

    return result

def get_completion_stats(
    question_id: int,
    run_id: int,
    qid_runid_dict: dict,
    tokenizer,
) -> dict:
    """
    Given (question_id, run_id), the dict produced by load_qid_runid_dict,
    and a tokenizer, return a dict in the desired format:
    {
        "question_id": ...,
        "generation_id": ...,
        "old_logp": ...,
        "completion_ids": [...],
        "completion_mask": [...],
    }

    Assumes qid_runid_dict[(question_id, run_id)] contains:
        - "original_problem"
        - "system_prompt"
        - "response"
        - "logprobs"  (list of per-token logprobs)
    """
    if run_id > 3:
        return None
    rec = qid_runid_dict[(question_id, run_id)]

    response = rec["response"]
    logprobs = rec["logprobs"]  # <-- make sure you store this when building qid_runid_dict
    logprobs = logprobs[:-1]  # remove the logprob for the EOS token
    
    # completion_enc = tokenizer(
    # response,
    # return_tensors="pt",
    # add_special_tokens=False,
    #     )
    completion_ids = rec["completion_ids"]
    completion_ids = completion_ids[:-1]
    completion_mask = [1 for i in range(len(completion_ids))]

    # if not len(completion_ids) == len(logprobs):
    #     print(f"Warning: Mismatched sizes for question_id {question_id}, run_id {run_id}: {len(completion_ids)} vs {len(logprobs)}")
    #     return None
    if max(completion_ids) >= len(tokenizer):
        return None
    # 4) Sum old logprobs
    old_per_token_logps = logprobs  # (L_completion,)

    dict_with_right_format = {
        "completions": response,
        "question_id": question_id,
        "generation_id": run_id,
        "old_per_token_logps": old_per_token_logps,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "answer": rec["answer"],
    }
    return dict_with_right_format

def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def pad_sequences(sequences: List[torch.Tensor], 
                 padding_value: int = 0, 
                 padding_side: str = "right") -> torch.Tensor:
    """
    Pad a list of 1D tensors to the same length.
    
    Args:
        sequences: List of 1D tensors
        padding_value: Value to use for padding
        padding_side: "left" or "right"
    
    Returns:
        Padded tensor of shape (len(sequences), max_length)
    """
    if not sequences:
        return torch.empty(0)
    
    max_length = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)
    
    # Create output tensor
    padded = torch.full((batch_size, max_length), padding_value, 
                       dtype=sequences[0].dtype, device=sequences[0].device)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        if padding_side == "right":
            padded[i, :seq_len] = seq
        else:  # left padding
            padded[i, max_length - seq_len:] = seq
    
    return padded


class GRPORPGExternalTrainer(GRPOTrainer):
    """
    GRPOTrainer with FIFO replay buffer and importance-sampling correction.
    
    This trainer maintains a replay buffer and performs importance-sampled
    updates on ALL experiences in the buffer, processing them in mini-batches
    to avoid memory issues.
    
    Compatible with trl==0.19.0.
    """
    
    def __init__(self, 
                 *args,
                 buffer_size: int = 1024,
                 replay_batch_size: int = 6,  # Size of mini-batches for replay
                 replay_frequency: int = 1,   # Replay every N training steps
                 min_buffer_size: int = 4,   # Minimum buffer size before replay starts
                 detach_denominator: bool = False,
                 log_is_metrics: bool = True,
                 cur_model_weight = 0.5,
                 past_model_weight = 0.5,
                 behavior_policy_dir = '/home/shuai14/scratch/rollouts_qwen2.5_math_7b_instruct_temp0.6_3_flatten.jsonl',
                 if_tokenize_external = True,
                 **kwargs):
        """
        Args:
            buffer_size: Maximum size of replay buffer
            is_ratio_lower_bound: Lower bound for importance sampling ratio filtering
            is_ratio_upper_bound: Upper bound for importance sampling ratio filtering
            replay_batch_size: Number of replay samples to process at once
            replay_frequency: Perform replay every N training steps
            min_buffer_size: Minimum samples in buffer before replay starts
            log_is_metrics: Whether to log IS-related metrics to wandb
            filter_replay: Whether to filter samples based on IS ratio bounds
        """
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.replay_batch_size = replay_batch_size
        self.replay_frequency = replay_frequency
        self.min_buffer_size = min_buffer_size
        self.log_is_metrics = log_is_metrics
        self.replay_losses_accumulated = []
        self.cur_model_weight = cur_model_weight
        self.past_model_weight = past_model_weight
        
        # Get padding token ID
        self.replay_pad_token_id = self._get_pad_token_id()
        self.replay_counter = 0

        # NEW: slice pointer that cycles over [0, steps_per_generation-1]
        self._replay_slice_idx = 0
        self._pending_buffer: List[Dict[str, Any]] = []
        print(f'replay_batch_size={self.replay_batch_size}')

        self.detach_denominator = detach_denominator
        if self.detach_denominator:
            print("Detaching denominator in importance sampling weights.")
        print(f'self.importance_sampling_level: {self.importance_sampling_level}')
        print(f'self.curw: {self.cur_model_weight}, self.pastw: {self.past_model_weight}')

        self.offline_data = load_jsonl(behavior_policy_dir)


    @torch.inference_mode()
    def _compute_per_token_logp(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute per-token log probability of completions given prompts.

        Args:
            model: The language model
            batch: Dictionary containing prompt and completion ids/masks
            
        Returns:
            Joint log probabilities of shape (batch_size,)
        """
       # 1) stitch prompt + completion
        ids_p, m_p = batch["prompt_ids"],     batch["prompt_mask"]
        ids_c, m_c = batch["completion_ids"], batch["completion_mask"]
        # import pdb; pdb.set_trace()
        ids_all  = torch.cat((ids_p,  ids_c), 1)
        mask_all = torch.cat([m_p,   m_c], 1)
        # 2) forward
        logits = model(input_ids=ids_all, attention_mask=mask_all).logits
        # 3) shift & log‑softmax
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = ids_all[:, 1:]                          # (B, P+C-1)
        token_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # 4) mask off prompt, keep completion
        # comp_mask = torch.cat([torch.zeros_like(m_p), m_c], 1)[:, 1:]
        # return (token_lp * comp_mask).sum(1)  # joint log‑prob (B,)
        return token_lp  # per-token log‑prob (B, C)
    
    def _get_pad_token_id(self) -> int:
        """Extract padding token ID from various possible sources."""
        if hasattr(self, 'pad_token_id') and self.pad_token_id is not None:
            return self.pad_token_id
        elif hasattr(self.processing_class, 'pad_token_id'):
            return self.processing_class.pad_token_id
        elif hasattr(self.processing_class, 'tokenizer'):
            return self.processing_class.tokenizer.pad_token_id
        else:
            # Fallback - adjust based on your tokenizer
            return 151645

    def _stage_batch(self, inputs: Dict[str, Any], per_token_logps: torch.Tensor):
        """
        Copy per-sample tensors to CPU and hold them until flush.
        
        Args:
            inputs: Dictionary containing batch data
            per_token_logps: Per-token log probabilities from behavior policy
        """
        batch_size = inputs["prompt_ids"].shape[0]
        for i in range(batch_size):
            store = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    store[k] = v[i].detach().cpu()
                else:
                    store[k] = copy.deepcopy(v)
            # Store the per-token log probabilities instead of joint log prob
            store["old_per_token_logps"] = per_token_logps[i].detach().cpu()
            self._pending_buffer.append(store)
            print(f"staged buffer: {len(self._pending_buffer)-1} -> {len(self._pending_buffer)}")

    def _flush_pending_to_buffer(self):
        """Move staged samples into the main deque buffer."""
        if not self._pending_buffer:
            return
        for s in self._pending_buffer:
            self.buffer.append(s)
        flushed = len(self._pending_buffer)
        self._pending_buffer.clear()
        if self.log_is_metrics and wandb.run is not None:
            print(f"Flushed {flushed} samples to buffer of size {len(self.buffer)}")
            wandb.log({"replay/flushed_pending": flushed, "replay/buffer_size": len(self.buffer)})

    def _store_to_buffer(self, inputs: Dict[str, Any], per_token_logps: torch.Tensor):
        """
        Store current batch to replay buffer.
        
        Args:
            inputs: Dictionary containing batch data
            per_token_logps: Per-token log probabilities from behavior policy
        """
        batch_size = inputs["prompt_ids"].shape[0]
        for i in range(batch_size):
            store = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    store[k] = v[i].detach().cpu()
                else:
                    store[k] = copy.deepcopy(v)
            # Store the per-token log probabilities
            store["old_per_token_logps"] = per_token_logps[i].detach().cpu()
            self.buffer.append(store)

    def _prepare_replay_sample(self, sample: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Create a batch from multiple replay samples.
        
        Args:
            samples: List of sample dictionaries from buffer
            device: Device to place tensors on
            
        Returns:
            Batched dictionary ready for forward pass
        """
        prepared = {}
        for key in sample.keys():
            
            if isinstance(sample[key], torch.Tensor) and not key == "is_conversational":
                prepared[key] = sample[key].unsqueeze(0).to(device)
            else:
                # Non-tensor values (shouldn't happen in typical usage)
                prepared[key] = sample[key]

        return prepared

    def _process_replay_slice(
        self, model: torch.nn.Module, device: torch.device, replay_examples: List[Dict[str, Any]], **kw
    ) -> tuple:
        """
        Process one slice of the buffer, performing a backward pass for each
        mini-batch to accumulate gradients. Returns the average loss and IS
        ratio statistics for logging.
        """
        

        total_weighted_loss = 0.0
        total_valid_samples = 0
        all_slice_ratios = []
        
        # print(f'self._replay_slice_idx: {self._replay_slice_idx}')
        for s in range(0, len(replay_examples)):
            # pdb.set_trace()
            batch_sample = replay_examples[s]

            replay_sample = self._prepare_replay_sample(batch_sample, device)

            # Get current per-token log probabilities
            with torch.no_grad():
                # Get input_ids and attention_mask for the full sequence
                input_ids = torch.cat([replay_sample["prompt_ids"], replay_sample["completion_ids"]], dim=1)
                attention_mask = torch.cat([replay_sample["prompt_mask"], replay_sample["completion_mask"]], dim=1)
                completion_ids = replay_sample["completion_ids"]
                logits_to_keep = completion_ids.size(1) # only need logits for completion tokens
                # pdb.set_trace()
                # Compute current per-token log probabilities for the full sequence
                per_token_logps_current, _ = self._get_per_token_logps_and_entropies(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep=logits_to_keep,
                    compute_entropy=False,
                    pixel_values=replay_sample.get("pixel_values"),
                    image_grid_thw=replay_sample.get("image_grid_thw"),
                    pixel_attention_mask=replay_sample.get("pixel_attention_mask"),
                    image_sizes=replay_sample.get("image_sizes"),
                )
            
            # Compute importance sampling ratios
            # The old_per_token_logps are already detached
            old_per_token_logps = replay_sample["old_per_token_logps"]
            prompts = replay_sample["raw_inputs"]["prompt"]
            # replay_completion_ids_list = [
            #     [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(replay_sample["completion_ids"], replay_sample["completion_mask"])
            # ]
            completions_text = replay_sample['completions']
            rewards_per_func = torch.zeros([1,1],dtype=torch.float32,device=device)
            rewards_per_func[0,0] = correctness_reward_func_original(completions_text, replay_sample['answer'])
            # rewards_per_func = self._calculate_rewards(replay_sample["raw_inputs"], replay_sample['prompts'], completions, replay_completion_ids_list)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = replay_sample["mean_grouped_rewards"].to(device)
            # GRPO advantage computation
            # Compute grouped-wise rewards
            # mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            # std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # # Normalize the rewards to compute the advantages
            # mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            # std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            # advantages = rewards - mean_grouped_rewards
            # if self.scale_rewards:
            #     advantages = advantages / (std_grouped_rewards + 1e-4)

            advantages = rewards - mean_grouped_rewards


            # pdb.set_trace()
            # Slice to keep only the local part of the data
            # process_slice = slice(
            #     self.accelerator.process_index * len(prompts),
            #     (self.accelerator.process_index + 1) * len(prompts),
            # )

            # advantages = advantages[process_slice]
            replay_sample["advantages"] = advantages
            # print(f"[DEBUG] rank={self.accelerator.local_process_index} replay advantages shape process replay slice={replay_sample['advantages'].shape}")
                
            # Create a mask for completion tokens only (assuming the per_token_logps are for the full sequence)
            # We need to mask out the prompt portion for importance ratio calculation
            # prompt_length = replay_sample["prompt_ids"].shape[1]
            full_completion_mask = replay_sample["completion_mask"]
            # Compute importance ratios only for completion tokens
            log_importance_ratios = (per_token_logps_current - old_per_token_logps) * full_completion_mask
            
            # Aggregate to get per-sample importance ratios (sum in log space = product in probability space)
            log_importance_ratios_sum = log_importance_ratios.sum(dim=1)
            importance_ratios = log_importance_ratios_sum.exp()

            all_slice_ratios.extend(importance_ratios.detach().cpu().numpy())

            # if self.filter_replay:
            #     valid_mask = (importance_ratios >= self.is_ratio_lower_bound) & \
            #                  (importance_ratios <= self.is_ratio_upper_bound)
            # else:
            #     valid_mask = torch.ones_like(importance_ratios, dtype=torch.bool)
            
            # num_valid =  torch.ones_like(importance_ratios, dtype=torch.bool).sum().item()
            # if self.log_is_metrics and wandb.run is not None:
            #     num_discarded = importance_ratios.size(0) - num_valid
            #     wandb.log({"replay/discarded_per_minibatch": num_discarded})
            #     print(f"replay/discarded_per_minibatch: {num_discarded}")

            # if num_valid == 0:
            #     continue

            # Filter the batch based on valid samples
            # filtered_replay_batch = {}
            # for key, tensor in replay_batch.items():
            #     if isinstance(tensor, torch.Tensor) and tensor.size(0) == len(valid_mask):
            #         filtered_replay_batch[key] = tensor[valid_mask]
            #     else:
            #         filtered_replay_batch[key] = tensor
            
            # Apply importance sampling weights to advantages
            # filtered_ratios = importance_ratios[valid_mask]
            # filtered_replay_batch["advantages"] = filtered_replay_batch["advantages"] * filtered_ratios
            
            # Compute loss for the filtered mini-batch
            batch_loss = self.compute_replay_loss(model, replay_sample, return_outputs=False, **kw)
            # pdb.set_trace()
            scaled_loss = 0.5 * batch_loss / len(replay_examples)  # Scale loss for gradient accumulation
            self.accelerator.backward(scaled_loss)

            total_weighted_loss += batch_loss.detach().cpu().item()
            # total_valid_samples += num_valid
            
            del replay_sample, batch_loss, scaled_loss
            torch.cuda.empty_cache()

        # if total_valid_samples == 0:
        #     return None, None

        avg_loss = total_weighted_loss
        # ratio_stats = {
        #     "mean": np.mean(all_slice_ratios),
        #     "min": np.min(all_slice_ratios),
        #     "max": np.max(all_slice_ratios)
        # }
        return avg_loss

    def compute_loss(self, 
                     model: torch.nn.Module, 
                     inputs: Dict[str, Any], 
                     return_outputs=False, 
                     **kw) -> torch.Tensor:
        """
        Compute live loss, trigger replay backward pass, and return live loss.
        """
        # pdb.set_trace()
        device = inputs["prompt_ids"].device

         # Compute the per-token log probabilities for the model
        # prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        # completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens


        # Compute per-token log probabilities for the current batch
        # with torch.no_grad():
        #     input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        #     attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            
        #     per_token_logps, _ = self._get_per_token_logps_and_entropies(
        #     model,
        #     input_ids,
        #     attention_mask,
        #     logits_to_keep,
        #     compute_entropy=False,
        #     pixel_values=inputs.get("pixel_values"),
        #     image_grid_thw=inputs.get("image_grid_thw"),
        #     pixel_attention_mask=inputs.get("pixel_attention_mask"),
        #     image_sizes=inputs.get("image_sizes"),
        # )

        # if len(self.buffer) < num_slices * self.args.per_device_train_batch_size:
        #     print(f"Buffer size {len(self.buffer)} is less than threshold. Storing to buffer.")
        #     self._store_to_buffer(inputs, per_token_logps)
        #     live_loss = super().compute_loss(model, inputs, return_outputs, **kw)
        #     return live_loss

        # Stage current batch for later addition to buffer
        # self._stage_batch(inputs, per_token_logps)

        # Process replay slice
        # print('call self._process_replay_slice')
        # pdb.set_trace()
        question_ids = inputs.get("question_ids", None)
        generation_ids = inputs.get("generation_ids", None)
        replay_examples = []
        if question_ids is not None and generation_ids is not None:
            for i in range(len(question_ids)):
                external_experience = {}
                qid = question_ids[i]
                gid = generation_ids[i]
                external_sample = self.offline_data.get((qid, gid), None)
                if external_sample is not None:
                    completion_stats = get_completion_stats(
                        qid,
                        gid,
                        self.offline_data,
                        self.processing_class,
                    )
                    # Update inputs with external completion stats
                    external_experience["completion_ids"] = torch.tensor(
                        completion_stats["completion_ids"], dtype=inputs["completion_ids"].dtype
                    )
                    external_experience["completion_mask"] = torch.tensor(
                        completion_stats["completion_mask"], dtype=inputs["completion_mask"].dtype
                    )
                    external_experience["old_per_token_logps"] = torch.tensor(
                        completion_stats["old_per_token_logps"], dtype=torch.float32
                    )
                    external_experience["answer"] = completion_stats["answer"]
                    external_experience["completions"] = completion_stats["completions"]
                    external_experience["raw_inputs"] = inputs["raw_inputs"][i]
                    
                    external_experience["prompt_ids"] = inputs["prompt_ids"][i].detach().cpu()
                    external_experience["prompt_mask"] = inputs["prompt_mask"][i].detach().cpu()
                    external_experience["is_conversational"] = inputs["conversational"][i]
                    replay_examples.append(external_experience)
                    external_experience["mean_grouped_rewards"] = inputs["mean_grouped_rewards"][i]
        # pdb.set_trace()
        replay_loss_val = self._process_replay_slice(model, device, replay_examples, **kw)
            
        # Compute live loss
        live_loss = super().compute_loss(model, inputs, return_outputs, **kw)
        print(f"Live loss: {live_loss.item():.4f}, Replay loss: {replay_loss_val:.4f}")
        # Advance slice pointer and flush if cycle is complete
            # buffer_list = list(self.buffer)
            # random.shuffle(buffer_list)
            # self.buffer = deque(buffer_list, maxlen=self.buffer_size)
            # print(f"Shuffled buffer of size {len(buffer_list)}")

        # Logging
        if self.log_is_metrics and wandb.run is not None and replay_loss_val is not None:
            log_data = {
                "loss/live": live_loss.item(),
                "loss/replay_slice": replay_loss_val,
            }
            
            wandb.log(log_data)

        # Return scaled live loss
        return 0.5 * live_loss
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # pdb.set_trace()

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None
        # pdb.set_trace()
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps",None)
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            # log_importance_weights = log_ratio
            cur_log_prob = per_token_logps.clamp(min=-10,max=10)
            old_log_prob = old_per_token_logps.clamp(min=-10,max=10)
            cur_prob = torch.exp(cur_log_prob).clamp(min=1e-9)
            old_prob = torch.exp(old_log_prob).clamp(min=1e-9)
            cur_prob_detached = cur_prob.detach()
            if self.detach_denominator:
                # coef_1 = 0.5 * (cur_prob**2) / (((cur_prob_detached + old_prob) / 2))
                # coef_1 = cur_prob / (((cur_prob_detached + old_prob) / 2))
                coef_1 = old_prob / cur_prob_detached * self.past_model_weight
                coef_1 = coef_1 + self.cur_model_weight
                coef_1 = 1 / coef_1
                coef_1 = coef_1 * cur_log_prob
                coef_1 = torch.nan_to_num(coef_1, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                coef_1 = cur_prob / (((cur_prob + old_prob) / 2))
                coef_1 = torch.clamp(coef_1, max=2.0)

            # Compute masked sums for logging (token-level)
            cur_prob_sum = (cur_prob_detached * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1) # Sum over tokens for each sample
            old_prob_sum = (old_prob * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)
            coef_1_sum = (coef_1.detach() * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)

            # Take batch averages
            cur_prob_mean = cur_prob_sum.mean().item()
            old_prob_mean = old_prob_sum.mean().item()
            coef_1_mean = coef_1_sum.mean().item()
            
            # Get min and max values
            cur_prob_min = cur_prob_sum.min().item()
            cur_prob_max = cur_prob_sum.max().item()
            old_prob_min = old_prob_sum.min().item()
            old_prob_max = old_prob_sum.max().item()
            coef_1_min = coef_1_sum.min().item()
            coef_1_max = coef_1_sum.max().item()
            print("coef_1_min:", coef_1_min, "coef_1_max:", coef_1_max)
        elif self.importance_sampling_level == "sequence":
            cur_log_prob = (per_token_logps * completion_mask).sum(-1)/ completion_mask.sum(-1).clamp(min=1.0)
            old_log_prob = (old_per_token_logps * completion_mask).sum(-1)/ completion_mask.sum(-1).clamp(min=1.0)
            cur_prob = torch.exp(cur_log_prob.unsqueeze(-1)).clamp(min=-50,max=50)
            old_prob = torch.exp(old_log_prob.unsqueeze(-1)).clamp(min=-50,max=50)  
            cur_prob_detached = cur_prob.detach()
            if self.detach_denominator:
                coef_1 = cur_prob / (((cur_prob_detached + old_prob) / 2)+1e-9)
            else:
                coef_1 = cur_prob / (((cur_prob + old_prob) / 2)+1e-9)
            # log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            # log_importance_weights = log_importance_weights.unsqueeze(-1)
            # For sequence-level, squeeze and take batch statistics
            cur_prob_squeezed = cur_prob_detached.squeeze(-1)
            old_prob_squeezed = old_prob.squeeze(-1)
            coef_1_squeezed = coef_1.squeeze(-1)
            
            # Take batch averages
            cur_prob_mean = cur_prob_squeezed.mean().item()
            old_prob_mean = old_prob_squeezed.mean().item()
            coef_1_mean = coef_1_squeezed.mean().item()
            
            # Get min and max values
            cur_prob_min = cur_prob_squeezed.min().item()
            cur_prob_max = cur_prob_squeezed.max().item()
            old_prob_min = old_prob_squeezed.min().item()
            old_prob_max = old_prob_squeezed.max().item()
            coef_1_min = coef_1_squeezed.min().item()
            coef_1_max = coef_1_squeezed.max().item()
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        if self.log_is_metrics and wandb.run is not None:
            wandb.log({
                "importance_sampling/cur_prob_mean": cur_prob_mean,
                "importance_sampling/cur_prob_min": cur_prob_min,
                "importance_sampling/cur_prob_max": cur_prob_max,
                "importance_sampling/old_prob_mean": old_prob_mean,
                "importance_sampling/old_prob_min": old_prob_min,
                "importance_sampling/old_prob_max": old_prob_max,
                "importance_sampling/coef_1_mean": coef_1_mean,
                "importance_sampling/coef_1_min": coef_1_min,
                "importance_sampling/coef_1_max": coef_1_max,
            })
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        # coef_1 = torch.exp(log_importance_weights)
        # coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        

        # # Two-sided clipping
        # if self.args.delta is not None:
        #     coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        # per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = -coef_1 * advantages.unsqueeze(1)
        if_entropy_mask = 0
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
            if_entropy_mask = 1
        if wandb.run is not None:
            wandb.log({"if_entropy_mask": if_entropy_mask})
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        # is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        # is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        # is_region_clipped = is_low_clipped | is_high_clipped

        # low_clip = masked_batch_mean(is_low_clipped.float())
        # high_clip = masked_batch_mean(is_high_clipped.float())
        # clip_ratio = masked_batch_mean(is_region_clipped.float())

        # gathered_low_clip = self.accelerator.gather(low_clip)
        # self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        # self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        # gathered_high_clip = self.accelerator.gather(high_clip)
        # self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        # self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        # gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        # self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def _compute_replay_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # pdb.set_trace()

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )
        ref_per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            self.ref_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # advantage_shape = advantages.shape
        # print(f"[DEBUG] rank={self.accelerator.local_process_index} replay Advantages shape={advantage_shape}")
        # if torch.isnan(advantages).any() or torch.isinf(advantages).any():
        #     print("[NaN debug] advantages has NaN/Inf")
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"]
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            # log_importance_weights = log_ratio
            cur_log_prob = per_token_logps.clamp(min=-10,max=10)
            old_log_prob = old_per_token_logps.clamp(min=-10,max=10)
            cur_prob = torch.exp(cur_log_prob).clamp(min=1e-9)
            old_prob = torch.exp(old_log_prob).clamp(min=1e-9)
            # nan_debug(cur_prob, "cur_prob", self.accelerator.local_process_index)
            # nan_debug(old_prob, "old_prob", self.accelerator.local_process_index)
            cur_prob_detached = cur_prob.detach()
            if self.detach_denominator:
                # coef_1 = 0.5 * (cur_prob**2) / (((cur_prob_detached + old_prob) / 2))
                # coef_1 = cur_prob / (((cur_prob_detached + old_prob) / 2))
                coef_1 = old_prob / cur_prob_detached * self.past_model_weight
                # nan_debug(coef_1, "coef_1_1", self.accelerator.local_process_index)
            
                coef_1 = coef_1 + self.cur_model_weight
                coef_1 = 1 / coef_1
                # nan_debug(coef_1, "coef_1_2", self.accelerator.local_process_index)
            
                coef_1 = coef_1 * cur_log_prob
                # nan_debug(coef_1, "coef_1_3", self.accelerator.local_process_index)
            
                coef_1 = torch.nan_to_num(coef_1, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                coef_1 = cur_prob / (((cur_prob + old_prob) / 2))
                coef_1 = torch.clamp(coef_1, max=2.0)

            # Compute masked sums for logging (token-level)
            cur_prob_sum = (cur_prob_detached * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1) # Sum over tokens for each sample
            # nan_debug(cur_prob_sum, "cur_prob_sum", self.accelerator.local_process_index)
            old_prob_sum = (old_prob * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)
            # nan_debug(old_prob_sum, "old_prob_sum", self.accelerator.local_process_index)
            coef_1_sum = (coef_1.detach() * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)
            # nan_debug(coef_1_sum, "coef_1_sum", self.accelerator.local_process_index)
            # Take batch averages
            cur_prob_mean = cur_prob_sum.mean().item()
            old_prob_mean = old_prob_sum.mean().item()
            coef_1_mean = coef_1_sum.mean().item()
            
            # Get min and max values
            cur_prob_min = cur_prob_sum.min().item()
            cur_prob_max = cur_prob_sum.max().item()
            old_prob_min = old_prob_sum.min().item()
            old_prob_max = old_prob_sum.max().item()
            coef_1_min = coef_1_sum.min().item()
            coef_1_max = coef_1_sum.max().item()
            print("coef_1_min:", coef_1_min, "coef_1_max:", coef_1_max)
        elif self.importance_sampling_level == "sequence":
            cur_log_prob = (per_token_logps * completion_mask).sum(-1)/ completion_mask.sum(-1).clamp(min=1.0)
            old_log_prob = (old_per_token_logps * completion_mask).sum(-1)/ completion_mask.sum(-1).clamp(min=1.0)
            cur_prob = torch.exp(cur_log_prob.unsqueeze(-1)).clamp(min=-50,max=50)
            old_prob = torch.exp(old_log_prob.unsqueeze(-1)).clamp(min=-50,max=50)  
            cur_prob_detached = cur_prob.detach()
            if self.detach_denominator:
                coef_1 = cur_prob / (((cur_prob_detached + old_prob) / 2)+1e-9)
            else:
                coef_1 = cur_prob / (((cur_prob + old_prob) / 2)+1e-9)
            # log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            # log_importance_weights = log_importance_weights.unsqueeze(-1)
            # For sequence-level, squeeze and take batch statistics
            cur_prob_squeezed = cur_prob_detached.squeeze(-1)
            old_prob_squeezed = old_prob.squeeze(-1)
            coef_1_squeezed = coef_1.squeeze(-1)
            
            # Take batch averages
            cur_prob_mean = cur_prob_squeezed.mean().item()
            old_prob_mean = old_prob_squeezed.mean().item()
            coef_1_mean = coef_1_squeezed.mean().item()
            
            # Get min and max values
            cur_prob_min = cur_prob_squeezed.min().item()
            cur_prob_max = cur_prob_squeezed.max().item()
            old_prob_min = old_prob_squeezed.min().item()
            old_prob_max = old_prob_squeezed.max().item()
            coef_1_min = coef_1_squeezed.min().item()
            coef_1_max = coef_1_squeezed.max().item()
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        if self.log_is_metrics and wandb.run is not None:
            wandb.log({
                "importance_sampling/cur_prob_mean": cur_prob_mean,
                "importance_sampling/cur_prob_min": cur_prob_min,
                "importance_sampling/cur_prob_max": cur_prob_max,
                "importance_sampling/old_prob_mean": old_prob_mean,
                "importance_sampling/old_prob_min": old_prob_min,
                "importance_sampling/old_prob_max": old_prob_max,
                "importance_sampling/coef_1_mean": coef_1_mean,
                "importance_sampling/coef_1_min": coef_1_min,
                "importance_sampling/coef_1_max": coef_1_max,
            })
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        # coef_1 = torch.exp(log_importance_weights)
        # coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        

        # # Two-sided clipping
        # if self.args.delta is not None:
        #     coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        # per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = -coef_1 * advantages.unsqueeze(1)
        # per_token_loss_shape = per_token_loss.shape
        # print(f"[DEBUG] rank={self.accelerator.local_process_index} replay per_token_loss shape before entropy={per_token_loss_shape}")
        # nan_debug(per_token_loss, "per_token_loss before entropy mask", self.accelerator.local_process_index)
        if_entropy_mask = 0
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
            # per_token_loss_shape = per_token_loss.shape
            # print(f"[DEBUG] rank={self.accelerator.local_process_index} replay per_token_loss shape after entropy={per_token_loss_shape}")
            # nan_debug(per_token_loss, "per_token_loss after entropy mask", self.accelerator.local_process_index)
            if_entropy_mask = 1
        if wandb.run is not None:
            wandb.log({"if_entropy_mask": if_entropy_mask})
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
            # per_token_loss_shape = per_token_loss.shape
            # print(f"[DEBUG] rank={self.accelerator.local_process_index} replay per_token_loss shape={per_token_loss_shape}")
            
            # nan_debug(per_token_loss, "per_token_loss after adding KL", self.accelerator.local_process_index)

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            # nan_debug((per_token_loss * completion_mask).sum(-1), "numerator loss grpo", self.accelerator.local_process_index)
            # nan_debug(completion_mask.sum(-1).clamp(min=1.0), "denominator loss grpo", self.accelerator.local_process_index)
            # print(f"denominator loss grpo: {completion_mask.sum(-1).clamp(min=1.0)}")
            # nan_debug(loss, "loss grpo", self.accelerator.local_process_index)  
            # B, T = per_token_loss.shape
            # print(f"[DEBUG] rank={self.accelerator.local_process_index} replay B={B}, T={T}")

        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        # mode = "train" if self.model.training else "eval"

        # completion_token_count = completion_mask.sum().clamp(min=1.0)

        # def masked_batch_mean(x):
        #     if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
        #         return x.mean()
        #     else:
        #         return (x * completion_mask).sum() / completion_token_count

        # if self.beta != 0.0:
        #     mean_kl = masked_batch_mean(per_token_kl)
        #     self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # mean_entropy = masked_batch_mean(entropies)
        # self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        # is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        # is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        # is_region_clipped = is_low_clipped | is_high_clipped

        # low_clip = masked_batch_mean(is_low_clipped.float())
        # high_clip = masked_batch_mean(is_high_clipped.float())
        # clip_ratio = masked_batch_mean(is_region_clipped.float())

        # gathered_low_clip = self.accelerator.gather(low_clip)
        # self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        # self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        # gathered_high_clip = self.accelerator.gather(high_clip)
        # self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        # self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        # gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        # self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        # if torch.isnan(loss) or torch.isinf(loss):
        #     print("[NaN debug] Loss is NaN/Inf on rank",
        #                    self.accelerator.local_process_index)
        return loss



    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get statistics about the replay buffer."""
        if not self.buffer:
            return {"buffer_size": 0}
        
        advantages = [entry["advantages"].numpy() for entry in self.buffer]
        advantages_flat = np.concatenate(advantages)
        
        return {
            "buffer_size": len(self.buffer),
            "buffer_advantages_mean": float(np.mean(advantages_flat)),
            "buffer_advantages_std": float(np.std(advantages_flat)),
            "buffer_advantages_min": float(np.min(advantages_flat)),
            "buffer_advantages_max": float(np.max(advantages_flat)),
        }
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        # pdb.set_trace()
        prompts = [x["prompt"] for x in inputs]
        question_id = [x.get("question_id", None) for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        # apply chat template if needed - for tokenization
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # tokenized prompts
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs  = Trainer._prepare_inputs(self,prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            # The chat template inserts a single image token into the prompt text. However, when this text is later
            # tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original template.
            if self.image_token is not None:
                prompts_text = [
                    re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text) for text in prompts_text
                ]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        inputs_is_conversational = is_conversational(inputs[0])
        if inputs_is_conversational:
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        # pdb.set_trace()
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # pdb.set_trace()
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]
        print("advantages after slicing:", advantages)

        # pdb.set_trace()
        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if has_images:
            self._logs["image"].extend(gather_object(images))

        eff_batch_size = len(question_id)
        # generation_id = torch.arange(eff_batch_size) % self.num_generations
        generation_id = [i % self.num_generations for i in range(eff_batch_size)]
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "question_ids": question_id,
            "generation_ids": generation_id,
            "conversational": int(inputs_is_conversational)*torch.ones(len(generation_id), dtype=torch.int8, device=device),
            "prompts": original_prompts,
            "raw_inputs": inputs,
            "mean_grouped_rewards": mean_grouped_rewards[process_slice].detach().cpu(),
        }
        print("rewards:", rewards)
        print("advantages:", advantages)

        # pdb.set_trace()
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        return output
    
    @profiling_decorator
    def compute_replay_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_replay_loss(model, inputs)
