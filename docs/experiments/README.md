# Experiment Analysis Index

Each file documents a specific experiment: what was run, why, and what we learned.

| File | Experiment | Date | Status |
|------|-----------|------|--------|
| [exp01_multigpu_lora.md](exp01_multigpu_lora.md) | Multi-GPU LoRA compatibility test (DDP / ZeRO-2 / FSDP) | 2026-03-05 | Complete |
| [exp02_full_train_qwen05b.md](exp02_full_train_qwen05b.md) | Full-scale training: Qwen2.5-0.5B on 12K MATH problems | 2026-03-05 | Complete |
| [exp02_train_log.md](exp02_train_log.md) | Training log, eval results, and analysis for exp02 | 2026-03-06 | Complete |
| [algorithm.md](algorithm.md) | Full algorithm explanation: GRPO → Offline GRPO → code mapping | 2026-03-06 | Complete |
| [grpo_formulation.md](grpo_formulation.md) | GRPO equations with every symbol mapped to code | 2026-03-06 | Complete |
| [exp03_full_train_multigpu_lora.md](exp03_full_train_multigpu_lora.md) | Full-scale 4-GPU training + reference sync (refsync 0/16/1) | 2026-03-09 | Complete |
| [exp03_train_log.md](exp03_train_log.md) | Training log, eval results, and analysis for exp03 | 2026-03-09 | Complete |
| [peft_lora_multigpu_fix.md](peft_lora_multigpu_fix.md) | PEFT LoRA disable_adapter() fix for FSDP/ZeRO-3 | 2026-03-06 | Complete |
| [offline_grpo_mechanics.md](offline_grpo_mechanics.md) | How rewards, ratios, and losses work in offline GRPO | 2026-03-06 | Complete |
| [training_setup_reference.md](training_setup_reference.md) | All hyperparameters and setup in one place | 2026-03-09 | Complete |
| [deeplearning-wiki.md](deeplearning-wiki.md) | Deep learning concepts wiki (25 topics) | 2026-03-09 | Complete |
| [exp07_diagnose_method_B.md](exp07_diagnose_method_B.md) | Diagnose Method B offline loss: H1 (advantage collapse) vs H2 (IS clipping) | 2026-03-14 | Complete |
