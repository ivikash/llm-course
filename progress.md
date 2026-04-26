# Progress Tracker

Copy this to a personal `progress.md` or keep notes here. Tick off lessons as you complete them. Add a one-line note per lesson to force yourself to articulate what you learned.

```
Module 0 - Start Here
[ ] 00_what_is_this_course.md                     - note:
[ ] 01_demystified_glossary.md                     - note:
[ ] 02_tools_and_setup.md                           - note:
[ ] 03_what_is_an_llm.md                            - note:

Module 1 - Math Foundations
[ ] 01_vectors_and_matrices.md                     - note:
[ ] 02_matrix_multiplication.md                    - note:
[ ] 03_calculus_for_learning.md                    - note:
[ ] 04_probability_basics.md                       - note:
[ ] 05_numpy_and_tensors.md                        - note:
[ ] capstone_bigram.md                              - note:

Module 2 - Deep Learning
[ ] 01_a_neuron.md                                  - note:
[ ] 02_activations_nonlinearity.md                  - note:
[ ] 03_loss_functions.md                            - note:
[ ] 04_backprop_and_sgd.md                          - note:
[ ] 05_optimizers_adam_adamw.md                     - note:
[ ] 06_regularization_dropout_weightdecay.md        - note:
[ ] 07_training_loop_anatomy.md                     - note:
[ ] capstone_mlp.md                                 - note:

Module 3 - Transformer via nanoGPT
[ ] 01_tokenization.md                              - note:
[ ] 02_embeddings.md                                - note:
[ ] 03_attention_intuition.md                       - note:
[ ] 04_attention_mathematically.md                  - note:
[ ] 05_multi_head_attention.md                      - note:
[ ] 06_mlp_block.md                                 - note:
[ ] 07_layernorm_and_residual.md                    - note:
[ ] 08_full_block_walkthrough.md                    - note:
[ ] 09_positional_encoding.md                       - note:
[ ] 10_lm_head_and_sampling.md                      - note:
[ ] 11_train_py_walkthrough.md                      - note:
[ ] capstone_shakespeare.md                          - note:

Module 4 - Training Pipeline
[ ] 01_datasets_and_dataloaders.md                  - note:
[ ] 02_bpe_tokenizer_deep.md                        - note:
[ ] 03_mixed_precision_bf16_fp16.md                 - note:
[ ] 04_gradient_accumulation.md                     - note:
[ ] 05_learning_rate_schedules.md                   - note:
[ ] 06_gradient_clipping_nan_debugging.md           - note:
[ ] 07_checkpointing_and_resume.md                  - note:
[ ] 08_evaluation_val_loss_perplexity.md            - note:
[ ] capstone_custom_dataset.md                       - note:

Module 5 - nanochat Full Stack
[ ] 01_tour_of_nanochat.md                          - note:
[ ] 02_tokenizer_training.md                        - note:
[ ] 03_pretraining_base_train.md                    - note:
[ ] 04_midtraining_and_chat_formatting.md           - note:
[ ] 05_sft_supervised_fine_tuning.md                - note:
[ ] 06_rl_and_preferences.md                        - note:
[ ] 07_evaluation_core_mmlu_humaneval_gsm8k.md      - note:
[ ] 08_inference_engine_kv_cache.md                 - note:
[ ] 09_serving_chat_web.md                          - note:
[ ] 10_speedrun_end_to_end.md                       - note:
[ ] capstone_train_and_talk.md                       - note:

Module 6 - Infrastructure and Scaling
[ ] 01_cpu_vs_gpu_vs_tpu.md                         - note:
[ ] 02_gpu_memory_and_vram.md                       - note:
[ ] 03_distributed_training_ddp.md                  - note:
[ ] 04_fsdp_zero_tensor_parallelism.md              - note:
[ ] 05_flash_attention_fp8_performance.md           - note:
[ ] 06_the_cloud_lambda_aws_sagemaker.md            - note:
[ ] 07_experiment_tracking_wandb.md                 - note:
[ ] 08_huggingface_ecosystem.md                     - note:
[ ] 09_scaling_laws.md                              - note:
[ ] 10_cost_engineering.md                          - note:
[ ] capstone_cloud_run.md                           - note:

Module 7 - Research and Papers
[ ] 01_how_to_read_a_paper.md                       - note:
[ ] 02_core_papers_you_must_know.md                 - note:
[ ] 03_transformer_paper_walkthrough.md             - note:
[ ] 04_gpt_papers_gpt2_gpt3_gpt4.md                 - note:
[ ] 05_scaling_laws_papers.md                       - note:
[ ] 06_rlhf_instruct_dpo.md                         - note:
[ ] 07_modern_architectures.md                      - note:
[ ] 08_how_to_reproduce_a_paper.md                  - note:
[ ] 09_how_to_write_a_paper.md                      - note:
[ ] capstone_paper_replica.md                       - note:

Module 8 - Leadership
[ ] 01_what_ml_scientists_actually_do_day_to_day.md - note:
[ ] 02_the_research_loop.md                         - note:
[ ] 03_project_scoping_and_killing_projects.md      - note:
[ ] 04_evaluations_as_the_product.md                - note:
[ ] 05_managing_compute_and_budgets.md              - note:
[ ] 06_team_structure_and_hiring.md                 - note:
[ ] 07_working_with_product_and_eng.md              - note:
[ ] 08_communicating_results_up_and_out.md          - note:
[ ] 09_research_taste.md                            - note:
[ ] 10_staying_current_in_a_field_that_moves_weekly.md - note:
```

## Capstones overview

Each module ends in a practical capstone. Bold ones produce a concrete artifact you should keep.

- **Module 1**: neural bigram language model (Python script).
- **Module 2**: 2-layer MLP on character sequences (PyTorch).
- **Module 3**: trained Shakespeare nanoGPT (checkpoint + generated samples).
- **Module 4**: train nanoGPT on a custom dataset of your choice.
- **Module 5**: end-to-end train-and-chat with nanochat (web UI + your chatbot).
- **Module 6**: a cloud GPU run (with receipts).
- **Module 7**: reproduce a figure from a paper.
- Module 8: no capstone - the capstone is your career.
