{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "fc830717-fccd-4dc9-bea0-b18ab1c37caa",
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 13\u001b[0m\n\u001b[0;32m     10\u001b[0m dataset \u001b[38;5;241m=\u001b[39m load_dataset(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcsv\u001b[39m\u001b[38;5;124m\"\u001b[39m, data_files\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdata/processed_data.csv\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     12\u001b[0m \u001b[38;5;66;03m# ✅ Training arguments\u001b[39;00m\n\u001b[1;32m---> 13\u001b[0m training_args \u001b[38;5;241m=\u001b[39m \u001b[43mTrainingArguments\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m     14\u001b[0m \u001b[43m    \u001b[49m\u001b[43moutput_dir\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43m./gpt2-hotel-model\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m     15\u001b[0m \u001b[43m    \u001b[49m\u001b[43mper_device_train_batch_size\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m4\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m     16\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlogging_dir\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43m./logs\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m     17\u001b[0m \u001b[43m    \u001b[49m\u001b[43mnum_train_epochs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m2\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[0;32m     18\u001b[0m \u001b[43m    \u001b[49m\u001b[43msave_strategy\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mepoch\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\n\u001b[0;32m     19\u001b[0m \u001b[43m)\u001b[49m\n\u001b[0;32m     21\u001b[0m trainer \u001b[38;5;241m=\u001b[39m Trainer(model\u001b[38;5;241m=\u001b[39mmodel, args\u001b[38;5;241m=\u001b[39mtraining_args, train_dataset\u001b[38;5;241m=\u001b[39mdataset[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtrain\u001b[39m\u001b[38;5;124m\"\u001b[39m])\n\u001b[0;32m     22\u001b[0m trainer\u001b[38;5;241m.\u001b[39mtrain()\n",
      "File \u001b[1;32m<string>:135\u001b[0m, in \u001b[0;36m__init__\u001b[1;34m(self, output_dir, overwrite_output_dir, do_train, do_eval, do_predict, eval_strategy, prediction_loss_only, per_device_train_batch_size, per_device_eval_batch_size, per_gpu_train_batch_size, per_gpu_eval_batch_size, gradient_accumulation_steps, eval_accumulation_steps, eval_delay, torch_empty_cache_steps, learning_rate, weight_decay, adam_beta1, adam_beta2, adam_epsilon, max_grad_norm, num_train_epochs, max_steps, lr_scheduler_type, lr_scheduler_kwargs, warmup_ratio, warmup_steps, log_level, log_level_replica, log_on_each_node, logging_dir, logging_strategy, logging_first_step, logging_steps, logging_nan_inf_filter, save_strategy, save_steps, save_total_limit, save_safetensors, save_on_each_node, save_only_model, restore_callback_states_from_checkpoint, no_cuda, use_cpu, use_mps_device, seed, data_seed, jit_mode_eval, use_ipex, bf16, fp16, fp16_opt_level, half_precision_backend, bf16_full_eval, fp16_full_eval, tf32, local_rank, ddp_backend, tpu_num_cores, tpu_metrics_debug, debug, dataloader_drop_last, eval_steps, dataloader_num_workers, dataloader_prefetch_factor, past_index, run_name, disable_tqdm, remove_unused_columns, label_names, load_best_model_at_end, metric_for_best_model, greater_is_better, ignore_data_skip, fsdp, fsdp_min_num_params, fsdp_config, tp_size, fsdp_transformer_layer_cls_to_wrap, accelerator_config, deepspeed, label_smoothing_factor, optim, optim_args, adafactor, group_by_length, length_column_name, report_to, ddp_find_unused_parameters, ddp_bucket_cap_mb, ddp_broadcast_buffers, dataloader_pin_memory, dataloader_persistent_workers, skip_memory_metrics, use_legacy_prediction_loop, push_to_hub, resume_from_checkpoint, hub_model_id, hub_strategy, hub_token, hub_private_repo, hub_always_push, gradient_checkpointing, gradient_checkpointing_kwargs, include_inputs_for_metrics, include_for_metrics, eval_do_concat_batches, fp16_backend, evaluation_strategy, push_to_hub_model_id, push_to_hub_organization, push_to_hub_token, mp_parameters, auto_find_batch_size, full_determinism, torchdynamo, ray_scope, ddp_timeout, torch_compile, torch_compile_backend, torch_compile_mode, dispatch_batches, split_batches, include_tokens_per_second, include_num_input_tokens_seen, neftune_noise_alpha, optim_target_modules, batch_eval_metrics, eval_on_start, use_liger_kernel, eval_use_gather_object, average_tokens_across_devices)\u001b[0m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\transformers\\training_args.py:1808\u001b[0m, in \u001b[0;36mTrainingArguments.__post_init__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   1806\u001b[0m \u001b[38;5;66;03m# Initialize device before we proceed\u001b[39;00m\n\u001b[0;32m   1807\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mframework \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpt\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mand\u001b[39;00m is_torch_available():\n\u001b[1;32m-> 1808\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdevice\u001b[49m\n\u001b[0;32m   1810\u001b[0m \u001b[38;5;66;03m# Disable average tokens when using single device\u001b[39;00m\n\u001b[0;32m   1811\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39maverage_tokens_across_devices:\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\transformers\\training_args.py:2344\u001b[0m, in \u001b[0;36mTrainingArguments.device\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   2340\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   2341\u001b[0m \u001b[38;5;124;03mThe device used by this process.\u001b[39;00m\n\u001b[0;32m   2342\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m   2343\u001b[0m requires_backends(\u001b[38;5;28mself\u001b[39m, [\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtorch\u001b[39m\u001b[38;5;124m\"\u001b[39m])\n\u001b[1;32m-> 2344\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_setup_devices\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\transformers\\utils\\generic.py:62\u001b[0m, in \u001b[0;36mcached_property.__get__\u001b[1;34m(self, obj, objtype)\u001b[0m\n\u001b[0;32m     60\u001b[0m cached \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mgetattr\u001b[39m(obj, attr, \u001b[38;5;28;01mNone\u001b[39;00m)\n\u001b[0;32m     61\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m cached \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m---> 62\u001b[0m     cached \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfget\u001b[49m\u001b[43m(\u001b[49m\u001b[43mobj\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     63\u001b[0m     \u001b[38;5;28msetattr\u001b[39m(obj, attr, cached)\n\u001b[0;32m     64\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m cached\n",
      "File \u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\transformers\\training_args.py:2214\u001b[0m, in \u001b[0;36mTrainingArguments._setup_devices\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   2212\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m is_sagemaker_mp_enabled():\n\u001b[0;32m   2213\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m is_accelerate_available():\n\u001b[1;32m-> 2214\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mImportError\u001b[39;00m(\n\u001b[0;32m   2215\u001b[0m             \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mUsing the `Trainer` with `PyTorch` requires `accelerate>=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mACCELERATE_MIN_VERSION\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m`: \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   2216\u001b[0m             \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPlease run `pip install transformers[torch]` or `pip install \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124maccelerate>=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mACCELERATE_MIN_VERSION\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m`\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m   2217\u001b[0m         )\n\u001b[0;32m   2218\u001b[0m \u001b[38;5;66;03m# We delay the init of `PartialState` to the end for clarity\u001b[39;00m\n\u001b[0;32m   2219\u001b[0m accelerator_state_kwargs \u001b[38;5;241m=\u001b[39m {\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124menabled\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;28;01mTrue\u001b[39;00m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124muse_configured_state\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;28;01mFalse\u001b[39;00m}\n",
      "\u001b[1;31mImportError\u001b[0m: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`"
     ]
    }
   ],
   "source": [
    "from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer\n",
    "from datasets import load_dataset\n",
    "\n",
    "# ✅ Model & Tokenizer\n",
    "model_name = \"distilgpt2\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "model = AutoModelForCausalLM.from_pretrained(model_name)\n",
    "\n",
    "# ✅ Load dataset\n",
    "dataset = load_dataset(\"csv\", data_files=\"data/processed_data.csv\")\n",
    "\n",
    "# ✅ Training arguments\n",
    "training_args = TrainingArguments(\n",
    "    output_dir=\"./gpt2-hotel-model\",\n",
    "    per_device_train_batch_size=4,\n",
    "    logging_dir=\"./logs\",\n",
    "    num_train_epochs=2,\n",
    "    save_strategy=\"epoch\"\n",
    ")\n",
    "\n",
    "trainer = Trainer(model=model, args=training_args, train_dataset=dataset[\"train\"])\n",
    "trainer.train()\n",
    "\n",
    "# ✅ Save the trained model\n",
    "trainer.save_model(\"./gpt2-hotel-model\")\n",
    "tokenizer.save_pretrained(\"./gpt2-hotel-model\")\n",
    "\n",
    "print(\"🎉✅ Fine-tuning complete! Model saved to './gpt2-hotel-model'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a98715e-c9b2-4273-9e0b-a98e650f7377",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
