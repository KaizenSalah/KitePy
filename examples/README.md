# KitePy Examples

Short, focused, runnable examples (each runs in seconds on CPU or GPU).

| Example | What It Shows |
|---|---|
| [`01_hello_kitepy.py`](01_hello_kitepy.py) | One-line training + `top_k` prediction |
| [`vision/01_cnn_quickstart.py`](vision/01_cnn_quickstart.py) | Train, save, reload, and predict with a CNN |
| [`vision/02_transfer_learning_vit.py`](vision/02_transfer_learning_vit.py) | Freeze ViT backbone, differential LRs, and EMA |
| [`vision/03_power_user_custom_step.py`](vision/03_power_user_custom_step.py) | Custom `nn.Module`, `train_step` hook, callbacks, and `.unwrap()` |
| [`vision/04_segmentation_and_detection.py`](vision/04_segmentation_and_detection.py) | Semantic segmentation (`unet`) & object detection (`tiny_detector`) |
| [`language/01_llm_lora_and_generate.py`](language/01_llm_lora_and_generate.py) | Train a tiny GPT from scratch and generate text |
| [`language/02_text_classification.py`](language/02_text_classification.py) | Sequence classification with `TextClassifier` |
| [`language/03_instruction_finetune_jsonl.py`](language/03_instruction_finetune_jsonl.py) | 1-line ChatML / Alpaca JSONL fine-tuning (SLM & HuggingFace LoRA) |
| [`multimodal/01_clip_zero_shot.py`](multimodal/01_clip_zero_shot.py) | Contrastive `VLM` training + zero-shot image-text scoring |
| [`audio/01_audio_classification.py`](audio/01_audio_classification.py) | Train `Audio` classifier & predict from raw waveforms |
| [`tabular/01_tabular_csv_and_regression.py`](tabular/01_tabular_csv_and_regression.py) | Classification & regression on tabular data |
| [`generative/01_vae_and_ddpm.py`](generative/01_vae_and_ddpm.py) | Train a VAE / DDPM and `.sample()` new images |
