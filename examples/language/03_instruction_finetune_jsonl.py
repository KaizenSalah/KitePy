"""
Instruction / Chat Fine-Tuning from a JSONL file in 3 lines.

Swap "tiny_gpt" for any HuggingFace LLM (e.g., "Qwen/Qwen2.5-0.5B", pretrained=True)
on GPU / Google Colab to fine-tune a real 0.5B-7B model with LoRA!
"""

import json
import tempfile
from pathlib import Path
from kitepy import LLM

chat_data = [
    {"messages": [{"role": "user", "content": "What is KitePy?"}, {"role": "assistant", "content": "KitePy is a dead simple PyTorch framework."}]},
    {"messages": [{"role": "user", "content": "How do I freeze a ViT?"}, {"role": "assistant", "content": "Call vit.freeze_backbone() before train()."}]},
    {"instruction": "Summarize KitePy in one phrase.", "output": "Dead simple deep learning with full PyTorch control."},
    {"instruction": "Name KitePy's escape hatch.", "output": "Use model.unwrap() or custom train_step hooks."},
    {"instruction": "Which modalities does KitePy support?", "output": "Vision, Language, Multimodal, Audio, Tabular, and Generative."},
] * 3

with tempfile.TemporaryDirectory() as tmp:
    jsonl_path = Path(tmp) / "my_chat.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in chat_data), encoding="utf-8")

    # On GPU / Colab, simply use:
    # llm = LLM("Qwen/Qwen2.5-0.5B", pretrained=True, lora=True)
    llm = LLM("tiny_gpt")
    llm.train(str(jsonl_path), epochs=6, batch_size=4, lr=3e-3, checkpoint_dir=tmp, save_every_n_epochs=0)

    print("Reply:", repr(llm.generate("<|user|>\nWhat is KitePy?\n<|assistant|>\n", max_new_tokens=30)))
