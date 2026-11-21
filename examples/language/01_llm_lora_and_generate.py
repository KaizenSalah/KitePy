"""Language Quickstart: Train a Small Language Model (SLM) from scratch & generate text."""

import tempfile
from kitepy import LLM

corpus = [
    "kitepy makes deep learning dead simple",
    "kitepy makes training models fast and clean",
    "progressive disclosure keeps beginners fast and experts free",
    "pytorch powers the underlying execution engine",
    "callbacks and adapters scale cleanly across tasks",
] * 4  # Repeat mini corpus so 15 epochs fit in ~2 seconds on CPU

with tempfile.TemporaryDirectory() as tmp:
    # Train a tiny GPT-style SLM from scratch (pass lora=True when fine-tuning pretrained HF models)
    llm = LLM("tiny_gpt")
    llm.train(corpus, epochs=15, batch_size=4, lr=3e-3, checkpoint_dir=tmp, save_every_n_epochs=0)

    sample = llm.generate("kitepy makes", max_new_tokens=30, temperature=0.5)
    print("Generated:", repr(sample))
