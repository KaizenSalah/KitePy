"""Unit tests for Language Pillar (LLM, TextClassifier, LoRA, text generation)."""

from kitepy import LLM, TextClassifier


def test_llm_causal_lm_train_and_generate(tmp_path):
    model = LLM("tiny_gpt")
    model.train(
        "synthetic",
        epochs=1,
        batch_size=8,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert model.is_trained
    assert "train_perplexity" in model.training_history[-1]

    out = model.generate("Hello", max_new_tokens=5)
    assert isinstance(out, str) and len(out) > 0


def test_llm_with_lora_and_text_list(tmp_path):
    texts = [
        "kitepy makes deep learning dead simple",
        "pytorch powers custom research",
        "transformers scale across modalities",
        "callbacks and adapters keep code clean",
        "training loops with mixed precision",
    ]
    model = LLM("tiny_gpt", lora=True)
    model.train(
        texts,
        epochs=1,
        batch_size=2,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert model.is_trained
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    assert trainable < total


def test_text_classifier(tmp_path):
    clf = TextClassifier("tiny_gpt", num_classes=2)
    clf.train(
        "synthetic",
        epochs=1,
        batch_size=8,
        checkpoint_dir=str(tmp_path),
        save_every_n_epochs=0,
    )
    assert clf.is_trained
    preds = clf.predict("this framework is awesome", top_k=2)
    assert len(preds) == 2

    batch_preds = clf.predict(["short", "much longer sequence with many more tokens"], top_k=2)
    assert len(batch_preds) == 2 and all(len(row) == 2 for row in batch_preds)


def test_hf_causal_lm_jsonl_instruction_finetune_offline(tmp_path):
    """End-to-end test of the JSONL ChatML/Alpaca + tokenizer propagation + LoRA + HF generate() pipeline."""
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    import torch.nn as nn
    from kitepy.pillars.language.data import SimpleCharTokenizer

    jsonl_path = tmp_path / "my_chat.jsonl"
    rows = [
        {"instruction": "What is KitePy?", "output": "Simple deep learning."},
        {"messages": [{"role": "user", "content": "What is KitePy?"}, {"role": "assistant", "content": "Simple deep learning."}]},
    ] * 3
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # Try real HuggingFace GPT2LMHeadModel first; if local env lacks scipy for transformers.generation,
    # fall back to a faithful HF-compatible CausalLM stub so the full KitePy HF path is still verified.
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        hf_dir = tmp_path / "tiny_hf_llm"
        vocab = {
            "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
            "###": 4, "Instruction:": 5, "Response:": 6, "What": 7,
            "is": 8, "KitePy?": 9, "Simple": 10, "deep": 11, "learning.": 12,
        }
        tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tok_backend.pre_tokenizer = Whitespace()
        fast_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok_backend,
            bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
        )
        fast_tok.save_pretrained(str(hf_dir))

        hf_cfg = GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=2, n_head=2)
        GPT2LMHeadModel(hf_cfg).save_pretrained(str(hf_dir))

        llm = LLM(str(hf_dir), pretrained=True, lora=True)
        llm.train(str(jsonl_path), epochs=2, batch_size=2, checkpoint_dir=str(tmp_path / "ck"), save_every_n_epochs=0)
        reply = llm.generate("What is KitePy?", max_new_tokens=6)
        assert isinstance(reply, str) and len(reply) > 0
    except ModuleNotFoundError:
        class _FakeHFCausalLM(nn.Module):
            def __init__(self, vocab_size=260, dim=32):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, dim)
                self.q_proj = nn.Linear(dim, dim)
                self.v_proj = nn.Linear(dim, dim)
                self.lm_head = nn.Linear(dim, vocab_size)

            def forward(self, input_ids, attention_mask=None, labels=None):
                h = self.v_proj(torch.relu(self.q_proj(self.emb(input_ids))))
                logits = self.lm_head(h)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits[:, :-1, :].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                return SimpleNamespace(loss=loss, logits=logits)

            def generate(self, input_ids, max_new_tokens=6, **kwargs):
                extra = torch.full((input_ids.size(0), max_new_tokens), 65, dtype=torch.long, device=input_ids.device)
                return torch.cat([input_ids, extra], dim=1)

        tok = SimpleCharTokenizer(vocab_size=260)
        with patch("kitepy.pillars.vision.wrappers.create_hf_model", return_value=_FakeHFCausalLM()), \
             patch("kitepy.pillars.vision.wrappers.create_hf_tokenizer", return_value=tok):
            llm = LLM("Qwen/Qwen2.5-0.5B", pretrained=True, lora=True)
            llm.train(str(jsonl_path), epochs=2, batch_size=2, checkpoint_dir=str(tmp_path / "ck"), save_every_n_epochs=0)
            assert llm.is_trained
            reply = llm.generate("What is KitePy?", max_new_tokens=6)
            assert isinstance(reply, str) and len(reply) > 0


def test_llm_streaming_jsonl(tmp_path):
    import json

    jsonl_path = tmp_path / "stream.jsonl"
    rows = [{"text": f"sample streaming line number {i}"} for i in range(10)]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    llm = LLM("tiny_gpt")
    llm.train(
        str(jsonl_path),
        epochs=1,
        max_steps=2,
        batch_size=2,
        streaming=True,
        checkpoint_dir=str(tmp_path / "ck"),
        save_every_n_epochs=0,
    )
    assert llm.is_trained
