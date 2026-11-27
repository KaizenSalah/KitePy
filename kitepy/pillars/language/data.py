"""
Language data loading, chat/instruction formatting, and tokenization pipeline.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from kitepy.core.config import DataConfig, TrainingConfig


class SyntheticCausalLMDataset(Dataset):
    """Synthetic token sequences for offline causal LM testing."""

    def __init__(self, num_samples: int = 64, seq_len: int = 32, vocab_size: int = 260):
        self.tokens = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        ids = self.tokens[idx]
        return {"input_ids": ids, "labels": ids.clone()}


class SyntheticTextClassificationDataset(Dataset):
    """Synthetic token sequences + labels for offline sequence classification testing."""

    def __init__(self, num_samples: int = 64, seq_len: int = 32, vocab_size: int = 260, num_classes: int = 2):
        self.tokens = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {"input_ids": self.tokens[idx], "labels": self.labels[idx]}


class SimpleCharTokenizer:
    """Lossless byte-level UTF-8 fallback tokenizer (vocab: 0=PAD, 1=BOS, 2=EOS, 3..258=bytes)."""

    def __init__(self, vocab_size: int = 260):
        self.vocab_size = max(260, vocab_size)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def encode(self, text: str, max_length: int = 512) -> List[int]:
        raw_bytes = list(text.encode("utf-8"))[: max(0, max_length - 2)]
        return [self.bos_token_id] + [b + 3 for b in raw_bytes] + [self.eos_token_id]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        raw_bytes = bytearray()
        for i in ids:
            if i in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            b = i - 3
            if 0 <= b <= 255:
                raw_bytes.append(b)
        return raw_bytes.decode("utf-8", errors="replace")

    def __call__(
        self,
        texts,
        max_length: int = 512,
        padding: Any = True,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t, max_length=max_length) for t in texts]
        if padding == "max_length":
            target_len = max_length
        elif padding:
            target_len = max(len(x) for x in encoded)
        else:
            target_len = None

        if target_len is not None:
            encoded = [
                (x[:target_len] if len(x) > target_len else x + [self.pad_token_id] * (target_len - len(x)))
                for x in encoded
            ]
        tensor = torch.tensor(encoded, dtype=torch.long)
        attn = (tensor != self.pad_token_id).long()
        return {"input_ids": tensor, "attention_mask": attn}


def format_chat_record(record: Any, tokenizer: Optional[Any] = None) -> str:
    """
    Convert a JSON/JSONL dict record into a formatted training string.
    Supports:
    - OpenAI/ChatML: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    - Alpaca/Instruction: {"instruction": "...", "input": "...", "output": "..."} (or "response")
    - QA / Prompt-Completion: {"prompt": "...", "completion": "..."} or {"question": "...", "answer": "..."}
    - Raw text: {"text": "..."} or {"content": "..."}
    """
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return str(record)

    if "messages" in record and isinstance(record["messages"], list):
        msgs = record["messages"]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                return str(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            except Exception:
                pass
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    if "instruction" in record:
        instr = record["instruction"]
        inp = record.get("input", "")
        out = record.get("output", record.get("response", ""))
        if inp:
            return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        return f"### Instruction:\n{instr}\n\n### Response:\n{out}"

    if "prompt" in record and ("completion" in record or "response" in record):
        resp = record.get("completion", record.get("response", ""))
        return f"{record['prompt']}\n{resp}"

    if "question" in record and "answer" in record:
        return f"Question: {record['question']}\nAnswer: {record['answer']}"

    if "text" in record:
        return str(record["text"])
    if "content" in record:
        return str(record["content"])

    return json.dumps(record, ensure_ascii=False)


def _load_records_from_file(path: Path) -> Tuple[List[str], Optional[List[int]]]:
    """Load .txt, .jsonl, or .json records cleanly."""
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")

    if suffix == ".jsonl":
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    elif suffix == ".json":
        parsed = json.loads(raw_text)
        records = parsed if isinstance(parsed, list) else [parsed]
    else:
        return [line.strip() for line in raw_text.splitlines() if line.strip()], None

    labels = None
    if records and isinstance(records[0], dict) and "label" in records[0]:
        labels = [int(r["label"]) for r in records]
    return records, labels


class TextListDataset(Dataset):
    """Dataset wrapping strings or chat/instruction dicts for causal LM or text classification."""

    def __init__(
        self,
        items: List[Any],
        labels: Optional[List[int]],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.texts = [format_chat_record(it, tokenizer=tokenizer) for it in items]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        if input_ids.ndim == 0:
            input_ids = input_ids.unsqueeze(0)
        mask = enc.get("attention_mask")
        if mask is not None:
            mask = mask.squeeze(0)
        else:
            mask = torch.ones_like(input_ids)

        item: Dict[str, torch.Tensor] = {
            "input_ids": input_ids[: self.max_length],
            "attention_mask": mask[: self.max_length],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            item["labels"] = item["input_ids"].clone()
        return item


class StreamingJsonlDataset(IterableDataset):
    """True constant-memory streaming IterableDataset over local .jsonl / .txt files."""

    def __init__(self, file_path: Path, tokenizer: Any, max_length: int = 512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        is_jsonl = self.file_path.suffix.lower() == ".jsonl"
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s) if is_jsonl else s
                text = format_chat_record(rec, tokenizer=self.tokenizer)
                enc = self.tokenizer(text, max_length=self.max_length, padding=False, truncation=True, return_tensors="pt")
                ids = enc["input_ids"].squeeze(0)[: self.max_length]
                mask = enc.get("attention_mask")
                mask = mask.squeeze(0)[: self.max_length] if mask is not None else torch.ones_like(ids)
                yield {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}


def make_language_collate_fn(pad_token_id: int, is_causal_lm: bool = True):
    """Dynamic batch padding collate function (pads only to max sequence length in the batch)."""

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids_list, mask_list, labels_list = [], [], []

        for b in batch:
            seq_len = b["input_ids"].size(0)
            pad_len = max_len - seq_len
            if pad_len > 0:
                ids = torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                mask = torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            else:
                ids = b["input_ids"]
                mask = b["attention_mask"]

            input_ids_list.append(ids)
            mask_list.append(mask)

            if is_causal_lm:
                lbl = torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]) if pad_len > 0 else b["labels"].clone()
                lbl[mask == 0] = -100
                labels_list.append(lbl)
            else:
                labels_list.append(b["labels"])

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }

    return collate_fn


def load_language_data(
    dataset_name: Any,
    data_config: DataConfig,
    train_config: TrainingConfig,
    tokenizer: Optional[Any] = None,
    vocab_size: int = 260,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    """
    Load language data from:
    - 'synthetic'
    - .jsonl / .json / .txt file paths (supports ChatML messages, Alpaca instruction/output, prompt/completion, text)
    - Python list of strings or chat/instruction dicts
    - HuggingFace datasets (e.g. 'wikitext', 'hf://...')
    """
    bs = train_config.batch_size if isinstance(train_config.batch_size, int) else 16
    task = (getattr(train_config, "task", None) or "causal_lm").lower()
    is_causal_lm = task not in ("classification", "sequence_classification")
    max_len = getattr(data_config, "max_length", 512)
    tok = tokenizer or getattr(data_config, "_tokenizer", None) or SimpleCharTokenizer(vocab_size=vocab_size)
    pad_id = getattr(tok, "pad_token_id", 0)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", 0) or 0
    metadata: Dict[str, Any] = {}

    use_dynamic_collate = True
    train_ds: Dataset
    val_ds: Optional[Dataset] = None

    if dataset_name == "synthetic" or dataset_name is None:
        use_dynamic_collate = False
        if not is_causal_lm:
            train_ds = SyntheticTextClassificationDataset(num_samples=64, seq_len=32, vocab_size=vocab_size, num_classes=2)
            val_ds = SyntheticTextClassificationDataset(num_samples=32, seq_len=32, vocab_size=vocab_size, num_classes=2)
            metadata["num_classes"] = 2
        else:
            train_ds = SyntheticCausalLMDataset(num_samples=64, seq_len=32, vocab_size=vocab_size)
            val_ds = SyntheticCausalLMDataset(num_samples=32, seq_len=32, vocab_size=vocab_size)

    elif isinstance(dataset_name, list):
        labels = None
        if dataset_name and isinstance(dataset_name[0], dict) and "label" in dataset_name[0]:
            labels = [int(x["label"]) for x in dataset_name]
            metadata["num_classes"] = len(set(labels))
        n_val = max(1, int(len(dataset_name) * 0.2)) if len(dataset_name) >= 5 else 0
        train_items = dataset_name[n_val:] if n_val else dataset_name
        val_items = dataset_name[:n_val] if n_val else None
        train_lbls = labels[n_val:] if (labels and n_val) else labels
        val_lbls = labels[:n_val] if (labels and n_val) else None
        train_ds = TextListDataset(train_items, train_lbls, tok, max_length=max_len)
        val_ds = TextListDataset(val_items, val_lbls, tok, max_length=max_len) if val_items else None

    elif isinstance(dataset_name, str) and Path(dataset_name).is_file():
        p = Path(dataset_name)
        if getattr(data_config, "streaming", False) and p.suffix.lower() in (".jsonl", ".txt"):
            train_ds = StreamingJsonlDataset(p, tok, max_length=max_len)
            collate = make_language_collate_fn(pad_id, is_causal_lm=is_causal_lm)
            return DataLoader(train_ds, batch_size=bs, num_workers=0, collate_fn=collate), None, None, metadata
        records, labels = _load_records_from_file(p)
        if labels:
            metadata["num_classes"] = len(set(labels))
        n_val = max(1, int(len(records) * 0.2)) if len(records) >= 5 else 0
        train_items = records[n_val:] if n_val else records
        val_items = records[:n_val] if n_val else None
        train_lbls = labels[n_val:] if (labels and n_val) else labels
        val_lbls = labels[:n_val] if (labels and n_val) else None
        train_ds = TextListDataset(train_items, train_lbls, tok, max_length=max_len)
        val_ds = TextListDataset(val_items, val_lbls, tok, max_length=max_len) if val_items else None

    elif isinstance(dataset_name, str):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets required for HuggingFace text datasets. Install with: pip install datasets")
        name = dataset_name[5:] if dataset_name.startswith("hf://") else dataset_name
        if name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        else:
            ds = load_dataset(name)
        train_texts = [x for x in ds["train"]["text"] if x and x.strip()][:512]
        val_split = ds.get("validation") or ds.get("test")
        val_texts = [x for x in val_split["text"] if x and x.strip()][:128] if val_split else None
        train_ds = TextListDataset(train_texts, None, tok, max_length=max_len)
        val_ds = TextListDataset(val_texts, None, tok, max_length=max_len) if val_texts else None
    else:
        raise ValueError(f"Unsupported language dataset: {type(dataset_name)}")

    collate = make_language_collate_fn(pad_id, is_causal_lm=is_causal_lm) if use_dynamic_collate else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate) if val_ds is not None else None
    return train_loader, val_loader, None, metadata


__all__ = [
    "SyntheticCausalLMDataset",
    "SyntheticTextClassificationDataset",
    "SimpleCharTokenizer",
    "TextListDataset",
    "format_chat_record",
    "make_language_collate_fn",
    "load_language_data",
]
