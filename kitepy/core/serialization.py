"""
Atomic & Safe Checkpoint Serialization for KitePy.

Features:
- Crash-safe atomic file writes (temp file in target directory + os.replace)
- Pickle-free `safetensors` + JSON metadata bundle support
- Experiment lineage capture (KitePy version, PyTorch version, Python version, Git SHA, UTC timestamp)
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Tuple, Union

import torch


def get_lineage_metadata() -> Dict[str, Any]:
    """Capture reproducibility & institutional audit metadata."""
    git_sha = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if res.returncode == 0:
            git_sha = res.stdout.strip()
    except Exception:
        pass

    return {
        "kitepy_version": "0.2.1",
        "torch_version": str(torch.__version__),
        "python_version": str(sys.version.split()[0]),
        "git_sha": str(git_sha) if git_sha is not None else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def atomic_write_text(content: str, path: Union[str, Path]):
    """Atomically write text file using temp file + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=target.suffix, dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def atomic_torch_save(obj: Any, path: Union[str, Path]):
    """Atomically write a PyTorch checkpoint so crashes mid-save never corrupt files."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".pt", dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp_name)
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def save_safe_bundle(state_dict: Dict[str, torch.Tensor], metadata: Dict[str, Any], path: Union[str, Path]):
    """
    Save model weights with `safetensors` (zero pickle!) plus a `.meta.json` sidecar.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors is required for safe bundle export. Install with: pip install safetensors")

    target = Path(path)
    if target.suffix != ".safetensors":
        target = target.with_suffix(".safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)

    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".safetensors", dir=str(target.parent))
    os.close(fd)
    try:
        save_file(tensors, tmp_name)
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    meta_path = target.with_suffix(".meta.json")
    full_meta = {"lineage": get_lineage_metadata(), **metadata}
    atomic_write_text(json.dumps(full_meta, indent=2, default=str), meta_path)
    return target, meta_path


def load_safe_bundle(path: Union[str, Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load `safetensors` weights + `.meta.json` without executing any pickle code."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors is required. Install with: pip install safetensors")

    target = Path(path)
    state_dict = load_file(str(target), device="cpu")
    meta_path = target.with_suffix(".meta.json")
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return state_dict, metadata


__all__ = [
    "get_lineage_metadata",
    "atomic_write_text",
    "atomic_torch_save",
    "save_safe_bundle",
    "load_safe_bundle",
]
