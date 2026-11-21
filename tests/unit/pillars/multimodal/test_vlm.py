"""Unit tests for Multimodal (VLM) pillar."""

import torch

from kitepy import VLM


def test_vlm_contrastive_train_and_zero_shot(tmp_path):
    vlm = VLM("clip_tiny")
    vlm.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert vlm.is_trained

    probs = vlm.predict(torch.randn(1, 3, 64, 64), texts=["a cat", "a dog", "a car"])
    assert probs.shape == (1, 3)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)

    # PIL Image zero-shot prediction
    import numpy as np
    from PIL import Image
    pil_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    probs_pil = vlm.predict(pil_img, texts=["sky", "ocean"])
    assert probs_pil.shape == (1, 2)


def test_vlm_real_jsonl_manifest(tmp_path):
    import json
    import numpy as np
    from PIL import Image

    manifest = tmp_path / "pairs.jsonl"
    rows = []
    for i, caption in enumerate(["a red square", "a blue circle", "green grass", "night sky", "yellow sun"]):
        img_path = tmp_path / f"img_{i}.png"
        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(img_path)
        rows.append(json.dumps({"image": img_path.name, "text": caption}))
    manifest.write_text("\n".join(rows), encoding="utf-8")

    vlm = VLM("clip_tiny", img_size=32)
    vlm.train(str(manifest), epochs=1, batch_size=2, checkpoint_dir=str(tmp_path / "ck"), save_every_n_epochs=0)
    assert vlm.is_trained
