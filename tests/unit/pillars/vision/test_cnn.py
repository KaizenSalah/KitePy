"""Unit tests for CNN API (auto-class inference, real-world predict, save/load, export, transfer learning)."""

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from kitepy import CNN, CNNConfig


def test_cnn_creation_and_configs():
    m = CNN("resnet18")
    assert m.model_name == "resnet18" and m.modality == "vision"

    m_cfg = CNN("resnet18", config=CNNConfig(num_classes=12, dropout=0.3))
    assert m_cfg.config.num_classes == 12

    m_kw = CNN("resnet18", num_classes=7, epochs=3)
    assert m_kw.config.num_classes == 7
    assert m_kw.train_config.epochs == 3


def test_cnn_auto_infers_num_classes_from_folder(sample_image_folder, tmp_path):
    # No num_classes specified! Should auto-detect 3 classes from sample_image_folder
    model = CNN("resnet18")
    model.train(
        str(sample_image_folder),
        epochs=1,
        batch_size=4,
        fast_dev_run=True,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        save_every_n_epochs=0,
    )
    assert model.config.num_classes == 3
    assert model.class_names == ["birds", "cats", "dogs"]
    assert model.is_trained is True


def test_cnn_predict_tensor_pil_numpy_and_file(tmp_path):
    model = CNN("resnet18", num_classes=3)
    model.class_names = ["birds", "cats", "dogs"]
    model._build_model()

    # 1. 4D Tensor
    out1 = model.predict(torch.randn(2, 3, 224, 224))
    assert out1.shape == (2, 3)

    # 2. 3D Tensor (auto-unsqueezed)
    out2 = model.predict(torch.randn(3, 224, 224))
    assert out2.shape == (1, 3)

    # 3. PIL Image + top_k structured output
    pil_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    preds = model.predict(pil_img, top_k=2)
    assert len(preds) == 2
    assert preds[0]["label"] in model.class_names

    # 4. File path & list of paths
    img_path = tmp_path / "test.png"
    pil_img.save(img_path)
    out_file = model.predict(str(img_path), return_probs=True)
    assert out_file.shape == (1, 3)
    assert torch.isclose(out_file.sum(), torch.tensor(1.0), atol=1e-4)

    out_list = model.predict([str(img_path), pil_img])
    assert out_list.shape == (2, 3)


def test_cnn_save_load_export_and_quantize(tmp_path):
    model = CNN("resnet18", num_classes=4)
    model.class_names = ["a", "b", "c", "d"]
    model._build_model()

    ckpt_path = tmp_path / "model.pt"
    model.save(ckpt_path)
    loaded = CNN.load(ckpt_path)
    assert loaded.class_names == ["a", "b", "c", "d"]
    assert loaded.config.num_classes == 4

    # TorchScript export with verification
    ts_path = tmp_path / "scripted.pt"
    model.export(ts_path, format="torchscript", verify=True)
    assert ts_path.exists()

    # SafeTensors export
    st_path = tmp_path / "weights.safetensors"
    model.export(st_path, format="safetensors")
    assert st_path.exists()

    # Quantize
    model.quantize(bits=8)
    assert model.predict(torch.randn(1, 3, 224, 224)).shape == (1, 4)


def test_cnn_custom_model_and_errors():
    net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    )
    m = CNN(net)
    m.train("synthetic", epochs=1, fast_dev_run=True, num_workers=0, save_every_n_epochs=0)
    assert m.is_trained
    assert isinstance(m.unwrap(), nn.Module)

    with pytest.raises(ValueError):
        bad = CNN("nonexistent_model_xyz")
        bad._build_model()


def test_all_vision_presets_exist_in_timm():
    import timm
    from kitepy.core.presets import CNN_PRESETS, TRANSFORMER_PRESETS
    from kitepy.pillars.vision.wrappers import TIMM_NAME_MAPPING

    avail = set(timm.list_models())
    for preset_name in list(CNN_PRESETS) + list(TRANSFORMER_PRESETS):
        resolved = TIMM_NAME_MAPPING.get(preset_name, preset_name)
        assert resolved in avail, f"Preset '{preset_name}' -> '{resolved}' missing from timm"
