"""Unit tests for Vision data loading (ImageFolder val transform isolation, transforms, synthetic)."""

from kitepy.core.config import DataConfig, TrainingConfig
from kitepy.pillars.vision.data import get_train_transforms, get_val_transforms, load_vision_data_with_meta


def test_custom_folder_does_not_leak_train_augmentations_to_val(sample_image_folder):
    data_cfg = DataConfig(augmentation="heavy", val_split=0.25, img_size=64)
    train_cfg = TrainingConfig(batch_size=4, num_workers=0, pin_memory=False)
    train_loader, val_loader, _, meta = load_vision_data_with_meta(
        str(sample_image_folder), data_cfg, train_cfg
    )
    assert meta["num_classes"] == 3
    assert meta["class_names"] == ["birds", "cats", "dogs"]
    assert val_loader is not None

    # Verify underlying val ImageFolder uses val_transform (no RandomResizedCrop/RandomRotation)
    val_tf_names = [type(t).__name__ for t in val_loader.dataset.dataset.transform.transforms]
    train_tf_names = [type(t).__name__ for t in train_loader.dataset.dataset.transform.transforms]
    assert "RandomRotation" in train_tf_names
    assert "RandomRotation" not in val_tf_names


def test_transforms_handle_none_and_all_strategies():
    for aug in ["none", "light", "auto", "heavy", "randaugment", "trivialaugment", "autoaugment"]:
        tf = get_train_transforms(128, aug)
        assert len(tf.transforms) >= 3
    vtf = get_val_transforms(128)
    assert len(vtf.transforms) == 3
