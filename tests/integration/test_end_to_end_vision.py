"""Integration tests: complete end-to-end Vision pipelines."""

from kitepy import CNN, CSVLogger, EarlyStopping


def test_full_folder_to_save_reload_predict_and_evaluate(sample_image_folder, tmp_path):
    ckpt_file = tmp_path / "final.pt"
    log_dir = tmp_path / "logs"

    # 1. One-liner train on local ImageFolder (auto-detects 3 classes, class names, splits cleanly)
    model = CNN("resnet18", img_size=64)
    model.train(
        str(sample_image_folder),
        epochs=2,
        batch_size=4,
        num_workers=0,
        augmentation="light",
        ema=True,
        checkpoint_dir=str(tmp_path / "ckpts"),
        csv_log_dir=str(log_dir),
        callbacks=[EarlyStopping(patience=5)],
    )
    assert model.is_trained
    assert model.config.num_classes == 3
    assert len(model.training_history) == 2
    assert (log_dir / "metrics.csv").exists()

    # 2. Evaluate
    metrics = model.evaluate(str(sample_image_folder), batch_size=4, num_workers=0)
    assert "test_loss" in metrics and "test_accuracy" in metrics

    # 3. Save, reload, predict directly from an image file path
    model.save(ckpt_file)
    reloaded = CNN.load(ckpt_file)
    assert reloaded.class_names == ["birds", "cats", "dogs"]

    sample_img = next((sample_image_folder / "cats").glob("*.png"))
    preds = reloaded.predict(str(sample_img), top_k=3)
    assert len(preds) == 3
    assert abs(sum(p["confidence"] for p in preds) - 1.0) < 1e-4

    # 4. Export to TorchScript with inferred shape
    ts_path = tmp_path / "exported.pt"
    reloaded.export(ts_path, format="torchscript", verify=True)
    assert ts_path.exists()
