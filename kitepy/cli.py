"""
KitePy CLI:
- kitepy doctor
- kitepy models [modality]
- kitepy datasets [modality]
- kitepy train <config.yaml> [--data ...] [--epochs ...]
- kitepy predict <checkpoint.pt> <input> [--top-k 5]
- kitepy export <checkpoint.pt> <out_path> [--format auto]
- kitepy serve <checkpoint.pt> [--host 0.0.0.0 --port 8000]
"""

import argparse
import json
import sys
from typing import List, Optional

from kitepy.core.presets import list_datasets, list_models
from kitepy.core.utils import print_device_info, print_version_info


def _resolve_model_cls(modality: str, arch: str = ""):
    from kitepy import CNN, LLM, VLM, Audio, Generative, Tabular, Transformer, Vision

    if modality == "language":
        return LLM
    if modality == "multimodal":
        return VLM
    if modality == "audio":
        return Audio
    if modality == "tabular":
        return Tabular
    if modality == "generative":
        return Generative
    if arch in ("vit", "deit", "swin"):
        return Transformer
    return CNN


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="kitepy", description="KitePy - Dead Simple Deep Learning CLI")
    sub = parser.add_subparsers(dest="command")

    # doctor
    sub.add_parser("doctor", help="Print environment and GPU/CPU diagnostics")

    # models
    p_models = sub.add_parser("models", help="List available model presets")
    p_models.add_argument("modality", nargs="?", default="all", help="Modality filter (cnn, transformer, llm, vlm, audio, tabular, generative, all)")

    # datasets
    p_data = sub.add_parser("datasets", help="List available dataset presets")
    p_data.add_argument("modality", nargs="?", default="all")

    # train
    p_train = sub.add_parser("train", help="Train a model from a YAML/JSON config")
    p_train.add_argument("config", help="Path to YAML/JSON config file")
    p_train.add_argument("--data", default=None, help="Dataset name or path (overrides config)")
    p_train.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p_train.add_argument("--modality", default="vision", help="Model modality")
    p_train.add_argument("--model", default="resnet18", help="Model preset name")
    p_train.add_argument("--save", default=None, help="Optional output checkpoint path")

    # predict
    p_pred = sub.add_parser("predict", help="Run inference from a saved checkpoint")
    p_pred.add_argument("checkpoint", help="Path to saved .pt checkpoint")
    p_pred.add_argument("input", help="Input file path (e.g., image path)")
    p_pred.add_argument("--modality", default="vision")
    p_pred.add_argument("--top-k", type=int, default=5)

    # export
    p_exp = sub.add_parser("export", help="Export a saved checkpoint to TorchScript/ONNX/SafeTensors")
    p_exp.add_argument("checkpoint", help="Path to saved .pt checkpoint")
    p_exp.add_argument("output", help="Output path (.pt, .onnx, .safetensors)")
    p_exp.add_argument("--modality", default="vision")
    p_exp.add_argument("--format", default="auto")

    # serve
    p_serve = sub.add_parser("serve", help="Serve a saved checkpoint over FastAPI")
    p_serve.add_argument("checkpoint", help="Path to saved .pt checkpoint")
    p_serve.add_argument("--modality", default="vision")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "doctor":
        print_version_info()
        print_device_info()
        return 0

    if args.command == "models":
        print(json.dumps(list_models(args.modality), indent=2))
        return 0

    if args.command == "datasets":
        print(json.dumps(list_datasets(args.modality), indent=2))
        return 0

    if args.command == "train":
        cls = _resolve_model_cls(args.modality)
        model = cls(args.model, config=args.config)
        train_kwargs = {}
        if args.epochs is not None:
            train_kwargs["epochs"] = args.epochs
        model.train(args.data, **train_kwargs)
        if args.save:
            model.save(args.save)
        return 0

    if args.command == "predict":
        cls = _resolve_model_cls(args.modality)
        model = cls.load(args.checkpoint)
        preds = model.predict(args.input, top_k=args.top_k)
        print(json.dumps(preds, indent=2))
        return 0

    if args.command == "export":
        cls = _resolve_model_cls(args.modality)
        model = cls.load(args.checkpoint)
        model.export(args.output, format=args.format)
        return 0

    if args.command == "serve":
        import uvicorn
        cls = _resolve_model_cls(args.modality)
        model = cls.load(args.checkpoint)
        app = model.to_fastapi()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
