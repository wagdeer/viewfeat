from __future__ import annotations

import argparse
from pathlib import Path

import torch

from onnx_export.constants import (
    COSPLACE_BACKBONE,
    COSPLACE_DESCRIPTOR_DIM,
    COSPLACE_INPUT_SIZE,
    COSPLACE_ONNX,
    DEFAULT_OPSET,
    DEFAULT_OUTPUT_DIR,
)
from onnx_export.pipeline import export_and_finalize


def load_cosplace_model(backbone: str, descriptor_dim: int) -> torch.nn.Module:
    """Load pretrained CosPlace from torch.hub (requires network on first run)."""
    try:
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=descriptor_dim,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load CosPlace from torch.hub. "
            "Install torchvision and ensure network access for the first download. "
            f"Original error: {exc!r}"
        ) from exc
    return model.eval()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name or COSPLACE_ONNX
    return output_dir / name


def export_cosplace(args: argparse.Namespace) -> str:
    model = load_cosplace_model(args.backbone, args.descriptor_dim)
    print("Total number of params: ", sum(p.numel() for p in model.parameters()))

    size = args.input_size
    example = torch.randn(1, 3, size, size)
    onnx_path = resolve_output_path(args)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"images": {2: "image_height", 3: "image_width"}}

    input_shapes = (
        {"images": [1, 3, size, size]} if args.simplify and not args.dynamic else None
    )

    return export_and_finalize(
        model,
        example,
        onnx_path,
        input_names=["images"],
        output_names=["descriptors"],
        dynamic_axes=dynamic_axes,
        simplify=args.simplify,
        input_shapes=input_shapes,
        skip_simplify=args.dynamic,
        opset_version=args.opset,
        verbose=args.verbose,
    )


def add_cosplace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="ONNX output directory",
    )
    parser.add_argument("--output-file", type=Path, default=None, help="Explicit ONNX output path")
    parser.add_argument(
        "--output-name",
        default=COSPLACE_ONNX,
        help=f"Output filename under --output-dir (default: {COSPLACE_ONNX})",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=COSPLACE_INPUT_SIZE,
        help="Square input side length in pixels (default: 320)",
    )
    parser.add_argument(
        "--backbone",
        default=COSPLACE_BACKBONE,
        help="CosPlace backbone passed to torch.hub (default: ResNet18)",
    )
    parser.add_argument(
        "--descriptor-dim",
        type=int,
        default=COSPLACE_DESCRIPTOR_DIM,
        help="Global descriptor dimension (default: 512)",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="export dynamic image height/width (not used by default zsi_swarm runtime)",
    )
    parser.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="skip onnxsim static-shape simplification",
    )
    parser.set_defaults(simplify=True)
