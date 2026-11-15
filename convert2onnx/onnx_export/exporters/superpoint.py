from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import superpoint as superpoint_model
from onnx_export.constants import (
    DEFAULT_HEIGHT,
    DEFAULT_OPSET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WIDTH,
    SUPERPOINT_ONNX,
    SUPERPOINT_WEIGHT,
)
from onnx_export.pipeline import export_and_finalize


def build_superpoint_config(weight_file: str) -> dict:
    return {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "weights": weight_file,
    }


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        name = args.output_name
    else:
        weight_stem = Path(args.weight_file).stem
        name = f"{weight_stem}.onnx" if weight_stem else SUPERPOINT_ONNX
    return output_dir / name


def export_superpoint(args: argparse.Namespace) -> str:
    model = superpoint_model.SuperPoint(build_superpoint_config(args.weight_file)).eval()
    print("Total number of params: ", sum(p.numel() for p in model.parameters()))

    example = torch.randn(1, 1, args.height, args.width)
    onnx_path = resolve_output_path(args)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"image": {2: "image_height", 3: "image_width"}}

    input_shapes = {"image": [1, 1, args.height, args.width]} if args.simplify and not args.dynamic else None

    return export_and_finalize(
        model,
        example,
        onnx_path,
        input_names=["image"],
        output_names=["scores", "descriptors"],
        dynamic_axes=dynamic_axes,
        simplify=args.simplify,
        input_shapes=input_shapes,
        skip_simplify=args.dynamic,
        opset_version=args.opset,
        verbose=args.verbose,
    )


def add_superpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weight-file",
        default=str(DEFAULT_OUTPUT_DIR / SUPERPOINT_WEIGHT),
        help="PyTorch weight file (.pth); auto-downloaded from LightGlue if missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="ONNX output directory",
    )
    parser.add_argument("--output-file", type=Path, default=None, help="Explicit ONNX output path")
    parser.add_argument(
        "--output-name",
        default=None,
        help=f"Output filename under --output-dir (default: derived from weights or {SUPERPOINT_ONNX})",
    )
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="export dynamic image height/width",
    )
    parser.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="skip onnxsim static-shape simplification",
    )
    parser.set_defaults(simplify=True)
