from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models import lightglue as lightglue_model
from onnx_export.constants import (
    DEFAULT_HEIGHT,
    DEFAULT_NUM_KEYPOINTS,
    DEFAULT_OPSET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WIDTH,
)
from onnx_export.pipeline import export_and_finalize
from onnx_export.shapes import lightglue_input_shapes


def normalize_keypoints(keypoints: torch.Tensor, image_shape: tuple) -> torch.Tensor:
    """Normalize keypoint locations for export tracing (batch, H, W layout)."""
    _, _, height, width = image_shape
    one = keypoints.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.5
    return (keypoints - center[:, None, :]) / scaling[:, None, :]


def build_dummy_inputs(
    num_keypoints: int,
    descriptor_dim: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_shape = (1, 1, height, width)
    x0 = torch.from_numpy(np.random.randint(0, width, size=(1, num_keypoints)))
    y0 = torch.from_numpy(np.random.randint(0, height, size=(1, num_keypoints)))
    keypoints0 = normalize_keypoints(torch.stack((x0, y0), 2).float(), image_shape)
    descriptors0 = torch.randn(1, num_keypoints, descriptor_dim)

    x1 = torch.from_numpy(np.random.randint(0, width, size=(1, num_keypoints)))
    y1 = torch.from_numpy(np.random.randint(0, height, size=(1, num_keypoints)))
    keypoints1 = normalize_keypoints(torch.stack((x1, y1), 2).float(), image_shape)
    descriptors1 = torch.randn(1, num_keypoints, descriptor_dim)
    return keypoints0, keypoints1, descriptors0, descriptors1


def resolve_output_path(args: argparse.Namespace, default_stem: str) -> Path:
    if args.output_file:
        return Path(args.output_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        name = args.output_name
    elif args.weight_file:
        name = f"{Path(args.weight_file).stem}.onnx"
    else:
        name = f"{default_stem}.onnx"
    return output_dir / name


def export_lightglue(args: argparse.Namespace) -> str:
    model = lightglue_model.LightGlue(
        args.features,
        depth_confidence=-1,
        width_confidence=-1,
        flash=args.flash,
        weights=args.weight_file,
    ).eval()
    print("Total number of params: ", sum(p.numel() for p in model.parameters()))
    descriptor_dim = model.conf.input_dim
    print("Input descriptor dimension: ", descriptor_dim)

    example = build_dummy_inputs(args.num_keypoints, descriptor_dim, args.height, args.width)
    default_stem = f"{args.features}_lightglue"
    onnx_path = resolve_output_path(args, default_stem)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "keypoints0": {1: "feature_number_0"},
            "keypoints1": {1: "feature_number_1"},
            "descriptors0": {1: "feature_number_0"},
            "descriptors1": {1: "feature_number_1"},
        }

    input_shapes = (
        lightglue_input_shapes(args.num_keypoints, descriptor_dim)
        if args.simplify and not args.dynamic
        else None
    )

    return export_and_finalize(
        model,
        example,
        onnx_path,
        input_names=["keypoints0", "keypoints1", "descriptors0", "descriptors1"],
        output_names=["scores"],
        dynamic_axes=dynamic_axes,
        simplify=args.simplify,
        input_shapes=input_shapes,
        skip_simplify=args.dynamic,
        opset_version=args.opset,
        verbose=args.verbose,
    )


def add_lightglue_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        default="superpoint",
        choices=("superpoint", "disk", "aliked", "sift"),
        help="feature type for LightGlue weights and descriptor dimension",
    )
    parser.add_argument("--weight-file", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename under --output-dir (default: {features}_lightglue.onnx)",
    )
    parser.add_argument("--num-keypoints", type=int, default=DEFAULT_NUM_KEYPOINTS)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--flash", action="store_true")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="export dynamic keypoint-count axes",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="skip onnxsim static-shape simplification",
    )
    parser.set_defaults(simplify=True)
