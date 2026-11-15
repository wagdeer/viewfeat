from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.aliked import ALIKED
from models.ops import register_deform_conv2d_onnx_op
from onnx_export.constants import (
    CHECKPOINT_URL,
    DEFAULT_HEIGHT,
    DEFAULT_OPSET,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOP_K,
    DEFAULT_WIDTH,
    topk_label,
)
from onnx_export.pipeline import export_and_finalize


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model_name}-{topk_label(args.top_k)}-{args.width}x{args.height}.onnx"
    return output_dir / name


def load_state_dict(args: argparse.Namespace):
    if args.weight_file:
        return torch.load(args.weight_file, map_location=args.device)
    return torch.hub.load_state_dict_from_url(
        CHECKPOINT_URL.format(args.model_name),
        map_location=args.device,
    )


def load_model(args: argparse.Namespace) -> ALIKED:
    model = ALIKED(
        model_name=args.model_name,
        device=args.device,
        top_k=args.top_k,
        scores_th=args.scores_th,
        n_limit=args.n_limit,
        load_pretrained=False,
    )
    state_dict = load_state_dict(args)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device).eval()
    return model


def export_aliked(args: argparse.Namespace) -> str:
    if args.input_size is not None:
        args.height = args.input_size
        args.width = args.input_size

    if args.top_k <= 0:
        print(
            "Warning: top_k <= 0 produces variable keypoint counts "
            "and is not the TensorRT default path."
        )

    register_deform_conv2d_onnx_op(use_gathernd=not args.use_gather_elements)

    model = load_model(args)
    print("Total number of params: ", sum(p.numel() for p in model.parameters()))

    example = torch.randn(1, 3, args.height, args.width, device=args.device)
    onnx_path = resolve_output_path(args)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"image": {2: "height", 3: "width"}}

    input_shapes = {"image": [1, 3, args.height, args.width]} if args.simplify and not args.dynamic else None

    return export_and_finalize(
        model,
        example,
        onnx_path,
        input_names=["image"],
        output_names=["keypoints", "descriptors", "scores"],
        dynamic_axes=dynamic_axes,
        simplify=args.simplify,
        input_shapes=input_shapes,
        skip_simplify=args.dynamic,
        opset_version=args.opset,
        verbose=args.verbose,
        amp=args.amp,
        device=args.device,
    )


def add_aliked_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        default="aliked-n16rot",
        choices=("aliked-t16", "aliked-n16", "aliked-n16rot", "aliked-n32"),
    )
    parser.add_argument("--weight-file", default=None, help="Optional local .pth file")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="deprecated square input size; sets height and width when provided",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--scores-th", type=float, default=0.2)
    parser.add_argument("--n-limit", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--amp", action="store_true")
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
    parser.add_argument(
        "--use-gather-elements",
        action="store_true",
        help="use GatherElements instead of GatherND in deform_conv2d symbolic",
    )
    parser.set_defaults(simplify=True)
