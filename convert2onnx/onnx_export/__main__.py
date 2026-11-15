"""Unified CLI: python -m onnx_export <command> ..."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from onnx_export.compare import add_compare_arguments, run_compare
from onnx_export.constants import (
    ALIKED_ENGINE_SUFFIX,
    ALIKED_MAX_BUILDER_OPTIMIZATION_LEVEL,
    ALIKED_LIGHTGLUE_ENGINE,
    ALIKED_LIGHTGLUE_ONNX,
    ALIKED_MODEL,
    ALIKED_ONNX_SUFFIX,
    COSPLACE_BACKBONE,
    COSPLACE_DESCRIPTOR_DIM,
    COSPLACE_ENGINE,
    COSPLACE_INPUT_SIZE,
    COSPLACE_ONNX,
    EIGENPLACES_BACKBONE,
    EIGENPLACES_DESCRIPTOR_DIM,
    EIGENPLACES_ENGINE,
    EIGENPLACES_INPUT_SIZE,
    EIGENPLACES_ONNX,
    DEFAULT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_KEYPOINTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOP_K,
    DEFAULT_WIDTH,
    FLEXGLUE_DESCRIPTOR_ENGINE,
    FLEXGLUE_DESCRIPTOR_ONNX,
    FLEXGLUE_MATCHER_ENGINE,
    FLEXGLUE_MATCHER_ONNX,
    SUPERPOINT_ENGINE,
    SUPERPOINT_LIGHTGLUE_ENGINE,
    SUPERPOINT_LIGHTGLUE_ONNX,
    SUPERPOINT_ONNX,
    aliked_basename,
)
from onnx_export.exporters.aliked import add_aliked_arguments, export_aliked
from onnx_export.exporters.cosplace import add_cosplace_arguments, export_cosplace
from onnx_export.exporters.eigenplaces import add_eigenplaces_arguments, export_eigenplaces
from onnx_export.exporters.flexglue import add_flexglue_arguments, run_prepare_flexglue
from onnx_export.exporters.lightglue import add_lightglue_arguments, export_lightglue
from onnx_export.exporters.superpoint import add_superpoint_arguments, export_superpoint

TRT_MODEL_CHOICES = (
    "superpoint",
    "aliked",
    "superpoint_lightglue",
    "aliked_lightglue",
    "cosplace",
    "eigenplaces",
    "flexglue_descriptor",
    "flexglue_matcher",
)


def _trt_job_key(onnx_name: str, aliked_stem: str) -> str:
    if onnx_name == COSPLACE_ONNX:
        return "cosplace"
    if onnx_name == EIGENPLACES_ONNX:
        return "eigenplaces"
    if onnx_name == FLEXGLUE_DESCRIPTOR_ONNX:
        return "flexglue_descriptor"
    if onnx_name == FLEXGLUE_MATCHER_ONNX:
        return "flexglue_matcher"
    if onnx_name == SUPERPOINT_ONNX:
        return "superpoint"
    if onnx_name == SUPERPOINT_LIGHTGLUE_ONNX:
        return "superpoint_lightglue"
    if onnx_name == ALIKED_LIGHTGLUE_ONNX:
        return "aliked_lightglue"
    if onnx_name == aliked_stem + ALIKED_ONNX_SUFFIX:
        return "aliked"
    return onnx_name


def run_trt(args: argparse.Namespace) -> int:
    weights_dir = Path(args.weights_dir).resolve()
    if not weights_dir.is_dir():
        raise SystemExit(f"weights directory not found: {weights_dir}")

    aliked_stem = aliked_basename(args.model_name, args.top_k, args.width, args.height)
    jobs = [
        (SUPERPOINT_ONNX, SUPERPOINT_ENGINE, []),
        (aliked_stem + ALIKED_ONNX_SUFFIX, aliked_stem + ALIKED_ENGINE_SUFFIX, []),
        (SUPERPOINT_LIGHTGLUE_ONNX, SUPERPOINT_LIGHTGLUE_ENGINE, []),
        (ALIKED_LIGHTGLUE_ONNX, ALIKED_LIGHTGLUE_ENGINE, []),
        (COSPLACE_ONNX, COSPLACE_ENGINE, []),
        (EIGENPLACES_ONNX, EIGENPLACES_ENGINE, []),
        (FLEXGLUE_DESCRIPTOR_ONNX, FLEXGLUE_DESCRIPTOR_ENGINE, []),
        (
            FLEXGLUE_MATCHER_ONNX,
            FLEXGLUE_MATCHER_ENGINE,
            [
                "--minShapes=features0:1x256x64x64,features1:1x256x64x64,"
                "kpts0:1x2x2,kpts1:1x2x2,image_size0:1x2,image_size1:1x2",
                "--optShapes=features0:1x256x64x64,features1:1x256x64x64,"
                "kpts0:1x1024x2,kpts1:1x1024x2,image_size0:1x2,image_size1:1x2",
                "--maxShapes=features0:1x256x64x64,features1:1x256x64x64,"
                "kpts0:1x1024x2,kpts1:1x1024x2,image_size0:1x2,image_size1:1x2",
            ],
        ),
    ]

    if args.models:
        allowed = {name.lower() for name in args.models}
        jobs = [job for job in jobs if _trt_job_key(job[0], aliked_stem) in allowed]
        if not jobs:
            raise SystemExit(f"No TRT jobs matched --models {args.models}")

    extra = []
    if args.fp16:
        extra.append("--fp16")
    if args.extra_args:
        extra.extend(args.extra_args)

    for onnx_name, engine_name, model_args in jobs:
        onnx_path = weights_dir / onnx_name
        engine_path = weights_dir / engine_name
        if not onnx_path.is_file():
            print(f"Skipping (missing ONNX): {onnx_path}")
            continue
        level = args.builder_optimization_level
        if (
            _trt_job_key(onnx_name, aliked_stem) == "aliked"
            and level > ALIKED_MAX_BUILDER_OPTIMIZATION_LEVEL
        ):
            print(
                "ALIKED builderOptimizationLevel "
                f"{level} collapses keypoint y to the top 36 rows on TensorRT 10.9; "
                f"using {ALIKED_MAX_BUILDER_OPTIMIZATION_LEVEL}."
            )
            level = ALIKED_MAX_BUILDER_OPTIMIZATION_LEVEL
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--builderOptimizationLevel={level}",
            *extra,
            *model_args,
        ]
        print("Running:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


def run_export_all(args: argparse.Namespace) -> int:
    common = {
        "output_dir": args.output_dir,
        "height": args.height,
        "width": args.width,
        "simplify": args.simplify,
        "verbose": args.verbose,
        "opset": args.opset,
    }

    sp_args = argparse.Namespace(
        weight_file=str(Path(args.output_dir) / "superpoint_v1.pth"),
        output_file=None,
        output_name=SUPERPOINT_ONNX,
        dynamic=False,
        **common,
    )
    export_superpoint(sp_args)

    lg_sp_args = argparse.Namespace(
        features="superpoint",
        weight_file=None,
        output_file=None,
        output_name=SUPERPOINT_LIGHTGLUE_ONNX,
        num_keypoints=args.num_keypoints,
        flash=False,
        dynamic=False,
        **common,
    )
    export_lightglue(lg_sp_args)

    aliked_args = argparse.Namespace(
        model_name=args.model_name,
        weight_file=None,
        input_size=None,
        top_k=args.top_k,
        scores_th=0.2,
        n_limit=5000,
        device="cpu",
        output_file=None,
        amp=False,
        dynamic=False,
        use_gather_elements=False,
        **common,
    )
    export_aliked(aliked_args)

    lg_aliked_args = argparse.Namespace(
        features="aliked",
        weight_file=None,
        output_file=None,
        output_name=ALIKED_LIGHTGLUE_ONNX,
        num_keypoints=args.num_keypoints,
        flash=False,
        dynamic=False,
        **common,
    )
    export_lightglue(lg_aliked_args)

    cosplace_args = argparse.Namespace(
        output_dir=args.output_dir,
        output_file=None,
        output_name=COSPLACE_ONNX,
        input_size=COSPLACE_INPUT_SIZE,
        backbone=COSPLACE_BACKBONE,
        descriptor_dim=COSPLACE_DESCRIPTOR_DIM,
        dynamic=False,
        simplify=args.simplify,
        verbose=args.verbose,
        opset=args.opset,
    )
    export_cosplace(cosplace_args)

    eigenplaces_args = argparse.Namespace(
        output_dir=args.output_dir,
        output_file=None,
        output_name=EIGENPLACES_ONNX,
        input_size=EIGENPLACES_INPUT_SIZE,
        backbone=EIGENPLACES_BACKBONE,
        descriptor_dim=EIGENPLACES_DESCRIPTOR_DIM,
        simplify=args.simplify,
        verbose=args.verbose,
        opset=args.opset,
    )
    export_eigenplaces(eigenplaces_args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-export",
        description="Export and compare ONNX models for viewfeat",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="export a model to ONNX")
    export_sub = export.add_subparsers(dest="model", required=True)

    sp = export_sub.add_parser("superpoint", help="export SuperPoint extractor")
    add_superpoint_arguments(sp)
    sp.set_defaults(handler=export_superpoint)

    aliked = export_sub.add_parser("aliked", help="export ALIKED extractor")
    add_aliked_arguments(aliked)
    aliked.set_defaults(handler=export_aliked)

    lg = export_sub.add_parser("lightglue", help="export LightGlue matcher")
    add_lightglue_arguments(lg)
    lg.set_defaults(handler=export_lightglue)

    cosplace = export_sub.add_parser("cosplace", help="export CosPlace global descriptor model")
    add_cosplace_arguments(cosplace)
    cosplace.set_defaults(handler=export_cosplace)

    eigenplaces = export_sub.add_parser("eigenplaces", help="export EigenPlaces global descriptor model")
    add_eigenplaces_arguments(eigenplaces)
    eigenplaces.set_defaults(handler=export_eigenplaces)

    export_all = export_sub.add_parser(
        "all",
        help="export SuperPoint, ALIKED, LightGlue matchers, CosPlace, and EigenPlaces",
    )
    export_all.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    export_all.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    export_all.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    export_all.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    export_all.add_argument("--num-keypoints", type=int, default=DEFAULT_NUM_KEYPOINTS)
    export_all.add_argument("--model-name", default=ALIKED_MODEL)
    export_all.add_argument("--opset", type=int, default=17)
    export_all.add_argument("--verbose", action="store_true")
    export_all.add_argument("--no-simplify", dest="simplify", action="store_false")
    export_all.set_defaults(simplify=True, handler=run_export_all)

    compare = sub.add_parser("compare", help="compare ONNX models")
    add_compare_arguments(compare)
    compare.set_defaults(handler=run_compare)

    flexglue = sub.add_parser(
        "flexglue",
        help="download and prepare FlexGlue ONNX models for TensorRT",
    )
    add_flexglue_arguments(flexglue)
    flexglue.set_defaults(handler=run_prepare_flexglue)

    trt = sub.add_parser("trt", help="build TensorRT engines with trtexec")
    trt.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory containing ONNX files and engine outputs",
    )
    trt.add_argument("--model-name", default=ALIKED_MODEL)
    trt.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    trt.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    trt.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    trt.add_argument("--fp16", action="store_true", default=True)
    trt.add_argument("--no-fp16", dest="fp16", action="store_false")
    trt.add_argument(
        "--builder-optimization-level",
        type=int,
        default=DEFAULT_BUILDER_OPTIMIZATION_LEVEL,
        help="trtexec --builderOptimizationLevel (default: 5; ALIKED is capped at 3)",
    )
    trt.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to each trtexec invocation",
    )
    trt.add_argument("--dry-run", action="store_true", help="print commands without running")
    trt.add_argument(
        "--models",
        nargs="+",
        choices=TRT_MODEL_CHOICES,
        metavar="MODEL",
        help="build only selected engines (default: all ONNX files present)",
    )
    trt.set_defaults(handler=run_trt, models=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    result = handler(args)
    if result is None:
        return 0
    if isinstance(result, str):
        return 0
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
