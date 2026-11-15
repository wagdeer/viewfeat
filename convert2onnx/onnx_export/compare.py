#!/usr/bin/env python3
"""Compare two ONNX models, or compare one model before/after onnxsim."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from typing import Dict, List, Optional

import onnx
from onnx import numpy_helper

from onnx_export.shapes import simplify_onnx


def load_model(path: str) -> onnx.ModelProto:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return onnx.load(path)


def tensor_shape(value_info) -> List[object]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def io_summary(model: onnx.ModelProto) -> str:
    lines = []
    for item in list(model.graph.input) + list(model.graph.output):
        role = "input" if item in model.graph.input else "output"
        lines.append(f"  {role} {item.name}: {tensor_shape(item)}")
    return "\n".join(lines) if lines else "  (none)"


def op_histogram(model: onnx.ModelProto) -> Counter:
    counts = Counter()
    for node in model.graph.node:
        counts[node.op_type] += 1
    return counts


def initializer_bytes(model: onnx.ModelProto) -> int:
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total += arr.nbytes
    return total


def collect_stats(path: str, label: str) -> Dict[str, object]:
    model = load_model(path)
    graph = model.graph
    return {
        "label": label,
        "path": path,
        "file_bytes": os.path.getsize(path),
        "opset": [(op.domain, op.version) for op in model.opset_import],
        "nodes": len(graph.node),
        "initializers": len(graph.initializer),
        "initializer_bytes": initializer_bytes(model),
        "inputs": len(graph.input),
        "outputs": len(graph.output),
        "ops": op_histogram(model),
        "io_summary": io_summary(model),
    }


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def pct_delta(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    delta = (after - before) / before * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def print_scalar_compare(name: str, left: int, right: int) -> None:
    print(f"  {name:<18} {left:>10} -> {right:>10}  ({pct_delta(left, right)})")


def print_op_diff(left_ops: Counter, right_ops: Counter) -> None:
    all_ops = sorted(set(left_ops) | set(right_ops))
    changed = []
    for op in all_ops:
        before = left_ops.get(op, 0)
        after = right_ops.get(op, 0)
        if before != after:
            changed.append((op, before, after, after - before))
    if not changed:
        print("  op types unchanged")
        return
    print("  op type changes (only deltas):")
    for op, before, after, delta in sorted(changed, key=lambda x: abs(x[3]), reverse=True):
        sign = "+" if delta >= 0 else ""
        print(f"    {op:<18} {before:>6} -> {after:>6}  ({sign}{delta})")


def maybe_compare_runtime_outputs(
    left_path: str,
    right_path: str,
    input_shapes: Optional[Dict[str, List[int]]] = None,
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("\nRuntime check skipped: install onnxruntime to compare inference outputs.")
        return

    left_sess = ort.InferenceSession(left_path, providers=["CPUExecutionProvider"])
    right_sess = ort.InferenceSession(right_path, providers=["CPUExecutionProvider"])

    feeds = {}
    for inp in left_sess.get_inputs():
        shape = list(inp.shape)
        if input_shapes and inp.name in input_shapes:
            shape = input_shapes[inp.name]
        resolved = []
        for dim in shape:
            if isinstance(dim, str) or dim is None:
                print("\nRuntime check skipped: dynamic input shape requires --input-shapes.")
                return
            resolved.append(int(dim))
        feeds[inp.name] = np.random.randn(*resolved).astype(np.float32)

    left_out = left_sess.run(None, feeds)
    right_out = right_sess.run(None, feeds)
    if len(left_out) != len(right_out):
        print("\nRuntime outputs differ in count; numerical check skipped.")
        return

    print("\nRuntime numerical check (random inputs, CPU ORT):")
    for idx, (a, b) in enumerate(zip(left_out, right_out)):
        if a.shape != b.shape:
            print(f"  output[{idx}] shape mismatch: {a.shape} vs {b.shape}")
            continue
        abs_diff = np.max(np.abs(a - b))
        rel_diff = abs_diff / max(float(np.max(np.abs(a))), 1e-12)
        print(f"  output[{idx}] max_abs_diff={abs_diff:.6e}, max_rel_diff={rel_diff:.6e}")


def compare_two(
    left_path: str,
    right_path: str,
    input_shapes: Optional[Dict[str, List[int]]] = None,
) -> None:
    left = collect_stats(left_path, "before")
    right = collect_stats(right_path, "after")

    print(f"Before: {left['path']}")
    print(f"After : {right['path']}")
    print("\nSummary:")
    print(
        f"  {'file size':<18} {format_bytes(left['file_bytes']):>10} -> "
        f"{format_bytes(right['file_bytes']):>10}  "
        f"({pct_delta(left['file_bytes'], right['file_bytes'])})"
    )
    print_scalar_compare("nodes", left["nodes"], right["nodes"])
    print_scalar_compare("initializers", left["initializers"], right["initializers"])
    print_scalar_compare("weight bytes", left["initializer_bytes"], right["initializer_bytes"])
    print("\nOp histogram:")
    print_op_diff(left["ops"], right["ops"])
    print("\nBefore IO:")
    print(left["io_summary"])
    print("\nAfter IO:")
    print(right["io_summary"])

    maybe_compare_runtime_outputs(left_path, right_path, input_shapes)


def compare_with_inline_simplify(
    path: str,
    input_shapes: Optional[Dict[str, List[int]]] = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="onnx_simp_") as tmp_dir:
        raw_copy = os.path.join(tmp_dir, "raw.onnx")
        simp_copy = os.path.join(tmp_dir, "simplified.onnx")
        shutil.copy2(path, raw_copy)
        shutil.copy2(path, simp_copy)
        if not simplify_onnx(simp_copy, input_shapes):
            print("Inline simplification skipped; compare two exported files instead.")
            return
        compare_two(raw_copy, simp_copy, input_shapes)


def parse_input_shapes(spec: str) -> Dict[str, List[int]]:
    shapes: Dict[str, List[int]] = {}
    if not spec:
        return shapes
    for item in spec.split(","):
        name, shape_text = item.split(":", 1)
        dims = [int(x) for x in shape_text.lower().split("x") if x]
        shapes[name.strip()] = dims
    return shapes


def run_compare(args: argparse.Namespace) -> int:
    input_shapes = parse_input_shapes(args.input_shapes) or None

    if args.inline:
        if not args.before:
            raise SystemExit("--inline requires the source ONNX path as the first positional argument")
        compare_with_inline_simplify(args.before, input_shapes)
        return 0

    if not args.before or not args.after:
        raise SystemExit("provide BEFORE AFTER paths, or use --inline with one ONNX path")
    compare_two(args.before, args.after, input_shapes)
    return 0


def add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("before", nargs="?", help="original ONNX path")
    parser.add_argument("after", nargs="?", help="simplified ONNX path")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="copy one ONNX file and run onnxsim in a temp dir, then compare",
    )
    parser.add_argument(
        "--input-shapes",
        default="",
        help="optional ORT feed shapes, e.g. image:1x1x288x512 or keypoints0:1x1024x2,...",
    )
