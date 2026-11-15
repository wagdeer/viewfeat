"""Download and prepare FlexGlue ONNX models for TensorRT.

FlexGlue's released matcher uses ``com.microsoft::MultiHeadAttention`` from
ONNX Runtime. TensorRT does not ship that operator, so this prepares a semantic
equivalent standard-ONNX graph before building the TensorRT engine.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from onnx_export.constants import (
    FLEXGLUE_DESCRIPTOR_ONNX,
    FLEXGLUE_MATCHER_ONNX,
)

RELEASE_URL = "https://github.com/cvg/FlexGlue/releases/download/v0.1/"
MODEL_SHA256 = {
    FLEXGLUE_DESCRIPTOR_ONNX: "93fab8a4ccde1b7331640915357d96cff0929ea87134872f37f7b0150d9914ff",
    FLEXGLUE_MATCHER_ONNX: "a0baf84f00b32fc613206511a464fc3e71f4321e9d34df3a83362bd7d2a73012",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_model(path: Path, force: bool) -> None:
    expected = MODEL_SHA256[path.name]
    if path.is_file() and not force and _sha256(path) == expected:
        return
    print(f"Downloading {RELEASE_URL + path.name} to {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(RELEASE_URL + path.name, temporary)
    actual = _sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {path.name}: {actual} != {expected}")
    temporary.replace(path)


def _rewrite_fused_attention(model: onnx.ModelProto) -> int:
    """Replace ORT MultiHeadAttention nodes with standard ONNX operators."""
    num_heads = 8
    head_dim = 64
    initializers = [
        helper.make_tensor("fg_reshape_heads", TensorProto.INT64, [4], [0, 0, num_heads, head_dim]),
        helper.make_tensor("fg_reshape_merged", TensorProto.INT64, [3], [0, 0, num_heads * head_dim]),
        helper.make_tensor("fg_attn_scale", TensorProto.FLOAT, [], [1.0 / (head_dim**0.5)]),
    ]
    nodes = []
    replaced = 0
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "MultiHeadAttention":
            nodes.append(node)
            continue

        query, key, value = node.input[:3]
        output = node.output[0]
        prefix = f"fg_mha_{replaced}"
        nodes.extend(
            [
                helper.make_node("Reshape", [query, "fg_reshape_heads"], [f"{prefix}_q4"]),
                helper.make_node("Reshape", [key, "fg_reshape_heads"], [f"{prefix}_k4"]),
                helper.make_node("Reshape", [value, "fg_reshape_heads"], [f"{prefix}_v4"]),
                helper.make_node("Transpose", [f"{prefix}_q4"], [f"{prefix}_qt"], perm=[0, 2, 1, 3]),
                helper.make_node("Transpose", [f"{prefix}_k4"], [f"{prefix}_kt"], perm=[0, 2, 1, 3]),
                helper.make_node("Transpose", [f"{prefix}_v4"], [f"{prefix}_vt"], perm=[0, 2, 1, 3]),
                helper.make_node("Transpose", [f"{prefix}_kt"], [f"{prefix}_kt_t"], perm=[0, 1, 3, 2]),
                helper.make_node("MatMul", [f"{prefix}_qt", f"{prefix}_kt_t"], [f"{prefix}_scores"]),
                helper.make_node("Mul", [f"{prefix}_scores", "fg_attn_scale"], [f"{prefix}_scaled"]),
                helper.make_node("Softmax", [f"{prefix}_scaled"], [f"{prefix}_prob"], axis=-1),
                helper.make_node("MatMul", [f"{prefix}_prob", f"{prefix}_vt"], [f"{prefix}_ctx"]),
                helper.make_node("Transpose", [f"{prefix}_ctx"], [f"{prefix}_ctx_seq"], perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", [f"{prefix}_ctx_seq", "fg_reshape_merged"], [output]),
            ]
        )
        replaced += 1

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(initializers)
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 18)])
    return replaced


def prepare_flexglue(output_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Prepare descriptor and standard-ONNX matcher files for TensorRT."""
    import onnxsim

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = output_dir / FLEXGLUE_DESCRIPTOR_ONNX
    matcher_path = output_dir / FLEXGLUE_MATCHER_ONNX
    _download_model(descriptor_path, force)
    if force or not matcher_path.is_file():
        _download_model(matcher_path, True)

    try:
        model = onnx.load(str(matcher_path), load_external_data=False)
    except Exception:
        _download_model(matcher_path, True)
        model = onnx.load(str(matcher_path), load_external_data=False)
    replaced = _rewrite_fused_attention(model)
    if replaced == 0:
        return descriptor_path, matcher_path

    simplified, ok = onnxsim.simplify(
        model,
        check_n=0,
        certify=False,
        check_hazards=False,
        check_gradients=False,
        external_data_threshold="4GB",
    )
    if not ok:
        raise RuntimeError("onnxsim rejected the standard-ONNX FlexGlue matcher")
    onnx.checker.check_model(simplified)
    onnx.save(simplified, str(matcher_path))
    print(
        f"Prepared FlexGlue matcher for TensorRT: replaced {replaced} ORT MHA nodes, "
        f"{len(simplified.graph.node)} nodes, {os.path.getsize(matcher_path) / 1e6:.1f} MB"
    )
    return descriptor_path, matcher_path


def add_flexglue_arguments(parser) -> None:
    from onnx_export.constants import DEFAULT_OUTPUT_DIR

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="redownload and rewrite the models")


def run_prepare_flexglue(args) -> int:
    prepare_flexglue(args.output_dir.resolve(), force=args.force)
    return 0
