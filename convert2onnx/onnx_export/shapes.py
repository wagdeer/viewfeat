from __future__ import annotations

from typing import Dict, List, Optional, Union

import onnx

ShapeDict = Dict[str, List[Union[int, str]]]


def simplify_onnx(onnx_path: str, input_shapes: Optional[ShapeDict] = None) -> bool:
    try:
        import onnxruntime
        from onnxsim import simplify
    except ModuleNotFoundError:
        print("onnxsim or onnxruntime is not installed; skipping simplification.")
        return False

    if input_shapes is None:
        input_shapes = infer_static_input_shapes(onnx_path)
        if input_shapes is None:
            return False

    model = onnx.load(onnx_path)
    model_simp, check = simplify(
        model,
        overwrite_input_shapes=input_shapes,
        test_input_shapes=input_shapes,
    )
    if not check:
        raise RuntimeError("onnxsim failed to validate the simplified ONNX model")
    onnx.save(model_simp, onnx_path)
    print("Simplified ONNX model has been checked successfully.")
    return True


def infer_static_input_shapes(onnx_path: str) -> Optional[ShapeDict]:
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_shapes: ShapeDict = {}
    for inp in session.get_inputs():
        shape: List[Union[int, str]] = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                print(
                    f"Skipping simplification: input '{inp.name}' has dynamic shape {inp.shape}."
                )
                return None
            shape.append(int(dim))
        input_shapes[inp.name] = shape
    return input_shapes


def image_input_shapes(onnx_path: str, height: int, width: int) -> ShapeDict:
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return {
        inp.name: [int(inp.shape[0]), int(inp.shape[1]), height, width]
        for inp in session.get_inputs()
    }


def lightglue_input_shapes(num_keypoints: int, descriptor_dim: int) -> ShapeDict:
    return {
        "keypoints0": [1, num_keypoints, 2],
        "keypoints1": [1, num_keypoints, 2],
        "descriptors0": [1, num_keypoints, descriptor_dim],
        "descriptors1": [1, num_keypoints, descriptor_dim],
    }
