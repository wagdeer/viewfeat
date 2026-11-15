from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import onnx
import torch
from torch.onnx import OperatorExportTypes

from onnx_export.shapes import ShapeDict, simplify_onnx


def export_and_finalize(
    model: torch.nn.Module,
    example_inputs: Any,
    onnx_path: Union[str, Path],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Optional[dict] = None,
    simplify: bool = True,
    input_shapes: Optional[ShapeDict] = None,
    skip_simplify: bool = False,
    opset_version: int = 17,
    verbose: bool = False,
    operator_export_type=OperatorExportTypes.ONNX,
    amp: bool = False,
    device: Optional[str] = None,
) -> str:
    """Export PyTorch model to ONNX, validate, and optionally simplify."""
    onnx_path = str(onnx_path)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        export_params=True,
        verbose=verbose,
        opset_version=opset_version,
        operator_export_type=operator_export_type,
        do_constant_folding=True,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
    )

    with torch.no_grad():
        if amp and device is not None:
            with torch.cuda.amp.autocast(enabled=True):
                torch.onnx.export(model, example_inputs, onnx_path, **export_kwargs)
        else:
            torch.onnx.export(model, example_inputs, onnx_path, **export_kwargs)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Exported ONNX model has been checked successfully.")

    if simplify and not skip_simplify:
        if input_shapes is None:
            print("Skipping simplification: no static input_shapes provided.")
        elif simplify_onnx(onnx_path, input_shapes):
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
    elif skip_simplify:
        print("Skipping simplification because dynamic axes were enabled.")

    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path
