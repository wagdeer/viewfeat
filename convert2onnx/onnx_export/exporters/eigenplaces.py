from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from onnx_export.constants import (
    DEFAULT_OPSET,
    DEFAULT_OUTPUT_DIR,
    EIGENPLACES_BACKBONE,
    EIGENPLACES_DESCRIPTOR_DIM,
    EIGENPLACES_INPUT_SIZE,
    EIGENPLACES_ONNX,
    EIGENPLACES_WEIGHT_URL,
)
from onnx_export.pipeline import export_and_finalize

CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


class L2Norm(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=self.dim)


class GeM(nn.Module):
    def __init__(self, p: float = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kernel is the feature-map size. Export with a fixed input so this becomes a constant.
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class EigenPlacesNet(nn.Module):
    """Same inference graph as eigenplaces_model.GeoLocalizationNet_ without the CosPlace hub init."""

    def __init__(self, backbone: str, fc_output_dim: int) -> None:
        super().__init__()
        if backbone not in CHANNELS_NUM_IN_LAST_CONV:
            raise ValueError(f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV)}")
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone]
        raw = getattr(torchvision.models, backbone.lower())()
        if backbone.startswith("ResNet"):
            layers = list(raw.children())[:-2]
        elif backbone == "VGG16":
            layers = list(raw.features.children())[:-2]
        else:
            raise ValueError(f"unsupported backbone {backbone}")
        self.backbone = nn.Sequential(*layers)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def _weight_url(backbone: str, descriptor_dim: int) -> str:
    if backbone == EIGENPLACES_BACKBONE and descriptor_dim == EIGENPLACES_DESCRIPTOR_DIM:
        return EIGENPLACES_WEIGHT_URL
    return (
        "https://github.com/gmberton/EigenPlaces/releases/download/v1.0/"
        f"{backbone}_{descriptor_dim}_eigenplaces.pth"
    )


def _load_release_state_dict(backbone: str, descriptor_dim: int) -> EigenPlacesNet:
    model = EigenPlacesNet(backbone, descriptor_dim)
    state = torch.hub.load_state_dict_from_url(_weight_url(backbone, descriptor_dim), map_location="cpu")
    model.load_state_dict(state)
    return model


def load_eigenplaces_model(backbone: str, descriptor_dim: int) -> torch.nn.Module:
    """Load EigenPlaces. Hub init also downloads CosPlace; fall back to the release state_dict."""
    try:
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=descriptor_dim,
            trust_repo=True,
        )
    except Exception as exc:
        print(
            "torch.hub EigenPlaces load failed "
            f"({exc!r}). Building GeoLocalizationNet_ locally and loading "
            f"{_weight_url(backbone, descriptor_dim)}."
        )
        try:
            model = _load_release_state_dict(backbone, descriptor_dim)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Could not load EigenPlaces from torch.hub or from the release state_dict. "
                f"Hub error: {exc!r}. Fallback error: {fallback_exc!r}"
            ) from fallback_exc
    return model.eval()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name or EIGENPLACES_ONNX
    return output_dir / name


def export_eigenplaces(args: argparse.Namespace) -> str:
    model = load_eigenplaces_model(args.backbone, args.descriptor_dim)
    print("Total number of params: ", sum(p.numel() for p in model.parameters()))

    size = args.input_size
    example = torch.randn(1, 3, size, size)
    onnx_path = resolve_output_path(args)
    input_shapes = {"images": [1, 3, size, size]} if args.simplify else None

    return export_and_finalize(
        model,
        example,
        onnx_path,
        input_names=["images"],
        output_names=["descriptors"],
        dynamic_axes=None,
        simplify=args.simplify,
        input_shapes=input_shapes,
        skip_simplify=False,
        opset_version=args.opset,
        verbose=args.verbose,
    )


def add_eigenplaces_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="ONNX output directory",
    )
    parser.add_argument("--output-file", type=Path, default=None, help="Explicit ONNX output path")
    parser.add_argument(
        "--output-name",
        default=EIGENPLACES_ONNX,
        help=f"Output filename under --output-dir (default: {EIGENPLACES_ONNX})",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=EIGENPLACES_INPUT_SIZE,
        help="Square input side length in pixels (default: 512)",
    )
    parser.add_argument(
        "--backbone",
        default=EIGENPLACES_BACKBONE,
        help="EigenPlaces backbone passed to torch.hub (default: ResNet18)",
    )
    parser.add_argument(
        "--descriptor-dim",
        type=int,
        default=EIGENPLACES_DESCRIPTOR_DIM,
        help="Global descriptor dimension (default: 512)",
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
