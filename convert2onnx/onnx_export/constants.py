from pathlib import Path

DEFAULT_HEIGHT = 288
DEFAULT_WIDTH = 512
DEFAULT_TOP_K = 1024
DEFAULT_NUM_KEYPOINTS = 1024
DEFAULT_OPSET = 17
DEFAULT_BUILDER_OPTIMIZATION_LEVEL = 5
# TensorRT 10.9 levels 4 and 5 fuse ALIKED's scale-8 bilinear upsample.
# The score map stays 36 rows (H/8 of 288) and keypoint y only covers the top of the image.
ALIKED_MAX_BUILDER_OPTIMIZATION_LEVEL = 3

DEFAULT_OUTPUT_DIR = Path("../weights")
DEFAULT_WEIGHTS_DIR = Path("../weights")

SUPERPOINT_WEIGHT = "superpoint_v1.pth"
SUPERPOINT_ONNX = "superpoint_v1.onnx"
SUPERPOINT_ENGINE = "superpoint_v1.engine"

SUPERPOINT_LIGHTGLUE_ONNX = "superpoint_lightglue.onnx"
SUPERPOINT_LIGHTGLUE_ENGINE = "superpoint_lightglue.engine"

ALIKED_MODEL = "aliked-n16rot"
def topk_label(top_k: int) -> str:
    if top_k > 0 and top_k % 1000 == 0:
        return f"top{top_k // 1000}k"
    if top_k > 0:
        return f"top{top_k}"
    return "threshold"


def aliked_basename(model: str, top_k: int, width: int, height: int) -> str:
    return f"{model}-{topk_label(top_k)}-{width}x{height}"


ALIKED_ONNX_SUFFIX = ".onnx"
ALIKED_ENGINE_SUFFIX = ".engine"

ALIKED_LIGHTGLUE_ONNX = "aliked_lightglue.onnx"
ALIKED_LIGHTGLUE_ENGINE = "aliked_lightglue.engine"

CHECKPOINT_URL = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"

COSPLACE_ONNX = "cosplace.onnx"
COSPLACE_ENGINE = "cosplace.engine"
COSPLACE_INPUT_SIZE = 320
COSPLACE_BACKBONE = "ResNet18"
COSPLACE_DESCRIPTOR_DIM = 512

EIGENPLACES_ONNX = "eigenplaces.onnx"
EIGENPLACES_ENGINE = "eigenplaces.engine"
EIGENPLACES_INPUT_SIZE = 512
EIGENPLACES_BACKBONE = "ResNet18"
EIGENPLACES_DESCRIPTOR_DIM = 512
EIGENPLACES_WEIGHT_URL = (
    "https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet18_512_eigenplaces.pth"
)

FLEXGLUE_DESCRIPTOR_ONNX = "flexglue_descriptor.onnx"
FLEXGLUE_DESCRIPTOR_ENGINE = "flexglue_descriptor.engine"
FLEXGLUE_MATCHER_ONNX = "flexglue_matcher.onnx"
FLEXGLUE_MATCHER_ENGINE = "flexglue_matcher.engine"
