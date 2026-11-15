#ifndef VIEWFEAT_RUNTIME_CONFIG_H_
#define VIEWFEAT_RUNTIME_CONFIG_H_

#include <memory>
#include <string>
#include <vector>

namespace viewfeat {

struct SuperPointConfig {
  int image_width{};
  int image_height{};
  int max_keypoints{};
  double keypoint_threshold{};
  int remove_borders{};
  int dla_core{};
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct SuperPointLightGlueConfig {
  int num_keypoints{};
  double filter_threshold{};
  int dla_core{};
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct AlikedConfig {
  int image_width{};
  int image_height{};
  int top_k{};
  int descriptor_dim{};
  int dla_core{};
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

// CosPlace and EigenPlaces share this layout. The yaml section and file names stay separate.
struct PlaceDescriptorConfig {
  int input_size{};
  int descriptor_dim{};
  int dla_core{};
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

using CosPlaceConfig = PlaceDescriptorConfig;
using EigenPlacesConfig = PlaceDescriptorConfig;

struct AlikedLightGlueConfig {
  int descriptor_dim{};
  double filter_threshold{};
  int dla_core{};
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct FlexGlueConfig {
  int input_size{512};
  int feature_dim{256};
  int feature_width{64};
  int feature_height{64};
  int max_keypoints{1024};
  double filter_threshold{0.1};
  double max_distance_px{3.0};
  double min_covisibility{0.0};
  std::vector<std::string> descriptor_input_tensor_names;
  std::vector<std::string> descriptor_output_tensor_names;
  std::vector<std::string> matcher_input_tensor_names;
  std::vector<std::string> matcher_output_tensor_names;
  std::string descriptor_engine_file;
  std::string matcher_engine_file;
};

struct Configs {
  std::string feature_type = "superpoint";

  SuperPointConfig superpoint_config;
  SuperPointLightGlueConfig superpoint_lightglue_config;
  AlikedConfig aliked_config;
  AlikedLightGlueConfig aliked_lightglue_config;
  CosPlaceConfig cosplace_config;
  EigenPlacesConfig eigenplaces_config;
  FlexGlueConfig flexglue_config;

  Configs() = default;
};

// A missing file returns an empty Configs. A file that exists but is not valid
// YAML, or a present section with a missing or mistyped key, returns nullptr.
// A section that is absent leaves that model's tensor names empty.
std::shared_ptr<const Configs> LoadConfigs(const std::string &config_file, const std::string &model_dir);

}  // namespace viewfeat

#endif  // VIEWFEAT_RUNTIME_CONFIG_H_
