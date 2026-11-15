#include "runtime/config.h"

#include <iostream>
#include <memory>
#include <string>

#include <yaml-cpp/yaml.h>

#include "runtime/paths.h"

namespace viewfeat {

namespace {

template <typename T>
bool ReadScalar(const YAML::Node &section, const char *section_name, const char *key, T &out) {
  const YAML::Node node = section[key];
  if (!node || node.IsNull() || !node.IsScalar() || !YAML::convert<T>::decode(node, out)) {
    std::cerr << "[Configs] Config section '" << section_name << "' is missing key '" << key << "' or the value has the wrong type." << std::endl;
    return false;
  }
  return true;
}

bool ReadOptionalInt(const YAML::Node &section, const char *section_name, const char *key, int &out) {
  const YAML::Node node = section[key];
  if (!node || node.IsNull()) {
    return true;
  }
  if (!node.IsScalar() || !YAML::convert<int>::decode(node, out)) {
    std::cerr << "[Configs] Config section '" << section_name << "' key '" << key << "' has the wrong type." << std::endl;
    return false;
  }
  return true;
}

bool ReadStringList(const YAML::Node &section, const char *section_name, const char *key, std::vector<std::string> &out) {
  const YAML::Node node = section[key];
  if (!node || node.IsNull() || !node.IsSequence()) {
    std::cerr << "[Configs] Config section '" << section_name << "' is missing key '" << key << "' or the value has the wrong type." << std::endl;
    return false;
  }
  out.clear();
  for (const YAML::Node &item : node) {
    std::string value;
    if (!item.IsScalar() || !YAML::convert<std::string>::decode(item, value)) {
      std::cerr << "[Configs] Config section '" << section_name << "' key '" << key << "' has an entry with the wrong type." << std::endl;
      return false;
    }
    out.push_back(value);
  }
  return true;
}

bool ReadFiles(const YAML::Node &section, const char *section_name, const char *onnx_key, const char *engine_key, const std::string &model_dir, std::string &onnx_file,
               std::string &engine_file) {
  std::string onnx_name;
  std::string engine_name;
  if (!ReadScalar(section, section_name, onnx_key, onnx_name) || !ReadScalar(section, section_name, engine_key, engine_name)) {
    return false;
  }
  onnx_file = ConcatenateFolderAndFileName(model_dir, onnx_name);
  engine_file = ConcatenateFolderAndFileName(model_dir, engine_name);
  return true;
}

bool ReadEngineFile(const YAML::Node &section, const char *section_name, const char *engine_key, const std::string &model_dir, std::string &engine_file) {
  std::string engine_name;
  if (!ReadScalar(section, section_name, engine_key, engine_name)) {
    return false;
  }
  engine_file = ConcatenateFolderAndFileName(model_dir, engine_name);
  return true;
}

bool ParseSuperPoint(const YAML::Node &node, const std::string &model_dir, SuperPointConfig &config) {
  return ReadScalar(node, "superpoint", "image_width", config.image_width) && ReadScalar(node, "superpoint", "image_height", config.image_height) &&
         ReadScalar(node, "superpoint", "max_keypoints", config.max_keypoints) &&
         ReadScalar(node, "superpoint", "keypoint_threshold", config.keypoint_threshold) &&
         ReadScalar(node, "superpoint", "remove_borders", config.remove_borders) && ReadScalar(node, "superpoint", "dla_core", config.dla_core) &&
         ReadStringList(node, "superpoint", "input_tensor_names", config.input_tensor_names) &&
         ReadStringList(node, "superpoint", "output_tensor_names", config.output_tensor_names) &&
         ReadFiles(node, "superpoint", "onnx_file", "engine_file", model_dir, config.onnx_file, config.engine_file);
}

bool ParseSuperPointLightGlue(const YAML::Node &node, const std::string &model_dir, int default_num_keypoints, SuperPointLightGlueConfig &config) {
  config.num_keypoints = default_num_keypoints;
  return ReadOptionalInt(node, "superpoint_lightglue", "num_keypoints", config.num_keypoints) &&
         ReadScalar(node, "superpoint_lightglue", "filter_threshold", config.filter_threshold) &&
         ReadScalar(node, "superpoint_lightglue", "dla_core", config.dla_core) &&
         ReadStringList(node, "superpoint_lightglue", "input_tensor_names", config.input_tensor_names) &&
         ReadStringList(node, "superpoint_lightglue", "output_tensor_names", config.output_tensor_names) &&
         ReadFiles(node, "superpoint_lightglue", "onnx_file", "engine_file", model_dir, config.onnx_file, config.engine_file);
}

bool ParseAliked(const YAML::Node &node, const std::string &model_dir, AlikedConfig &config) {
  return ReadScalar(node, "aliked", "image_width", config.image_width) && ReadScalar(node, "aliked", "image_height", config.image_height) &&
         ReadScalar(node, "aliked", "top_k", config.top_k) && ReadScalar(node, "aliked", "descriptor_dim", config.descriptor_dim) &&
         ReadScalar(node, "aliked", "dla_core", config.dla_core) && ReadStringList(node, "aliked", "input_tensor_names", config.input_tensor_names) &&
         ReadStringList(node, "aliked", "output_tensor_names", config.output_tensor_names) &&
         ReadFiles(node, "aliked", "onnx_file", "engine_file", model_dir, config.onnx_file, config.engine_file);
}

bool ParseAlikedLightGlue(const YAML::Node &node, const std::string &model_dir, AlikedLightGlueConfig &config) {
  return ReadScalar(node, "aliked_lightglue", "descriptor_dim", config.descriptor_dim) &&
         ReadScalar(node, "aliked_lightglue", "filter_threshold", config.filter_threshold) &&
         ReadScalar(node, "aliked_lightglue", "dla_core", config.dla_core) &&
         ReadStringList(node, "aliked_lightglue", "input_tensor_names", config.input_tensor_names) &&
         ReadStringList(node, "aliked_lightglue", "output_tensor_names", config.output_tensor_names) &&
         ReadFiles(node, "aliked_lightglue", "onnx_file", "engine_file", model_dir, config.onnx_file, config.engine_file);
}

bool ParsePlaceDescriptor(const YAML::Node &node, const char *section_name, const std::string &model_dir, PlaceDescriptorConfig &config) {
  return ReadScalar(node, section_name, "input_size", config.input_size) && ReadScalar(node, section_name, "descriptor_dim", config.descriptor_dim) &&
         ReadScalar(node, section_name, "dla_core", config.dla_core) &&
         ReadStringList(node, section_name, "input_tensor_names", config.input_tensor_names) &&
         ReadStringList(node, section_name, "output_tensor_names", config.output_tensor_names) &&
         ReadFiles(node, section_name, "onnx_file", "engine_file", model_dir, config.onnx_file, config.engine_file);
}

bool ParseCosPlace(const YAML::Node &node, const std::string &model_dir, CosPlaceConfig &config) {
  return ParsePlaceDescriptor(node, "cosplace", model_dir, config);
}

bool ParseEigenPlaces(const YAML::Node &node, const std::string &model_dir, EigenPlacesConfig &config) {
  return ParsePlaceDescriptor(node, "eigenplaces", model_dir, config);
}

bool ParseFlexGlue(const YAML::Node &node, const std::string &model_dir, FlexGlueConfig &config) {
  return ReadScalar(node, "flexglue", "input_size", config.input_size) && ReadScalar(node, "flexglue", "feature_dim", config.feature_dim) &&
         ReadScalar(node, "flexglue", "feature_width", config.feature_width) && ReadScalar(node, "flexglue", "feature_height", config.feature_height) &&
         ReadOptionalInt(node, "flexglue", "max_keypoints", config.max_keypoints) && ReadScalar(node, "flexglue", "filter_threshold", config.filter_threshold) &&
         ReadScalar(node, "flexglue", "max_distance_px", config.max_distance_px) && ReadScalar(node, "flexglue", "min_covisibility", config.min_covisibility) &&
         ReadStringList(node, "flexglue", "descriptor_input_tensor_names", config.descriptor_input_tensor_names) &&
         ReadStringList(node, "flexglue", "descriptor_output_tensor_names", config.descriptor_output_tensor_names) &&
         ReadStringList(node, "flexglue", "matcher_input_tensor_names", config.matcher_input_tensor_names) &&
         ReadStringList(node, "flexglue", "matcher_output_tensor_names", config.matcher_output_tensor_names) &&
         ReadEngineFile(node, "flexglue", "descriptor_engine_file", model_dir, config.descriptor_engine_file) &&
         ReadEngineFile(node, "flexglue", "matcher_engine_file", model_dir, config.matcher_engine_file);
}

template <typename Parse>
bool ParseIfPresent(const YAML::Node &root, const char *section_name, Parse parse) {
  const YAML::Node node = root[section_name];
  if (!node || node.IsNull()) {
    return true;
  }
  if (!node.IsMap()) {
    std::cerr << "[Configs] Config section '" << section_name << "' must be a map." << std::endl;
    return false;
  }
  return parse(node);
}

bool FillConfigs(const YAML::Node &root, const std::string &model_dir, Configs &configs) {
  if (root["feature_type"] && !root["feature_type"].IsNull()) {
    if (!ReadScalar(root, "config", "feature_type", configs.feature_type)) {
      return false;
    }
  }

  if (!ParseIfPresent(root, "superpoint", [&](const YAML::Node &node) { return ParseSuperPoint(node, model_dir, configs.superpoint_config); })) {
    return false;
  }
  if (!ParseIfPresent(root, "superpoint_lightglue", [&](const YAML::Node &node) {
        return ParseSuperPointLightGlue(node, model_dir, configs.superpoint_config.max_keypoints, configs.superpoint_lightglue_config);
      })) {
    return false;
  }
  if (!ParseIfPresent(root, "aliked", [&](const YAML::Node &node) { return ParseAliked(node, model_dir, configs.aliked_config); })) {
    return false;
  }
  if (!ParseIfPresent(root, "aliked_lightglue", [&](const YAML::Node &node) { return ParseAlikedLightGlue(node, model_dir, configs.aliked_lightglue_config); })) {
    return false;
  }
  if (!ParseIfPresent(root, "cosplace", [&](const YAML::Node &node) { return ParseCosPlace(node, model_dir, configs.cosplace_config); })) {
    return false;
  }
  if (!ParseIfPresent(root, "eigenplaces", [&](const YAML::Node &node) { return ParseEigenPlaces(node, model_dir, configs.eigenplaces_config); })) {
    return false;
  }
  if (!ParseIfPresent(root, "flexglue", [&](const YAML::Node &node) { return ParseFlexGlue(node, model_dir, configs.flexglue_config); })) {
    return false;
  }
  return true;
}

}  // namespace

std::shared_ptr<const Configs> LoadConfigs(const std::string &config_file, const std::string &model_dir) {
  std::cout << "[Configs] Config file is " << config_file << "." << std::endl;
  if (!FileExists(config_file)) {
    std::cerr << "[Configs] Config file " << config_file << " does not exist." << std::endl;
    return std::make_shared<Configs>();
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(config_file);
  } catch (const YAML::Exception &ex) {
    std::cerr << "[Configs] Config file " << config_file << " could not be parsed: " << ex.what() << "." << std::endl;
    return nullptr;
  }
  if (!root || !root.IsMap()) {
    std::cerr << "[Configs] Config file " << config_file << " must be a YAML map." << std::endl;
    return nullptr;
  }

  auto configs = std::make_shared<Configs>();
  if (!FillConfigs(root, model_dir, *configs)) {
    return nullptr;
  }
  return configs;
}

}  // namespace viewfeat
