#include "viewfeat/retriever.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "models/cosplace.h"
#include "models/eigenplaces.h"
#include "runtime/search_paths.h"

namespace viewfeat {

namespace {

constexpr const char *kRetrieverCosPlace = "cosplace";
constexpr const char *kRetrieverEigenPlaces = "eigenplaces";

std::string NormalizeType(std::string type, const std::string &default_type) {
  std::transform(type.begin(), type.end(), type.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (type.empty() || type == "default") {
    return default_type;
  }
  return type;
}

}  // namespace

struct GlobalRetriever::Impl {
  std::unique_ptr<CosPlace> cosplace;
  std::unique_ptr<EigenPlaces> eigenplaces;
  std::string type_name;
  int descriptor_dim = 0;
};

GlobalRetriever::GlobalRetriever(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GlobalRetriever::GlobalRetriever(GlobalRetriever &&) noexcept = default;

GlobalRetriever &GlobalRetriever::operator=(GlobalRetriever &&) noexcept = default;

GlobalRetriever::~GlobalRetriever() = default;

std::unique_ptr<GlobalRetriever> GlobalRetriever::Create(const std::string &family) {
  const std::string normalized_type = NormalizeType(family, kRetrieverCosPlace);
  if (normalized_type != kRetrieverCosPlace && normalized_type != kRetrieverEigenPlaces) {
    std::cerr << "[GlobalRetriever] Unsupported global retriever type: " << family << "." << std::endl;
    return nullptr;
  }

  const std::shared_ptr<const Configs> configs = GetConfigs();
  if (!configs) {
    std::cerr << "[GlobalRetriever] Failed to load config from " << ResolveConfigPath() << "." << std::endl;
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  if (normalized_type == kRetrieverEigenPlaces) {
    if (configs->eigenplaces_config.input_tensor_names.empty() || configs->eigenplaces_config.output_tensor_names.empty()) {
      std::cerr << "[GlobalRetriever] Invalid or missing eigenplaces config section in " << ResolveConfigPath() << "." << std::endl;
      return nullptr;
    }
    auto retriever = std::make_unique<EigenPlaces>(configs->eigenplaces_config);
    if (!retriever->Build()) {
      std::cerr << "[GlobalRetriever] Failed to initialize EigenPlaces retriever." << std::endl;
      return nullptr;
    }
    impl->eigenplaces = std::move(retriever);
    impl->type_name = kRetrieverEigenPlaces;
    impl->descriptor_dim = configs->eigenplaces_config.descriptor_dim;
  } else {
    if (configs->cosplace_config.input_tensor_names.empty() || configs->cosplace_config.output_tensor_names.empty()) {
      std::cerr << "[GlobalRetriever] Invalid or missing cosplace config section in " << ResolveConfigPath() << "." << std::endl;
      return nullptr;
    }
    auto retriever = std::make_unique<CosPlace>(configs->cosplace_config);
    if (!retriever->Build()) {
      std::cerr << "[GlobalRetriever] Failed to initialize CosPlace retriever." << std::endl;
      return nullptr;
    }
    impl->cosplace = std::move(retriever);
    impl->type_name = kRetrieverCosPlace;
    impl->descriptor_dim = configs->cosplace_config.descriptor_dim;
  }
  return std::unique_ptr<GlobalRetriever>(new GlobalRetriever(std::move(impl)));
}

bool GlobalRetriever::Infer(const cv::Mat &image, std::vector<float> &descriptor) {
  if (!impl_ || (!impl_->cosplace && !impl_->eigenplaces)) {
    std::cerr << "[GlobalRetriever] Retriever is not initialized." << std::endl;
    return false;
  }
  Eigen::VectorXf eigen_descriptor;
  const bool ok = impl_->cosplace ? impl_->cosplace->Infer(image, eigen_descriptor) : impl_->eigenplaces->Infer(image, eigen_descriptor);
  if (!ok) {
    std::cerr << "[GlobalRetriever] Infer failed." << std::endl;
    return false;
  }
  descriptor.assign(eigen_descriptor.data(), eigen_descriptor.data() + eigen_descriptor.size());
  return true;
}

int GlobalRetriever::DescriptorDim() const { return impl_ ? impl_->descriptor_dim : 0; }

const std::string &GlobalRetriever::Family() const {
  static const std::string kEmpty;
  if (!impl_) {
    return kEmpty;
  }
  return impl_->type_name;
}

}  // namespace viewfeat
