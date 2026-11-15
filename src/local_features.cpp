#include "feature_payload.h"

#include <utility>

namespace viewfeat {

struct LocalFeatures::Impl {
  std::shared_ptr<void> backend_blob;
  std::string backend_tag;
  std::type_index backend_type{typeid(void)};
  std::shared_ptr<std::vector<float>> flexglue_features;
  int flexglue_feature_height = 0;
  int flexglue_feature_width = 0;
  int flexglue_feature_dim = 0;
  int original_width = 0;
  int original_height = 0;
  int infer_width = 0;
  int infer_height = 0;
  bool coords_in_original = false;
  std::string pipeline_family;
};

LocalFeatures::Impl &FeaturePayload::Ensure(LocalFeatures &features) {
  if (!features.impl_) {
    features.impl_ = std::make_unique<LocalFeatures::Impl>();
  }
  return *features.impl_;
}

void FeaturePayload::AttachBackend(LocalFeatures &features, std::shared_ptr<void> blob, std::string tag, std::type_index type) {
  LocalFeatures::Impl &impl = Ensure(features);
  impl.backend_blob = std::move(blob);
  impl.backend_tag = std::move(tag);
  impl.backend_type = type;
}

std::shared_ptr<void> FeaturePayload::BackendOf(const LocalFeatures &features, const char *family, std::type_index type) {
  if (!features.impl_ || features.impl_->backend_tag != family || !features.impl_->backend_blob) {
    return nullptr;
  }
  if (features.impl_->backend_type != type) {
    return nullptr;
  }
  return features.impl_->backend_blob;
}

void FeaturePayload::AttachFlexGlue(LocalFeatures &features, std::shared_ptr<std::vector<float>> blob, int h, int w, int dim) {
  LocalFeatures::Impl &impl = Ensure(features);
  impl.flexglue_features = std::move(blob);
  impl.flexglue_feature_height = h;
  impl.flexglue_feature_width = w;
  impl.flexglue_feature_dim = dim;
}

std::shared_ptr<std::vector<float>> FeaturePayload::FlexGlueOf(const LocalFeatures &features, int &h, int &w, int &dim) {
  if (!features.impl_) {
    return nullptr;
  }
  h = features.impl_->flexglue_feature_height;
  w = features.impl_->flexglue_feature_width;
  dim = features.impl_->flexglue_feature_dim;
  return features.impl_->flexglue_features;
}

void FeaturePayload::AttachImageInfo(LocalFeatures &features, int original_width, int original_height, int infer_width, int infer_height, bool coords_in_original) {
  LocalFeatures::Impl &impl = Ensure(features);
  impl.original_width = original_width;
  impl.original_height = original_height;
  impl.infer_width = infer_width;
  impl.infer_height = infer_height;
  impl.coords_in_original = coords_in_original;
}

void FeaturePayload::AttachPipelineFamily(LocalFeatures &features, std::string pipeline_family) {
  Ensure(features).pipeline_family = std::move(pipeline_family);
}

std::string FeaturePayload::PipelineFamilyOf(const LocalFeatures &features) {
  if (!features.impl_) {
    return {};
  }
  return features.impl_->pipeline_family;
}

bool FeaturePayload::ImageInfo(const LocalFeatures &features, int &original_width, int &original_height, int &infer_width, int &infer_height, bool &coords_in_original) {
  if (!features.impl_) {
    return false;
  }
  original_width = features.impl_->original_width;
  original_height = features.impl_->original_height;
  infer_width = features.impl_->infer_width;
  infer_height = features.impl_->infer_height;
  coords_in_original = features.impl_->coords_in_original;
  return original_width > 0 && original_height > 0;
}

LocalFeatures::LocalFeatures() : impl_(nullptr) {}

LocalFeatures::LocalFeatures(LocalFeatures &&other) noexcept
    : keypoints(std::move(other.keypoints)),
      descriptors(std::move(other.descriptors)),
      image_size(other.image_size),
      family(std::move(other.family)),
      impl_(std::move(other.impl_)) {}

LocalFeatures &LocalFeatures::operator=(LocalFeatures &&other) noexcept {
  if (this != &other) {
    keypoints = std::move(other.keypoints);
    descriptors = std::move(other.descriptors);
    image_size = other.image_size;
    family = std::move(other.family);
    impl_ = std::move(other.impl_);
  }
  return *this;
}

LocalFeatures::~LocalFeatures() = default;

}  // namespace viewfeat
