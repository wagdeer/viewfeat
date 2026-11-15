#include "viewfeat/pipeline.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/imgproc.hpp>

#include "feature_payload.h"
#include "models/aliked.h"
#include "models/aliked_lightglue.h"
#include "models/flexglue.h"
#include "models/superpoint.h"
#include "models/superpoint_lightglue.h"
#include "runtime/search_paths.h"

namespace viewfeat {

namespace {

constexpr const char *kFamilySuperPoint = "superpoint";
constexpr const char *kFamilyAliked = "aliked";
constexpr const char *kMatcherSuperPointLightGlue = "superpoint_lightglue";
constexpr const char *kMatcherAlikedLightGlue = "aliked_lightglue";
constexpr const char *kMatcherFlexGlue = "flexglue";
constexpr const char *kPipelineSuperPointFlexGlue = "superpoint_flexglue";
constexpr const char *kPipelineAlikedFlexGlue = "aliked_flexglue";
constexpr int kFeatureCoordRows = 2;

using SuperPointFeatureMatrix = Eigen::Matrix<float, SuperPoint::kFeatureRows, Eigen::Dynamic>;
using AlikedFeatureMatrix = Eigen::Matrix<float, Aliked::kFeatureRows, Eigen::Dynamic>;
using FlexGlueFeatureVector = std::vector<float>;

std::string LowercaseAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool IsPipelineFamily(const std::string &family) {
  return family == kFamilySuperPoint || family == kFamilyAliked || family == kPipelineSuperPointFlexGlue || family == kPipelineAlikedFlexGlue;
}

bool ValidateMatcherFamily(const LocalFeatures &features, const char *expected_family) {
  return features.family.empty() || features.family == expected_family;
}

bool HasFlexGlueFeatures(const LocalFeatures &features) {
  int height = 0;
  int width = 0;
  int dim = 0;
  return static_cast<bool>(FeaturePayload::FlexGlueOf(features, height, width, dim));
}

bool PipelineFamilyMatches(const LocalFeatures &features, const std::string &pipeline_family) {
  const std::string origin = FeaturePayload::PipelineFamilyOf(features);
  return origin.empty() || origin == pipeline_family;
}

template <typename MatrixType>
void FillKeypointsFromEigen(const MatrixType &features, LocalFeatures &local) {
  const int cols = features.cols();
  local.keypoints.resize(static_cast<size_t>(cols));
  for (int i = 0; i < cols; ++i) {
    local.keypoints[static_cast<size_t>(i)] = cv::Point2f(features(0, i), features(1, i));
  }
}

void ResolveMatcherImageSize(const LocalFeatures &features, int fallback_width, int fallback_height, int &width, int &height) {
  width = fallback_width;
  height = fallback_height;

  int original_width = 0;
  int original_height = 0;
  int infer_width = 0;
  int infer_height = 0;
  bool coords_in_original = false;
  if (!FeaturePayload::ImageInfo(features, original_width, original_height, infer_width, infer_height, coords_in_original)) {
    return;
  }

  if (coords_in_original && original_width > 0 && original_height > 0) {
    width = original_width;
    height = original_height;
    return;
  }
  if (!coords_in_original && infer_width > 0 && infer_height > 0) {
    width = infer_width;
    height = infer_height;
  }
}

template <typename MatrixType, typename MatcherType>
bool MatchLocalFeatures(MatcherType &matcher, const char *family, const LocalFeatures &features0, const LocalFeatures &features1, std::vector<cv::DMatch> &matches,
                        bool outlier_rejection) {
  if (!ValidateMatcherFamily(features0, family) || !ValidateMatcherFamily(features1, family)) {
    return false;
  }

  const auto backend0 = FeaturePayload::BackendOf<MatrixType>(features0, family);
  const auto backend1 = FeaturePayload::BackendOf<MatrixType>(features1, family);
  if (!backend0 || !backend1) {
    std::cerr << "[FeaturePipeline] Feature backend is missing or has the wrong type." << std::endl;
    return false;
  }

  int image0_width = 0;
  int image0_height = 0;
  int image1_width = 0;
  int image1_height = 0;
  ResolveMatcherImageSize(features0, 0, 0, image0_width, image0_height);
  ResolveMatcherImageSize(features1, 0, 0, image1_width, image1_height);

  if (image0_width <= 0 || image0_height <= 0 || image1_width <= 0 || image1_height <= 0) {
    std::cerr << "[FeaturePipeline] Matcher image size is missing." << std::endl;
    return false;
  }

  return matcher.MatchingPoints(*backend0, *backend1, matches, image0_width, image0_height, image1_width, image1_height, outlier_rejection);
}

bool ConvertImageToGrayscale(const cv::Mat &image, cv::Mat &gray) {
  if (image.channels() == 1) {
    gray = image;
  } else if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    return false;
  }
  return true;
}

template <typename MatrixType, typename ExtractorType>
bool ExtractLocalFeatures(const char *family, ExtractorType &extractor, const cv::Mat &image, LocalFeatures &features, int model_width, int model_height,
                          bool to_grayscale) {
  cv::Mat prepared;
  if (to_grayscale) {
    if (!ConvertImageToGrayscale(image, prepared)) {
      std::cerr << "[FeaturePipeline] ConvertImageToGrayscale failed." << std::endl;
      return false;
    }
  } else if (image.channels() == 1 || image.channels() == 3 || image.channels() == 4) {
    prepared = image;
  } else {
    std::cerr << "[FeaturePipeline] Unsupported image channel count." << std::endl;
    return false;
  }

  cv::Mat model_input = prepared;
  const bool need_resize = model_width > 0 && model_height > 0 && (prepared.cols != model_width || prepared.rows != model_height);
  if (need_resize) {
    cv::resize(prepared, model_input, cv::Size(model_width, model_height), 0.0, 0.0, cv::INTER_LINEAR);
  }

  MatrixType eigen_features;
  Eigen::Matrix<float, 1, Eigen::Dynamic> eigen_scores;
  if (!extractor.Infer(model_input, eigen_features, eigen_scores)) {
    std::cerr << "[FeaturePipeline] extractor.Infer failed." << std::endl;
    return false;
  }
  (void)eigen_scores;

  if (need_resize && model_input.cols > 0 && model_input.rows > 0) {
    const float scale_x = static_cast<float>(prepared.cols) / static_cast<float>(model_input.cols);
    const float scale_y = static_cast<float>(prepared.rows) / static_cast<float>(model_input.rows);
    for (int i = 0; i < eigen_features.cols(); ++i) {
      eigen_features(0, i) *= scale_x;
      eigen_features(1, i) *= scale_y;
    }
  }

  auto backend = std::make_shared<MatrixType>(std::move(eigen_features));
  FillKeypointsFromEigen(*backend, features);
  features.family = family;
  features.image_size = cv::Size(prepared.cols, prepared.rows);
  FeaturePayload::AttachBackend(features, backend, family);
  FeaturePayload::AttachImageInfo(features, prepared.cols, prepared.rows, model_input.cols, model_input.rows, true);
  return true;
}

bool ExtractFlexGlueFeatures(const cv::Mat &image, FlexGlueDescriptor *descriptor, LocalFeatures &features) {
  if (!descriptor) {
    return true;
  }
  auto dense = std::make_shared<FlexGlueFeatureVector>();
  if (!descriptor->Infer(image, *dense)) {
    std::cerr << "[FeaturePipeline] descriptor.Infer failed." << std::endl;
    return false;
  }
  FeaturePayload::AttachFlexGlue(features, dense, descriptor->FeatureHeight(), descriptor->FeatureWidth(), descriptor->FeatureDim());
  return true;
}

class LocalExtractor {
 public:
  virtual ~LocalExtractor() = default;
  virtual bool Extract(const cv::Mat &image, LocalFeatures &features) = 0;
  virtual int DescriptorDim() const = 0;
};

class FeatureMatcher {
 public:
  virtual ~FeatureMatcher() = default;
  virtual bool Match(const LocalFeatures &features0, const LocalFeatures &features1, std::vector<cv::DMatch> &matches, bool outlier_rejection) = 0;
  virtual FlexGlueMatcher *GetFlexGlueMatcher() { return nullptr; }
};

class SuperPointExtractor final : public LocalExtractor {
 public:
  explicit SuperPointExtractor(SuperPointConfig config, FlexGlueDescriptor *flexglue_descriptor = nullptr, int model_width = 0, int model_height = 0)
      : extractor_(std::make_unique<SuperPoint>(std::move(config))),
        flexglue_descriptor_(flexglue_descriptor),
        model_width_(model_width),
        model_height_(model_height) {}

  bool Build() {
    if (!extractor_ || !extractor_->Build()) {
      return false;
    }
    return !flexglue_descriptor_ || flexglue_descriptor_->Build();
  }

  bool Extract(const cv::Mat &image, LocalFeatures &features) override {
    if (!extractor_) {
      return false;
    }
    if (!ExtractLocalFeatures<SuperPointFeatureMatrix>(kFamilySuperPoint, *extractor_, image, features, model_width_, model_height_, true)) {
      return false;
    }
    return ExtractFlexGlueFeatures(image, flexglue_descriptor_, features);
  }

  int DescriptorDim() const override { return SuperPoint::kFeatureRows - kFeatureCoordRows; }

 private:
  std::unique_ptr<SuperPoint> extractor_;
  FlexGlueDescriptor *flexglue_descriptor_;
  int model_width_;
  int model_height_;
};

class AlikedExtractor final : public LocalExtractor {
 public:
  explicit AlikedExtractor(AlikedConfig config, FlexGlueDescriptor *flexglue_descriptor = nullptr) : flexglue_descriptor_(flexglue_descriptor) {
    const int model_width = config.image_width;
    const int model_height = config.image_height;
    extractor_ = std::make_unique<Aliked>(std::move(config));
    model_width_ = model_width;
    model_height_ = model_height;
  }

  bool Build() {
    if (!extractor_ || !extractor_->Build()) {
      return false;
    }
    return !flexglue_descriptor_ || flexglue_descriptor_->Build();
  }

  bool Extract(const cv::Mat &image, LocalFeatures &features) override {
    if (!extractor_) {
      return false;
    }
    if (!ExtractLocalFeatures<AlikedFeatureMatrix>(kFamilyAliked, *extractor_, image, features, model_width_, model_height_, false)) {
      return false;
    }
    return ExtractFlexGlueFeatures(image, flexglue_descriptor_, features);
  }

  int DescriptorDim() const override { return Aliked::kFeatureRows - kFeatureCoordRows; }

 private:
  std::unique_ptr<Aliked> extractor_;
  FlexGlueDescriptor *flexglue_descriptor_;
  int model_width_;
  int model_height_;
};

class SuperPointLightGlueMatcher final : public FeatureMatcher {
 public:
  explicit SuperPointLightGlueMatcher(SuperPointLightGlueConfig config) : matcher_(std::make_unique<SuperPointLightGlue>(std::move(config))) {}

  bool Build() {
    if (!matcher_) {
      return false;
    }
    return matcher_->Build();
  }

  bool Match(const LocalFeatures &features0, const LocalFeatures &features1, std::vector<cv::DMatch> &matches, bool outlier_rejection) override {
    if (!matcher_) {
      return false;
    }
    return MatchLocalFeatures<SuperPointFeatureMatrix>(*matcher_, kFamilySuperPoint, features0, features1, matches, outlier_rejection);
  }

 private:
  std::unique_ptr<SuperPointLightGlue> matcher_;
};

class AlikedLightGlueMatcher final : public FeatureMatcher {
 public:
  explicit AlikedLightGlueMatcher(AlikedLightGlueConfig config) : matcher_(std::make_unique<AlikedLightGlue>(std::move(config))) {}

  bool Build() {
    if (!matcher_) {
      return false;
    }
    return matcher_->Build();
  }

  bool Match(const LocalFeatures &features0, const LocalFeatures &features1, std::vector<cv::DMatch> &matches, bool outlier_rejection) override {
    if (!matcher_) {
      return false;
    }
    return MatchLocalFeatures<AlikedFeatureMatrix>(*matcher_, kFamilyAliked, features0, features1, matches, outlier_rejection);
  }

 private:
  std::unique_ptr<AlikedLightGlue> matcher_;
};

class FlexGlueFeatureMatcher final : public FeatureMatcher {
 public:
  FlexGlueFeatureMatcher(FlexGlueConfig config, std::string expected_family)
      : matcher_(std::make_unique<FlexGlueMatcher>(std::move(config))), expected_family_(std::move(expected_family)) {}

  bool Build() { return matcher_ && matcher_->Build(); }

  bool Match(const LocalFeatures &features0, const LocalFeatures &features1, std::vector<cv::DMatch> &matches, bool outlier_rejection) override {
    if (!ValidateMatcherFamily(features0, expected_family_.c_str()) || !ValidateMatcherFamily(features1, expected_family_.c_str())) {
      return false;
    }

    int height0 = 0;
    int width0 = 0;
    int dim0 = 0;
    int height1 = 0;
    int width1 = 0;
    int dim1 = 0;
    const auto dense0 = FeaturePayload::FlexGlueOf(features0, height0, width0, dim0);
    const auto dense1 = FeaturePayload::FlexGlueOf(features1, height1, width1, dim1);
    if (!dense0 || !dense1 || height0 != height1 || width0 != width1 || dim0 != dim1) {
      std::cerr << "[FlexGlueMatcher] Missing or mismatched image feature maps." << std::endl;
      return false;
    }

    const int image0_width = features0.image_size.width;
    const int image0_height = features0.image_size.height;
    const int image1_width = features1.image_size.width;
    const int image1_height = features1.image_size.height;
    if (image0_width <= 0 || image0_height <= 0 || image1_width <= 0 || image1_height <= 0) {
      std::cerr << "[FlexGlueMatcher] Image size is missing." << std::endl;
      return false;
    }

    const auto &vectors0 = *dense0;
    const auto &vectors1 = *dense1;
    return matcher_->Infer(vectors0, vectors1, features0.keypoints, features1.keypoints, image0_width, image0_height, image1_width, image1_height, matches,
                           outlier_rejection);
  }

  FlexGlueMatcher *GetFlexGlueMatcher() override { return matcher_.get(); }

 private:
  std::unique_ptr<FlexGlueMatcher> matcher_;
  std::string expected_family_;
};

template <typename Config>
bool HasValidTensorConfig(const Config &config) {
  return !config.input_tensor_names.empty() && !config.output_tensor_names.empty();
}

template <typename Impl, typename Config, typename Factory>
std::unique_ptr<Impl> CreateAndBuild(const Config &config, Factory &&make_impl) {
  if (!HasValidTensorConfig(config)) {
    return nullptr;
  }
  auto impl = make_impl(config);
  if (!impl->Build()) {
    return nullptr;
  }
  return impl;
}

std::unique_ptr<LocalExtractor> CreateLocalFeatureExtractorFromConfigs(const std::string &normalized_type, const Configs &configs,
                                                                        FlexGlueDescriptor *flexglue_descriptor = nullptr) {
  if (normalized_type == kFamilySuperPoint) {
    return CreateAndBuild<SuperPointExtractor>(configs.superpoint_config, [&](const SuperPointConfig &cfg) {
      return std::make_unique<SuperPointExtractor>(cfg, flexglue_descriptor, cfg.image_width, cfg.image_height);
    });
  }
  if (normalized_type == kFamilyAliked) {
    if (HasValidTensorConfig(configs.aliked_config) && configs.aliked_config.descriptor_dim != Aliked::kDescriptorDim) {
      std::cerr << "[FeaturePipeline] aliked.descriptor_dim is " << configs.aliked_config.descriptor_dim << "; expected " << Aliked::kDescriptorDim << "."
                << std::endl;
      return nullptr;
    }
    return CreateAndBuild<AlikedExtractor>(configs.aliked_config, [&](const AlikedConfig &cfg) { return std::make_unique<AlikedExtractor>(cfg, flexglue_descriptor); });
  }
  return nullptr;
}

std::unique_ptr<FeatureMatcher> CreateFeatureMatcherFromConfigs(const std::string &normalized_type, const Configs &configs,
                                                                const char *feature_family = nullptr) {
  if (normalized_type == kMatcherSuperPointLightGlue) {
    return CreateAndBuild<SuperPointLightGlueMatcher>(configs.superpoint_lightglue_config,
                                                      [](const SuperPointLightGlueConfig &cfg) { return std::make_unique<SuperPointLightGlueMatcher>(cfg); });
  }
  if (normalized_type == kMatcherAlikedLightGlue) {
    if (HasValidTensorConfig(configs.aliked_lightglue_config) && configs.aliked_lightglue_config.descriptor_dim != AlikedLightGlue::kDescriptorDim) {
      std::cerr << "[FeaturePipeline] aliked_lightglue.descriptor_dim is " << configs.aliked_lightglue_config.descriptor_dim << "; expected "
                << AlikedLightGlue::kDescriptorDim << "." << std::endl;
      return nullptr;
    }
    return CreateAndBuild<AlikedLightGlueMatcher>(configs.aliked_lightglue_config,
                                                  [](const AlikedLightGlueConfig &cfg) { return std::make_unique<AlikedLightGlueMatcher>(cfg); });
  }
  if (normalized_type == kMatcherFlexGlue) {
    if (feature_family == nullptr || feature_family[0] == '\0') {
      return nullptr;
    }
    auto matcher = std::make_unique<FlexGlueFeatureMatcher>(configs.flexglue_config, feature_family);
    if (!matcher->Build()) {
      return nullptr;
    }
    return matcher;
  }
  return nullptr;
}

template <typename MatrixType>
bool ExportDescriptors(const LocalFeatures &features, const char *family, cv::Mat &descriptors) {
  const auto backend = FeaturePayload::BackendOf<MatrixType>(features, family);
  if (!backend) {
    return false;
  }
  const int num_points = backend->cols();
  const int descriptor_dim = backend->rows() - kFeatureCoordRows;
  if (descriptor_dim <= 0) {
    return false;
  }
  descriptors.create(num_points, descriptor_dim, CV_32F);
  for (int i = 0; i < num_points; ++i) {
    float *row = descriptors.ptr<float>(i);
    for (int d = 0; d < descriptor_dim; ++d) {
      row[d] = (*backend)(kFeatureCoordRows + d, i);
    }
  }
  return true;
}

template <typename MatrixType>
bool ImportLocalFeatures(const std::vector<cv::Point2f> &keypoints, const cv::Mat &descriptors, const cv::Size &image_size, const char *family, int expected_dim,
                         LocalFeatures &features) {
  if (image_size.width <= 0 || image_size.height <= 0) {
    return false;
  }
  if (static_cast<int>(keypoints.size()) != descriptors.rows) {
    return false;
  }

  features.keypoints = keypoints;
  features.descriptors = descriptors;
  features.image_size = image_size;
  features.family = family;
  FeaturePayload::AttachImageInfo(features, image_size.width, image_size.height, image_size.width, image_size.height, true);

  if (keypoints.empty()) {
    return true;
  }
  if (descriptors.type() != CV_32F || descriptors.cols != expected_dim) {
    return false;
  }
  if (expected_dim != MatrixType::RowsAtCompileTime - kFeatureCoordRows) {
    return false;
  }

  const cv::Mat contiguous = descriptors.isContinuous() ? descriptors : descriptors.clone();
  auto backend = std::make_shared<MatrixType>(MatrixType::RowsAtCompileTime, static_cast<int>(keypoints.size()));
  for (int i = 0; i < backend->cols(); ++i) {
    (*backend)(0, i) = keypoints[static_cast<size_t>(i)].x;
    (*backend)(1, i) = keypoints[static_cast<size_t>(i)].y;
    const float *row = contiguous.ptr<float>(i);
    for (int d = 0; d < expected_dim; ++d) {
      (*backend)(kFeatureCoordRows + d, i) = row[d];
    }
  }
  FeaturePayload::AttachBackend(features, backend, family);
  return true;
}

bool ExportDescriptorsForFamily(const LocalFeatures &features, const std::string &family, cv::Mat &descriptors) {
  if (family == kFamilyAliked) {
    return ExportDescriptors<AlikedFeatureMatrix>(features, kFamilyAliked, descriptors);
  }
  if (family == kFamilySuperPoint) {
    return ExportDescriptors<SuperPointFeatureMatrix>(features, kFamilySuperPoint, descriptors);
  }
  return false;
}

bool ImportLocalFeaturesForFamily(const std::vector<cv::Point2f> &keypoints, const cv::Mat &descriptors, const cv::Size &image_size, const std::string &family, int expected_dim,
                                  LocalFeatures &features) {
  if (family == kFamilyAliked) {
    return ImportLocalFeatures<AlikedFeatureMatrix>(keypoints, descriptors, image_size, kFamilyAliked, expected_dim, features);
  }
  if (family == kFamilySuperPoint) {
    return ImportLocalFeatures<SuperPointFeatureMatrix>(keypoints, descriptors, image_size, kFamilySuperPoint, expected_dim, features);
  }
  return false;
}

}  // namespace

struct FeaturePipeline::Impl {
  std::string family;
  std::string local_family;
  int descriptor_dim = 0;
  std::unique_ptr<FlexGlueDescriptor> flexglue_descriptor;
  std::unique_ptr<LocalExtractor> extractor;
  std::unique_ptr<FeatureMatcher> matcher;
  FlexGlueMatcher *flexglue_matcher = nullptr;
};

FeaturePipeline::FeaturePipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

FeaturePipeline::FeaturePipeline(FeaturePipeline &&) noexcept = default;

FeaturePipeline &FeaturePipeline::operator=(FeaturePipeline &&) noexcept = default;

FeaturePipeline::~FeaturePipeline() = default;

void FeaturePipeline::SetSearchPaths(const std::string &config_path, const std::string &model_dir) { ::viewfeat::SetSearchPaths(config_path, model_dir); }

std::unique_ptr<FeaturePipeline> FeaturePipeline::Create(const std::string &family) {
  const std::string requested = LowercaseAscii(family);
  const bool use_feature_type = requested.empty() || requested == "default";

  std::shared_ptr<const Configs> configs;
  std::string normalized = requested;
  if (use_feature_type) {
    configs = GetConfigs();
    if (!configs) {
      std::cerr << "[FeaturePipeline] Failed to load config from " << ResolveConfigPath() << "." << std::endl;
      return nullptr;
    }
    normalized = LowercaseAscii(configs->feature_type);
  }
  if (!IsPipelineFamily(normalized)) {
    std::cerr << "[FeaturePipeline] Unsupported feature family: " << (use_feature_type ? configs->feature_type : family) << "." << std::endl;
    return nullptr;
  }

  if (!configs) {
    configs = GetConfigs();
    if (!configs) {
      std::cerr << "[FeaturePipeline] Failed to load config from " << ResolveConfigPath() << "." << std::endl;
      return nullptr;
    }
  }

  std::string local_family;
  std::unique_ptr<FlexGlueDescriptor> flexglue_descriptor;
  std::unique_ptr<LocalExtractor> extractor;
  std::unique_ptr<FeatureMatcher> matcher;

  if (normalized == kFamilySuperPoint) {
    local_family = kFamilySuperPoint;
    extractor = CreateLocalFeatureExtractorFromConfigs(kFamilySuperPoint, *configs);
    matcher = CreateFeatureMatcherFromConfigs(kMatcherSuperPointLightGlue, *configs);
  } else if (normalized == kFamilyAliked) {
    local_family = kFamilyAliked;
    extractor = CreateLocalFeatureExtractorFromConfigs(kFamilyAliked, *configs);
    matcher = CreateFeatureMatcherFromConfigs(kMatcherAlikedLightGlue, *configs);
  } else {
    local_family = normalized == kPipelineSuperPointFlexGlue ? kFamilySuperPoint : kFamilyAliked;
    flexglue_descriptor = std::make_unique<FlexGlueDescriptor>(configs->flexglue_config);
    extractor = CreateLocalFeatureExtractorFromConfigs(local_family, *configs, flexglue_descriptor.get());
    matcher = CreateFeatureMatcherFromConfigs(kMatcherFlexGlue, *configs, local_family.c_str());
  }

  if (!extractor || !matcher) {
    std::cerr << "[FeaturePipeline] Failed to build " << normalized << " using " << ResolveConfigPath() << "." << std::endl;
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->family = normalized;
  impl->local_family = local_family;
  impl->descriptor_dim = extractor->DescriptorDim();
  impl->extractor = std::move(extractor);
  impl->matcher = std::move(matcher);
  impl->flexglue_descriptor = std::move(flexglue_descriptor);
  impl->flexglue_matcher = impl->matcher->GetFlexGlueMatcher();
  return std::unique_ptr<FeaturePipeline>(new FeaturePipeline(std::move(impl)));
}

bool FeaturePipeline::Extract(const cv::Mat &image, LocalFeatures &features) {
  if (!impl_ || !impl_->extractor || image.empty()) {
    return false;
  }
  features = LocalFeatures();
  if (!impl_->extractor->Extract(image, features) || !ExportDescriptorsForFamily(features, impl_->local_family, features.descriptors)) {
    features = LocalFeatures();
    return false;
  }
  FeaturePayload::AttachPipelineFamily(features, impl_->family);
  return true;
}

bool FeaturePipeline::Match(const LocalFeatures &features0, const LocalFeatures &features1, std::vector<cv::DMatch> &matches, bool outlier_rejection) {
  matches.clear();
  if (!impl_ || !impl_->matcher) {
    return false;
  }
  if (!ValidateMatcherFamily(features0, impl_->local_family.c_str()) || !ValidateMatcherFamily(features1, impl_->local_family.c_str()) ||
      !PipelineFamilyMatches(features0, impl_->family) || !PipelineFamilyMatches(features1, impl_->family)) {
    std::cerr << "[FeaturePipeline] Feature family does not match pipeline " << impl_->family << "." << std::endl;
    return false;
  }

  const bool flexglue_pipeline = impl_->family == kPipelineSuperPointFlexGlue || impl_->family == kPipelineAlikedFlexGlue;
  if (flexglue_pipeline) {
    return impl_->matcher->Match(features0, features1, matches, outlier_rejection);
  }
  if (HasFlexGlueFeatures(features0) || HasFlexGlueFeatures(features1)) {
    return false;
  }

  LocalFeatures rebuilt0;
  LocalFeatures rebuilt1;
  if (!ImportLocalFeaturesForFamily(features0.keypoints, features0.descriptors, features0.image_size, impl_->local_family, impl_->descriptor_dim, rebuilt0) ||
      !ImportLocalFeaturesForFamily(features1.keypoints, features1.descriptors, features1.image_size, impl_->local_family, impl_->descriptor_dim, rebuilt1)) {
    return false;
  }
  return impl_->matcher->Match(rebuilt0, rebuilt1, matches, outlier_rejection);
}

bool FeaturePipeline::Match(const std::vector<cv::Point2f> &keypoints0, const cv::Mat &descriptors0, const cv::Size &image0_size, const std::vector<cv::Point2f> &keypoints1,
                            const cv::Mat &descriptors1, const cv::Size &image1_size, std::vector<cv::DMatch> &matches, bool outlier_rejection) {
  matches.clear();
  if (!impl_ || !impl_->matcher) {
    return false;
  }

  LocalFeatures features0;
  LocalFeatures features1;
  if (!ImportLocalFeaturesForFamily(keypoints0, descriptors0, image0_size, impl_->local_family, impl_->descriptor_dim, features0) ||
      !ImportLocalFeaturesForFamily(keypoints1, descriptors1, image1_size, impl_->local_family, impl_->descriptor_dim, features1)) {
    return false;
  }
  return Match(features0, features1, matches, outlier_rejection);
}

bool FeaturePipeline::MatchImages(const cv::Mat &image0, const cv::Mat &image1, std::vector<cv::DMatch> &matches, bool outlier_rejection) {
  matches.clear();
  if (!impl_ || !impl_->extractor || !impl_->matcher || image0.empty() || image1.empty()) {
    return false;
  }
  LocalFeatures features0;
  LocalFeatures features1;
  return impl_->extractor->Extract(image0, features0) && impl_->extractor->Extract(image1, features1) &&
         impl_->matcher->Match(features0, features1, matches, outlier_rejection);
}

bool FeaturePipeline::MatchWithImages(const cv::Mat &image0, const std::vector<cv::Point2f> &keypoints0, const cv::Mat &image1, const std::vector<cv::Point2f> &keypoints1,
                                      std::vector<cv::DMatch> &matches, bool outlier_rejection) {
  matches.clear();
  if (!impl_ || !impl_->flexglue_descriptor || !impl_->flexglue_matcher || image0.empty() || image1.empty()) {
    return false;
  }

  std::vector<float> features0;
  std::vector<float> features1;
  if (!impl_->flexglue_descriptor->Infer(image0, features0) || !impl_->flexglue_descriptor->Infer(image1, features1)) {
    return false;
  }
  return impl_->flexglue_matcher->Infer(features0, features1, keypoints0, keypoints1, image0.cols, image0.rows, image1.cols, image1.rows, matches, outlier_rejection);
}

const std::string &FeaturePipeline::Family() const {
  static const std::string kEmpty;
  if (!impl_) {
    return kEmpty;
  }
  return impl_->family;
}

int FeaturePipeline::DescriptorDim() const { return impl_ ? impl_->descriptor_dim : 0; }

}  // namespace viewfeat
