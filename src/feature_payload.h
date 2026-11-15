#ifndef VIEWFEAT_FEATURE_PAYLOAD_H_
#define VIEWFEAT_FEATURE_PAYLOAD_H_

#include <memory>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include "viewfeat/types.h"

namespace viewfeat {

class FeaturePayload {
 public:
  static void AttachBackend(LocalFeatures &features, std::shared_ptr<void> blob, std::string tag, std::type_index type);

  template <typename MatrixType>
  static void AttachBackend(LocalFeatures &features, std::shared_ptr<MatrixType> blob, std::string tag) {
    AttachBackend(features, std::shared_ptr<void>(std::move(blob)), std::move(tag), std::type_index(typeid(MatrixType)));
  }

  static std::shared_ptr<void> BackendOf(const LocalFeatures &features, const char *family, std::type_index type);

  template <typename MatrixType>
  static std::shared_ptr<MatrixType> BackendOf(const LocalFeatures &features, const char *family) {
    return std::static_pointer_cast<MatrixType>(BackendOf(features, family, std::type_index(typeid(MatrixType))));
  }

  static void AttachFlexGlue(LocalFeatures &features, std::shared_ptr<std::vector<float>> blob, int h, int w, int dim);
  static std::shared_ptr<std::vector<float>> FlexGlueOf(const LocalFeatures &features, int &h, int &w, int &dim);
  static void AttachImageInfo(LocalFeatures &features, int original_width, int original_height, int infer_width, int infer_height, bool coords_in_original);
  static void AttachPipelineFamily(LocalFeatures &features, std::string pipeline_family);
  static std::string PipelineFamilyOf(const LocalFeatures &features);
  static bool ImageInfo(const LocalFeatures &features, int &original_width, int &original_height, int &infer_width, int &infer_height, bool &coords_in_original);

 private:
  static LocalFeatures::Impl &Ensure(LocalFeatures &features);
};

}  // namespace viewfeat

#endif  // VIEWFEAT_FEATURE_PAYLOAD_H_
