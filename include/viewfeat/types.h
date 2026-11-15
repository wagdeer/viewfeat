#ifndef VIEWFEAT_TYPES_H_
#define VIEWFEAT_TYPES_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace viewfeat {

class FeaturePayload;  // Defined in src/feature_payload.h. Not installed.

// Public output of local feature extraction.
// keypoints are in the coordinate frame of image_size.
// descriptors is N x D, CV_32F, and is always filled after a successful Extract.
// image_size is that frame, in pixels.
// family is "superpoint" or "aliked". It only names this sparse descriptor.
class LocalFeatures {
 public:
  std::vector<cv::Point2f> keypoints;
  cv::Mat descriptors;
  cv::Size image_size;
  std::string family;

  LocalFeatures();
  LocalFeatures(LocalFeatures &&) noexcept;
  LocalFeatures &operator=(LocalFeatures &&) noexcept;
  LocalFeatures(const LocalFeatures &) = delete;
  LocalFeatures &operator=(const LocalFeatures &) = delete;
  ~LocalFeatures();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class FeaturePayload;
};

// Process-wide config file and weight directory.
// FeaturePipeline::SetSearchPaths forwards to this function. GlobalRetriever reads the same paths.
void SetSearchPaths(const std::string &config_path, const std::string &model_dir);

}  // namespace viewfeat

#endif  // VIEWFEAT_TYPES_H_
