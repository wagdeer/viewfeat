#ifndef VIEWFEAT_PIPELINE_H_
#define VIEWFEAT_PIPELINE_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "viewfeat/types.h"

namespace viewfeat {

// Local feature extraction and matching.
// family: "", "default", "superpoint", "aliked", "superpoint_flexglue", "aliked_flexglue".
// An empty string or "default" uses configs.feature_type. A missing feature_type key still selects SuperPoint.
// SetSearchPaths is process-wide and forwards to the function in types.h. GlobalRetriever reads the same paths.
class FeaturePipeline {
 public:
  static void SetSearchPaths(const std::string &config_path, const std::string &model_dir);

  static std::unique_ptr<FeaturePipeline> Create(const std::string &family);

  FeaturePipeline(FeaturePipeline &&) noexcept;
  FeaturePipeline &operator=(FeaturePipeline &&) noexcept;
  ~FeaturePipeline();

  FeaturePipeline(const FeaturePipeline &) = delete;
  FeaturePipeline &operator=(const FeaturePipeline &) = delete;

  bool Extract(const cv::Mat &image, LocalFeatures &features);

  bool Match(const LocalFeatures &features0, const LocalFeatures &features1, std::vector<cv::DMatch> &matches, bool outlier_rejection = false);

  bool Match(const std::vector<cv::Point2f> &keypoints0, const cv::Mat &descriptors0, const cv::Size &image0_size,
             const std::vector<cv::Point2f> &keypoints1, const cv::Mat &descriptors1, const cv::Size &image1_size, std::vector<cv::DMatch> &matches,
             bool outlier_rejection = false);

  bool MatchImages(const cv::Mat &image0, const cv::Mat &image1, std::vector<cv::DMatch> &matches, bool outlier_rejection = false);

  bool MatchWithImages(const cv::Mat &image0, const std::vector<cv::Point2f> &keypoints0, const cv::Mat &image1,
                       const std::vector<cv::Point2f> &keypoints1, std::vector<cv::DMatch> &matches, bool outlier_rejection = false);

  const std::string &Family() const;
  int DescriptorDim() const;

 private:
  struct Impl;
  explicit FeaturePipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace viewfeat

#endif  // VIEWFEAT_PIPELINE_H_
