#ifndef VIEWFEAT_MODELS_OUTLIER_REJECTION_H_
#define VIEWFEAT_MODELS_OUTLIER_REJECTION_H_

#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>

namespace viewfeat {

// points0, points1, and matches are parallel. Fewer than 8 points, or a failed fundamental-matrix
// estimate, leaves matches unchanged.
inline void RejectFundamentalOutliers(const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1,
                                      std::vector<cv::DMatch> &matches) {
  if (points0.size() < 8 || points0.size() != points1.size() || points0.size() != matches.size()) {
    return;
  }

  std::vector<uchar> inliers;
  cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3.0, 0.99, inliers);
  if (inliers.size() != matches.size()) {
    return;
  }

  size_t kept = 0;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (inliers[i] != 0) {
      matches[kept++] = matches[i];
    }
  }
  matches.resize(kept);
}

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_OUTLIER_REJECTION_H_
