#ifndef VIEWFEAT_MODELS_FLEXGLUE_H_
#define VIEWFEAT_MODELS_FLEXGLUE_H_

#include <NvInfer.h>

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

#include "buffers.h"
#include "runtime/config.h"

namespace viewfeat {

class FlexGlueDescriptor {
 public:
  explicit FlexGlueDescriptor(FlexGlueConfig config);
  ~FlexGlueDescriptor();

  bool Build();
  bool Infer(const cv::Mat &image, std::vector<float> &features);

  int FeatureHeight() const { return config_.feature_height; }
  int FeatureWidth() const { return config_.feature_width; }
  int FeatureDim() const { return config_.feature_dim; }

 private:
  bool DeserializeEngine();
  bool EnsureBuffers();
  bool ProcessInput(const samplesCommon::BufferManager &buffers, const cv::Mat &image);

  FlexGlueConfig config_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<samplesCommon::BufferManager> buffers_;
  cudaStream_t stream_{nullptr};
};

class FlexGlueMatcher {
 public:
  explicit FlexGlueMatcher(FlexGlueConfig config);
  ~FlexGlueMatcher();

  bool Build();
  bool Infer(const std::vector<float> &features0, const std::vector<float> &features1, const std::vector<cv::Point2f> &keypoints0,
             const std::vector<cv::Point2f> &keypoints1, int image0_width, int image0_height, int image1_width, int image1_height,
             std::vector<cv::DMatch> &matches, bool outlier_rejection = false, float *covisibility_score = nullptr);

 private:
  bool DeserializeEngine();
  bool EnsureBuffers(int keypoints0, int keypoints1);

  FlexGlueConfig config_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<samplesCommon::BufferManager> buffers_;
  int buffer_keypoints0_{0};
  int buffer_keypoints1_{0};
  cudaStream_t stream_{nullptr};
};

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_FLEXGLUE_H_
