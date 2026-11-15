#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.h"
#include "viewfeat/pipeline.h"

// Two-image local feature matching via FeaturePipeline.

int RunFeatureMatch(const std::string &feature_family, const std::string &image0_path, const std::string &image1_path, const std::string &output_path, int iterations,
                    int target_width, int target_height) {
  cv::Mat raw_image0 = cv::imread(image0_path, cv::IMREAD_COLOR);
  cv::Mat raw_image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
  if (raw_image0.empty() || raw_image1.empty()) {
    std::cerr << "Input image is empty. Please check the image path." << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat image0;
  cv::Mat image1;
  // Resize to the same resolution used at library build time (config superpoint / aliked).
  cv::resize(raw_image0, image0, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_LINEAR);
  cv::resize(raw_image1, image1, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_LINEAR);
  std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl;
  std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl;

  std::cout << "Building " << feature_family << " feature pipeline......" << std::endl;
  // One call wires a matched extractor + matcher ("superpoint", "aliked", or a FlexGlue family).
  const auto pipeline = viewfeat::FeaturePipeline::Create(feature_family);
  if (!pipeline) {
    std::cerr << "Failed to create feature pipeline for: " << feature_family << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Pipeline ready. family=" << pipeline->Family() << std::endl;

  // LocalFeatures holds OpenCV keypoints and an N x D CV_32F descriptor matrix.
  viewfeat::LocalFeatures features0;
  viewfeat::LocalFeatures features1;
  std::vector<cv::DMatch> matches;

  long image0_time_count = 0;
  long image1_time_count = 0;
  long match_time_count = 0;
  std::cout << feature_family << " test in " << iterations << " times." << std::endl;
  for (int i = 0; i < iterations; ++i) {
    std::cout << "---------------------------------------------------------" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    if (!pipeline->Extract(image0, features0)) {  // BGR cv::Mat in
      std::cerr << "Failed when extracting features from first image." << std::endl;
      return EXIT_FAILURE;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (i > 0) {
      std::cout << "First image feature points number: " << features0.keypoints.size() << std::endl;
      image0_time_count += duration.count();
      std::cout << "First image infer cost " << image0_time_count / i / 1000.0 << " ms" << std::endl;
    }

    start = std::chrono::high_resolution_clock::now();
    if (!pipeline->Extract(image1, features1)) {
      std::cerr << "Failed when extracting features from second image." << std::endl;
      return EXIT_FAILURE;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (i > 0) {
      std::cout << "Second image feature points number: " << features1.keypoints.size() << std::endl;
      image1_time_count += duration.count();
      std::cout << "Second image infer cost " << image1_time_count / i / 1000.0 << " ms" << std::endl;
    }

    start = std::chrono::high_resolution_clock::now();
    // Match() needs features from the same pipeline; normalization uses LocalFeatures metadata.
    if (!pipeline->Match(features0, features1, matches, false)) {
      std::cerr << "Failed when matching features." << std::endl;
      return EXIT_FAILURE;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (i > 0) {
      match_time_count += duration.count();
      std::cout << "Matches number: " << matches.size() << std::endl;
      std::cout << "Match image cost " << match_time_count / i / 1000.0 << " ms" << std::endl;
    }
  }

  // After Match(), use OpenCV types only (example helper — not part of viewfeat).
  cv::Mat match_image;
  const std::vector<cv::KeyPoint> keypoints0 = ToKeypoints(features0);
  const std::vector<cv::KeyPoint> keypoints1 = ToKeypoints(features1);
  VisualizeMatching(image0, keypoints0, image1, keypoints1, matches, match_image);
  if (!cv::imwrite(output_path, match_image)) {
    std::cerr << "Failed to write output image: " << output_path << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Output image saved to " << output_path << std::endl;
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 8) {
    std::cerr << "Usage: " << argv[0]
              << " first_image_path second_image_path [output_image_path] [iterations] "
                 "[superpoint|aliked|superpoint_flexglue|aliked_flexglue] [width] [height]"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string image0_path = argv[1];
  const std::string image1_path = argv[2];
  const std::string output_path = argc >= 4 ? argv[3] : "match_image.png";
  const int iterations = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 100;
  const std::string feature_family = argc >= 6 ? argv[5] : "superpoint";
  const int target_width = argc >= 7 ? std::max(1, std::atoi(argv[6])) : 512;
  const int target_height = argc >= 8 ? std::max(1, std::atoi(argv[7])) : 288;

  if (!FileExists(image0_path) || !FileExists(image1_path)) {
    std::cerr << "Input image does not exist. Please check image paths." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Feature family: " << feature_family << std::endl;
  return RunFeatureMatch(feature_family, image0_path, image1_path, output_path, iterations, target_width, target_height);
}
