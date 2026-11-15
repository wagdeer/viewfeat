#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.h"
#include "viewfeat/pipeline.h"

// Sequential adjacent-frame matching via FeaturePipeline (reuse features across pairs).

namespace {

double ElapsedMs(const std::chrono::high_resolution_clock::time_point &start) {
  using namespace std::chrono;
  return duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 1000.0;
}

bool PrepareOutputFolder(const std::string &output_path) {
  const std::string mkdir_cmd = "mkdir -p \"" + output_path + "\"";
  return system(mkdir_cmd.c_str()) == 0;
}

cv::Mat LoadResizedImage(const std::string &path, int width, int height) {
  cv::Mat raw = cv::imread(path, cv::IMREAD_COLOR);
  if (raw.empty()) {
    return raw;
  }
  cv::Mat resized;
  cv::resize(raw, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
  return resized;
}

}  // namespace

int RunSequence(const std::string &feature_family, const std::vector<std::string> &image_names, const std::string &output_path, int target_width, int target_height) {
  std::cout << "Building " << feature_family << " feature pipeline......" << std::endl;
  // Create once; the pipeline stays loaded for the whole sequence.
  const auto pipeline = viewfeat::FeaturePipeline::Create(feature_family);
  if (!pipeline) {
    std::cerr << "Failed to create feature pipeline for: " << feature_family << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Pipeline ready. family=" << pipeline->Family() << std::endl;

  cv::Mat prev_image = LoadResizedImage(image_names[0], target_width, target_height);
  if (prev_image.empty()) {
    std::cerr << "First image in the image folder is empty." << std::endl;
    return EXIT_FAILURE;
  }

  viewfeat::LocalFeatures prev_features;
  if (!pipeline->Extract(prev_image, prev_features)) {  // seed frame for pair (0,1)
    std::cerr << "Failed when extracting features from first image." << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t index = 1; index < image_names.size(); ++index) {
    cv::Mat curr_image = LoadResizedImage(image_names[index], target_width, target_height);
    if (curr_image.empty()) {
      std::cerr << "Skip empty image: " << image_names[index] << std::endl;
      continue;
    }

    viewfeat::LocalFeatures curr_features;
    std::vector<cv::DMatch> matches;

    const auto extract_start = std::chrono::high_resolution_clock::now();
    if (!pipeline->Extract(curr_image, curr_features)) {
      std::cerr << "Failed when extracting features from image: " << image_names[index] << std::endl;
      return EXIT_FAILURE;
    }
    const double extract_time = ElapsedMs(extract_start);

    const auto match_start = std::chrono::high_resolution_clock::now();
    // Match consecutive frames only; both feature sets must come from this pipeline family.
    if (!pipeline->Match(prev_features, curr_features, matches, false)) {
      std::cerr << "Failed when matching adjacent images." << std::endl;
      return EXIT_FAILURE;
    }
    const double match_time = ElapsedMs(match_start);

    const double cost_time = extract_time + match_time;
    std::cout << "Pair " << index - 1 << " -> " << index << ", keypoints: " << prev_features.keypoints.size() << ":" << curr_features.keypoints.size()
              << ", matches: " << matches.size() << ", extract: " << extract_time << " ms, match: " << match_time << " ms, cost: " << cost_time << " ms" << std::endl;

    cv::Mat match_image;
    const std::vector<cv::KeyPoint> keypoints0 = ToKeypoints(prev_features);
    const std::vector<cv::KeyPoint> keypoints1 = ToKeypoints(curr_features);
    VisualizeMatching(prev_image, keypoints0, curr_image, keypoints1, matches, match_image, cost_time);
    cv::imwrite(output_path + "/" + std::to_string(index - 1) + "_" + std::to_string(index) + ".png", match_image);

    prev_image = curr_image;
    prev_features = std::move(curr_features);  // slide window: current becomes previous
  }
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 6) {
    std::cerr << "Usage: " << argv[0]
              << " image_folder_absolute_path output_folder_path "
                 "[superpoint|aliked|superpoint_flexglue|aliked_flexglue] [width] [height]"
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string image_path = argv[1];
  const std::string output_path = argv[2];
  const std::string feature_family = argc >= 4 ? argv[3] : "superpoint";
  const int target_width = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 512;
  const int target_height = argc >= 6 ? std::max(1, std::atoi(argv[5])) : 288;

  if (!PrepareOutputFolder(output_path)) {
    std::cerr << "Failed to create output folder: " << output_path << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> image_names;
  if (!GetFileNames(image_path, image_names) || image_names.size() < 2) {
    std::cerr << "Image folder must contain at least two images: " << image_path << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Feature family: " << feature_family << std::endl;
  std::cout << "Image count: " << image_names.size() << std::endl;
  std::cout << "Resize target: " << target_width << "x" << target_height << std::endl;
  return RunSequence(feature_family, image_names, output_path, target_width, target_height);
}
