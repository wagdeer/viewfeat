#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "viewfeat/pipeline.h"
#include "viewfeat/retriever.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string &message) {
  if (condition) {
    return;
  }
  std::cerr << "CHECK failed: " << message << std::endl;
  ++g_failures;
}

bool RegularFile(const std::string &path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

bool HaveNvidiaDevice() {
  struct stat info;
  // Native Linux exposes /dev/nvidia*. WSL2 exposes the GPU through /dev/dxg.
  return stat("/dev/nvidia0", &info) == 0 || stat("/dev/nvidiactl", &info) == 0 || stat("/dev/dxg", &info) == 0;
}

int Skip(const std::string &reason) {
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

cv::Mat ReadColor(const std::string &path) {
  return cv::imread(path, cv::IMREAD_COLOR);
}

struct ChannelImage {
  const char *name;
  int channels;
  cv::Mat image;
};

std::vector<ChannelImage> ChannelVariants(const cv::Mat &bgr) {
  cv::Mat gray;
  cv::Mat bgra;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
  return {{"gray", 1, gray}, {"bgr", 3, bgr}, {"bgra", 4, bgra}};
}

void CheckChannelExtractAndMatch(viewfeat::FeaturePipeline &pipeline, const std::string &name, const cv::Mat &image0, const cv::Mat &image1) {
  const std::vector<ChannelImage> variants0 = ChannelVariants(image0);
  const std::vector<ChannelImage> variants1 = ChannelVariants(image1);
  for (size_t i = 0; i < variants0.size(); ++i) {
    const ChannelImage &left = variants0[i];
    const ChannelImage &right = variants1[i];
    const std::string label = name + " " + left.name;
    Check(left.image.channels() == left.channels && right.image.channels() == right.channels, label + " channel count");
    viewfeat::LocalFeatures features0;
    viewfeat::LocalFeatures features1;
    if (!pipeline.Extract(left.image, features0) || !pipeline.Extract(right.image, features1)) {
      Check(false, label + " Extract failed");
      continue;
    }
    Check(!features0.keypoints.empty() || !features0.descriptors.empty(), label + " image0 features empty");
    Check(!features1.keypoints.empty() || !features1.descriptors.empty(), label + " image1 features empty");
    std::vector<cv::DMatch> matches;
    if (!pipeline.Match(features0, features1, matches, false)) {
      Check(false, label + " Match failed");
      continue;
    }
    Check(!matches.empty(), label + " matches empty");
    std::cout << label << " channels=" << left.image.channels() << " keypoints=" << features0.keypoints.size() << "/" << features1.keypoints.size()
              << " matches=" << matches.size() << std::endl;
  }
}

void CheckRetrieverChannels(viewfeat::GlobalRetriever &retriever, const cv::Mat &image, const std::string &family = "cosplace") {
  for (const ChannelImage &variant : ChannelVariants(image)) {
    const std::string label = family + " " + variant.name;
    Check(variant.image.channels() == variant.channels, label + " channel count");
    std::vector<float> descriptor;
    if (!retriever.Infer(variant.image, descriptor)) {
      Check(false, label + " Infer failed");
      continue;
    }
    Check(!descriptor.empty(), label + " descriptor empty");
    Check(static_cast<int>(descriptor.size()) == retriever.DescriptorDim(),
          label + " descriptor length " + std::to_string(descriptor.size()) + " != " + std::to_string(retriever.DescriptorDim()));
    std::cout << label << " channels=" << variant.image.channels() << " dim=" << descriptor.size() << std::endl;
  }
}

void CheckLocalFeatures(const std::string &name, const viewfeat::LocalFeatures &features, const cv::Mat &image, int descriptor_dim, const std::string &family) {
  const int count = static_cast<int>(features.keypoints.size());
  Check(count >= 100, name + " keypoints " + std::to_string(count) + " < 100");
  Check(features.family == family, name + " family is " + features.family);
  Check(features.image_size.width == image.cols && features.image_size.height == image.rows, name + " image_size");
  Check(features.descriptors.rows == count, name + " descriptor rows do not match keypoints");
  Check(features.descriptors.cols == descriptor_dim, name + " descriptor cols " + std::to_string(features.descriptors.cols));
  Check(features.descriptors.type() == CV_32F, name + " descriptors are not CV_32F");
  for (int i = 0; i < count; ++i) {
    const cv::Point2f &point = features.keypoints[static_cast<size_t>(i)];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < -1.0F || point.y < -1.0F || point.x > static_cast<float>(image.cols) ||
        point.y > static_cast<float>(image.rows)) {
      Check(false, name + " keypoint " + std::to_string(i) + " is outside the image");
      break;
    }
  }
}

void CheckMatches(const std::string &name, const std::vector<cv::DMatch> &matches, int count0, int count1, int minimum) {
  Check(static_cast<int>(matches.size()) >= minimum, name + " matches " + std::to_string(matches.size()) + " < " + std::to_string(minimum));
  for (size_t i = 0; i < matches.size(); ++i) {
    const cv::DMatch &match = matches[i];
    if (match.queryIdx < 0 || match.queryIdx >= count0 || match.trainIdx < 0 || match.trainIdx >= count1) {
      Check(false, name + " match " + std::to_string(i) + " has an index outside the keypoints");
      break;
    }
  }
}

bool ExtractPair(viewfeat::FeaturePipeline &pipeline, const std::string &name, const cv::Mat &image0, const cv::Mat &image1, viewfeat::LocalFeatures &features0,
                 viewfeat::LocalFeatures &features1) {
  if (!pipeline.Extract(image0, features0) || !pipeline.Extract(image1, features1)) {
    Check(false, name + " Extract failed");
    return false;
  }
  return true;
}

void CheckCrossPipeline(viewfeat::FeaturePipeline &flexglue, const cv::Mat &image0, const cv::Mat &image1) {
  const auto lightglue = viewfeat::FeaturePipeline::Create("superpoint");
  const auto aliked = viewfeat::FeaturePipeline::Create("aliked");
  if (!lightglue || !aliked) {
    Check(false, "cross-pipeline Create returned null");
    return;
  }

  viewfeat::LocalFeatures flexglue0;
  viewfeat::LocalFeatures flexglue1;
  if (!ExtractPair(flexglue, "flexglue cross", image0, image1, flexglue0, flexglue1)) {
    return;
  }
  std::vector<cv::DMatch> matches;
  Check(!lightglue->Match(flexglue0, flexglue1, matches, false), "LightGlue pipeline matched FlexGlue features");

  viewfeat::LocalFeatures lightglue0;
  viewfeat::LocalFeatures lightglue1;
  if (!ExtractPair(*lightglue, "superpoint cross", image0, image1, lightglue0, lightglue1)) {
    return;
  }
  Check(!flexglue.Match(lightglue0, lightglue1, matches, false), "FlexGlue pipeline matched LightGlue features");

  viewfeat::LocalFeatures aliked0;
  viewfeat::LocalFeatures aliked1;
  if (!ExtractPair(*aliked, "aliked cross", image0, image1, aliked0, aliked1)) {
    return;
  }
  Check(!flexglue.Match(aliked0, aliked1, matches, false), "FlexGlue pipeline matched ALIKED features");
  Check(!lightglue->Match(aliked0, aliked1, matches, false), "LightGlue pipeline matched ALIKED features");
}

bool MatchPair(viewfeat::FeaturePipeline &pipeline, const std::string &name, const std::string &feature_family, const cv::Mat &image0, const cv::Mat &image1,
               int minimum_matches) {
  viewfeat::LocalFeatures features0;
  viewfeat::LocalFeatures features1;
  if (!pipeline.Extract(image0, features0) || !pipeline.Extract(image1, features1)) {
    Check(false, name + " Extract failed");
    return false;
  }
  const int descriptor_dim = pipeline.DescriptorDim();
  CheckLocalFeatures(name + " image0", features0, image0, descriptor_dim, feature_family);
  CheckLocalFeatures(name + " image1", features1, image1, descriptor_dim, feature_family);

  std::vector<cv::DMatch> matches;
  if (!pipeline.Match(features0, features1, matches, false)) {
    Check(false, name + " Match failed");
    return false;
  }
  std::cout << name << " keypoints=" << features0.keypoints.size() << "/" << features1.keypoints.size() << " matches=" << matches.size() << std::endl;
  CheckMatches(name, matches, static_cast<int>(features0.keypoints.size()), static_cast<int>(features1.keypoints.size()), minimum_matches);
  return true;
}

int RunLocal(const std::string &requested_family, const std::string &pipeline_family, const std::string &feature_family, const cv::Mat &image0, const cv::Mat &image1,
             const cv::Mat &frame0, const cv::Mat &frame1) {
  const auto pipeline = viewfeat::FeaturePipeline::Create(requested_family);
  if (!pipeline) {
    Check(false, "Create(" + requested_family + ") returned null");
    return 1;
  }
  Check(pipeline->Family() == pipeline_family, "Family() is " + pipeline->Family());

  MatchPair(*pipeline, requested_family + " pair", feature_family, image0, image1, 30);
  MatchPair(*pipeline, requested_family + " freiburg", feature_family, frame0, frame1, 50);
  CheckChannelExtractAndMatch(*pipeline, requested_family + " channels", image0, image1);
  if (requested_family == "superpoint_flexglue") {
    CheckCrossPipeline(*pipeline, image0, image1);
  }
  if (g_failures != 0) {
    return 1;
  }
  std::cout << requested_family << ": all checks passed" << std::endl;
  return 0;
}

double Dot(const std::vector<float> &a, const std::vector<float> &b) {
  double sum = 0.0;
  const size_t count = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < count; ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return sum;
}

double Norm(const std::vector<float> &values) {
  return std::sqrt(Dot(values, values));
}

bool InferDescriptor(viewfeat::GlobalRetriever &retriever, const std::string &name, const cv::Mat &image, std::vector<float> &descriptor,
                     const std::string &dim_label = "CosPlace") {
  if (!retriever.Infer(image, descriptor)) {
    Check(false, name + " Infer failed");
    return false;
  }
  Check(static_cast<int>(descriptor.size()) == retriever.DescriptorDim(), name + " descriptor length");
  Check(retriever.DescriptorDim() == 512, dim_label + " DescriptorDim is " + std::to_string(retriever.DescriptorDim()));
  const double norm = Norm(descriptor);
  Check(std::abs(norm - 1.0) < 1e-3, name + " L2 norm " + std::to_string(norm));
  return true;
}

int RunCosPlace(const cv::Mat &image0, const cv::Mat &frame0, const cv::Mat &frame1) {
  const auto retriever = viewfeat::GlobalRetriever::Create("cosplace");
  if (!retriever) {
    Check(false, "GlobalRetriever::Create returned null");
    return 1;
  }
  Check(retriever->Family() == "cosplace", "Family is " + retriever->Family());

  std::vector<float> first;
  std::vector<float> second;
  std::vector<float> near0;
  std::vector<float> near1;
  if (!InferDescriptor(*retriever, "image0", image0, first) || !InferDescriptor(*retriever, "image0 again", image0, second) ||
      !InferDescriptor(*retriever, "frame0", frame0, near0) || !InferDescriptor(*retriever, "frame1", frame1, near1)) {
    return 1;
  }

  CheckRetrieverChannels(*retriever, image0);

  const double same = Dot(first, second);
  const double adjacent = Dot(near0, near1);
  std::cout << "cosplace same=" << same << " adjacent=" << adjacent << std::endl;
  Check(same > 0.99, "same image cosine " + std::to_string(same));
  Check(adjacent > 0.80, "adjacent Freiburg cosine " + std::to_string(adjacent));
  if (g_failures != 0) {
    return 1;
  }
  std::cout << "cosplace: all checks passed" << std::endl;
  return 0;
}

bool FiniteDescriptor(const std::vector<float> &values) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return !values.empty();
}

int RunEigenPlaces(const cv::Mat &image0, const cv::Mat &frame0, const cv::Mat &frame1) {
  const auto retriever = viewfeat::GlobalRetriever::Create("eigenplaces");
  if (!retriever) {
    Check(false, "GlobalRetriever::Create(eigenplaces) returned null");
    return 1;
  }
  Check(retriever->Family() == "eigenplaces", "Family is " + retriever->Family());

  std::vector<float> first;
  std::vector<float> second;
  std::vector<float> near0;
  std::vector<float> near1;
  if (!InferDescriptor(*retriever, "image0", image0, first, "EigenPlaces") ||
      !InferDescriptor(*retriever, "image0 again", image0, second, "EigenPlaces") ||
      !InferDescriptor(*retriever, "frame0", frame0, near0, "EigenPlaces") ||
      !InferDescriptor(*retriever, "frame1", frame1, near1, "EigenPlaces")) {
    return 1;
  }

  CheckRetrieverChannels(*retriever, image0, "eigenplaces");

  const double same = Dot(first, second);
  const double adjacent = Dot(near0, near1);
  std::cout << "eigenplaces same=" << same << " adjacent=" << adjacent << std::endl;
  Check(same > 0.99, "same image cosine " + std::to_string(same));
  Check(FiniteDescriptor(first) && FiniteDescriptor(near0) && FiniteDescriptor(near1), "descriptor has a non-finite value");
  Check(std::isfinite(adjacent) && adjacent >= -1.0 && adjacent <= 1.0, "adjacent cosine " + std::to_string(adjacent));
  if (g_failures != 0) {
    return 1;
  }
  std::cout << "eigenplaces: all checks passed" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0] << " family config weights image0 image1 frame0 frame1" << std::endl;
    return 1;
  }

  const std::string family = argv[1];
  const std::string config_path = argv[2];
  const std::string model_dir = argv[3];
  const std::string image0_path = argv[4];
  const std::string image1_path = argv[5];
  const std::string frame0_path = argv[6];
  const std::string frame1_path = argv[7];

  if (!HaveNvidiaDevice()) {
    return Skip("no NVIDIA device");
  }
  const std::string paths[] = {config_path, image0_path, image1_path, frame0_path, frame1_path};
  for (const std::string &path : paths) {
    if (!RegularFile(path)) {
      return Skip("missing file " + path);
    }
  }
  struct stat model_dir_info;
  if (stat(model_dir.c_str(), &model_dir_info) != 0 || !S_ISDIR(model_dir_info.st_mode)) {
    return Skip("missing model dir " + model_dir);
  }

  viewfeat::FeaturePipeline::SetSearchPaths(config_path, model_dir);

  const cv::Mat image0 = ReadColor(image0_path);
  const cv::Mat image1 = ReadColor(image1_path);
  const cv::Mat frame0 = ReadColor(frame0_path);
  const cv::Mat frame1 = ReadColor(frame1_path);
  if (image0.empty() || image1.empty() || frame0.empty() || frame1.empty()) {
    std::cerr << "CHECK failed: imread returned an empty image" << std::endl;
    return 1;
  }

  if (family == "default") {
    return RunLocal(family, "superpoint", "superpoint", image0, image1, frame0, frame1);
  }
  if (family == "aliked") {
    return RunLocal(family, "aliked", "aliked", image0, image1, frame0, frame1);
  }
  if (family == "superpoint_flexglue") {
    return RunLocal(family, "superpoint_flexglue", "superpoint", image0, image1, frame0, frame1);
  }
  if (family == "cosplace") {
    return RunCosPlace(image0, frame0, frame1);
  }
  if (family == "eigenplaces") {
    return RunEigenPlaces(image0, frame0, frame1);
  }

  std::cerr << "CHECK failed: unknown family " << family << std::endl;
  return 1;
}
