// Correctness and steady-state timing for Extract, Match, and global descriptors.
// imread runs before the timer. Warmup calls are not part of the reported samples.
// A missing baseline file skips the byte-level compare. --write-baseline records one.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>

#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>

#include "feature_payload.h"
#include "viewfeat/pipeline.h"
#include "viewfeat/retriever.h"

namespace {

constexpr int kWarmup = 10;
constexpr int kIterations = 50;
constexpr uint32_t kMagic = 0x31504656u;  // 'VFP1' little-endian

int g_failures = 0;
bool g_skipped = false;

struct Record {
  std::string name;
  std::vector<float> values;
};

std::vector<Record> g_records;

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
  return stat("/dev/nvidia0", &info) == 0 || stat("/dev/nvidiactl", &info) == 0 || stat("/dev/dxg", &info) == 0;
}

int Skip(const std::string &reason) {
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

std::string BaseName(const std::string &path) {
  const std::string::size_type slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return path;
  }
  return path.substr(slash + 1);
}

bool AllEngines(const std::string &model_dir, const std::vector<std::string> &names) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (!RegularFile(model_dir + "/" + names[i])) {
      return false;
    }
  }
  return true;
}

void AddRecord(const std::string &name, std::vector<float> values) {
  Record record;
  record.name = name;
  record.values.swap(values);
  g_records.push_back(std::move(record));
}

const Record *FindRecord(const std::string &name) {
  for (size_t i = 0; i < g_records.size(); ++i) {
    if (g_records[i].name == name) {
      return &g_records[i];
    }
  }
  return nullptr;
}

double MaxAbsDiff(const std::vector<float> &left, const std::vector<float> &right) {
  double max_diff = 0.0;
  const size_t count = left.size() < right.size() ? left.size() : right.size();
  for (size_t i = 0; i < count; ++i) {
    const double diff = std::abs(static_cast<double>(left[i]) - static_cast<double>(right[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

bool SameValues(const std::string &label, const std::vector<float> &left, const std::vector<float> &right) {
  if (left.size() != right.size()) {
    Check(false, label + " length " + std::to_string(left.size()) + " != " + std::to_string(right.size()));
    return false;
  }
  const double max_diff = MaxAbsDiff(left, right);
  if (max_diff != 0.0) {
    Check(false, label + " max_abs_diff=" + std::to_string(max_diff));
    return false;
  }
  std::cout << "STABLE " << label << " max_abs_diff=0" << std::endl;
  return true;
}

bool FiniteValues(const std::vector<float> &values) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

double SumValues(const std::vector<float> &values) {
  double sum = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    sum += static_cast<double>(values[i]);
  }
  return sum;
}

void PrintSample(const std::string &name, const std::vector<float> &values) {
  const size_t shown = std::min<size_t>(8, values.size());
  std::cout << "SAMPLE " << name << " n=" << values.size() << " sum=" << SumValues(values) << std::hexfloat;
  for (size_t i = 0; i < shown; ++i) {
    std::cout << " " << values[i];
  }
  std::cout << std::defaultfloat << std::endl;
}

std::vector<float> CopyKeypoints(const std::vector<cv::Point2f> &points) {
  std::vector<float> values;
  values.reserve(points.size() * 2);
  for (size_t i = 0; i < points.size(); ++i) {
    values.push_back(points[i].x);
    values.push_back(points[i].y);
  }
  return values;
}

std::vector<float> CopyDescriptors(const cv::Mat &descriptors) {
  std::vector<float> values;
  if (descriptors.empty() || descriptors.type() != CV_32F) {
    return values;
  }
  values.reserve(static_cast<size_t>(descriptors.rows) * static_cast<size_t>(descriptors.cols));
  for (int row = 0; row < descriptors.rows; ++row) {
    const float *ptr = descriptors.ptr<float>(row);
    values.insert(values.end(), ptr, ptr + descriptors.cols);
  }
  return values;
}

std::vector<float> CopyMatches(const std::vector<cv::DMatch> &matches) {
  std::vector<float> values;
  values.reserve(matches.size() * 3);
  for (size_t i = 0; i < matches.size(); ++i) {
    values.push_back(static_cast<float>(matches[i].queryIdx));
    values.push_back(static_cast<float>(matches[i].trainIdx));
    values.push_back(matches[i].distance);
  }
  return values;
}

std::vector<float> CopyDense(const viewfeat::LocalFeatures &features) {
  int height = 0;
  int width = 0;
  int dim = 0;
  const std::shared_ptr<std::vector<float>> dense = viewfeat::FeaturePayload::FlexGlueOf(features, height, width, dim);
  if (!dense) {
    return std::vector<float>();
  }
  Check(height == 64 && width == 64 && dim == 256, "FlexGlue dense shape " + std::to_string(dim) + "x" + std::to_string(height) + "x" + std::to_string(width));
  Check(dense->size() == static_cast<size_t>(dim * height * width), "FlexGlue dense length " + std::to_string(dense->size()));
  return *dense;
}

template <typename Fn>
bool TimeOp(const std::string &pipeline, const std::string &op, const std::string &image, Fn fn) {
  for (int i = 0; i < kWarmup; ++i) {
    if (!fn()) {
      Check(false, pipeline + " " + op + " warmup failed");
      return false;
    }
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(kIterations));
  for (int i = 0; i < kIterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    if (!fn()) {
      Check(false, pipeline + " " + op + " timed call failed");
      return false;
    }
    const auto stop = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::vector<double> ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const size_t mid = ordered.size() / 2;
  const double median = ordered.size() % 2 == 0 ? 0.5 * (ordered[mid - 1] + ordered[mid]) : ordered[mid];
  double sum = 0.0;
  for (size_t i = 0; i < samples.size(); ++i) {
    sum += samples[i];
  }
  const double mean = sum / static_cast<double>(samples.size());
  std::cout << "PERF_TIME pipeline=" << pipeline << " op=" << op << " image=" << image << " warmup=" << kWarmup << " iters=" << kIterations
            << " median_ms=" << median << " mean_ms=" << mean << " min_ms=" << ordered.front() << " max_ms=" << ordered.back() << std::endl;
  return true;
}

bool WriteBaseline(const std::string &path) {
  std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!output) {
    Check(false, "cannot write baseline " + path);
    return false;
  }
  const uint32_t magic = kMagic;
  const uint32_t count = static_cast<uint32_t>(g_records.size());
  output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  output.write(reinterpret_cast<const char *>(&count), sizeof(count));
  for (size_t i = 0; i < g_records.size(); ++i) {
    const uint32_t name_len = static_cast<uint32_t>(g_records[i].name.size());
    const uint32_t value_count = static_cast<uint32_t>(g_records[i].values.size());
    output.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
    output.write(g_records[i].name.data(), static_cast<std::streamsize>(name_len));
    output.write(reinterpret_cast<const char *>(&value_count), sizeof(value_count));
    if (value_count > 0) {
      output.write(reinterpret_cast<const char *>(g_records[i].values.data()), static_cast<std::streamsize>(value_count * sizeof(float)));
    }
  }
  output.flush();
  if (!output) {
    Check(false, "baseline write failed " + path);
    return false;
  }
  std::cout << "BASELINE wrote " << path << " records=" << g_records.size() << std::endl;
  return true;
}

bool ReadBaseline(const std::string &path, std::map<std::string, std::vector<float>> &records) {
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input) {
    Check(false, "cannot read baseline " + path);
    return false;
  }
  uint32_t magic = 0;
  uint32_t count = 0;
  input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (!input || magic != kMagic) {
    Check(false, "baseline magic mismatch");
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t name_len = 0;
    input.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    if (!input || name_len > 1024) {
      Check(false, "baseline record name is invalid");
      return false;
    }
    std::string name(name_len, '\0');
    if (name_len > 0) {
      input.read(&name[0], static_cast<std::streamsize>(name_len));
    }
    uint32_t value_count = 0;
    input.read(reinterpret_cast<char *>(&value_count), sizeof(value_count));
    if (!input) {
      Check(false, "baseline record is truncated");
      return false;
    }
    std::vector<float> values(value_count);
    if (value_count > 0) {
      input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(value_count * sizeof(float)));
    }
    if (!input) {
      Check(false, "baseline values are truncated");
      return false;
    }
    records[name].swap(values);
  }
  return static_cast<bool>(input);
}

bool CompareBaseline(const std::string &path) {
  std::map<std::string, std::vector<float>> expected;
  if (!ReadBaseline(path, expected)) {
    return false;
  }
  double worst = 0.0;
  std::string worst_name;
  for (size_t i = 0; i < g_records.size(); ++i) {
    const std::map<std::string, std::vector<float>>::const_iterator found = expected.find(g_records[i].name);
    if (found == expected.end()) {
      Check(false, "baseline has no record " + g_records[i].name);
      continue;
    }
    if (found->second.size() != g_records[i].values.size()) {
      Check(false, g_records[i].name + " length " + std::to_string(g_records[i].values.size()) + " != baseline " + std::to_string(found->second.size()));
      continue;
    }
    const double diff = MaxAbsDiff(found->second, g_records[i].values);
    if (diff > worst) {
      worst = diff;
      worst_name = g_records[i].name;
    }
    if (diff != 0.0) {
      Check(false, g_records[i].name + " max_abs_diff=" + std::to_string(diff));
    }
  }
  for (std::map<std::string, std::vector<float>>::const_iterator it = expected.begin(); it != expected.end(); ++it) {
    if (FindRecord(it->first) == nullptr) {
      Check(false, "run is missing baseline record " + it->first);
    }
  }
  std::cout << "BASELINE compare " << path << " records=" << expected.size() << " worst=" << worst_name << " max_abs_diff=" << worst << std::endl;
  return g_failures == 0;
}

void CheckLocalFeatures(const std::string &name, const viewfeat::LocalFeatures &features, const cv::Mat &image, int descriptor_dim) {
  const int count = static_cast<int>(features.keypoints.size());
  Check(count >= 100, name + " keypoints " + std::to_string(count));
  Check(features.descriptors.rows == count, name + " descriptor rows");
  Check(features.descriptors.cols == descriptor_dim, name + " descriptor cols " + std::to_string(features.descriptors.cols));
  Check(features.descriptors.type() == CV_32F, name + " descriptor type");
  Check(features.image_size.width == image.cols && features.image_size.height == image.rows, name + " image_size");
  const std::vector<float> descriptors = CopyDescriptors(features.descriptors);
  Check(FiniteValues(descriptors), name + " descriptors are not finite");
  const std::vector<float> points = CopyKeypoints(features.keypoints);
  Check(FiniteValues(points), name + " keypoints are not finite");
}

bool RunLocal(const std::string &family, const std::string &feature_family, bool expect_dense, const std::vector<std::string> &engines, const std::string &model_dir,
              const cv::Mat &image0, const cv::Mat &image1, const std::string &image0_name, const std::string &image1_name) {
  const int failures_before = g_failures;
  std::unique_ptr<viewfeat::FeaturePipeline> pipeline = viewfeat::FeaturePipeline::Create(family);
  if (!pipeline) {
    if (!AllEngines(model_dir, engines)) {
      std::cout << "SKIP: " << family << " missing engine or weights" << std::endl;
      g_skipped = true;
      return false;
    }
    Check(false, family + " Create returned null");
    return false;
  }
  Check(pipeline->Family() == family, family + " Family is " + pipeline->Family());

  viewfeat::LocalFeatures features0;
  viewfeat::LocalFeatures features1;
  viewfeat::LocalFeatures features0_again;
  if (!pipeline->Extract(image0, features0) || !pipeline->Extract(image1, features1) || !pipeline->Extract(image0, features0_again)) {
    Check(false, family + " Extract failed");
    return false;
  }
  const int descriptor_dim = pipeline->DescriptorDim();
  Check(features0.family == feature_family, family + " feature family " + features0.family);
  CheckLocalFeatures(family + " image0", features0, image0, descriptor_dim);
  CheckLocalFeatures(family + " image1", features1, image1, descriptor_dim);
  SameValues(family + " extract repeat", CopyDescriptors(features0.descriptors), CopyDescriptors(features0_again.descriptors));
  SameValues(family + " keypoint repeat", CopyKeypoints(features0.keypoints), CopyKeypoints(features0_again.keypoints));

  std::vector<cv::DMatch> matches;
  std::vector<cv::DMatch> matches_again;
  if (!pipeline->Match(features0, features1, matches, false) || !pipeline->Match(features0, features1, matches_again, false)) {
    Check(false, family + " Match failed");
    return false;
  }
  Check(static_cast<int>(matches.size()) >= 30, family + " matches " + std::to_string(matches.size()));
  SameValues(family + " match repeat", CopyMatches(matches), CopyMatches(matches_again));

  const std::vector<float> keypoints0 = CopyKeypoints(features0.keypoints);
  const std::vector<float> descriptors0 = CopyDescriptors(features0.descriptors);
  const std::vector<float> keypoints1 = CopyKeypoints(features1.keypoints);
  const std::vector<float> descriptors1 = CopyDescriptors(features1.descriptors);
  const std::vector<float> match_values = CopyMatches(matches);
  std::vector<float> dense0;
  std::vector<float> dense1;
  if (expect_dense) {
    dense0 = CopyDense(features0);
    dense1 = CopyDense(features1);
    Check(!dense0.empty() && !dense1.empty(), family + " dense features missing");
    Check(FiniteValues(dense0) && FiniteValues(dense1), family + " dense features are not finite");
    viewfeat::LocalFeatures dense_again_features;
    if (!pipeline->Extract(image0, dense_again_features)) {
      Check(false, family + " dense repeat Extract failed");
    } else {
      SameValues(family + " dense repeat", dense0, CopyDense(dense_again_features));
    }
  }

  std::cout << "RESULT " << family << " keypoints=" << features0.keypoints.size() << "/" << features1.keypoints.size() << " matches=" << matches.size()
            << " desc_dim=" << descriptor_dim << std::endl;
  PrintSample(family + ".image0.keypoints", keypoints0);
  PrintSample(family + ".image0.descriptors", descriptors0);
  PrintSample(family + ".image1.descriptors", descriptors1);
  PrintSample(family + ".matches", match_values);
  if (expect_dense) {
    PrintSample(family + ".image0.dense", dense0);
    PrintSample(family + ".image1.dense", dense1);
  }

  AddRecord(family + ".image0.keypoints", keypoints0);
  AddRecord(family + ".image0.descriptors", descriptors0);
  AddRecord(family + ".image1.keypoints", keypoints1);
  AddRecord(family + ".image1.descriptors", descriptors1);
  AddRecord(family + ".matches", match_values);
  if (expect_dense) {
    AddRecord(family + ".image0.dense", std::move(dense0));
    AddRecord(family + ".image1.dense", std::move(dense1));
  }

  const std::string pair_name = image0_name + "+" + image1_name;
  TimeOp(family, "extract", image0_name, [&]() {
    viewfeat::LocalFeatures timed;
    return pipeline->Extract(image0, timed);
  });
  TimeOp(family, "match", pair_name, [&]() {
    std::vector<cv::DMatch> timed;
    return pipeline->Match(features0, features1, timed, false);
  });
  return g_failures == failures_before;
}

bool RunGlobal(const std::string &family, const std::string &engine_file, const std::string &model_dir, const cv::Mat &image0, const cv::Mat &image1,
               const std::string &image0_name) {
  const int failures_before = g_failures;
  std::unique_ptr<viewfeat::GlobalRetriever> retriever = viewfeat::GlobalRetriever::Create(family);
  if (!retriever) {
    if (!RegularFile(model_dir + "/" + engine_file)) {
      std::cout << "SKIP: " << family << " missing engine or weights" << std::endl;
      g_skipped = true;
      return false;
    }
    Check(false, family + " Create returned null");
    return false;
  }
  Check(retriever->Family() == family, family + " Family is " + retriever->Family());
  Check(retriever->DescriptorDim() == 512, family + " dim " + std::to_string(retriever->DescriptorDim()));

  std::vector<float> first;
  std::vector<float> second;
  std::vector<float> again;
  if (!retriever->Infer(image0, first) || !retriever->Infer(image1, second) || !retriever->Infer(image0, again)) {
    Check(false, family + " Infer failed");
    return false;
  }
  Check(static_cast<int>(first.size()) == 512 && second.size() == 512, family + " descriptor length");
  Check(FiniteValues(first) && FiniteValues(second), family + " descriptor is not finite");
  double sum_sq = 0.0;
  for (size_t i = 0; i < first.size(); ++i) {
    sum_sq += static_cast<double>(first[i]) * static_cast<double>(first[i]);
  }
  const double norm = std::sqrt(sum_sq);
  Check(std::abs(norm - 1.0) < 1e-3, family + " L2 norm " + std::to_string(norm));
  SameValues(family + " infer repeat", first, again);

  std::cout << "RESULT " << family << " dim=" << first.size() << " norm=" << norm << std::endl;
  PrintSample(family + ".image0", first);
  PrintSample(family + ".image1", second);
  AddRecord(family + ".image0", first);
  AddRecord(family + ".image1", std::move(second));

  TimeOp(family, "infer", image0_name, [&]() {
    std::vector<float> timed;
    return retriever->Infer(image0, timed);
  });
  return g_failures == failures_before;
}

std::string GpuName() {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return "unknown";
  }
  return prop.name;
}

int ClipIndex(int val, int max_val) {
  if (val < 0) {
    return 0;
  }
  return std::min(val, max_val - 1);
}

// CPU reference for SuperPoint bilinear sampling. cell is 8. This stays in the
// test so a GPU sampler can be checked against the same formula.
void SampleDescriptorReference(const float *input, int x, int y, float *dest, int dim, int h, int w) {
  const int cell = 8;
  const float shift = cell / 2 - 0.5F;
  const float x_mul = 2.0F / (w * cell - cell / 2 - 0.5F);
  const float y_mul = 2.0F / (h * cell - cell / 2 - 0.5F);
  const float gx = x * x_mul + (-1.0F - shift * x_mul);
  const float gy = y * y_mul + (-1.0F - shift * y_mul);
  const float ix = ((gx + 1.0F) / 2.0F) * (w - 1);
  const float iy = ((gy + 1.0F) / 2.0F) * (h - 1);
  const int ix_nw = ClipIndex(static_cast<int>(std::floor(ix)), w);
  const int iy_nw = ClipIndex(static_cast<int>(std::floor(iy)), h);
  const int ix_ne = ClipIndex(ix_nw + 1, w);
  const int iy_ne = ClipIndex(iy_nw, h);
  const int ix_sw = ClipIndex(ix_nw, w);
  const int iy_sw = ClipIndex(iy_nw + 1, h);
  const int ix_se = ClipIndex(ix_nw + 1, w);
  const int iy_se = ClipIndex(iy_nw + 1, h);
  const float nw = (ix_se - ix) * (iy_se - iy);
  const float ne = (ix - ix_sw) * (iy_sw - iy);
  const float sw = (ix_ne - ix) * (iy - iy_ne);
  const float se = (ix - ix_nw) * (iy - iy_nw);
  float sum = 0.0F;
  for (int i = 0; i < dim; ++i) {
    const float *plane = input + i * h * w;
    const float value = plane[iy_nw * w + ix_nw] * nw + plane[iy_ne * w + ix_ne] * ne + plane[iy_sw * w + ix_sw] * sw + plane[iy_se * w + ix_se] * se;
    dest[i] = value;
    sum += value * value;
  }
  const float inv_norm = 1.0F / std::sqrt(sum);
  for (int i = 0; i < dim; ++i) {
    dest[i] *= inv_norm;
  }
}

void CheckSamplingReference() {
  const int h = 4;
  const int w = 4;
  const int dim = 2;
  std::vector<float> input(static_cast<size_t>(dim * h * w), 1.0F);
  input[0] = 3.0F;
  input[static_cast<size_t>(h * w)] = 4.0F;
  float dest[2] = {0.0F, 0.0F};
  SampleDescriptorReference(input.data(), 0, 0, dest, dim, h, w);
  const float norm = std::sqrt(dest[0] * dest[0] + dest[1] * dest[1]);
  Check(std::abs(norm - 1.0F) < 1e-6F, "sampling reference L2 norm " + std::to_string(norm));
  Check(std::isfinite(dest[0]) && std::isfinite(dest[1]), "sampling reference is finite");

  std::vector<float> grid(static_cast<size_t>(dim * h * w), 0.0F);
  for (int i = 0; i < h * w; ++i) {
    grid[static_cast<size_t>(i)] = 1.0F;
  }
  float unit[2] = {0.0F, 0.0F};
  SampleDescriptorReference(grid.data(), 3, 2, unit, dim, h, w);
  Check(unit[1] == 0.0F && std::abs(std::abs(unit[0]) - 1.0F) < 1e-6F, "sampling reference normalizes a constant plane");
  std::cout << "SAMPLING_REF cpu formula checked" << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " config weights image0 image1 [--baseline path] [--write-baseline path]" << std::endl;
    return 1;
  }

  const std::string config_path = argv[1];
  const std::string model_dir = argv[2];
  const std::string image0_path = argv[3];
  const std::string image1_path = argv[4];
  std::string baseline_path;
  std::string write_baseline_path;
  for (int i = 5; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--baseline" || arg == "--write-baseline") && i + 1 >= argc) {
      std::cerr << "CHECK failed: " << arg << " needs a path" << std::endl;
      return 1;
    }
    if (arg == "--baseline") {
      baseline_path = argv[++i];
    } else if (arg == "--write-baseline") {
      write_baseline_path = argv[++i];
    } else {
      std::cerr << "CHECK failed: unknown argument " << arg << std::endl;
      return 1;
    }
  }

  if (!HaveNvidiaDevice()) {
    return Skip("no NVIDIA device");
  }
  if (!RegularFile(config_path) || !RegularFile(image0_path) || !RegularFile(image1_path)) {
    return Skip("missing config or image");
  }
  struct stat model_dir_info;
  if (stat(model_dir.c_str(), &model_dir_info) != 0 || !S_ISDIR(model_dir_info.st_mode)) {
    return Skip("missing model dir " + model_dir);
  }

  const cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_COLOR);
  const cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
  if (image0.empty() || image1.empty()) {
    std::cerr << "CHECK failed: imread returned an empty image" << std::endl;
    return 1;
  }

  const std::string image0_name = BaseName(image0_path);
  const std::string image1_name = BaseName(image1_path);
  std::cout << "PERF_META gpu=" << GpuName() << std::endl;
  std::cout << "PERF_META images=" << image0_name << "," << image1_name << " size=" << image0.cols << "x" << image0.rows << std::endl;
  std::cout << "PERF_META warmup=" << kWarmup << " iters=" << kIterations << std::endl;

  CheckSamplingReference();

  viewfeat::FeaturePipeline::SetSearchPaths(config_path, model_dir);

  const std::vector<std::string> superpoint_engines = {"superpoint_v1.engine", "superpoint_lightglue.engine"};
  const std::vector<std::string> aliked_engines = {"aliked-n16rot-top1024-512x288.engine", "aliked_lightglue.engine"};
  std::vector<std::string> flexglue_engines = superpoint_engines;
  flexglue_engines.push_back("flexglue_descriptor.engine");
  flexglue_engines.push_back("flexglue_matcher.engine");

  RunLocal("superpoint", "superpoint", false, superpoint_engines, model_dir, image0, image1, image0_name, image1_name);
  RunLocal("aliked", "aliked", false, aliked_engines, model_dir, image0, image1, image0_name, image1_name);
  RunLocal("superpoint_flexglue", "superpoint", true, flexglue_engines, model_dir, image0, image1, image0_name, image1_name);
  RunGlobal("cosplace", "cosplace.engine", model_dir, image0, image1, image0_name);
  RunGlobal("eigenplaces", "eigenplaces.engine", model_dir, image0, image1, image0_name);

  if (g_skipped) {
    if (!write_baseline_path.empty() || (!baseline_path.empty() && RegularFile(baseline_path))) {
      std::cerr << "CHECK failed: a model was skipped, so the baseline was not written or compared" << std::endl;
      return 1;
    }
    return Skip("one or more models had no engine");
  }

  if (!write_baseline_path.empty() && !WriteBaseline(write_baseline_path)) {
    return 1;
  }
  if (!baseline_path.empty()) {
    if (!RegularFile(baseline_path)) {
      std::cout << "BASELINE missing " << baseline_path << " (comparison skipped)" << std::endl;
    } else if (!CompareBaseline(baseline_path)) {
      return 1;
    }
  }
  if (g_failures != 0) {
    return 1;
  }
  std::cout << "perf: all checks passed" << std::endl;
  return 0;
}
