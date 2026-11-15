#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "utils.h"
#include "viewfeat/retriever.h"

// Global place descriptor (CosPlace) via GlobalRetriever — no local features / matching.

namespace {

struct Options {
  std::string image_dir;
  std::string output_csv = "cosplace_loop_similarity.csv";
  double threshold = 0.80;
  int max_images = 0;
};

struct FrameResult {
  int index = 0;
  std::string filename;
  double similarity = 0.0;
};

bool HasImageExtension(const std::string &path) {
  const auto dot = path.find_last_of('.');
  if (dot == std::string::npos) {
    return false;
  }
  std::string ext = path.substr(dot);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<std::string> ListImages(const std::string &image_dir, int max_images) {
  std::vector<std::string> all_files;
  if (!GetFileNames(image_dir, all_files)) {
    throw std::runtime_error("Failed to list images in: " + image_dir);
  }

  std::vector<std::string> images;
  images.reserve(all_files.size());
  for (const auto &path : all_files) {
    if (HasImageExtension(path)) {
      images.push_back(path);
    }
  }
  std::sort(images.begin(), images.end());

  if (max_images > 0 && static_cast<int>(images.size()) > max_images) {
    images.resize(static_cast<size_t>(max_images));
  }
  return images;
}

double CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size() || a.empty()) {
    return -1.0;
  }
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double VectorNorm(const std::vector<float> &v) {
  double sum = 0.0;
  for (float x : v) {
    sum += static_cast<double>(x) * x;
  }
  return std::sqrt(sum);
}

std::string Basename(const std::string &path) {
  const auto slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

void PrintUsage(const char *program) {
  std::cerr << "Usage:" << std::endl
            << "  " << program << " <image_sequence_dir> [options]" << std::endl
            << std::endl
            << "Compare frame 0 against all later frames using CosPlace TRT descriptors." << std::endl
            << "Descriptors are L2-normalized; similarity is the dot product (cosine)." << std::endl
            << std::endl
            << "Options:" << std::endl
            << "  --threshold <float>     Loop similarity threshold (default: 0.80)" << std::endl
            << "  --output <csv_path>     Output CSV path (default: cosplace_loop_similarity.csv)" << std::endl
            << "  --max-images <int>      Limit number of frames, 0 = no limit (default: 0)" << std::endl
            << "  --help                  Show this message" << std::endl;
}

Options ParseArgs(int argc, char **argv) {
  Options options;
  if (argc < 2) {
    PrintUsage(argv[0]);
    throw std::runtime_error("Missing required arguments.");
  }

  options.image_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--threshold") {
      options.threshold = std::stod(require_value(arg));
    } else if (arg == "--output") {
      options.output_csv = require_value(arg);
    } else if (arg == "--max-images") {
      options.max_images = std::stoi(require_value(arg));
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("Unknown option: " + arg);
    } else {
      throw std::runtime_error("Unexpected positional argument: " + arg);
    }
  }

  if (options.threshold < -1.0 || options.threshold > 1.0) {
    throw std::runtime_error("--threshold must be in [-1, 1].");
  }
  if (options.max_images < 0) {
    throw std::runtime_error("--max-images must be non-negative.");
  }
  return options;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = ParseArgs(argc, argv);

    const auto image_paths = ListImages(options.image_dir, options.max_images);
    if (image_paths.size() < 2) {
      std::cerr << "Need at least 2 images in sequence dir: " << options.image_dir << std::endl;
      return EXIT_FAILURE;
    }

    // "default" resolves to CosPlace; config/weights are resolved at runtime.
    auto retriever = viewfeat::GlobalRetriever::Create("default");
    if (!retriever) {
      std::cerr << "Failed to create CosPlace global retriever." << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "Retriever type: " << retriever->Family() << std::endl;
    std::cout << "Reference frame: " << Basename(image_paths.front()) << " (index 0)" << std::endl;
    std::cout << "Sequence frames: " << image_paths.size() << std::endl;
    std::cout << "Threshold: " << options.threshold << std::endl;

    cv::Mat ref_image = cv::imread(image_paths.front(), cv::IMREAD_COLOR);
    if (ref_image.empty()) {
      std::cerr << "Failed to read reference image: " << image_paths.front() << std::endl;
      return EXIT_FAILURE;
    }

    std::vector<float> ref_descriptor;
    // GlobalRetriever::Infer — L2-normalized float vector (cosine similarity = dot product).
    if (!retriever->Infer(ref_image, ref_descriptor)) {
      std::cerr << "CosPlace inference failed on reference frame." << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Reference descriptor dim=" << ref_descriptor.size() << " norm=" << VectorNorm(ref_descriptor) << std::endl;

    std::vector<FrameResult> results;
    results.reserve(image_paths.size() - 1);

    for (size_t i = 1; i < image_paths.size(); ++i) {
      cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_COLOR);
      if (image.empty()) {
        std::cerr << "Failed to read image: " << image_paths[i] << std::endl;
        return EXIT_FAILURE;
      }

      std::vector<float> descriptor;
      if (!retriever->Infer(image, descriptor)) {  // same retriever, compare all frames to frame 0
        std::cerr << "CosPlace inference failed on frame " << i << std::endl;
        return EXIT_FAILURE;
      }

      FrameResult row;
      row.index = static_cast<int>(i);
      row.filename = Basename(image_paths[i]);
      row.similarity = CosineSimilarity(ref_descriptor, descriptor);
      results.push_back(row);

      if ((i % 50) == 0 || i + 1 == image_paths.size()) {
        std::cout << "Processed " << i + 1 << '/' << image_paths.size() << " frames" << std::endl;
      }
    }

    std::vector<FrameResult> ranked = results;
    std::sort(ranked.begin(), ranked.end(), [](const FrameResult &a, const FrameResult &b) {
      return a.similarity > b.similarity;
    });

    std::ofstream csv(options.output_csv);
    if (!csv) {
      std::cerr << "Failed to open output CSV: " << options.output_csv << std::endl;
      return EXIT_FAILURE;
    }
    csv << "frame_index,image,similarity,is_loop" << std::endl;

    int loop_count = 0;
    std::cout << std::endl << "--- similarity vs frame 0 ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (const auto &row : results) {
      const bool is_loop = row.similarity >= options.threshold;
      if (is_loop) {
        ++loop_count;
      }
      csv << row.index << ',' << row.filename << ',' << row.similarity << ',' << (is_loop ? 1 : 0) << std::endl;
      std::cout << "frame=" << row.index << " sim=" << row.similarity << " image=" << row.filename;
      if (is_loop) {
        std::cout << " [LOOP]";
      }
      std::cout << std::endl;
    }

    const int preview = std::min<int>(10, static_cast<int>(ranked.size()));
    std::cout << std::endl << "--- top " << preview << " matches to frame 0 ---" << std::endl;
    for (int i = 0; i < preview; ++i) {
      std::cout << "rank=" << (i + 1) << " frame=" << ranked[static_cast<size_t>(i)].index << " sim=" << ranked[static_cast<size_t>(i)].similarity
                << " image=" << ranked[static_cast<size_t>(i)].filename << std::endl;
    }

    std::cout << std::endl << "Loop candidates (sim >= " << options.threshold << "): " << loop_count << std::endl;
    std::cout << "CSV: " << options.output_csv << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception &exc) {
    std::cerr << "Error: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
}
