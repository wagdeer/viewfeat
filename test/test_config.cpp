#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "runtime/search_paths.h"
#include "viewfeat/pipeline.h"
#include "viewfeat/retriever.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef VIEWFEAT_SAMPLE_CONFIG
#define VIEWFEAT_SAMPLE_CONFIG "config/config.yaml"
#endif

namespace {

int g_failures = 0;

void Check(bool condition, const std::string &message) {
  if (condition) {
    return;
  }
  std::cerr << "CHECK failed: " << message << std::endl;
  ++g_failures;
}

class CerrCapture {
 public:
  CerrCapture() : previous_(std::cerr.rdbuf(buffer_.rdbuf())) {}
  ~CerrCapture() { std::cerr.rdbuf(previous_); }

  std::string str() const { return buffer_.str(); }

 private:
  std::ostringstream buffer_;
  std::streambuf *previous_;
};

class TempDir {
 public:
  TempDir() {
    std::string pattern = "/tmp/viewfeat-config-XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (mkdtemp(buffer.data()) == nullptr) {
      return;
    }
    path_.assign(buffer.data());
  }

  ~TempDir() {
    if (path_.empty()) {
      return;
    }
    for (size_t i = 0; i < files_.size(); ++i) {
      std::remove(files_[i].c_str());
    }
    rmdir(path_.c_str());
  }

  bool ok() const { return !path_.empty(); }

  std::string File(const std::string &name) {
    const std::string path = path_ + "/" + name;
    files_.push_back(path);
    return path;
  }

 private:
  std::string path_;
  std::vector<std::string> files_;
};

bool WriteFile(const std::string &path, const std::string &contents) {
  std::ofstream output(path.c_str());
  if (!output) {
    return false;
  }
  output << contents;
  return static_cast<bool>(output);
}

bool SetMtime(const std::string &path, time_t seconds) {
  struct timespec times[2];
  times[0].tv_sec = seconds;
  times[0].tv_nsec = 0;
  times[1].tv_sec = seconds;
  times[1].tv_nsec = 0;
  return utimensat(AT_FDCWD, path.c_str(), times, 0) == 0;
}

std::string CosPlaceYaml(int descriptor_dim) {
  std::ostringstream yaml;
  yaml << "cosplace:\n"
        << "  input_size: 320\n"
        << "  descriptor_dim: " << descriptor_dim << "\n"
        << "  dla_core: -1\n"
        << "  input_tensor_names:\n"
        << "    - \"images\"\n"
        << "  output_tensor_names:\n"
        << "    - \"descriptors\"\n"
        << "  onnx_file: \"cosplace.onnx\"\n"
        << "  engine_file: \"cosplace.engine\"\n";
  return yaml.str();
}

const char *kAlikedSection =
    "aliked:\n"
    "  image_width: 512\n"
    "  image_height: 288\n"
    "  top_k: 1024\n"
    "  descriptor_dim: 128\n"
    "  dla_core: -1\n"
    "  input_tensor_names:\n"
    "    - \"image\"\n"
    "  output_tensor_names:\n"
    "    - \"keypoints\"\n"
    "    - \"descriptors\"\n"
    "    - \"scores\"\n"
    "  onnx_file: \"aliked.onnx\"\n"
    "  engine_file: \"aliked.engine\"\n";

const char *kLightGlueWithoutNumKeypoints =
    "superpoint:\n"
    "  image_width: 512\n"
    "  image_height: 288\n"
    "  max_keypoints: 100\n"
    "  keypoint_threshold: 0.01\n"
    "  remove_borders: 4\n"
    "  dla_core: -1\n"
    "  input_tensor_names:\n"
    "    - \"image\"\n"
    "  output_tensor_names:\n"
    "    - \"scores\"\n"
    "    - \"descriptors\"\n"
    "  onnx_file: \"superpoint.onnx\"\n"
    "  engine_file: \"superpoint.engine\"\n"
    "superpoint_lightglue:\n"
    "  filter_threshold: 0.1\n"
    "  dla_core: -1\n"
    "  input_tensor_names:\n"
    "    - \"keypoints0\"\n"
    "  output_tensor_names:\n"
    "    - \"scores\"\n"
    "  onnx_file: \"lightglue.onnx\"\n"
    "  engine_file: \"lightglue.engine\"\n";

void Use(const std::string &config_path) { viewfeat::FeaturePipeline::SetSearchPaths(config_path, "unit-weights"); }

void TestSampleConfig() {
  Use(VIEWFEAT_SAMPLE_CONFIG);
  const std::shared_ptr<const viewfeat::Configs> configs = viewfeat::GetConfigs();
  Check(static_cast<bool>(configs), "sample config loads");
  if (!configs) {
    return;
  }
  Check(configs->feature_type == "superpoint", "sample feature_type");
  Check(configs->superpoint_config.max_keypoints == 1024, "sample superpoint keypoints");
  Check(configs->superpoint_config.input_tensor_names.size() == 1, "sample superpoint inputs");
  Check(configs->superpoint_config.image_width == 512, "sample superpoint width");
  Check(configs->superpoint_config.image_height == 288, "sample superpoint height");
  Check(configs->superpoint_lightglue_config.num_keypoints == 1024, "sample lightglue keypoints");
  Check(configs->aliked_config.descriptor_dim == 128, "sample aliked dim");
  Check(configs->cosplace_config.descriptor_dim == 512, "sample cosplace dim");
  Check(configs->eigenplaces_config.input_size == 512, "sample eigenplaces size");
  Check(configs->eigenplaces_config.descriptor_dim == 512, "sample eigenplaces dim");
  Check(configs->eigenplaces_config.input_tensor_names.size() == 1, "sample eigenplaces inputs");
  Check(configs->eigenplaces_config.onnx_file == "unit-weights/eigenplaces.onnx", "sample eigenplaces onnx");
  Check(configs->flexglue_config.feature_dim == 256, "sample flexglue dim");
  Check(configs->superpoint_config.engine_file == "unit-weights/superpoint_v1.engine", "sample engine path");

  const std::shared_ptr<const viewfeat::Configs> again = viewfeat::GetConfigs();
  Check(again.get() == configs.get(), "unchanged file stays cached");
}

void TestMissingFileThenAppears(TempDir &dir) {
  const std::string path = dir.File("later.yaml");
  Use(path);
  const std::shared_ptr<const viewfeat::Configs> missing = viewfeat::GetConfigs();
  Check(static_cast<bool>(missing), "missing file still returns an empty config");
  Check(missing && missing->cosplace_config.input_tensor_names.empty(), "missing file has no cosplace tensors");

  Check(WriteFile(path, CosPlaceYaml(512)), "write appeared config");
  const std::shared_ptr<const viewfeat::Configs> loaded = viewfeat::GetConfigs();
  Check(loaded && loaded->cosplace_config.descriptor_dim == 512, "file that appears later is loaded");
  Check(missing && missing->cosplace_config.input_tensor_names.empty(), "the earlier empty config stays empty");
  Check(loaded.get() != missing.get(), "appearance replaces the cached object");
}

void TestRewriteKeepsHeldConfig(TempDir &dir) {
  const std::string path = dir.File("rewrite.yaml");
  Check(WriteFile(path, CosPlaceYaml(512)), "write first config");
  Check(SetMtime(path, 1700000000), "stamp first config");
  Use(path);
  const std::shared_ptr<const viewfeat::Configs> first = viewfeat::GetConfigs();
  Check(first && first->cosplace_config.descriptor_dim == 512, "first load");

  Check(WriteFile(path, CosPlaceYaml(256)), "rewrite config");
  Check(SetMtime(path, 1700000001), "stamp rewritten config");
  const std::shared_ptr<const viewfeat::Configs> second = viewfeat::GetConfigs();
  Check(second && second->cosplace_config.descriptor_dim == 256, "rewrite is visible");
  Check(first && first->cosplace_config.descriptor_dim == 512, "held config is unchanged");
}

void TestMissingSuperPointSection(TempDir &dir) {
  const std::string path = dir.File("no-superpoint.yaml");
  Check(WriteFile(path, std::string(kAlikedSection) + CosPlaceYaml(512)), "write partial config");
  Use(path);
  const std::shared_ptr<const viewfeat::Configs> configs = viewfeat::GetConfigs();
  Check(static_cast<bool>(configs), "missing superpoint section still loads");
  if (!configs) {
    return;
  }
  Check(configs->superpoint_config.input_tensor_names.empty(), "absent superpoint section stays empty");
  Check(configs->aliked_config.descriptor_dim == 128, "aliked section loaded");
  Check(configs->aliked_config.input_tensor_names.size() == 1, "aliked tensor names loaded");
  Check(configs->cosplace_config.descriptor_dim == 512, "cosplace section loaded");

  std::unique_ptr<viewfeat::FeaturePipeline> pipeline;
  std::string log;
  try {
    CerrCapture capture;
    pipeline = viewfeat::FeaturePipeline::Create("superpoint");
    log = capture.str();
  } catch (const std::exception &ex) {
    Check(false, std::string("Create threw: ") + ex.what());
  }
  Check(!pipeline, "superpoint pipeline is null without its section");
  Check(log.find("Failed to load config") == std::string::npos, "absent section is not a parse failure");
  Check(log.find("Engine not found") == std::string::npos, "absent section does not open an engine");
}

void TestOptionalNumKeypoints(TempDir &dir) {
  const std::string path = dir.File("optional.yaml");
  Check(WriteFile(path, kLightGlueWithoutNumKeypoints), "write optional key config");
  Use(path);
  const std::shared_ptr<const viewfeat::Configs> configs = viewfeat::GetConfigs();
  Check(configs && configs->superpoint_lightglue_config.num_keypoints == 100, "omitted num_keypoints uses max_keypoints");
}

void TestBrokenYamlDoesNotThrow(TempDir &dir) {
  const std::string path = dir.File("broken.yaml");
  Check(WriteFile(path, "superpoint: [\n"), "write broken yaml");
  Check(SetMtime(path, 1700000100), "stamp broken yaml");
  Use(path);

  std::string first_log;
  {
    CerrCapture capture;
    std::shared_ptr<const viewfeat::Configs> configs;
    try {
      configs = viewfeat::GetConfigs();
    } catch (const std::exception &ex) {
      Check(false, std::string("GetConfigs threw: ") + ex.what());
    }
    first_log = capture.str();
    Check(!configs, "broken yaml returns null");
  }
  Check(first_log.find("could not be parsed") != std::string::npos, "broken yaml is logged");
  Check(first_log.find(path) != std::string::npos, "broken yaml log names the file");

  {
    CerrCapture capture;
    const std::shared_ptr<const viewfeat::Configs> cached = viewfeat::GetConfigs();
    Check(!cached, "failed parse stays cached");
    Check(capture.str().empty(), "cached failure does not log again");
  }

  std::unique_ptr<viewfeat::FeaturePipeline> pipeline;
  std::unique_ptr<viewfeat::GlobalRetriever> retriever;
  try {
    pipeline = viewfeat::FeaturePipeline::Create("aliked");
    retriever = viewfeat::GlobalRetriever::Create("cosplace");
  } catch (const std::exception &ex) {
    Check(false, std::string("Create threw on broken yaml: ") + ex.what());
  }
  Check(!pipeline && !retriever, "Create turns a parse failure into null");

  Check(WriteFile(path, CosPlaceYaml(128)), "repair broken yaml");
  Check(SetMtime(path, 1700000101), "stamp repaired yaml");
  const std::shared_ptr<const viewfeat::Configs> repaired = viewfeat::GetConfigs();
  Check(repaired && repaired->cosplace_config.descriptor_dim == 128, "rewritten file replaces a cached failure");
}

void TestMissingKey(TempDir &dir) {
  const std::string path = dir.File("missing-key.yaml");
  const std::string yaml = std::string("superpoint:\n  keypoint_threshold: 0.01\n") + CosPlaceYaml(512);
  Check(WriteFile(path, yaml), "write missing key");
  Use(path);
  std::string log;
  std::shared_ptr<const viewfeat::Configs> configs;
  try {
    CerrCapture capture;
    configs = viewfeat::GetConfigs();
    log = capture.str();
  } catch (const std::exception &ex) {
    Check(false, std::string("missing key threw: ") + ex.what());
  }
  Check(!configs, "missing key returns null");
  Check(log.find("superpoint") != std::string::npos && log.find("image_width") != std::string::npos, "missing key log names the section and key");
}

void TestWrongType(TempDir &dir) {
  const std::string path = dir.File("wrong-type.yaml");
  Check(WriteFile(path, "feature_type: [superpoint]\n"), "write wrong type");
  Use(path);
  std::string log;
  std::shared_ptr<const viewfeat::Configs> configs;
  try {
    CerrCapture capture;
    configs = viewfeat::GetConfigs();
    log = capture.str();
  } catch (const std::exception &ex) {
    Check(false, std::string("wrong type threw: ") + ex.what());
  }
  Check(!configs, "wrong type returns null");
  Check(log.find("feature_type") != std::string::npos, "wrong type log names the key");
}

void CreateAndCapture(const std::string &family, std::unique_ptr<viewfeat::FeaturePipeline> &pipeline, std::string &log) {
  pipeline.reset();
  log.clear();
  try {
    CerrCapture capture;
    pipeline = viewfeat::FeaturePipeline::Create(family);
    log = capture.str();
  } catch (const std::exception &ex) {
    Check(false, std::string("Create threw: ") + ex.what());
  }
}

void TestFeatureTypeSelectsDefault(TempDir &dir) {
  std::string wrong_dim = kAlikedSection;
  const std::string dim_key = "descriptor_dim: 128";
  const size_t dim_pos = wrong_dim.find(dim_key);
  Check(dim_pos != std::string::npos, "fixture contains descriptor_dim");
  if (dim_pos != std::string::npos) {
    wrong_dim.replace(dim_pos, dim_key.size(), "descriptor_dim: 64");
  }

  const std::string selected = dir.File("feature-type-aliked.yaml");
  Check(WriteFile(selected, "feature_type: \"aliked\"\n" + wrong_dim), "write feature_type aliked");
  Use(selected);

  std::unique_ptr<viewfeat::FeaturePipeline> pipeline;
  std::string log;
  CreateAndCapture("default", pipeline, log);
  Check(!pipeline, "Create(default) follows feature_type");
  Check(log.find("aliked.descriptor_dim is 64") != std::string::npos, "Create(default) selects aliked");
  Check(log.find("Engine not found") == std::string::npos, "rejected descriptor_dim does not open an engine");

  CreateAndCapture("", pipeline, log);
  Check(log.find("aliked.descriptor_dim is 64") != std::string::npos, "empty family selects aliked");

  CreateAndCapture("DEFAULT", pipeline, log);
  Check(log.find("aliked.descriptor_dim is 64") != std::string::npos, "Default is case-insensitive");

  CreateAndCapture("superpoint", pipeline, log);
  Check(!pipeline, "explicit superpoint ignores feature_type");
  Check(log.find("aliked.descriptor_dim") == std::string::npos, "explicit family does not read feature_type");
  Check(log.find("Unsupported feature family") == std::string::npos, "explicit superpoint stays supported");

  const std::string folded = dir.File("feature-type-folded.yaml");
  Check(WriteFile(folded, "feature_type: \"ALIKED\"\n" + wrong_dim), "write folded feature_type");
  Use(folded);
  CreateAndCapture("default", pipeline, log);
  Check(log.find("aliked.descriptor_dim is 64") != std::string::npos, "feature_type is case-insensitive");

  const std::string flexglue = dir.File("feature-type-flexglue.yaml");
  Check(WriteFile(flexglue, "feature_type: \"superpoint_flexglue\"\n"), "write flexglue feature_type");
  Use(flexglue);
  CreateAndCapture("default", pipeline, log);
  Check(!pipeline, "flexglue feature_type does not build without an engine");
  Check(log.find("Unsupported feature family") == std::string::npos, "superpoint_flexglue is a known family");
  Check(log.find("FlexGlueMatcher") != std::string::npos, "default selects the flexglue matcher");

  const std::string missing = dir.File("feature-type-missing.yaml");
  Check(WriteFile(missing, wrong_dim), "write config without feature_type");
  Use(missing);
  CreateAndCapture("default", pipeline, log);
  Check(!pipeline, "missing feature_type does not build aliked");
  Check(log.find("aliked.descriptor_dim") == std::string::npos, "missing feature_type stays superpoint");
  Check(log.find("Unsupported feature family") == std::string::npos, "missing feature_type is still a known family");

  const std::string bad = dir.File("feature-type-bad.yaml");
  Check(WriteFile(bad, "feature_type: \"not-a-family\"\n" + wrong_dim), "write bad feature_type");
  Use(bad);
  CreateAndCapture("default", pipeline, log);
  Check(!pipeline, "unknown feature_type is null");
  Check(log.find("Unsupported feature family") != std::string::npos, "unknown feature_type is unsupported");
  Check(log.find("not-a-family") != std::string::npos, "unknown feature_type is named");
  Check(log.find("Engine not found") == std::string::npos, "unknown feature_type stops before an engine");

  const std::string loop = dir.File("feature-type-default.yaml");
  Check(WriteFile(loop, "feature_type: \"default\"\n"), "write feature_type default");
  Use(loop);
  CreateAndCapture("", pipeline, log);
  Check(!pipeline, "feature_type default is null");
  Check(log.find("Unsupported feature family") != std::string::npos, "feature_type default is not a family");
}

void TestUnknownFamilyDoesNotThrow(TempDir &dir) {
  const std::string path = dir.File("unused.yaml");
  Check(WriteFile(path, ":\n"), "write unused broken yaml");
  Use(path);
  std::unique_ptr<viewfeat::FeaturePipeline> pipeline;
  std::string log;
  try {
    CerrCapture capture;
    pipeline = viewfeat::FeaturePipeline::Create("not-a-family");
    log = capture.str();
  } catch (const std::exception &ex) {
    Check(false, std::string("unknown family threw: ") + ex.what());
  }
  Check(!pipeline, "unknown family is null");
  Check(log.find("Unsupported feature family") != std::string::npos, "unknown family keeps its own error");
}

}  // namespace

int main() {
  TempDir dir;
  Check(dir.ok(), "temp dir");
  if (!dir.ok()) {
    return 1;
  }

  TestSampleConfig();
  TestMissingFileThenAppears(dir);
  TestRewriteKeepsHeldConfig(dir);
  TestMissingSuperPointSection(dir);
  TestOptionalNumKeypoints(dir);
  TestBrokenYamlDoesNotThrow(dir);
  TestMissingKey(dir);
  TestWrongType(dir);
  TestFeatureTypeSelectsDefault(dir);
  TestUnknownFamilyDoesNotThrow(dir);

  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed" << std::endl;
    return 1;
  }
  std::cout << "config tests passed" << std::endl;
  return 0;
}
