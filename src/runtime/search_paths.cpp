#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "runtime/search_paths.h"

#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "runtime/paths.h"

#ifndef VIEWFEAT_CONFIG_PATH
#define VIEWFEAT_CONFIG_PATH "config/config.yaml"
#endif

#ifndef VIEWFEAT_MODEL_DIR
#define VIEWFEAT_MODEL_DIR "weights"
#endif

#ifndef VIEWFEAT_SOURCE_CONFIG_PATH
#define VIEWFEAT_SOURCE_CONFIG_PATH "config/config.yaml"
#endif

#ifndef VIEWFEAT_SOURCE_MODEL_DIR
#define VIEWFEAT_SOURCE_MODEL_DIR "weights"
#endif

namespace viewfeat {

namespace {

std::string g_config_path_override;
std::string g_model_dir_override;

struct FileStamp {
  bool exists = false;
  std::int64_t mtime_sec = 0;
  std::int64_t mtime_nsec = 0;
  std::int64_t size = 0;

  bool operator==(const FileStamp &other) const {
    return exists == other.exists && mtime_sec == other.mtime_sec && mtime_nsec == other.mtime_nsec && size == other.size;
  }
};

struct CachedConfig {
  bool valid = false;
  std::string config_path;
  std::string model_dir;
  FileStamp stamp;
  std::shared_ptr<const Configs> configs;
};

std::mutex &PathMutex() {
  static std::mutex mutex;
  return mutex;
}

CachedConfig &Cache() {
  static CachedConfig cache;
  return cache;
}

const char *EnvOrNull(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return nullptr;
  }
  return value;
}

FileStamp StampFile(const std::string &path) {
  FileStamp stamp;
  struct stat file_status {};
  if (stat(path.c_str(), &file_status) != 0 || (file_status.st_mode & S_IFREG) == 0) {
    return stamp;
  }
  stamp.exists = true;
  stamp.mtime_sec = static_cast<std::int64_t>(file_status.st_mtim.tv_sec);
  stamp.mtime_nsec = static_cast<std::int64_t>(file_status.st_mtim.tv_nsec);
  stamp.size = static_cast<std::int64_t>(file_status.st_size);
  return stamp;
}

std::string ResolveConfigPathUnlocked() {
  if (!g_config_path_override.empty()) {
    return g_config_path_override;
  }
  if (const char *env = EnvOrNull("VIEWFEAT_CONFIG_PATH")) {
    return env;
  }
  if (FileExists(VIEWFEAT_CONFIG_PATH)) {
    return VIEWFEAT_CONFIG_PATH;
  }
  if (FileExists(VIEWFEAT_SOURCE_CONFIG_PATH)) {
    return VIEWFEAT_SOURCE_CONFIG_PATH;
  }
  return VIEWFEAT_CONFIG_PATH;
}

std::string ResolveModelDirUnlocked() {
  if (!g_model_dir_override.empty()) {
    return g_model_dir_override;
  }
  if (const char *env = EnvOrNull("VIEWFEAT_MODEL_DIR")) {
    return env;
  }
  if (DirectoryExists(VIEWFEAT_MODEL_DIR)) {
    return VIEWFEAT_MODEL_DIR;
  }
  if (DirectoryExists(VIEWFEAT_SOURCE_MODEL_DIR)) {
    return VIEWFEAT_SOURCE_MODEL_DIR;
  }
  return VIEWFEAT_MODEL_DIR;
}

}  // namespace

void SetSearchPaths(const std::string &config_path, const std::string &model_dir) {
  std::lock_guard<std::mutex> lock(PathMutex());
  g_config_path_override = config_path;
  g_model_dir_override = model_dir;
}

std::string ResolveConfigPath() {
  std::lock_guard<std::mutex> lock(PathMutex());
  return ResolveConfigPathUnlocked();
}

std::string ResolveModelDir() {
  std::lock_guard<std::mutex> lock(PathMutex());
  return ResolveModelDirUnlocked();
}

std::shared_ptr<const Configs> GetConfigs() {
  std::lock_guard<std::mutex> lock(PathMutex());
  const std::string config_path = ResolveConfigPathUnlocked();
  const std::string model_dir = ResolveModelDirUnlocked();
  const FileStamp stamp = StampFile(config_path);

  CachedConfig &cache = Cache();
  if (cache.valid && cache.config_path == config_path && cache.model_dir == model_dir && cache.stamp == stamp) {
    return cache.configs;
  }

  cache.configs = LoadConfigs(config_path, model_dir);
  cache.config_path = config_path;
  cache.model_dir = model_dir;
  cache.stamp = stamp;
  cache.valid = true;
  return cache.configs;
}

}  // namespace viewfeat
