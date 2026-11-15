#ifndef VIEWFEAT_RUNTIME_SEARCH_PATHS_H_
#define VIEWFEAT_RUNTIME_SEARCH_PATHS_H_

#include <memory>
#include <string>

#include "runtime/config.h"

namespace viewfeat {

void SetSearchPaths(const std::string &config_path, const std::string &model_dir);
std::string ResolveConfigPath();
std::string ResolveModelDir();
// The returned object stays alive after a later reload. nullptr means the
// resolved file exists but could not be parsed.
std::shared_ptr<const Configs> GetConfigs();

}  // namespace viewfeat

#endif  // VIEWFEAT_RUNTIME_SEARCH_PATHS_H_
