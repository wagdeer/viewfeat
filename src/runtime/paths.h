#ifndef VIEWFEAT_RUNTIME_PATHS_H_
#define VIEWFEAT_RUNTIME_PATHS_H_

#include <sys/stat.h>
#include <string>

namespace viewfeat {

inline bool FileExists(const std::string &file) {
  struct stat file_status {};
  return stat(file.c_str(), &file_status) == 0 && (file_status.st_mode & S_IFREG);
}

inline bool DirectoryExists(const std::string &path) {
  struct stat file_status {};
  return stat(path.c_str(), &file_status) == 0 && (file_status.st_mode & S_IFDIR);
}

inline std::string ConcatenateFolderAndFileName(const std::string &folder, const std::string &file_name) {
  std::string path = folder;
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  return path + file_name;
}

}  // namespace viewfeat

#endif  // VIEWFEAT_RUNTIME_PATHS_H_
