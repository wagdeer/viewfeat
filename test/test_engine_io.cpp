#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "runtime/engine_io.h"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    const char *env = std::getenv("TMPDIR");
    std::string root = (env != nullptr && env[0] != '\0') ? env : "/tmp";
    while (!root.empty() && root[root.size() - 1] == '/') {
      root.erase(root.size() - 1);
    }
    if (root.empty()) {
      root = "/tmp";
    }
    std::string pattern = root + "/viewfeat-engine-io-XXXXXX";
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
    std::string path = path_ + "/" + name;
    files_.push_back(path);
    return path;
  }

 private:
  std::string path_;
  std::vector<std::string> files_;
};

bool WriteFile(const std::string &path, const std::string &bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  if (!bytes.empty()) {
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  out.close();
  return static_cast<bool>(out);
}

bool HaveNvidiaDevice() {
  struct stat info;
  // Native Linux exposes /dev/nvidia*. WSL2 exposes the GPU through /dev/dxg.
  return stat("/dev/nvidia0", &info) == 0 || stat("/dev/nvidiactl", &info) == 0 || stat("/dev/dxg", &info) == 0;
}

void ExpectNull(const std::string &check_name, const std::string &path, const char *label,
                const std::string &expected_log) {
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  std::string log;
  {
    CerrCapture capture;
    engine = viewfeat::LoadSerializedEngine(path, label);
    log = capture.str();
  }
  Check(engine == nullptr, check_name + " returned an engine");
  if (log == expected_log) {
    return;
  }
  std::cerr << "CHECK failed: " << check_name << " log mismatch" << std::endl;
  std::cerr << "  expected: " << expected_log;
  std::cerr << "  actual: " << log;
  ++g_failures;
}

int RunFileChecks(TempDir &temp) {
  const std::string missing = temp.File("missing.engine");
  ExpectNull("missing file", missing, "Test",
             std::string("[Test] Engine not found: ") + missing + ".\n");

  ExpectNull("empty path", "", "Test", "[Test] Engine not found: .\n");

  const std::string missing_null_label = temp.File("missing-null-label.engine");
  ExpectNull("nullptr label", missing_null_label, nullptr,
             std::string("[Engine] Engine not found: ") + missing_null_label + ".\n");

  const std::string empty = temp.File("empty.engine");
  if (!WriteFile(empty, "")) {
    Check(false, "write empty fixture " + empty);
  } else {
    ExpectNull("empty file", empty, "Test",
               std::string("[Test] Failed to read engine: ") + empty + ".\n");
  }

  if (g_failures != 0) {
    return 1;
  }
  std::cout << "engine_io: all checks passed" << std::endl;
  return 0;
}

void ExitChild(int code) {
  std::cout.flush();
  std::cerr.flush();
  _exit(code);
}

void GarbageChild(const std::string &path) {
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  std::string log;
  try {
    CerrCapture capture;
    engine = viewfeat::LoadSerializedEngine(path, "Garbage");
    log = capture.str();
  } catch (const std::exception &ex) {
    if (!HaveNvidiaDevice()) {
      std::cout << "SKIP: garbage file threw without an NVIDIA device: " << ex.what() << std::endl;
      ExitChild(77);
    }
    std::cerr << "CHECK failed: garbage file threw: " << ex.what() << std::endl;
    ExitChild(1);
  } catch (...) {
    if (!HaveNvidiaDevice()) {
      std::cout << "SKIP: garbage file threw without an NVIDIA device" << std::endl;
      ExitChild(77);
    }
    std::cerr << "CHECK failed: garbage file threw" << std::endl;
    ExitChild(1);
  }

  if (!log.empty()) {
    std::cerr << log;
  }
  if (engine != nullptr) {
    std::cerr << "CHECK failed: garbage file returned an engine" << std::endl;
    ExitChild(1);
  }
  if (log.find("[Garbage] Engine not found:") != std::string::npos ||
      log.find("[Garbage] Failed to read engine:") != std::string::npos) {
    std::cerr << "CHECK failed: garbage file hit the file-error path" << std::endl;
    ExitChild(1);
  }
  ExitChild(0);
}

// Parent-side deadline so a CUDA init hang cannot block ctest. The child is the
// only process that calls TensorRT; fork keeps a missing GPU from aborting the
// file-error cases, which run as a separate ctest invocation.
bool WaitChild(pid_t pid, int timeout_seconds, int *status) {
  const std::time_t deadline = std::time(nullptr) + timeout_seconds;
  for (;;) {
    pid_t waited = waitpid(pid, status, WNOHANG);
    if (waited == pid) {
      return true;
    }
    if (waited < 0 && errno != EINTR) {
      return false;
    }
    if (std::time(nullptr) >= deadline) {
      if (kill(pid, SIGKILL) < 0 && errno != ESRCH) {
        return false;
      }
      do {
        waited = waitpid(pid, status, 0);
      } while (waited < 0 && errno == EINTR);
      return waited == pid;
    }
    usleep(100000);
  }
}

int FinishGarbageStatus(int status) {
  if (WIFEXITED(status)) {
    const int code = WEXITSTATUS(status);
    if (code == 0) {
      std::cout << "engine_io_garbage: all checks passed" << std::endl;
    }
    return code;
  }
  if (WIFSIGNALED(status)) {
    const int signal_number = WTERMSIG(status);
    if (!HaveNvidiaDevice()) {
      std::cout << "SKIP: garbage file ended with signal " << signal_number
                << " and no NVIDIA device is present" << std::endl;
      return 77;
    }
    std::cerr << "CHECK failed: garbage file ended with signal " << signal_number << std::endl;
    return 1;
  }
  std::cerr << "CHECK failed: garbage child ended unexpectedly" << std::endl;
  return 1;
}

int RunGarbage(TempDir &temp) {
  const std::string path = temp.File("garbage.engine");
  if (!WriteFile(path, "not-a-tensorrt-engine")) {
    std::cerr << "CHECK failed: write garbage fixture " << path << std::endl;
    return 1;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "CHECK failed: fork: " << std::strerror(errno) << std::endl;
    return 1;
  }
  if (pid == 0) {
    GarbageChild(path);
    ExitChild(1);
  }

  int status = 0;
  if (!WaitChild(pid, 30, &status)) {
    std::cerr << "CHECK failed: waitpid: " << std::strerror(errno) << std::endl;
    return 1;
  }
  return FinishGarbageStatus(status);
}

}  // namespace

int main(int argc, char **argv) {
  const bool garbage_only = argc > 1 && std::string(argv[1]) == "--garbage";
  TempDir temp;
  if (!temp.ok()) {
    std::cerr << "CHECK failed: mkdtemp: " << std::strerror(errno) << std::endl;
    return 1;
  }
  if (garbage_only) {
    return RunGarbage(temp);
  }
  return RunFileChecks(temp);
}
