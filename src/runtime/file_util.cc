/*!
 *  Copyright (c) 2019 by Contributors if not otherwise specified
 * \file file_util.cc
 */
#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <decord/runtime/serializer.h>
#include <fstream>
#include <vector>

#include "file_util.h"

namespace decord {
namespace runtime {

std::string GetFileFormat(const std::string& file_name,
                          const std::string& format) {
  std::string fmt = format;
  if (fmt.length() == 0) {
    if (file_name.find(".signed.so") != std::string::npos) return "sgx";
    size_t pos = file_name.find_last_of(".");
    if (pos != std::string::npos) {
      return file_name.substr(pos + 1, file_name.length() - pos - 1);
    } else {
      return "";
    }
  } else {
    return format;
  }
}

std::string GetCacheDir() {
  char* env_cache_dir;
  if ((env_cache_dir = getenv("DECORD_CACHE_DIR"))) return env_cache_dir;
  if ((env_cache_dir = getenv("XDG_CACHE_HOME"))) {
    return std::string(env_cache_dir) + "/decord";
  }
  if ((env_cache_dir = getenv("HOME"))) {
    return std::string(env_cache_dir) + "/.cache/decord";
  }
  return ".";
}

std::string GetFileBasename(const std::string& file_name) {
  size_t last_slash = file_name.find_last_of("/");
  if (last_slash == std::string::npos) return file_name;
  return file_name.substr(last_slash + 1);
}

std::string GetMetaFilePath(const std::string& file_name) {
  size_t pos  = file_name.find_last_of(".");
  if (pos != std::string::npos) {
    return file_name.substr(0, pos) + ".decord_meta.json";
  } else {
    return file_name + ".decord_meta.json";
  }
}

void LoadBinaryFromFile(const std::string& file_name,
                        std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  CHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

void SaveBinaryToFile(
    const std::string& file_name,
    const std::string& data) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  CHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.write(&data[0], data.length());
}

void RemoveFile(const std::string& file_name) {
  std::remove(file_name.c_str());
}

}  // namespace runtime
}  // namespace decord
