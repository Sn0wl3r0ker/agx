// Compatibility shim for isaac_ros_common/cuda_stream.hpp (4.x -> 3.2)
// This file bridges the API gap between Isaac ROS 4.x and release-3.2
#pragma once

#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(cu_result, fmt, ...) \
  do { \
    cudaError_t err_ = (cu_result); \
    if (err_ != cudaSuccess) { \
      char user_msg_[1024]; \
      std::snprintf(user_msg_, sizeof(user_msg_), fmt, ## __VA_ARGS__); \
      std::ostringstream oss_; \
      oss_ << user_msg_ << ", cuda_error: " << cudaGetErrorName(err_) \
           << ", error_str: " << cudaGetErrorString(err_); \
      throw std::runtime_error(oss_.str()); \
    } \
  } while (0)
#endif

namespace nvidia {
namespace isaac_ros {
namespace common {

inline cudaError_t initNamedCudaStream(cudaStream_t & stream, const std::string & /*name*/) {
  return cudaStreamCreate(&stream);
}

inline void nameExistingCudaStream(cudaStream_t & /*stream*/, const std::string & /*name*/) {
  // NVTX naming not available in 3.2, no-op
}

}  // namespace common
}  // namespace isaac_ros
}  // namespace nvidia
