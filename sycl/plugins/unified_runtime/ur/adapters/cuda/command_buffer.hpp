//===--------- command_buffer.hpp - CUDA Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <ur/ur.hpp>
#include <ur_api.h>

#include <cuda_runtime.h>

// Trace an internal PI call; returns in case of an error.
#define UR_CALL(Call)                                                          \
  {                                                                            \
    if (PrintTrace)                                                            \
      fprintf(stderr, "UR ---> %s\n", #Call);                                  \
    ur_result_t Result = (Call);                                               \
    if (PrintTrace)                                                            \
      fprintf(stderr, "UR <--- %s(%s)\n", #Call, getUrResultString(Result));   \
    if (Result != UR_RESULT_SUCCESS)                                           \
      return Result;                                                           \
  }

// This wrapper around std::atomic is created to limit operations with reference
// counter and to make allowed operations more transparent in terms of
// thread-safety in the plugin. increment() and load() operations do not need a
// mutex guard around them since the underlying data is already atomic.
// decrementAndTest() method is used to guard a code which needs to be
// executed when object's ref count becomes zero after release. This method also
// doesn't need a mutex guard because decrement operation is atomic and only one
// thread can reach ref count equal to zero, i.e. only a single thread can pass
// through this check.
struct ReferenceCounter {
  ReferenceCounter() : RefCount{1} {}

  // Reset the counter to the initial value.
  void reset() { RefCount = 1; }

  // Used when retaining an object.
  void increment() { RefCount++; }

  // Supposed to be used in pi*GetInfo* methods where ref count value is
  // requested.
  uint32_t load() { return RefCount.load(); }

  // This method allows to guard a code which needs to be executed when object's
  // ref count becomes zero after release. It is important to notice that only a
  // single thread can pass through this check. This is true because of several
  // reasons:
  //   1. Decrement operation is executed atomically.
  //   2. It is not allowed to retain an object after its refcount reaches zero.
  //   3. It is not allowed to release an object more times than the value of
  //   the ref count.
  // 2. and 3. basically means that we can't use an object at all as soon as its
  // refcount reaches zero. Using this check guarantees that code for deleting
  // an object and releasing its resources is executed once by a single thread
  // and we don't need to use any mutexes to guard access to this object in the
  // scope after this check. Of course if we access another objects in this code
  // (not the one which is being deleted) then access to these objects must be
  // guarded, for example with a mutex.
  bool decrementAndTest() { return --RefCount == 0; }

private:
  std::atomic<uint32_t> RefCount;
};

// Base class to store common data
struct _ur_object {
  _ur_object() : RefCount{} {}

  // Must be atomic to prevent data race when incrementing/decrementing.
  ReferenceCounter RefCount;
};

/// Stub implementation of command-buffers for CUDA

struct ur_exp_command_buffer_handle_t_ : public _ur_object {

  ur_exp_command_buffer_handle_t_(ur_context_handle_t Context,
                                  ur_device_handle_t Device);

  ~ur_exp_command_buffer_handle_t_();

  // UR context associated with this command-buffer
  ur_context_handle_t Context;
  // Device associated with this command buffer
  ur_device_handle_t Device;
  // Cuda Graph handle
  cudaGraph_t cudaGraph;
  // Cuda Graph Exec handle
  cudaGraphExec_t cudaGraphExec;

  // Queue properties from command-buffer descriptor
  // TODO: Do we need these?
  ur_queue_properties_t QueueProperties;
};
