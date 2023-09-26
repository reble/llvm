// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out

// CHECK: piextCommandBufferPrefetchUSM

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/prefetch.cpp"
