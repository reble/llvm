// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out

// CHECK: piextCommandBufferAdviseUSM

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/memadvise.cpp"
