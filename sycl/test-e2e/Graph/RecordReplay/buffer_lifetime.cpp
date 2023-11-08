// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_EXTEND_BUFFER_LIFETIMES=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env SYCL_GRAPH_EXTEND_BUFFER_LIFETIMES=1 env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/buffer_lifetime.cpp"
