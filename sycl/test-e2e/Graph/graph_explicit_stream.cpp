// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER

// Expected fail as sycl streams aren't implemented yet
// XFAIL: *

// This test checks that we can use a stream when explicitly adding a
// command_graph node

#include "graph_common.hpp"

class stream_kernel;

int main() {
  queue testQueue;

  using T = int;

  size_t work_items = 16;
  std::vector<T> dataIn(work_items);

  // Initialize the data
  std::iota(dataIn.begin(), dataIn.end(), 1);

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};

  T *ptrIn = malloc_device<T>(work_items, testQueue);
  testQueue.copy(dataIn.data(), ptrIn, work_items);

  // Vector add to temporary output buffer
  graph.add([&](handler &cgh) {
    sycl::stream out(work_items * 16, 16, cgh);
    cgh.parallel_for<stream_kernel>(range<1>(work_items), [=](item<1> id) {
      out << "Val: " << ptrIn[id.get_linear_id()] << sycl::endl;
    });
  });

  auto graphExec = graph.finalize();

  testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });

  // Perform a wait on all graph submissions.
  testQueue.wait_and_throw();

  testQueue.copy(ptrIn, dataIn.data(), size);

  free(ptrIn, testQueue);

  return 0;
}

// CHECK-DAG: Val: 1
// CHECK-DAG: Val: 2
// CHECK-DAG: Val: 3
// CHECK-DAG: Val: 4
// CHECK-DAG: Val: 5
// CHECK-DAG: Val: 6
// CHECK-DAG: Val: 7
// CHECK-DAG: Val: 8
// CHECK-DAG: Val: 9
// CHECK-DAG: Val: 10
// CHECK-DAG: Val: 11
// CHECK-DAG: Val: 12
// CHECK-DAG: Val: 13
// CHECK-DAG: Val: 14
// CHECK-DAG: Val: 15
// CHECK-DAG: Val: 16
