// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test recording commands to a queue in a threaded situation. We don't
// submit the graph to verify the results as ordering of graph nodes isn't
// defined.

#include "graph_common.hpp"
#include <thread>

int main() {
  queue testQueue;

  using T = int;

  const unsigned iterations = std::thread::hardware_concurrency();
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.wait_and_throw();

  graph.begin_recording(testQueue);
  auto recordGraph = [&]() {
    // Record commands to graph
    run_kernels_usm(testQueue, size, ptrA, ptrB, ptrC);
  };
  graph.end_recording();

  std::vector<std::thread> threads;
  threads.reserve(iterations);
  for (unsigned i = 0; i < iterations; ++i) {
    threads.emplace_back(recordGraph);
  }

  for (unsigned i = 0; i < iterations; ++i) {
    threads[i].join();
  }

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);

  return 0;
}
