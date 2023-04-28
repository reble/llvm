// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as thread safety not yet implemented
// XFAIL: *

// Test each thread recording the same graph to a different queue.

#include "graph_common.hpp"

#include <thread>

using namespace sycl;

int main() {
  queue testQueue;

  using T = int;

  const size_t size = 1024;
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

  auto recordGraph = [&]() {
    queue myQueue;

    // Record commands to graph
    graph.begin_recording(myQueue);
    run_kernels_usm(myQueue, size, ptrA, ptrB, ptrC);
    graph.end_recording(myQueue);
  };

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
