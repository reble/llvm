// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as executable update isn't implemented yet
// XFAIL: *

// Test updating a graph in a threaded situation

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

  auto dataA2 = dataA;
  auto dataB2 = dataB;
  auto dataC2 = dataC;

  exp_ext::command_graph graphA{testQueue.get_context(),
                                testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.wait_and_throw();

  graphA.begin_recording(testQueue);

  // Record commands to graph
  run_kernels_usm(testQueue, size, ptrA, ptrB, ptrC);

  graphA.end_recording();

  auto graphExec = graphA.finalize();

  exp_ext::command_graph graphB{testQueue.get_context(),
                                testQueue.get_device()};

  T *ptrA2 = malloc_device<T>(size, testQueue);
  T *ptrB2 = malloc_device<T>(size, testQueue);
  T *ptrC2 = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA2.data(), ptrA2, size);
  testQueue.copy(dataB2.data(), ptrB2, size);
  testQueue.copy(dataC2.data(), ptrC2, size);
  testQueue.wait_and_throw();

  graphB.begin_recording(testQueue);

  // Record commands to graph
  run_kernels_usm(testQueue, size, ptrA2, ptrB2, ptrC2);

  graphB.end_recording();

  auto updateGraph = [&]() { graphExec.update(graphB); };

  std::vector<std::thread> threads;
  threads.reserve(iterations);

  for (unsigned i = 0; i < iterations; ++i) {
    threads.emplace_back(updateGraph);
  }

  for (unsigned i = 0; i < iterations; ++i) {
    threads[i].join();
  }

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);

  return 0;
}
