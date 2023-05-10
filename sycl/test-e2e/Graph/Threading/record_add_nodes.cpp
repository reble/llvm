// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test recording commands to a queue in a threaded situation. We don't
// submit the graph to verify the results as ordering of graph nodes isn't
// defined.

#include "../graph_common.hpp"
#include <thread>

int main() {
  queue TestQueue;

  using T = int;

  const unsigned iterations = std::thread::hardware_concurrency();
  std::vector<T> DataA(size), DataB(size), DataC(size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);
  auto recordGraph = [&]() {
    // Record commands to graph
    run_kernels_usm(TestQueue, size, PtrA, PtrB, PtrC);
  };
  Graph.end_recording();

  std::vector<std::thread> Threads;
  Threads.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    Threads.emplace_back(recordGraph);
  }

  for (size_t i = 0; i < iterations; ++i) {
    Threads[i].join();
  }

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  return 0;
}
