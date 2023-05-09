// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test each thread recording the same graph to a different queue.

#include "graph_common.hpp"

#include <thread>

using namespace sycl;

int main() {
  queue TestQueue;

  using T = int;

  const size_t size = 1024;
  const unsigned iterations = std::thread::hardware_concurrency();
  std::vector<T> DataA(size), DataB(size), DataC(size);

  // Initialize the data
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

  auto RecordGraph = [&]() {
    queue MyQueue;

    // Record commands to graph
    Graph.begin_recording(MyQueue);
    run_kernels_usm(MyQueue, size, PtrA, PtrB, PtrC);
    Graph.end_recording(MyQueue);
  };

  std::vector<std::thread> Threads;
  Threads.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    Threads.emplace_back(RecordGraph);
  }

  for (size_t i = 0; i < iterations; ++i) {
    Threads[i].join();
  }

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  return 0;
}
