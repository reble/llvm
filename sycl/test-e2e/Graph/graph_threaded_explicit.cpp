// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test each thread adding of nodes to same graph.

#include "graph_common.hpp"
#include <thread>

int main() {
  queue TestQueue;

  using T = int;

  const size_t Elements = 1024;
  const unsigned NumThreads = std::thread::hardware_concurrency();
  std::vector<T> DataA(Elements), DataB(Elements), DataC(Elements);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(Elements, TestQueue);
  T *PtrB = malloc_device<T>(Elements, TestQueue);
  T *PtrC = malloc_device<T>(Elements, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, Elements);
  TestQueue.copy(DataB.data(), PtrB, Elements);
  TestQueue.copy(DataC.data(), PtrC, Elements);
  TestQueue.wait_and_throw();

  auto AddNodesToGraph = [&]() {
    add_kernels_usm(Graph, Elements, PtrA, PtrB, PtrC);
  };

  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);
  for (size_t i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(AddNodesToGraph);
  }

  for (unsigned i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  return 0;
}
