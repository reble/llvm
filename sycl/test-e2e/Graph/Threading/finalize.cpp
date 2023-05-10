// REQUIRES: level_zero, gpu, TEMPORARY_DISABLED
// Disabled as thread safety not yet implemented

// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test finalizing and submitting a graph in a threaded situation.

#include "../graph_common.hpp"

#include <thread>

int main() {
  queue TestQueue;

  using T = int;

  const unsigned NumThreads = std::thread::hardware_concurrency();
  std::vector<T> DataA(size), DataB(size), DataC(size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(NumThreads, size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);
  run_kernels_usm(TestQueue, size, PtrA, PtrB, PtrC);
  Graph.end_recording();

  auto FinalizeGraph = [&]() {
    auto GraphExec = graph.finalize();
    TestQueue.submit(
        [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  };

  std::vector<std::thread> Threads;
  Threads.reserve(NumThreads);

  for (size_t i = 0; i < NumThreads; ++i) {
    Threads.emplace_back(FinalizeGraph);
  }

  for (size_t i = 0; i < NumThreads; ++i) {
    Threads[i].join();
  }

  // Perform a wait on all graph submissions.
  TestQueue.wait_and_throw();

  TestQueue.copy(PtrA, DataA.data(), size);
  TestQueue.copy(PtrB, DataB.data(), size);
  TestQueue.copy(PtrC, DataC.data(), size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
