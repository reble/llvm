// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test attempts creating multiple executable graphs from one modifiable
// graph.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(size), DataB(size), DataC(size);

  // Initialize the data
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(iterations, size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.wait_and_throw();

  run_kernels_usm(TestQueue, size, PtrA, PtrB, PtrC);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);
  // Record commands to graph
  run_kernels_usm(TestQueue, size, PtrA, PtrB, PtrC);
  Graph.end_recording();

  // Execute several iterations of the graph (first iteration has already
  // run before graph recording)
  for (size_t n = 1; n < iterations; n++) {
    auto GraphExec = Graph.finalize();
    TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
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
