// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as executable graph update not yet implemented
// XFAIL: *

// Tests whole graph update by creating a double buffering scenario, where a
// single graph is repeatedly executed then updated to swap between two sets of
// USM pointers.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(size), DataB(size), DataC(size);
  std::vector<T> DataA2(size), DataB2(size), DataC2(size);
  // Initialize the data
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::iota(DataA2.begin(), DataA2.end(), 3);
  std::iota(DataB2.begin(), DataB2.end(), 13);
  std::iota(DataC2.begin(), DataC2.end(), 1333);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  std::vector<T> ReferenceA2(DataA2), ReferenceB2(DataB2), ReferenceC2(DataC2);
  // Calculate Reference data
  calculate_reference_data(iterations, size, ReferenceA, ReferenceB,
                           ReferenceC);
  calculate_reference_data(iterations, size, ReferenceA2, ReferenceB2,
                           ReferenceC2);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);

  T *PtrA2 = malloc_device<T>(size, TestQueue);
  T *PtrB2 = malloc_device<T>(size, TestQueue);
  T *PtrC2 = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);

  TestQueue.copy(DataA2.data(), PtrA, size);
  TestQueue.copy(DataB2.data(), PtrB, size);
  TestQueue.copy(DataC2.data(), PtrC, size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);
  run_kernels_usm(TestQueue, size, PtrA, PtrB, PtrC);
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  // Create second graph using other buffer set
  exp_ext::command_graph GraphUpdate{TestQueue.get_context(),
                                     TestQueue.get_device()};
  GraphUpdate.begin_recording(TestQueue);
  run_kernels_usm(TestQueue, size, PtrA2, PtrB2, PtrC2);
  GraphUpdate.end_recording();

  event Event;
  for (size_t i = 0; i < iterations; i++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
    // Update to second set of buffers
    ExecGraph.update(GraphUpdate);

    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
    // Reset back to original buffers
    ExecGraph.update(Graph);
  }

  TestQueue.wait_and_throw();

  TestQueue.copy(PtrA, DataA.data(), size);
  TestQueue.copy(PtrB, DataB.data(), size);
  TestQueue.copy(PtrC, DataC.data(), size);

  TestQueue.copy(PtrA2, DataA2.data(), size);
  TestQueue.copy(PtrB2, DataB2.data(), size);
  TestQueue.copy(PtrC2, DataC2.data(), size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  free(PtrA2, TestQueue);
  free(PtrB2, TestQueue);
  free(PtrC2, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  assert(ReferenceA2 == DataA2);
  assert(ReferenceB2 == DataB2);
  assert(ReferenceC2 == DataC2);

  return 0;
}
