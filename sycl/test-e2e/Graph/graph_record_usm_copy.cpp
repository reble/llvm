// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as mem copy not implemented yet
// XFAIL: *

// Tests recording and submission of a graph containing usm memcpy commands.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  const T modValue = 7;
  std::vector<T> DataA(size), DataB(size), DataC(size);

  // Initialize the data
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (size_t i = 0; i < iterations; i++) {
    for (size_t j = 0; j < size; j++) {
      ReferenceA[j] = ReferenceB[j];
      ReferenceA[j] += modValue;
      ReferenceB[j] = ReferenceA[j];
      ReferenceB[j] += modValue;
      ReferenceC[j] = ReferenceB[j];
    }
  }

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);

  // Record commands to graph
  // memcpy from B to A
  auto EventA = TestQueue.copy(PtrB, PtrA, size);
  // Read & write A
  auto EventB = TestQueue.submit([&](handler &CGH) {
    CGH.depends_on(EventA);
    CGH.parallel_for(range<1>(size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrA[LinID] += modValue;
    });
  });

  // memcpy from A to B
  auto EventC = TestQueue.copy(PtrA, PtrB, size, EventB);

  // Read and write B
  auto EventD = TestQueue.submit([&](handler &CGH) {
    CGH.depends_on(EventC);
    CGH.parallel_for(range<1>(size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      PtrB[LinID] += modValue;
    });
  });

  // memcpy from B to C
  TestQueue.copy(PtrB, PtrC, size, EventD);

  Graph.end_recording();
  auto GraphExec = graph.finalize();

  // Execute graph over n iterations
  for (size_t n = 0; n < iterations; n++) {
    TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }
  // Perform a wait on all graph submissions.
  TestQueue.wait_and_throw();

  TestQueue.copy(PtrA, DataA.data(), size);
  TestQueue.copy(PtrB, DataB.data(), size);
  TestQueue.copy(PtrC, DataC.data(), size);

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
