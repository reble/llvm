// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as memcopy not implemented yet
// XFAIL: *

// Tests adding a usm memcpy node using the explicit API and submitting
// the graph.

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

  // memcpy from B to A
  auto NodeA = graph.add([&](handler &CGH) { CGH.copy(PtrB, PtrA, size); });

  // Read & write A
  auto NodeB = graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrA[LinID] += modValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // memcpy from B to A
  auto NodeC = graph.add([&](handler &CGH) { CGH.copy(PtrA, PtrB, size); },
                         {exp_ext::property::node::depends_on(NodeB)});

  // Read and write B
  auto nodeD = graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(size), [=](item<1> id) {
          auto LinID = id.get_linear_id();
          PtrB[LinID] += modValue;
        });
      },
      {exp_ext::property::node::depends_on(NodeC)});

  // memcpy from B to C
  graph.add([&](handler &CGH) { CGH.copy(PtrB, PtrC, size); },
            {exp_ext::property::node::depends_on(NodeB)});

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
