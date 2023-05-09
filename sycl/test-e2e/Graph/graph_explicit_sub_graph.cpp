// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test creates a graph, finalizes it, then submits that as a subgraph of
// another graph and executes that second graph.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  // Values used to modify data inside kernels.
  const int ModValue = 7;
  std::vector<T> DataA(size), DataB(size), DataC(size), DataOut(size);

  // Initialize the data
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataOut.begin(), DataOut.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA);
  std::vector<T> ReferenceB(DataB);
  std::vector<T> ReferenceC(DataC);
  std::vector<T> ReferenceOut(DataOut);
  for (size_t n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      ReferenceA[i] += ModValue;
      ReferenceB[i] += ModValue;
      ReferenceC[i] = (ReferenceA[i] + ReferenceB[i]);
      ReferenceC[i] -= ModValue;
      ReferenceOut[i] = ReferenceC[i] + ModValue;
    }
  }

  exp_ext::command_graph SubGraph{TestQueue.get_context(),
                                  TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);
  T *PtrOut = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.copy(DataOut.data(), PtrOut, size);
  TestQueue.wait_and_throw();

  // Add some operations to a graph which will later be submitted as part
  // of another graph.

  // Vector add two values
  auto NodeSubA = SubGraph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>(size),
                     [=](item<1> id) { PtrC[id] = PtrA[id] + PtrB[id]; });
  });

  // Modify the output value with some other value
  SubGraph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(size),
                         [=](item<1> id) { PtrC[id] -= ModValue; });
      },
      {exp_ext::property::node::depends_on(NodeSubA)});

  auto SubGraphExec = SubGraph.finalize();

  exp_ext::command_graph MainGraph{TestQueue.get_context(),
                                   TestQueue.get_device()};

  // Modify the input values.
  auto NodeMainA = MainGraph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>(size), [=](item<1> id) {
      PtrA[id] += ModValue;
      PtrB[id] += ModValue;
    });
  });

  auto NodeMainB =
      MainGraph.add([&](handler &CGH) { CGH.ext_oneapi_graph(SubGraphExec); },
                    {exp_ext::property::node::depends_on(NodeMainA)});

  // Copy to another output buffer.
  MainGraph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(size),
                         [=](item<1> id) { PtrOut[id] = PtrC[id] + ModValue; });
      },
      {exp_ext::property::node::depends_on(NodeMainB)});

  // Finalize a graph with the additional kernel for writing out to
  auto MainGraphExec = MainGraph.finalize();

  // Execute several iterations of the graph
  for (size_t n = 0; n < iterations; n++) {
    TestQueue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(MainGraphExec); });
  }
  // Perform a wait on all graph submissions.
  TestQueue.wait_and_throw();

  TestQueue.copy(PtrA, DataA.data(), size);
  TestQueue.copy(PtrB, DataB.data(), size);
  TestQueue.copy(PtrC, DataC.data(), size);
  TestQueue.copy(PtrOut, DataOut.data(), size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);
  free(PtrOut, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);
  assert(ReferenceOut == DataOut);
  return 0;
}
