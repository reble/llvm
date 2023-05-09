// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as host tasks aren't implemented yet.
// XFAIL: *

// This test uses a host_task when explicitly adding a command_graph node.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  if (!TestQueue.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return 0;
  }

  const T modValue = T{7};
  std::vector<T> DataA(size), DataB(size), DataC(size);

  // Initialize the data
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> Reference(DataC);
  for (size_t n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      Reference[i] += (DataA[i] + DataB[i]) + modValue + 1;
    }
  }

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_shared<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.wait_and_throw();

  // Vector add to output
  auto NodeA = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>(size),
                     [=](item<1> id) { PtrC[id] += PtrA[id] + PtrB[id]; });
  });

  // Modify the output values in a host_task
  auto NodeB = Graph.add(
      [&](handler &CGH) {
        CGH.host_task([=]() {
          for (size_t i = 0; i < size; i++) {
            PtrC[i] += modValue;
          }
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // Modify temp buffer and write to output buffer
  Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(size), [=](item<1> id) { PtrC[id] += 1; });
      },
      {exp_ext::property::node::depends_on(NodeB)});

  auto GraphExec = Graph.finalize();

  // Execute several iterations of the graph
  for (size_t n = 0; n < iterations; n++) {
    TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }
  // Perform a wait on all graph submissions.
  TestQueue.wait_and_throw();

  TestQueue.copy(PtrC, DataC.data(), size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  assert(Reference == DataC);

  return 0;
}
