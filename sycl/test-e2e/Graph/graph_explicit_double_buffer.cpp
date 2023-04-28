// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as executable graph update isn't implemented yet
// XFAIL: *

// Tests whole graph update by creating a double buffering scenario, where a
// single graph is repeatedly executed then updated to swap between two sets of
// buffers.

#include "graph_common.hpp"

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);
  std::vector<T> dataA2(size), dataB2(size), dataC2(size);
  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  std::iota(dataA2.begin(), dataA2.end(), 3);
  std::iota(dataB2.begin(), dataB2.end(), 13);
  std::iota(dataC2.begin(), dataC2.end(), 1333);

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  std::vector<T> referenceA2(dataA2), referenceB2(dataB2), referenceC2(dataC2);
  // Calculate reference data
  calculate_reference_data(iterations, size, referenceA, referenceB,
                           referenceC);
  calculate_reference_data(iterations, size, referenceA2, referenceB2,
                           referenceC2);

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);

  T *ptrA2 = malloc_device<T>(size, testQueue);
  T *ptrB2 = malloc_device<T>(size, testQueue);
  T *ptrC2 = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);

  testQueue.copy(dataA2.data(), ptrA, size);
  testQueue.copy(dataB2.data(), ptrB, size);
  testQueue.copy(dataC2.data(), ptrC, size);
  testQueue.wait_and_throw();

  add_kernels_usm(graph, size, ptrA, ptrB, ptrC);

  auto execGraph = graph.finalize();

  // Create second graph using other buffer set
  exp_ext::command_graph graphUpdate{testQueue.get_context(),
                                     testQueue.get_device()};
  add_kernels_usm(graphUpdate, size, ptrA, ptrB, ptrC);

  event e;
  for (size_t i = 0; i < iterations; i++) {
    e = testQueue.submit([&](handler &cgh) {
      cgh.depends_on(e);
      cgh.ext_oneapi_graph(graphExec);
    });
    // Update to second set of buffers
    execGraph.update(graphUpdate);
    e = testQueue.submit([&](handler &cgh) {
      cgh.depends_on(e);
      cgh.ext_oneapi_graph(graphExec);
    });
    // Reset back to original buffers
    execGraph.update(graph);
  }

  testQueue.wait_and_throw();

  testQueue.copy(ptrA, dataA.data(), size);
  testQueue.copy(ptrB, dataB.data(), size);
  testQueue.copy(ptrC, dataC.data(), size);

  testQueue.copy(ptrA2, dataA2.data(), size);
  testQueue.copy(ptrB2, dataB2.data(), size);
  testQueue.copy(ptrC2, dataC2.data(), size);
  testQueue.wait_and_throw();

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);

  free(ptrA2, testQueue);
  free(ptrB2, testQueue);
  free(ptrC2, testQueue);

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  assert(referenceA2 == dataA2);
  assert(referenceB2 == dataB2);
  assert(referenceC2 == dataC2);

  return 0;
}
