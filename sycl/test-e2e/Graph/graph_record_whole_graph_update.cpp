// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as executable graph update not implemented yet
// XFAIL: *

// Tests whole graph update by recording two graphs with different buffers and
// attempting to update one from the other.

#include "graph_common.hpp"

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  auto dataA2 = dataA;
  auto dataB2 = dataB;
  auto dataC2 = dataC;

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  calculate_reference_data(iterations, size, referenceA, referenceB,
                           referenceC);

  exp_ext::command_graph graphA{testQueue.get_context(),
                                testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.wait_and_throw();

  graphA.begin_recording(testQueue);

  // Record commands to graph
  run_kernels_usm(testQueue, size, ptrA, ptrB, ptrC);

  graphA.end_recording();

  auto graphExec = graphA.finalize();

  exp_ext::command_graph graphB{testQueue.get_context(),
                                testQueue.get_device()};

  T *ptrA2 = malloc_device<T>(size, testQueue);
  T *ptrB2 = malloc_device<T>(size, testQueue);
  T *ptrC2 = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA2.data(), ptrA2, size);
  testQueue.copy(dataB2.data(), ptrB2, size);
  testQueue.copy(dataC2.data(), ptrC2, size);
  testQueue.wait_and_throw();

  graphB.begin_recording(testQueue);

  // Record commands to graph

  run_kernels_usm(testQueue, size, ptrA2, ptrB2, ptrC2);

  graphB.end_recording();
  // Execute several iterations of the graph for 1st set of buffers
  for (unsigned n = 0; n < iterations; n++) {
    testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
  }

  graphExec.update(graphB);

  // Execute several iterations of the graph for 2nd set of buffers
  for (unsigned n = 0; n < iterations; n++) {
    testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
  }

  // Perform a wait on all graph submissions.
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

  assert(referenceA == dataA2);
  assert(referenceB == dataB2);
  assert(referenceC == dataC2);

  return 0;
}
