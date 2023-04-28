// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests queue shortcuts for executing a graph

#include "graph_common.hpp"

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  calculate_reference_data(iterations, size, referenceA, referenceB,
                           referenceC);

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};

  buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
  buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
  buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.wait_and_throw();

  graph.begin_recording(testQueue);

  // Record commands to graph
  run_kernels_usm(testQueue, size, ptrA, ptrB, ptrC);

  graph.end_recording();
  auto graphExec = graph.finalize();

  // Execute several iterations of the graph using the different shortcuts
  event e = testQueue.ext_oneapi_graph(graphExec);

  assert(iterations > 2);
  const unsigned loop_iterations = iterations - 2;
  std::vector<event> events(loop_iterations);
  for (unsigned n = 0; n < loop_iterations; n++) {
    events[n] = testQueue.ext_oneapi_graph(graphExec, e);
  }

  testQueue.ext_oneapi_graph(graphExec, events).wait();

  testQueue.copy(ptrA, dataA.data(), size);
  testQueue.copy(ptrB, dataB.data(), size);
  testQueue.copy(ptrC, dataC.data(), size);
  testQueue.wait_and_throw();

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  return 0;
}
