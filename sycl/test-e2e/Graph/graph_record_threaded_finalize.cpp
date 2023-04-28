// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test finalizing and submitting a graph in a threaded situation

#include "graph_common.hpp"

#include <thread>

int main() {
  queue testQueue;

  using T = int;

  const unsigned iterations = std::thread::hardware_concurrency();
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
  auto finalizeGraph = [&]() {
    auto graphExec = graph.finalize();
    testQueue.submit(
        [&](sycl::handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
  };

  std::vector<std::thread> threads;
  threads.reserve(iterations);

  for (unsigned i = 0; i < iterations; ++i) {
    threads.emplace_back(finalizeGraph);
  }

  for (unsigned i = 0; i < iterations; ++i) {
    threads[i].join();
  }

  // Perform a wait on all graph submissions.
  testQueue.wait_and_throw();

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
