// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** Tests basic recording and submission of a graph using USM pointers for
 * inputs and outputs.
 */

#include "graph_common.hpp"

using namespace sycl;

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

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph{testQueue.get_context(), testQueue.get_device()};
    auto ptrA = malloc_device<T>(dataA.size(), testQueue);
    testQueue.memcpy(ptrA, dataA.data(), dataA.size() * sizeof(T)).wait();
    auto ptrB = malloc_device<T>(dataB.size(), testQueue);
    testQueue.memcpy(ptrB, dataB.data(), dataB.size() * sizeof(T)).wait();
    auto ptrC = malloc_device<T>(dataC.size(), testQueue);
    testQueue.memcpy(ptrC, dataC.data(), dataC.size() * sizeof(T)).wait();

    graph.begin_recording(testQueue);

    // Record commands to graph

    run_kernels_usm(testQueue, size, ptrA, ptrB, ptrC);

    graph.end_recording();
    auto graphExec = graph.finalize();

    // Execute several iterations of the graph
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }
    // Perform a wait on all graph submissions.
    testQueue.wait_and_throw();

    testQueue.memcpy(dataA.data(), ptrA, dataA.size() * sizeof(T)).wait();
    testQueue.memcpy(dataB.data(), ptrB, dataB.size() * sizeof(T)).wait();
    testQueue.memcpy(dataC.data(), ptrC, dataC.size() * sizeof(T)).wait();

    free(ptrA, testQueue.get_context());
    free(ptrB, testQueue.get_context());
    free(ptrC, testQueue.get_context());
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  return 0;
}
