// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test creates a graph, finalizes it, then to adds a new nodes
// with the explicit API to the graph before finalizing and executing the
// second graph.

#include "graph_common.hpp"

class vector_plus_equals;
class write_to_output;

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size), dataOut(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);
  std::iota(dataOut.begin(), dataOut.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceC(dataC);
  std::vector<T> referenceOut(dataOut);
  for (unsigned n = 0; n < iterations * 2; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceC[i] += (dataA[i] + dataB[i]);
      if (n >= iterations)
        referenceOut[i] += referenceC[i] + 1;
    }
  }

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);
  T *ptrOut = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.copy(dataOut.data(), ptrOut, size);
  testQueue.wait_and_throw();

  // Vector add to some buffer
  auto nodeA = graph.add([&](handler &cgh) {
    cgh.parallel_for<vector_plus_equals>(
        range<1>(size), [=](item<1> id) { ptrC[id] += ptrA[id] + ptrB[id]; });
  });

  auto graphExec = graph.finalize();

  // Read and modify previous output and write to output buffer
  graph.add(
      [&](handler &cgh) {
        cgh.parallel_for<write_to_output>(
            range<1>(size), [=](item<1> id) { ptrOut[id] += ptrC[id] + 1; });
      },
      {exp_ext::property::node::depends_on(nodeA)});

  // Finalize a graph with the additional kernel for writing out to
  auto graphExecAdditional = graph.finalize();

  // Execute several iterations of the graph
  event e;
  for (unsigned n = 0; n < iterations; n++) {
    e = testQueue.submit([&](handler &cgh) {
      cgh.depends_on(e);
      cgh.ext_oneapi_graph(graphExec);
    });
  }
  // Execute the extended graph.
  for (unsigned n = 0; n < iterations; n++) {
    e = testQueue.submit([&](handler &cgh) {
      cgh.depends_on(e);
      cgh.ext_oneapi_graph(graphExecAdditional);
    });
  }
  // Perform a wait on all graph submissions.
  testQueue.wait_and_throw();

  testQueue.copy(ptrC, dataC.data(), size);
  testQueue.copy(ptrOut, dataOut.data(), size);
  testQueue.wait_and_throw();

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);
  free(ptrOut, testQueue);

  assert(referenceC == dataC);
  assert(referenceOut == dataOut);

  return 0;
}
