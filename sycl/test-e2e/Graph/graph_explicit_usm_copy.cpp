// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as memcopy not implemented yet
// XFAIL: *

// Tests adding a usm memcpy node using the explicit API and submitting
// the graph.

#include "graph_common.hpp"

class kernel_mod_a;
class kernel_mod_b;

int main() {
  queue testQueue;

  using T = int;

  const T modValue = 7;
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  for (size_t i = 0; i < iterations; i++) {
    for (size_t j = 0; j < size; j++) {
      referenceA[j] = referenceB[j];
      referenceA[j] += modValue;
      referenceB[j] = referenceA[j];
      referenceB[j] += modValue;
      referenceC[j] = referenceB[j];
    }
  }

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.wait_and_throw();

  // memcpy from B to A
  auto nodeA = graph.add([&](handler &cgh) { cgh.copy(ptrB, ptrA, size); });

  // Read & write A
  auto nodeB = graph.add(
      [&](handler &cgh) {
        cgh.parallel_for<kernel_mod_a>(range<1>(size), [=](item<1> id) {
          auto linID = id.get_linear_id();
          ptrA[linID] += modValue;
        });
      },
      {exp_ext::property::node::depends_on(nodeA)});

  // memcpy from B to A
  auto nodeC = graph.add([&](handler &cgh) { cgh.copy(ptrA, ptrB, size); },
                         {exp_ext::property::node::depends_on(nodeB)});

  // Read and write B
  auto nodeD = graph.add(
      [&](handler &cgh) {
        cgh.parallel_for<kernel_mod_b>(range<1>(size), [=](item<1> id) {
          auto linID = id.get_linear_id();
          ptrB[linID] += modValue;
        });
      },
      {exp_ext::property::node::depends_on(nodeC)});

  // memcpy from B to C
  graph.add([&](handler &cgh) { cgh.copy(ptrB, ptrC, size); },
            {exp_ext::property::node::depends_on(nodeB)});

  auto graphExec = graph.finalize();

  // Execute graph over n iterations
  for (unsigned n = 0; n < iterations; n++) {
    testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
  }
  // Perform a wait on all graph submissions.
  testQueue.wait_and_throw();

  testQueue.copy(ptrA, dataA.data(), size);
  testQueue.copy(ptrB, dataB.data(), size);
  testQueue.copy(ptrC, dataC.data(), size);

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  return 0;
}
