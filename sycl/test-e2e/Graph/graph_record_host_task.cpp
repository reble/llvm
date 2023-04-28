// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as host tasks are not implemented yet
// XFAIL: *

// This test uses a host_task within a command_graph recording

#include "graph_common.hpp"

class host_task_add;
class host_task_inc;

int main() {
  queue testQueue;

  using T = int;

  if (!testQueue.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return 0;
  }

  const T modValue = T{7};
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceC(dataC);
  for (unsigned n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceC[i] += (dataA[i] + dataB[i]) + modValue + 1;
    }
  }

  exp_ext::command_graph<exp_ext::graph_state::modifiable> graph{
      testQueue.get_context(), testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_shared<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.wait_and_throw();

  graph.begin_recording(testQueue);

  // Vector add to output
  event node1 = testQueue.submit([&](handler &cgh) {
    cgh.parallel_for<host_task_add>(
        range<1>(size), [=](item<1> id) { ptrC[id] += ptrA[id] + ptrB[id]; });
  });

  // Modify the output values in a host_task
  auto node2 = testQueue.submit([&](handler &cgh) {
    cgh.depends_on(node1);
    cgh.host_task([=]() {
      for (size_t i = 0; i < size; i++) {
        ptrC[i] += modValue;
      }
    });
  });

  // Modify temp buffer and write to output buffer
  testQueue.submit([&](handler &cgh) {
    cgh.depends_on(node2);
    cgh.parallel_for<host_task_inc>(range<1>(size),
                                    [=](item<1> id) { ptrC[id] += 1; });
  });
  graph.end_recording();

  auto graphExec = graph.finalize();

  // Execute several iterations of the graph
  for (unsigned n = 0; n < iterations; n++) {
    testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
  }
  // Perform a wait on all graph submissions.
  testQueue.wait_and_throw();

  testQueue.copy(ptrC, dataC.data(), size);
  testQueue.wait_and_throw();

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);

  assert(referenceC == dataC);

  return 0;
}
