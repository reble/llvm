// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test creates a graph, finalizes it, then submits that as a subgraph of
// another graph and executes that second graph.

#include "graph_common.hpp"

class sub_vec_add_kernel;
class sub_subtract_kernel;
class mod_input_kernel;
class copy_out_kernel;

int main() {
  queue testQueue;

  using T = int;

  // Values used to modify data inside kernels.
  const int mod_value = 7;
  std::vector<T> dataA(size), dataB(size), dataC(size), dataOut(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);
  std::iota(dataOut.begin(), dataOut.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceA(dataA);
  std::vector<T> referenceB(dataB);
  std::vector<T> referenceC(dataC);
  std::vector<T> referenceOut(dataOut);
  for (unsigned n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceA[i] += mod_value;
      referenceB[i] += mod_value;
      referenceC[i] = (referenceA[i] + referenceB[i]);
      referenceC[i] -= mod_value;
      referenceOut[i] = referenceC[i] + mod_value;
    }
  }

  exp_ext::command_graph subGraph{testQueue.get_context(),
                                  testQueue.get_device()};

  T *ptrA = malloc_device<T>(size, testQueue);
  T *ptrB = malloc_device<T>(size, testQueue);
  T *ptrC = malloc_device<T>(size, testQueue);
  T *ptrOut = malloc_device<T>(size, testQueue);

  testQueue.copy(dataA.data(), ptrA, size);
  testQueue.copy(dataB.data(), ptrB, size);
  testQueue.copy(dataC.data(), ptrC, size);
  testQueue.copy(dataOut.data(), ptrOut, size);
  testQueue.wait_and_throw();

  // Add some operations to a graph which will later be submitted as part
  // of another graph.

  // Vector add two values
  auto nodeSubA = subGraph.add([&](handler &cgh) {
    cgh.parallel_for<sub_vec_add_kernel>(
        range<1>(size), [=](item<1> id) { ptrC[id] = ptrA[id] + ptrB[id]; });
  });

  // Modify the output value with some other value
  subGraph.add(
      [&](handler &cgh) {
        cgh.parallel_for<sub_subtract_kernel>(
            range<1>(size), [=](item<1> id) { ptrC[id] -= mod_value; });
      },
      {exp_ext::property::node::depends_on(nodeSubA)});

  auto subGraphExec = subGraph.finalize();

  exp_ext::command_graph mainGraph{testQueue.get_context(),
                                   testQueue.get_device()};

  // Modify the input values.
  auto nodeMainA = mainGraph.add([&](handler &cgh) {
    cgh.parallel_for<mod_input_kernel>(range<1>(size), [=](item<1> id) {
      ptrA[id] += mod_value;
      ptrB[id] += mod_value;
    });
  });

  auto nodeMainB =
      mainGraph.add([&](handler &cgh) { cgh.ext_oneapi_graph(subGraphExec); },
                    {exp_ext::property::node::depends_on(nodeMainA)});

  // Copy to another output buffer.
  mainGraph.add(
      [&](handler &cgh) {
        cgh.parallel_for<copy_out_kernel>(range<1>(size), [=](item<1> id) {
          ptrOut[id] = ptrC[id] + mod_value;
        });
      },
      {exp_ext::property::node::depends_on(nodeMainB)});

  // Finalize a graph with the additional kernel for writing out to
  auto mainGraphExec = mainGraph.finalize();

  // Execute several iterations of the graph
  for (unsigned n = 0; n < iterations; n++) {
    testQueue.submit(
        [&](handler &cgh) { cgh.ext_oneapi_graph(mainGraphExec); });
  }
  // Perform a wait on all graph submissions.
  testQueue.wait_and_throw();

  testQueue.copy(ptrA, dataA.data(), size);
  testQueue.copy(ptrB, dataB.data(), size);
  testQueue.copy(ptrC, dataC.data(), size);
  testQueue.copy(ptrOut, dataOut.data(), size);
  testQueue.wait_and_throw();

  free(ptrA, testQueue);
  free(ptrB, testQueue);
  free(ptrC, testQueue);
  free(ptrOut, testQueue);

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);
  assert(referenceOut == dataOut);
  return 0;
}
