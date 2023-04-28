// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as buffer accessors not yet supported
// XFAIL: *

// Tests basic recording and submission of a graph using buffers and accessors
// for inputs and outputs.

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

  {
    exp_ext::command_graph graph{testQueue.get_context(),
                                 testQueue.get_device()};
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    bufferA.set_write_back(false);
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    bufferB.set_write_back(false);
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};
    bufferC.set_write_back(false);

    graph.begin_recording(testQueue);

    // Record commands to graph

    run_kernels(testQueue, size, bufferA, bufferB, bufferC);

    graph.end_recording();
    auto graphExec = graph.finalize();

    // Execute several iterations of the graph
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }
    // Perform a wait on all graph submissions.
    testQueue.wait_and_throw();

    host_accessor hostAccA(bufferA);
    host_accessor hostAccB(bufferB);
    host_accessor hostAccC(bufferC);

    for (size_t i = 0; i < size; i++) {
      assert(referenceA[i] == hostAccA[i]);
      assert(referenceB[i] == hostAccB[i]);
      assert(referenceC[i] == hostAccC[i]);
    }

  }

  return 0;
}
