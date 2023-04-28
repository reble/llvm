// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as buffer accessors not yet supported
// XFAIL: *

// This test creates a temporary buffer (which is reinterpreted from the main
// application buffers) which is used in kernels but destroyed before
// finalization and execution of the graph. The original buffers lifetime
// extends until after execution of the graph.

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
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    graph.begin_recording(testQueue);

    // Create some temporary buffers only for recording
    {
      auto bufferA2 = bufferA.reinterpret<T, 1>(bufferA.get_range());
      auto bufferB2 = bufferB.reinterpret<T, 1>(bufferB.get_range());
      auto bufferC2 = bufferC.reinterpret<T, 1>(bufferC.get_range());

      // Record commands to graph
      run_kernels(testQueue, size, bufferA2, bufferB2, bufferC2);

      graph.end_recording();
    }
    auto graphExec = graph.finalize();

    // Execute several iterations of the graph
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
    }
    // Perform a wait on all graph submissions.
    testQueue.wait();
  }

  assert(referenceA == dataA);
  assert(referenceB == dataB);
  assert(referenceC == dataC);

  return 0;
}
