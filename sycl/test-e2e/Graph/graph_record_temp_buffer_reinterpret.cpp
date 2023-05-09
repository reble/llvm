// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test creates a temporary buffer (which is reinterpreted from the main
// application buffers) which is used in kernels but destroyed before
// finalization and execution of the graph. The original buffers lifetime
// extends until after execution of the graph.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(size), DataB(size), DataC(size);

  // Initialize the data
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(iterations, size, ReferenceA, ReferenceB,
                           ReferenceC);

  {
    exp_ext::command_graph Graph{TestQueue.get_context(),
                                 TestQueue.get_device()};
    buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
    buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
    buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};

    Graph.begin_recording(TestQueue);

    // Create some temporary buffers only for recording
    {
      auto BufferA2 = BufferA.reinterpret<T, 1>(BufferA.get_range());
      auto BufferB2 = BufferB.reinterpret<T, 1>(BufferB.get_range());
      auto BufferC2 = BufferC.reinterpret<T, 1>(BufferC.get_range());

      // Record commands to graph
      run_kernels(TestQueue, size, BufferA2, BufferB2, BufferC2);

      Graph.end_recording();
    }
    auto GraphExec = Graph.finalize();

    // Execute several iterations of the graph
    for (size_t n = 0; n < iterations; n++) {
      TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    }
    // Perform a wait on all graph submissions.
    TestQueue.wait();
  }

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
