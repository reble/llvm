// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests adding nodes to a graph, and submitting the graph using buffers and
// accessors for inputs and outputs.

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
    BufferA.set_write_back(false);
    buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
    BufferB.set_write_back(false);
    buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
    BufferC.set_write_back(false);

    // Add commands to graph
    add_kernels(Graph, size, BufferA, BufferB, BufferC);

    auto GraphExec = Graph.finalize();

    // Execute several iterations of the graph
    for (size_t n = 0; n < iterations; n++) {
      TestQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    }
    // Perform a wait on all graph submissions.
    TestQueue.wait_and_throw();

    host_accessor HostAccA(BufferA);
    host_accessor HostAccB(BufferB);
    host_accessor HostAccC(BufferC);

    for (size_t i = 0; i < size; i++) {
      assert(ReferenceA[i] == HostAccA[i]);
      assert(ReferenceB[i] == HostAccB[i]);
      assert(ReferenceC[i] == HostAccC[i]);
    }
  }

  return 0;
}
