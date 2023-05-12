// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests basic queue recording and submission of a graph using buffers for
// inputs and outputs.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(size), DataB(size), DataC(size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

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

    Graph.begin_recording(TestQueue);
    run_kernels(TestQueue, size, BufferA, BufferB, BufferC);
    Graph.end_recording();

    auto GraphExec = Graph.finalize();

    event Event;
    for (size_t n = 0; n < iterations; n++) {
      Event = TestQueue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }
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
