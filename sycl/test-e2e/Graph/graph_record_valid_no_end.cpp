// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests obtaining a finalized, executable graph from a graph which is
// currently being recorded to (no end_recording called)

#include "graph_common.hpp"

int main() {
  queue testQueue;

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};
  {
    queue myQueue;
    graph.begin_recording(myQueue);
  }

  try {
    auto graphExec = graph.finalize();
    testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
  } catch (sycl::exception &e) {
    assert(false && "Exception thrown on finalize or submission.\n");
  }
  testQueue.wait();
  return 0;
}
