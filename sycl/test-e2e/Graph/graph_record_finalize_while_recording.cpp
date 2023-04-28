// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//  Tests the ability to finalize a command graph while it is currently being
// recorded to.

#include "graph_common.hpp"

int main() {
  queue testQueue;

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};
  graph.begin_recording(testQueue);

  try {
    graph.finalize();
  } catch (sycl::exception &e) {
    assert(false && "Exception thrown on finalize.\n");
  }

  graph.end_recording();
  return 0;
}
