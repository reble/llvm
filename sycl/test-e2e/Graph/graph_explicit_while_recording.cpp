// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected Fail as exception not implemented yet
// XFAIL: *

// Tests attempting to add a node to a command_graph while it is being
// recorded to by a queue is an error.

#include "graph_common.hpp"

int main() {
  queue testQueue;

  bool success = false;

  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};
  graph.begin_recording(testQueue);

  try {
    graph.add([&](handler &h) {});
  } catch (sycl::exception &e) {
    auto stdErrc = e.code().value();
    if (stdErrc == static_cast<int>(errc::invalid)) {
      success = true;
    }
  }

  graph.end_recording();
  assert(success);
  return 0;
}
