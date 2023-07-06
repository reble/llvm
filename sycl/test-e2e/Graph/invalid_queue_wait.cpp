// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that waiting on a Queue in recording mode throws.

#include "graph_common.hpp"

int main() {
  queue Queue;

  ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                 Queue.get_device()};
  Graph.begin_recording(Queue);

  try {
    Queue.wait();
  } catch (const sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid);
  }

  return 0;
}
