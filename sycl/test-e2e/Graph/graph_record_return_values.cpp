// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests the return values from queue graph functions which change the
// internal queue state

#include "graph_common.hpp"

int main() {
  queue testQueue;
  exp_ext::command_graph graph{testQueue.get_context(), testQueue.get_device()};

  bool changedState = graph.end_recording();
  assert(changedState == false);

  changedState = graph.begin_recording(testQueue);
  assert(changedState == true);

  // Recording to same graph is not an exception
  changedState = graph.begin_recording(testQueue);
  assert(changedState == false);

  changedState = graph.end_recording();
  assert(changedState == true);

  changedState = graph.end_recording();
  assert(changedState == false);

  return 0;
}
