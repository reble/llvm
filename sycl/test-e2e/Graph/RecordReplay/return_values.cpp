// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests the return values from queue graph functions which change the
// internal queue state.

#include "../graph_common.hpp"

int main() {
  queue Queue;
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  bool ChangedState = Graph.end_recording();
  assert(ChangedState == false);

  ChangedState = Graph.begin_recording(Queue);
  assert(ChangedState == true);

  ChangedState = Graph.begin_recording(Queue);
  assert(ChangedState == false);

  ChangedState = Graph.end_recording();
  assert(ChangedState == true);

  ChangedState = Graph.end_recording();
  assert(ChangedState == false);

  return 0;
}
