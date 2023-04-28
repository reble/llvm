// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests attempting to begin recording to a new graph when recording is
// already in progress on another.

#include "graph_common.hpp"

int main() {
  queue testQueue;

  bool success = false;

  exp_ext::command_graph graphA{testQueue.get_context(),
                                testQueue.get_device()};
  graphA.begin_recording(testQueue);

  try {
    exp_ext::command_graph graphB{testQueue.get_context(),
                                  testQueue.get_device()};
    graphB.begin_recording(testQueue);
  } catch (sycl::exception &e) {
    auto stdErrc = e.code().value();
    if (stdErrc == static_cast<int>(errc::invalid)) {
      success = true;
    }
  }

  graphA.end_recording();

  assert(success);
  return 0;
}
