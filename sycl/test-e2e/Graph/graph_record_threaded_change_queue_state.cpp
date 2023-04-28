// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test finalizing and submitting a graph in a threaded situation

#include "graph_common.hpp"

#include <thread>

int main() {
  queue testQueue;

  const unsigned iterations = std::thread::hardware_concurrency();

  auto recordGraph = [&]() {
    exp_ext::command_graph graph{testQueue.get_context(),
                                 testQueue.get_device()};
    try {
      graph.begin_recording(testQueue);
    } catch (sycl::exception &e) {
      // Can throw if graph is already being recorded to
    }
    graph.end_recording();
  };

  std::vector<std::thread> threads;
  threads.reserve(iterations);
  for (unsigned i = 0; i < iterations; ++i) {
    threads.emplace_back(recordGraph);
  }

  for (unsigned i = 0; i < iterations; ++i) {
    threads[i].join();
  }

  return 0;
}
