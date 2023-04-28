// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/**  Tests calling finalize() more than once on the same command_graph.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  ext::oneapi::experimental::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
      graph{testQueue.get_context(), testQueue.get_device()};
  auto graphExec = graph.finalize();
  auto graphExec2 = graph.finalize();

  return 0;
}