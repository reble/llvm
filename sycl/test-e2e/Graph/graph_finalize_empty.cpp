// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/**  Tests the ability to finalize a command graph without recording any nodes
 * to it.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  ext::oneapi::experimental::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
      graph{testQueue.get_context(), testQueue.get_device()};
  auto graphExec = graph.finalize();

  testQueue.submit([&](handler &cgh) { cgh.ext_oneapi_graph(graphExec); });
  testQueue.wait();

  return 0;
}
