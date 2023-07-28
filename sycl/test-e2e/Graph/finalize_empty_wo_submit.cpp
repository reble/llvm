// REQUIRES: cuda, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ability to finalize a empty command graph
// without submitting the graph.

#include "graph_common.hpp"

int GetCudaBackend(const sycl::device &Dev) {
  // Return 1 if the device backend is "cuda" or 0 else.
  // 0 does not prevent another device to be picked as a second choice
  return Dev.get_backend() ==
         backend::ext_oneapi_cuda; // backend::ext_oneapi_level_zero;
}

int main() {
  sycl::device CudaDev{GetCudaBackend};
  queue Queue{CudaDev};

  exp_ext::command_graph Graph{Queue.get_context(), CudaDev};
  auto GraphExec = Graph.finalize();

  return 0;
}
