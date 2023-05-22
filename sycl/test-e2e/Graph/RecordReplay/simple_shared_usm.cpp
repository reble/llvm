// REQUIRES: level_zero, gpu
<<<<<<< HEAD
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
=======
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
>>>>>>> sycl-graph-develop

// Tests using shared USM memory with graphs.

#include "../graph_common.hpp"

int main() {
  queue Queue{default_selector_v};

  if (!Queue.get_device().has(sycl::aspect::usm_shared_allocations)) {
    return 0;
  }

  const size_t N = 10;
  const float ExpectedValue = 7.f;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  float *Arr = malloc_shared<float>(N, Queue);

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = ExpectedValue;
    });
  });

  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  for (size_t i = 0; i < N; i++) {
    assert(Arr[i] == ExpectedValue);
  }

  sycl::free(Arr, Queue);

  return 0;
}
