// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  // Test passing empty property list, which is the default
  property_list EmptyProperties;
  exp_ext::command_graph Graph(Queue.get_context(), Queue.get_device(),
                               EmptyProperties);

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = 1;
    });
  });

  auto ExecGraph = Graph.finalize(EmptyProperties);

  auto Event1 = Queue.ext_oneapi_graph(ExecGraph);
  auto Event2 = Queue.ext_oneapi_graph(ExecGraph, Event1);
  auto Event3 = Queue.ext_oneapi_graph(ExecGraph, Event1);
  Queue.ext_oneapi_graph(ExecGraph, {Event2, Event3}).wait();

  std::vector<float> Output(N);
  Queue.memcpy(Output.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == 1);

  sycl::free(Arr, Queue);

  return 0;
}
