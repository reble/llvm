// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_shared<float>(N, Queue);
  for (int i = 0; i < N; i++) {
    Arr[i] = 0;
  }

  Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Arr[i] = 3.14f;
    });
  });

  for (int i = 0; i < N; i++) {
    assert(Arr[i] == 0);
  }

  auto ExecGraph = Graph.finalize();

  for (int i = 0; i < N; i++) {
    assert(Arr[i] == 0);
  }

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  for (int i = 0; i < N; i++)
    assert(Arr[i] == 3.14f);

  sycl::free(Arr, Queue);

  return 0;
}
