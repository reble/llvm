// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Arr = malloc_shared<float>(N, Queue);

  float Pattern = 3.14f;
  auto NodeA = Graph.add([&](handler &CGH) { CGH.fill(Arr, Pattern, N); });

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  for (int i = 0; i < N; i++)
    assert(Arr[i] == Pattern);

  sycl::free(Arr, Queue);

  return 0;
}
