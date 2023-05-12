// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../graph_common.hpp"

int main() {

  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *X = malloc_shared<float>(N, Queue);

  auto Init = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      X[i] = 2.0f;
    });
  });

  auto Add = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      X[i] += 2.0f;
    });
  });

  auto Mult = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      X[i] *= 3.0f;
    });
  });

  Graph.make_edge(Init, Mult);
  Graph.make_edge(Mult, Add);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  for (int i = 0; i < N; i++)
    assert(X[i] == 8.0f);

  sycl::free(X, Queue);

  return 0;
}
