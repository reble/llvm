// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  float *Input = malloc_shared<float>(N, Queue);
  float *Output = malloc_shared<float>(1, Queue);
  for (size_t i = 0; i < N; i++) {
    Input[i] = i;
  }

  auto Event = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, reduction(Output, 0.0f, std::plus()),
                     [=](id<1> idx, auto &Sum) { Sum += Input[idx]; });
  });

  auto ExecGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecGraph).wait();

  assert(*Output == 45);

  sycl::free(Input, Queue);
  sycl::free(Output, Queue);

  return 0;
}
