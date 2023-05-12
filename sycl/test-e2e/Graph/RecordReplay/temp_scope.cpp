// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../graph_common.hpp"

const size_t N = 10;
const float ExpectedValue = 42.0f;

void run_some_kernel(queue Queue, float *Data) {
  // 'Data' is captured by ref here but will have gone out of scope when the
  // CGF is later run when the graph is executed.
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      size_t i = idx;
      Data[i] = ExpectedValue;
    });
  });
}

int main() {

  queue Queue{default_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  float *Arr = malloc_shared<float>(N, Queue);

  Graph.begin_recording(Queue);
  run_some_kernel(Queue, Arr);
  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  for (size_t i = 0; i < N; i++) {
    assert(Arr[i] == ExpectedValue);
  }

  sycl::free(Arr, Queue);

  return 0;
}
