// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {
  namespace exp_ext = sycl::ext::oneapi::experimental;
  sycl::queue TestQueue;

  exp_ext::queue_state State = TestQueue.ext_oneapi_get_state();
  assert(State == exp_ext::queue_state::executing);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};
  Graph.begin_recording(TestQueue);
  State = TestQueue.ext_oneapi_get_state();
  assert(State == exp_ext::queue_state::recording);

  Graph.end_recording();
  State = TestQueue.ext_oneapi_get_state();
  assert(State == exp_ext::queue_state::executing);

  return 0;
}
