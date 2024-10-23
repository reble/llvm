// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Tests that whole graph update works when a node uses local memory.

#include "graph_common.hpp"
// Queue submissions that can be recorded to a graph, using a device function
// that expects a local memory pointer.
template <class T> void RecordGraph(queue &Queue, size_t Size, T *InOut) {
  sycl::range global_size = {Size};
  sycl::range local_size = {128};
  Queue.submit([&](handler &CGH) {
    local_accessor<T, 1> local_mem(local_size, CGH);
    CGH.parallel_for(nd_range(global_size, local_size), [=](nd_item<1> item) {
      local_mem[item.get_local_linear_id()] = item.get_global_linear_id();
      InOut[item.get_global_linear_id()] +=
          local_mem[item.get_local_linear_id()];
    });
  });
}
int main() {
  queue Queue{};
  using T = int;
  // USM allocations for GraphA
  T *InOut = malloc_device<T>(Size, Queue);
  // Initialize USM allocations
  T Pattern1 = 0xA;
  Queue.fill(InOut, Pattern1, Size);
  Queue.wait();
  // Define GraphA
  exp_ext::command_graph GraphA{Queue};
  GraphA.begin_recording(Queue);
  RecordGraph(Queue, Size, InOut);
  GraphA.end_recording();
  // Finalize, run, and validate GraphA
  auto GraphExecA = GraphA.finalize(exp_ext::property::graph::updatable{});
  Queue.ext_oneapi_graph(GraphExecA).wait();
  std::vector<T> HostOutput(Size);
  Queue.copy(InOut, HostOutput.data(), Size).wait();
  for (int i = 0; i < Size; i++) {
    T Ref = Pattern1 + i;
    (check_value(i, Ref, HostOutput[i], "InOut"));
  }
  // Create GraphB which will be used to update GraphA
  exp_ext::command_graph GraphB{Queue};
  // USM allocations for GraphB
  T *InOut2 = malloc_device<T>(Size, Queue);
  // Initialize GraphB allocations
  Pattern1 = -42;
  Queue.fill(InOut2, Pattern1, Size);
  Queue.wait();
  // Create GraphB
  GraphB.begin_recording(Queue);
  RecordGraph(Queue, Size, InOut2);
  GraphB.end_recording();
  // Update executable GraphA with GraphB, run, and validate
  GraphExecA.update(GraphB);
  Queue.ext_oneapi_graph(GraphExecA).wait();
  Queue.copy(InOut2, HostOutput.data(), Size).wait();
  for (int i = 0; i < Size; i++) {
    T Ref = Pattern1 + i;
    (check_value(i, Ref, HostOutput[i], "InOut2"));
  }
  free(InOut, Queue);
  free(InOut2, Queue);
  return 0;
}
