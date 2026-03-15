// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue1{sycl::property::queue::in_order{}};
  queue Queue2(Queue1.get_context(), Queue1.get_device(),
               sycl::property::queue::in_order());
  exp_ext::command_graph Graph{Queue1};

  std::vector<float> Data(Size, 1.0f);

  float *DevicePtr = sycl::malloc_device<float>(Size, Queue1);

  Graph.begin_recording(Queue1);

  Queue1.submit([&](handler &CGH) {
    CGH.memcpy(DevicePtr, Data.data(), Size * sizeof(float));
  });

  Queue1.submit([&](handler &CGH) {
    CGH.parallel_for(sycl::range<1>(Size),
                     [=](sycl::id<1> Id) { DevicePtr[Id] += 1.0f; });
  });

  Graph.end_recording(Queue1);

  auto GraphExec = Graph.finalize();

  auto Event = Queue1.ext_oneapi_graph(GraphExec);

  Queue2.submit([&](sycl::handler &CGH) {
#if 1 // Setting to zero hides the fail
    CGH.depends_on({Event});
#endif

#if 1 // Fail only appears with host-task
    CGH.host_task([=]() { volatile float b = 3.0; });
#else
    CGH.parallel_for(sycl::range<1>(Size),
                     [=](sycl::id<1> Id) { DevicePtr[Id] += 0.0f; });
#endif
  });

  std::vector<float> HostData(Size, 7.0f);
  Queue1.memcpy(HostData.data(), DevicePtr, Size * sizeof(float)).wait();
  bool IncorrectResult = false;
  for (size_t i = 0; i < Size; ++i) {
    IncorrectResult |= !(HostData[i] == 2.0f);
  if (IncorrectResult)
  {
    std::cout << "INCORRECT RESULT DETECTED! Value at " << i << " was " << HostData[i] << std::endl;
  }
  }
  
  sycl::free(DevicePtr, Queue1);

  return IncorrectResult;
}
