// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 2>&1
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// This test checks the profiling of an event returned
// from graph submission with event::get_profiling_info().
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "../graph_common.hpp"
#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

bool verifyProfiling(event Event) {
  auto Submit =
      Event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto Start =
      Event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto End =
      Event.get_profiling_info<sycl::info::event_profiling::command_end>();

  assert(Submit <= Start);
  assert(Start <= End);

  bool Pass = sycl::info::event_command_status::complete ==
              Event.get_info<sycl::info::event::command_execution_status>();

  return Pass;
}

// The test checks that get_profiling_info waits for command asccociated with
// event to complete execution.
int main() {
  device Dev;

  const size_t Size = 10000;
  int Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  int Values[Size] = {0};

  {
    buffer<int, 1> BufferFrom(Data, range<1>(Size));
    buffer<int, 1> BufferTo(Values, range<1>(Size));

    // buffer copy
    queue copyQueue{Dev, sycl::property::queue::enable_profiling()};

    exp_ext::command_graph copyGraph{copyQueue.get_context(),
                                     copyQueue.get_device()};
    copyGraph.begin_recording(copyQueue);

    copyQueue.submit([&](sycl::handler &Cgh) {
      accessor<int, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<int, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });

    copyGraph.end_recording(copyQueue);

    // kernel launch
    queue kernelQueue{Dev, sycl::property::queue::enable_profiling()};

    exp_ext::command_graph kernelGraph{kernelQueue.get_context(),
                                       kernelQueue.get_device()};
    kernelGraph.begin_recording(kernelQueue);

    kernelQueue.submit([&](sycl::handler &CGH) {
      CGH.single_task<class EmptyKernel>([=]() {});
    });

    kernelGraph.end_recording(kernelQueue);

    auto copyGraphExec = copyGraph.finalize();
    auto kernelGraphExec = kernelGraph.finalize();

    event copyEvent = copyQueue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(copyGraphExec); });
    event kernelEvent = kernelQueue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(kernelGraphExec); });

    copyQueue.wait_and_throw();
    kernelQueue.wait_and_throw();

    assert(verifyProfiling(copyEvent) && verifyProfiling(kernelEvent));
  }

  for (size_t I = 0; I < Size; ++I) {
    assert(Data[I] == Values[I]);
  }

  return 0;
}
