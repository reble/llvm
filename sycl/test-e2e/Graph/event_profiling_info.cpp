// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 2>&1
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// This test checks the profiling of an event returned
// from graph submission with event::get_profiling_info().
// It first tests a graph made exclusively of memory operations,
// then tests a graph made of kernels.
// The second run is to check that there are no leaks reported with the embedded
// ZE_DEBUG=4 testing capability.

#include "./graph_common.hpp"

bool verifyProfiling(event Event) {
  auto Submit =
      Event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto Start =
      Event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto End =
      Event.get_profiling_info<sycl::info::event_profiling::command_end>();

  assert((Submit && Start && End) && "Profiling information failed.");
  assert(Submit < Start);
  assert(Start < End);

  bool Pass = sycl::info::event_command_status::complete ==
              Event.get_info<sycl::info::event::command_execution_status>();

  return Pass;
}

// The test checks that get_profiling_info waits for command asccociated with
// event to complete execution.
int main() {
  device Dev;
  queue Queue{Dev, sycl::property::queue::enable_profiling()};

  const size_t Size = 10000;
  int Data[Size] = {0};
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = I;
  }
  int Values[Size] = {0};

  buffer<int, 1> BufferFrom(Data, range<1>(Size));
  buffer<int, 1> BufferTo(Values, range<1>(Size));

  buffer<int, 1> BufferA(Data, range<1>(Size));
  buffer<int, 1> BufferB(Values, range<1>(Size));
  buffer<int, 1> BufferC(Values, range<1>(Size));

  BufferFrom.set_write_back(false);
  BufferTo.set_write_back(false);
  BufferA.set_write_back(false);
  BufferB.set_write_back(false);
  BufferC.set_write_back(false);
  {
    // buffer copy
    exp_ext::command_graph CopyGraph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{},
         exp_ext::property::graph::assume_data_outlives_buffer{}}};
    CopyGraph.begin_recording(Queue);

    Queue.submit([&](sycl::handler &Cgh) {
      accessor<int, 1, access::mode::read, access::target::device> AccessorFrom(
          BufferFrom, Cgh, range<1>(Size));
      accessor<int, 1, access::mode::write, access::target::device> AccessorTo(
          BufferTo, Cgh, range<1>(Size));
      Cgh.copy(AccessorFrom, AccessorTo);
    });

    CopyGraph.end_recording(Queue);

    // kernel launch
    exp_ext::command_graph KernelGraph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{},
         exp_ext::property::graph::assume_data_outlives_buffer{}}};
    KernelGraph.begin_recording(Queue);

    run_kernels(Queue, Size, BufferA, BufferB, BufferC);

    KernelGraph.end_recording(Queue);

    auto CopyGraphExec = CopyGraph.finalize();
    auto KernelGraphExec = KernelGraph.finalize();

    // Run graphs
    event CopyEvent = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(CopyGraphExec); });
    event KernelEvent = Queue.submit(
        [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });

    Queue.wait_and_throw();

    // Checks profiling times
    assert(verifyProfiling(CopyEvent) && verifyProfiling(KernelEvent));
  }

  host_accessor HostData(BufferTo);
  for (size_t I = 0; I < Size; ++I) {
    assert(HostData[I] == Values[I]);
  }

  return 0;
}
