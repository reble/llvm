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

bool verifyProfiling(event Event) {
  auto Submit =
      Event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto Start =
      Event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto End =
      Event.get_profiling_info<sycl::info::event_profiling::command_end>();

  std::cout << "Submit =  " << Submit << std::endl;
  std::cout << "Start =  " << Submit << std::endl;
  std::cout << "End =  " << Submit << std::endl;

  assert(Submit <= Start);
  assert(Start <= End);

  bool Pass = sycl::info::event_command_status::complete ==
              Event.get_info<sycl::info::event::command_execution_status>();

  return Pass;
}

// The test checks that get_profiling_info waits for command asccociated with
// event to complete execution.
int main() {
  using T = int;

  const T ModValue = 7;
  const size_t Size = 10000;
  std::vector<T> Data(Size), Value(Size, 0);
  std::vector<T> DataA(Size);

  std::iota(Data.begin(), Data.end(), 1);
  std::iota(DataA.begin(), DataA.end(), 1);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA);
  for (size_t i = 0; i < Size; i++) {
    ReferenceA[i] += ModValue;
  }

  queue Queue{sycl::property::queue::enable_profiling()};
  queue KernelQueue{sycl::property::queue::enable_profiling()};

  buffer BufferFrom{Data};
  BufferFrom.set_write_back(false);
  buffer BufferTo{Value};
  BufferTo.set_write_back(false);
  buffer BufferA{DataA};
  BufferA.set_write_back(false);

  // buffer copy
  exp_ext::command_graph CopyGraph{Queue.get_context(), Queue.get_device()};
  CopyGraph.begin_recording(Queue);

  Queue.submit([&](sycl::handler &CGH) {
    auto AccessorFrom = BufferFrom.get_access(CGH);
    auto AccessorTo = BufferTo.get_access(CGH);
    CGH.copy(AccessorFrom, AccessorTo);
  });

  CopyGraph.end_recording(Queue);

  // kernel launch
  exp_ext::command_graph KernelGraph{Queue.get_context(), Queue.get_device()};
  KernelGraph.begin_recording(Queue);

  Queue.submit([&](sycl::handler &CGH) {
    auto AccA = BufferA.get_access(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> id) {
      auto LinID = id.get_linear_id();
      AccA[LinID] += ModValue;
    });
  });

  KernelGraph.end_recording(Queue);

  auto CopyGraphExec = CopyGraph.finalize();
  auto KernelGraphExec = KernelGraph.finalize();

  event CopyEvent = Queue.submit(
      [&](handler &CGH) { CGH.ext_oneapi_graph(CopyGraphExec); });
  event KernelEvent = Queue.submit(
      [&](handler &CGH) { CGH.ext_oneapi_graph(KernelGraphExec); });

  CopyEvent.wait();
  KernelEvent.wait();

  std::cout << "CopyEvent" << std::endl;
  assert(verifyProfiling(CopyEvent));
  std::cout << "KernelEvent" << std::endl;
  assert(verifyProfiling(KernelEvent));

  host_accessor HostAccA(BufferA);
  host_accessor HostAccTo(BufferTo);
  host_accessor HostAccFrom(BufferFrom);

  for (size_t i = 0; i < Size; i++) {
    assert(HostAccFrom[i] == HostAccTo[i]);
    assert(ReferenceA[i] == HostAccA[i]);
  }

  return 0;
}
