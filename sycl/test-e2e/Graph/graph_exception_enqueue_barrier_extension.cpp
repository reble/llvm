// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test checks that invalid exception is thrown
// when trying to use sycl_ext_oneapi_enqueue_barrier
// along with Graph.

#include "graph_common.hpp"

enum OperationPath { Explicit, RecordReplay, Shortcut };

template <OperationPath PathKind> void test() {
  sycl::context Context;
  sycl::queue Q1(Context, sycl::default_selector_v);

  exp_ext::command_graph Graph1{Q1.get_context(), Q1.get_device()};

  Graph1.begin_recording(Q1);

  Q1.submit([&](sycl::handler &cgh) {});
  Q1.submit([&](sycl::handler &cgh) {});

  // call queue::ext_oneapi_submit_barrier()
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q1.ext_oneapi_submit_barrier();
    }
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q1.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier(); });
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      Graph1.add([&](handler &CGH) { CGH.ext_oneapi_barrier(); });
    }

  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  Graph1.end_recording();

  sycl::queue Q2(Context, sycl::default_selector_v);
  sycl::queue Q3(Context, sycl::default_selector_v);

  exp_ext::command_graph Graph2{Q2.get_context(), Q2.get_device()};
  exp_ext::command_graph Graph3{Q3.get_context(), Q3.get_device()};

  Graph2.begin_recording(Q2);
  Graph3.begin_recording(Q3);

  auto Event1 = Q2.submit([&](sycl::handler &cgh) {});

  auto Event2 = Q3.submit([&](sycl::handler &cgh) {});

  // call handler::barrier(const std::vector<event> &WaitList)
  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q3.ext_oneapi_submit_barrier({Event1, Event2});
    }
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q3.submit([&](sycl::handler &CGH) {
        CGH.ext_oneapi_barrier({Event1, Event2});
      });
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      Graph3.add([&](handler &CGH) {
        CGH.ext_oneapi_barrier({Event1, Event2});
      });
    }

  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  Graph2.end_recording();
  Graph3.end_recording();
}

int main() {
  test<OperationPath::Explicit>();
  test<OperationPath::RecordReplay>();
  test<OperationPath::Shortcut>();
  return 0;
}
