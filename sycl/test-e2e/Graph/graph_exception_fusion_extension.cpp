// REQUIRES: fusion
// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out

// The test checks that invalid exception is thrown
// when a fusion is started on a queue that is in recording mode
// or when we try to record a queue in fusion mode.

#include "graph_common.hpp"

int main() {
  queue Q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  ext::codeplay::experimental::fusion_wrapper fw{Q};

  // Test: Start fusion on a queue that is in recording mode
  Graph.begin_recording(Q);

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    fw.start_fusion();
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  Graph.end_recording(Q);

  // Test: begin recording a queue in fusion mode

  fw.start_fusion();

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Graph.begin_recording(Q);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  return 0;
}
