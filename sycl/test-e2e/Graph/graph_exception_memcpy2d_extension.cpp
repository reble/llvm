// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test checks that invalid exception is thrown
// when trying to use sycl_ext_oneapi_enqueue_barrier
// alomg with Graph.

#include "graph_common.hpp"

enum OperationPath { Explicit, RecordReplay, Shortcut };

template <OperationPath PathKind>
void doMemcpy2D(
    sycl::ext::oneapi::experimental::detail::modifiable_command_graph &G,
    queue &Q, void *Dest, size_t DestPitch, const void *Src, size_t SrcPitch,
    size_t Width, size_t Height) {
  if constexpr (PathKind == OperationPath::RecordReplay) {
    Q.submit([&](handler &CGH) {
      CGH.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
    });
  }
  if constexpr (PathKind == OperationPath::Shortcut) {
    Q.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
  }
  if constexpr (PathKind == OperationPath::Explicit) {
    G.add([&](handler &CGH) {
      CGH.ext_oneapi_memcpy2d(Dest, DestPitch, Src, SrcPitch, Width, Height);
    });
  }
}

int main() {
  constexpr size_t RECT_WIDTH = 30;
  constexpr size_t RECT_HEIGHT = 21;
  constexpr size_t SRC_ELEMS = RECT_WIDTH * RECT_HEIGHT;
  constexpr size_t DST_ELEMS = SRC_ELEMS;

  using T = int;

  sycl::context Context;
  sycl::queue Q(Context, sycl::default_selector_v);

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};
  Graph.begin_recording(Q);

  T *USMMemSrc = malloc_device<T>(SRC_ELEMS, Q);
  T *USMMemDst = malloc_device<T>(DST_ELEMS, Q);

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    doMemcpy2D<OperationPath::RecordReplay>(
        Graph, Q, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    doMemcpy2D<OperationPath::Shortcut>(
        Graph, Q, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  Graph.end_recording();

  sycl::queue Q2(Context, sycl::default_selector_v);
  exp_ext::command_graph Graph2{Q2.get_context(), Q2.get_device()};

  ExceptionCode = make_error_code(sycl::errc::success);
  try {
    doMemcpy2D<OperationPath::Explicit>(
        Graph2, Q2, USMMemDst, RECT_WIDTH * sizeof(T), USMMemSrc,
        RECT_WIDTH * sizeof(T), RECT_WIDTH * sizeof(T), RECT_HEIGHT);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  sycl::free(USMMemSrc, Q);
  sycl::free(USMMemDst, Q);

  return 0;
}
