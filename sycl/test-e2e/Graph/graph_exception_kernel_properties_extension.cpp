// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The test checks that invalid exception is thrown
// when trying to use sycl_ext_oneapi_kernel_properties
// along with Graph.

#include "graph_common.hpp"

enum OperationPath { Explicit, RecordReplay, Shortcut };

enum class Variant { Function, Functor, FunctorAndProperty };

template <Variant KernelVariant, bool IsShortcut, size_t... Is>
class ReqdWGSizePositiveA;
template <Variant KernelVariant, bool IsShortcut> class ReqPositiveA;

template <size_t Dims> range<Dims> repeatRange(size_t Val);
template <> range<1> repeatRange<1>(size_t Val) { return range<1>{Val}; }
template <> range<2> repeatRange<2>(size_t Val) { return range<2>{Val, Val}; }
template <> range<3> repeatRange<3>(size_t Val) {
  return range<3>{Val, Val, Val};
}

template <size_t... Is> struct KernelFunctorWithWGSizeProp {
  void operator()(nd_item<sizeof...(Is)>) const {}
  void operator()(item<sizeof...(Is)>) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<Is...>};
  }
};

template <OperationPath PathKind, Variant KernelVariant, size_t... Is,
          typename PropertiesT, typename KernelType>
void test(sycl::ext::oneapi::experimental::detail::modifiable_command_graph &G,
          queue &Q, PropertiesT Props, KernelType KernelFunc) {
  constexpr size_t Dims = sizeof...(Is);

  // Test Parellel_for
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<ReqdWGSizePositiveA<KernelVariant, false, Is...>>(
            nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(Is...)), Props,
            KernelFunc);
      });
    }
    if constexpr (PathKind == OperationPath::Shortcut) {
      Q.parallel_for<ReqdWGSizePositiveA<KernelVariant, true, Is...>>(
          nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(Is...)), Props,
          KernelFunc);
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](handler &CGH) {
        CGH.parallel_for<ReqdWGSizePositiveA<KernelVariant, false, Is...>>(
            nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(Is...)), Props,
            KernelFunc);
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);
}

template <OperationPath PathKind, typename PropertiesT, typename KernelType>
void testSingleTask(
    sycl::ext::oneapi::experimental::detail::modifiable_command_graph &G,
    queue &Q, PropertiesT Props, KernelType KernelFunc) {

  // Test Single_task
  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    if constexpr (PathKind == OperationPath::RecordReplay) {
      Q.submit([&](sycl::handler &CGH) {
        CGH.single_task<ReqPositiveA<Variant::Function, false>>(Props,
                                                                KernelFunc);
      });
    }
    if constexpr (PathKind == OperationPath::Explicit) {
      G.add([&](sycl::handler &CGH) {
        CGH.single_task<ReqPositiveA<Variant::Function, false>>(Props,
                                                                KernelFunc);
      });
    }
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);
}

template <size_t... Is>
void testParallelFor(
    queue &Q,
    sycl::ext::oneapi::experimental::detail::modifiable_command_graph &G) {
  auto Props = ext::oneapi::experimental::properties{
      ext::oneapi::experimental::work_group_size<Is...>};
  auto KernelFunction = [](auto) {};

  KernelFunctorWithWGSizeProp<Is...> KernelFunctor;

  test<OperationPath::RecordReplay, Variant::Function, Is...>(G, Q, Props,
                                                              KernelFunction);
  test<OperationPath::RecordReplay, Variant::FunctorAndProperty, Is...>(
      G, Q, Props, KernelFunctor);

  test<OperationPath::Shortcut, Variant::Function, Is...>(G, Q, Props,
                                                          KernelFunction);
  test<OperationPath::Shortcut, Variant::FunctorAndProperty, Is...>(
      G, Q, Props, KernelFunctor);

  test<OperationPath::Explicit, Variant::Function, Is...>(G, Q, Props,
                                                          KernelFunction);
  test<OperationPath::Explicit, Variant::FunctorAndProperty, Is...>(
      G, Q, Props, KernelFunctor);
}

int main() {
  sycl::context Context;
  sycl::queue Q(Context, sycl::default_selector_v);

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  Graph.begin_recording(Q);

  // Test Parallel for entry point
  testParallelFor<4>(Q, Graph);
  testParallelFor<4, 4>(Q, Graph);
  testParallelFor<8, 4>(Q, Graph);
  testParallelFor<4, 8>(Q, Graph);
  testParallelFor<4, 4, 4>(Q, Graph);
  testParallelFor<4, 4, 8>(Q, Graph);
  testParallelFor<8, 4, 4>(Q, Graph);
  testParallelFor<4, 8, 4>(Q, Graph);

  // Test Single Task entry point
  auto Props = ext::oneapi::experimental::properties{
      ext::oneapi::experimental::work_group_size<4>};
  auto KernelFunction = [](auto) {};
  testSingleTask<OperationPath::Explicit>(Graph, Q, Props, KernelFunction);
  testSingleTask<OperationPath::RecordReplay>(Graph, Q, Props, KernelFunction);

  Graph.end_recording();

  return 0;
}
