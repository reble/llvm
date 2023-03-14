// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <iostream>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::property_list properties{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  const size_t n = 10;
  float *arr = sycl::malloc_shared<float>(n, q);
  for (int i = 0; i < n; i++) {
    arr[i] = 0;
  }

  g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      arr[i] += 1;
    });
  });

  bool check = true;
  for (int i = 0; i < n; i++) {
    if (arr[i] != 0)
      check = false;
  }

  auto executable_graph = g.finalize(q.get_context());

  for (int i = 0; i < n; i++) {
    if (arr[i] != 0)
      check = false;
  }

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(executable_graph); });

  for (int i = 0; i < n; i++) {
    if (arr[i] != 1)
      check = false;
  }

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(executable_graph); });

  for (int i = 0; i < n; i++)
    assert(arr[i] == 2);

  sycl::free(arr, q);

  return 0;
}
