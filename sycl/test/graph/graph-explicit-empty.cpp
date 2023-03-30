// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::property_list properties{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  const size_t n = 10;
  float *arr = sycl::malloc_device<float>(n, q);
  
  auto start = g.add();

  auto init = g.add([&](sycl::handler &h) {
      h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
        size_t i = idx;
        arr[i] = 0;
      });
  }, {start});
    
  auto empty = g.add({init});

  g.add([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      arr[i] = 1;
    });
  }, {empty});

  auto executable_graph = g.finalize(q.get_context());

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(executable_graph); }).wait();

  sycl::free(arr, q);

  return 0;
}
