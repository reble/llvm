// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
#include <CL/sycl.hpp>
#include <iostream>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {

  sycl::property_list properties{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  const size_t n = 1000;
  float *x;
  //void*& x = vec;
  
  auto a = g.add_malloc((void*&)x,n,sycl::usm::alloc::shared);

  g.add([=](sycl::handler& h){
          h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
              const size_t i = it[0];
              x[i] = 1.0f;
            });
  });

  auto executable_graph = g.finalize(q.get_context());

  q.submit([&](sycl::handler &h) { h.exec_graph(executable_graph); });

  float v = 2.0f;
  //auto vec = static_cast<float*>(x);
  x[0] = v;
  auto result = x[0];

  //sycl::free(x, q);

  std::cout << "done.\n";

  return 0;
}
