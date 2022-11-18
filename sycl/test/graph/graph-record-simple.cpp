// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <iostream>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

int main() {
  const size_t n = 10;
  const float expectedValue = 1.f;

  sycl::property_list properties{
      sycl::property::queue::in_order(),
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::default_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  float *arr = sycl::malloc_shared<float>(n, q);

  q.begin_recording(g);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      arr[i] = expectedValue;
    });
  });

  q.end_recording();

  auto exec_graph = g.finalize(q.get_context());

  q.submit(exec_graph);

  int errors = 0;
  // Verify results
  for (size_t i = 0; i < n; i++) {
    if (arr[i] != expectedValue) {
      std::cout << "Unexpected result at index: " << i
                << ", expected: " << expectedValue << " actual: " << arr[i]
                << "\n";
      errors++;
    }
  }

  std::cout << "done.\n";

  sycl::free(arr, q.get_context());

  return errors;
}