// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <iostream>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

const size_t n = 10;
const float expectedValue = 42.0f;

void run_some_kernel(sycl::queue q, float *data) {
  // data is captured by ref here but will have gone out of scope when the
  // CGF is later run when the graph is executed.
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
      size_t i = idx;
      data[i] = expectedValue;
    });
  });
}

int main() {

  sycl::property_list properties{
      sycl::property::queue::in_order(),
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::default_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  float *arr = sycl::malloc_shared<float>(n, q);

  g.begin_recording(q);
  run_some_kernel(q, arr);
  g.end_recording(q);

  auto exec_graph = g.finalize(q.get_context());

  q.submit([&](sycl::handler &h) { h.ext_oneapi_graph(exec_graph); });

  int errors = 0;
  // Verify results
  for (size_t i = 0; i < n; i++) {
    if (arr[i] != expectedValue) {
      std::cout << "Test failed: Unexpected result at index: " << i
                << ", expected: " << expectedValue << " actual: " << arr[i]
                << "\n";
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "Test passed successfuly.\n";
  }

  std::cout << "done.\n";

  sycl::free(arr, q.get_context());

  return errors;
}
