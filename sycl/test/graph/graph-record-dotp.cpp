#include <CL/sycl.hpp>
#include <iostream>
#include <thread>

#include <sycl/ext/oneapi/experimental/graph.hpp>

const size_t n = 10;

float host_gold_result() {
  float alpha = 1.0f;
  float beta = 2.0f;
  float gamma = 3.0f;

  float sum = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    sum += (alpha * 1.0f + beta * 2.0f) * (gamma * 3.0f + beta * 2.0f);
  }

  return sum;
}

int main() {
  float alpha = 1.0f;
  float beta = 2.0f;
  float gamma = 3.0f;

  float *x, *y, *z;

  sycl::property_list properties{
      sycl::property::queue::in_order(),
      sycl::ext::oneapi::property::queue::lazy_execution{}};

  sycl::queue q{sycl::gpu_selector_v, properties};

  sycl::ext::oneapi::experimental::command_graph g;

  float *dotp = sycl::malloc_shared<float>(1, q);

  x = sycl::malloc_shared<float>(n, q);
  y = sycl::malloc_shared<float>(n, q);
  z = sycl::malloc_shared<float>(n, q);

  q.begin_recording(g);

  /* init data on the device */
  q.submit([&](sycl::handler &h) {
    h.parallel_for(n, [=](sycl::id<1> it) {
      const size_t i = it[0];
      x[i] = 1.0f;
      y[i] = 2.0f;
      z[i] = 3.0f;
    });
  });

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
      const size_t i = it[0];
      x[i] = alpha * x[i] + beta * y[i];
    });
  });

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
      const size_t i = it[0];
      z[i] = gamma * z[i] + beta * y[i];
    });
  });

  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> it) {
      const size_t i = it[0];
      // Doing a manual reduction here because reduction objects cause issues
      // with graphs.
      if (i == 0) {
        for (size_t j = 0; j < n; j++) {
          dotp[0] += x[j] * z[j];
        }
      }
    });
  });

  q.end_recording();

  auto exec_graph = g.finalize(q.get_context());

  q.submit(exec_graph);

  if (dotp[0] != host_gold_result()) {
    std::cout << "Error unexpected result!\n";
  }

  sycl::free(dotp, q);
  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(z, q);

  std::cout << "done.\n";

  return 0;
}