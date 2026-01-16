// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests partitioned wait feature in SYCL Graph.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  property_list Properties{property::queue::in_order{}};
  queue Queue{Properties};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 100;
  int *A = malloc_device<int>(N, Queue);
  int *B = malloc_device<int>(N, Queue);
  int *C = malloc_device<int>(N, Queue);
  int *D = malloc_device<int>(N, Queue);

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      A[it] = static_cast<int>(it);
      B[it] = 0;
      C[it] = 0;
      D[it] = 0;
    });
  }).wait();

  // Begin recording the graph
  Graph.begin_recording(Queue);

  // Part 1: "Before" subgraph operations
  auto Event1 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      B[it] = A[it] * 2;
    });
  });

  auto Event2 = Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event1);
    CGH.parallel_for(N, [=](id<1> it) {
      C[it] = B[it] + 1;
    });
  });

  // should create a dummy barrier node in the graph
  Queue.wait();

  // Part 2: "After" subgraph operations 
  auto Event3 = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      D[it] = C[it] * 3;
    });
  });

  Queue.wait();

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      D[it] = D[it] + A[it]; 
    });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  // Verify results
  std::vector<int> OutputA(N), OutputB(N), OutputC(N), OutputD(N);
  Queue.memcpy(OutputA.data(), A, N * sizeof(int)).wait();
  Queue.memcpy(OutputB.data(), B, N * sizeof(int)).wait();
  Queue.memcpy(OutputC.data(), C, N * sizeof(int)).wait();
  Queue.memcpy(OutputD.data(), D, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int expected_a = static_cast<int>(i);
    int expected_b = expected_a * 2;
    int expected_c = expected_b + 1;
    int expected_d = expected_c * 3 + expected_a;

    assert(check_value(i, expected_a, OutputA[i], "A"));
    assert(check_value(i, expected_b, OutputB[i], "B"));
    assert(check_value(i, expected_c, OutputC[i], "C"));
    assert(check_value(i, expected_d, OutputD[i], "D"));
  }

  // Reset data and verify with new input
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(N, [=](id<1> it) {
      A[it] = static_cast<int>(it) + 10; // Different input
      B[it] = 0;
      C[it] = 0;
      D[it] = 0;
    });
  }).wait();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  Queue.memcpy(OutputA.data(), A, N * sizeof(int)).wait();
  Queue.memcpy(OutputB.data(), B, N * sizeof(int)).wait();
  Queue.memcpy(OutputC.data(), C, N * sizeof(int)).wait();
  Queue.memcpy(OutputD.data(), D, N * sizeof(int)).wait();

  for (size_t i = 0; i < N; i++) {
    int expected_a = static_cast<int>(i) + 10;
    int expected_b = expected_a * 2;
    int expected_c = expected_b + 1;
    int expected_d = expected_c * 3 + expected_a;

    assert(check_value(i, expected_a, OutputA[i], "A (second execution)"));
    assert(check_value(i, expected_b, OutputB[i], "B (second execution)"));
    assert(check_value(i, expected_c, OutputC[i], "C (second execution)"));
    assert(check_value(i, expected_d, OutputD[i], "D (second execution)"));
  }

  sycl::free(A, Queue);
  sycl::free(B, Queue);
  sycl::free(C, Queue);
  sycl::free(D, Queue);

  return 0;
}
