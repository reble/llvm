// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected fail as executable graph update not implemented yet
// XFAIL: *

// Tests executable graph update by recording two graphs with different buffers
// and attempting to update one from the other.

#include "../graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(size), DataB(size), DataC(size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(iterations, size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph GraphA{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.wait_and_throw();

  GraphA.begin_recording(TestQueue);
  run_kernels_usm(TestQueue, size, PtrA, PtrB, PtrC);
  GraphA.end_recording();

  auto GraphExec = GraphA.finalize();

  exp_ext::command_graph GraphB{TestQueue.get_context(),
                                TestQueue.get_device()};

  T *PtrA2 = malloc_device<T>(size, TestQueue);
  T *PtrB2 = malloc_device<T>(size, TestQueue);
  T *PtrC2 = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA2.data(), PtrA2, size);
  TestQueue.copy(DataB2.data(), PtrB2, size);
  TestQueue.copy(DataC2.data(), PtrC2, size);
  TestQueue.wait_and_throw();

  GraphB.begin_recording(TestQueue);
  run_kernels_usm(TestQueue, size, PtrA2, PtrB2, PtrC2);
  GraphB.end_recording();

  event Event;
  // Execute several iterations of the graph for 1st set of buffers
  for (size_t n = 0; n < iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  GraphExec.update(GraphB);

  // Execute several iterations of the graph for 2nd set of buffers
  for (size_t n = 0; n < iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  TestQueue.wait_and_throw();

  TestQueue.copy(PtrA, DataA.data(), size);
  TestQueue.copy(PtrB, DataB.data(), size);
  TestQueue.copy(PtrC, DataC.data(), size);

  TestQueue.copy(PtrA2, DataA2.data(), size);
  TestQueue.copy(PtrB2, DataB2.data(), size);
  TestQueue.copy(PtrC2, DataC2.data(), size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  free(PtrA2, TestQueue);
  free(PtrB2, TestQueue);
  free(PtrC2, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  assert(ReferenceA == DataA2);
  assert(ReferenceB == DataB2);
  assert(ReferenceC == DataC2);

  return 0;
}
