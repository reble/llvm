// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests queue shortcuts for executing a graph

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(size), DataB(size), DataC(size);

  // Initialize the data
  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(iterations, size, ReferenceA, ReferenceB,
                           ReferenceC);

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
  buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
  buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.wait_and_throw();

  Graph.begin_recording(TestQueue);

  // Record commands to graph
  run_kernels_usm(TestQueue, size, PtrA, PtrB, PtrC);

  Graph.end_recording();
  auto GraphExec = Graph.finalize();

  // Execute several iterations of the graph using the different shortcuts
  event Event = TestQueue.ext_oneapi_graph(GraphExec);

  assert(iterations > 2);
  const size_t LoopIterations = iterations - 2;
  std::vector<event> Events(LoopIterations);
  for (size_t n = 0; n < LoopIterations; n++) {
    Events[n] = TestQueue.ext_oneapi_graph(GraphExec, Event);
  }

  TestQueue.ext_oneapi_graph(GraphExec, Events).wait();

  TestQueue.copy(PtrA, DataA.data(), size);
  TestQueue.copy(PtrB, DataB.data(), size);
  TestQueue.copy(PtrC, DataC.data(), size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);

  assert(ReferenceA == DataA);
  assert(ReferenceB == DataB);
  assert(ReferenceC == DataC);

  return 0;
}
