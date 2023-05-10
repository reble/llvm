// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test adds a new node with the explicit API to an already finalized
// modifiable graph, before finalizing and executing the graph for a second
// time.

#include "graph_common.hpp"

int main() {
  queue TestQueue;

  using T = int;

  std::vector<T> DataA(size), DataB(size), DataC(size), DataOut(size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);
  std::iota(DataOut.begin(), DataOut.end(), 1000);

  std::vector<T> ReferenceC(DataC);
  std::vector<T> ReferenceOut(DataOut);
  for (size_t n = 0; n < iterations * 2; n++) {
    for (size_t i = 0; i < size; i++) {
      ReferenceC[i] += (DataA[i] + DataB[i]);
      if (n >= iterations)
        ReferenceOut[i] += ReferenceC[i] + 1;
    }
  }

  exp_ext::command_graph Graph{TestQueue.get_context(), TestQueue.get_device()};

  T *PtrA = malloc_device<T>(size, TestQueue);
  T *PtrB = malloc_device<T>(size, TestQueue);
  T *PtrC = malloc_device<T>(size, TestQueue);
  T *PtrOut = malloc_device<T>(size, TestQueue);

  TestQueue.copy(DataA.data(), PtrA, size);
  TestQueue.copy(DataB.data(), PtrB, size);
  TestQueue.copy(DataC.data(), PtrC, size);
  TestQueue.copy(DataOut.data(), PtrOut, size);
  TestQueue.wait_and_throw();

  auto NodeA = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>(size),
                     [=](item<1> id) { PtrC[id] += PtrA[id] + PtrB[id]; });
  });

  auto GraphExec = Graph.finalize();

  Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(size),
                         [=](item<1> id) { PtrOut[id] += PtrC[id] + 1; });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  auto GraphExecAdditional = Graph.finalize();

  event Event;
  for (size_t n = 0; n < iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExec);
    });
  }

  for (size_t n = 0; n < iterations; n++) {
    Event = TestQueue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.ext_oneapi_graph(GraphExecAdditional);
    });
  }

  TestQueue.wait_and_throw();

  TestQueue.copy(PtrC, DataC.data(), size);
  TestQueue.copy(PtrOut, DataOut.data(), size);
  TestQueue.wait_and_throw();

  free(PtrA, TestQueue);
  free(PtrB, TestQueue);
  free(PtrC, TestQueue);
  free(PtrOut, TestQueue);

  assert(ReferenceC == DataC);
  assert(ReferenceOut == DataOut);

  return 0;
}
