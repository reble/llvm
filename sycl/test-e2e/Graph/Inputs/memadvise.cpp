// Tests adding a USM mem advise operation as a graph node.
// Since Mem advise is only a memory hint that doesn't
// impact results but only performances, we verify
// that a node is correctly added by checking PI function calls

#include "../graph_common.hpp"

int NumNodes = 4;

struct Node {
  Node() : pNext(nullptr), Num(0xDEADBEEF) {}

  Node *pNext;
  uint32_t Num;
};

class foo;

int main() {

  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!Queue.get_device().get_info<info::device::usm_shared_allocations>()) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  Node *Head = (Node *)malloc_shared(sizeof(Node), Queue.get_device(),
                                     Queue.get_context());
  if (Head == nullptr) {
    return -1;
  }
  // Test handler::mem_advise
  add_node(Graph, Queue,
           [&](handler &CGH) { CGH.mem_advise(Head, sizeof(Node), 0); });
  Node *Cur = Head;

  for (int i = 0; i < NumNodes; i++) {
    Cur->Num = i * 2;

    if (i != (NumNodes - 1)) {
      Cur->pNext = (Node *)malloc_shared(sizeof(Node), Queue.get_device(),
                                         Queue.get_context());
      if (Cur->pNext == nullptr) {
        return -1;
      }
      // Test handler::mem_advise
      add_node(Graph, Queue, [&](handler &CGH) {
        CGH.mem_advise(Cur->pNext, sizeof(Node), 0);
      });
    } else {
      Cur->pNext = nullptr;
    }

    Cur = Cur->pNext;
  }

  auto E1 = add_node(Graph, Queue, [=](handler &CGH) {
    CGH.single_task<class foo>([=]() {
      Node *pHead = Head;
      while (pHead) {
        pHead->Num = pHead->Num * 2 + 1;
        pHead = pHead->pNext;
      }
    });
  });

  auto ExecGraph = Graph.finalize();
  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait_and_throw();

  Cur = Head;
  for (int i = 0; i < NumNodes; i++) {
    const int Want = i * 4 + 1;
    if (Cur->Num != Want) {
      return -2;
    }
    Node *Old = Cur;
    Cur = Cur->pNext;
    free(Old, Queue.get_context());
  }

  return 0;
}
