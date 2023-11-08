// Tests that extending buffer lifetimes with handler::get_access()
// works correctly.

#include "../graph_common.hpp"

int main() {

  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size), Result(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};
  {
    // Create a buffer in temporary scope to test lifetime extension
    buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
    BufferA.set_write_back(false);
    buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
    BufferB.set_write_back(false);
    buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
    BufferC.set_write_back(false);

    auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access(CGH);
      auto AccB = BufferB.get_access(CGH);
      auto AccC = BufferC.get_access(CGH);
      CGH.parallel_for(range<1>{Size},
                       [=](id<1> idx) { AccC[idx] += AccA[idx] + AccB[idx]; });
    });

    add_node(Graph, Queue, [&](handler &CGH) {
      auto AccC = BufferC.get_access(CGH);
      CGH.copy(AccC, Result.data());
    });
  }

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  Queue.wait();

  for (size_t i = 0; i < Size; i++) {
    T Expected = DataA[i] + DataB[i] + DataC[i];
    assert(check_value(i, Expected, Result[i], "Result"));
  }

  return 0;
}
