// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../graph_common.hpp"

int main() {
  queue Queue{gpu_selector_v};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  float DotpData = 0.f;

  const size_t N = 10;
  std::vector<float> XData(N);

  {
    buffer DotpBuf(&DotpData, range<1>(1));

    buffer XBuf(XData);
    float *Y = malloc_device<float>(N, Queue);
    float *Z = malloc_device<float>(N, Queue);

    auto NodeI = Graph.add([&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      CGH.parallel_for(N, [=](id<1> it) {
        const size_t i = it[0];
        X[i] = 1.0f;
        Y[i] = 2.0f;
        Z[i] = 3.0f;
      });
    });

    // Edge to NodeI from buffer accessor
    auto NodeA = Graph.add([&](handler &CGH) {
      auto X = XBuf.get_access(CGH);
      CGH.parallel_for(range<1>{N}, [=](id<1> it) {
        const size_t i = it[0];
        X[i] = Alpha * X[i] + Beta * Y[i];
      });
    });

    // Edge to NodeI explicitly added
    auto NodeB = Graph.add(
        [&](handler &CGH) {
          CGH.parallel_for(range<1>{N}, [=](id<1> it) {
            const size_t i = it[0];
            Z[i] = Gamma * Z[i] + Beta * Y[i];
          });
        },
        {exp_ext::property::node::depends_on(NodeI)});

    // Edge node_a from buffer accessor, and edge to NodeB explicitly added
    auto NodeC = Graph.add(
        [&](handler &CGH) {
          auto Dotp = DotpBuf.get_access(CGH);
          auto X = XBuf.get_access(CGH);
#ifdef TEST_GRAPH_REDUCTIONS
          CGH.parallel_for(range<1>{N},
                           reduction(DotpBuf, CGH, 0.0f, std::plus()),
                           [=](id<1> it, auto &Sum) {
                             const size_t i = it[0];
                             Sum += X[i] * Z[i];
                           });
#else
          CGH.single_task([=]() {
            // Doing a manual reduction here because reduction objects cause
            // issues with graphs.
            for (size_t j = 0; j < N; j++) {
              Dotp[0] += X[j] * Z[j];
            }
          });
#endif
        },
        {exp_ext::property::node::depends_on(NodeB)});

    auto ExecGraph = Graph.finalize();

    // Using shortcut for executing a graph of commands
    Queue.ext_oneapi_graph(ExecGraph).wait();

    sycl::free(Y, Queue);
    sycl::free(Z, Queue);
  }

  assert(DotpData == dotp_reference_result(N));
  return 0;
}
