// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../graph_common.hpp"

int main() {
  queue Queue;

  const size_t N = 10;
  float *Arr = malloc_device<float>(N, Queue);

  auto Init = Queue.submit(
			   [&](handler &CGH) {
			     CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
			       size_t i = idx;
			       Arr[i] = static_cast<float>(i);
			     });
			   });
  
  auto Empty1 = Queue.submit([&](handler &) {});
  auto Empty2 = Queue.submit([&](handler &CGH) {});
  
  Queue.submit(
	       [&](handler &CGH) {
		 CGH.depends_on({Empty1, Empty2});
		 CGH.depends_on({Init});
		 CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
		   size_t i = idx;
		   Arr[i] += 1.0f;
		 });
	       });

  std::vector<float> HostData(N);
  Queue.memcpy(HostData.data(), Arr, N * sizeof(float)).wait();
  for (int i = 0; i < N; i++)
    assert(HostData[i] == static_cast<float>(i) + 1.0f);
  
  free(Arr, Queue);
  
  return 0;
}
