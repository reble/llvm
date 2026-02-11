// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -lze_loader -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out

#include "../graph_common.hpp"

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/properties/all_properties.hpp>
#include <variant>

void ZE_APICALL HostFunction(void *UserData) {
  uint32_t *Data = static_cast<uint32_t *>(UserData);
  for (size_t i = 0; i < 1024; i++) {
    Data[i] = Data[i] * 3;
  }
}

int main() {
  // Initialize Level Zero
  ze_result_t status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (status != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed with error code: " << status << std::endl;
    return 1;
  }

  // Get Level Zero driver handle
  uint32_t DriverCount = 0;
  zeDriverGet(&DriverCount, nullptr);
  if (DriverCount == 0) {
    std::cout << "No Level Zero drivers found" << std::endl;
    return 1;
  }

  std::vector<ze_driver_handle_t> Drivers(DriverCount);
  zeDriverGet(&DriverCount, Drivers.data());
  ze_driver_handle_t ZeDriver = Drivers[0];

  // Load the experimental host function API
  typedef ze_result_t(ZE_APICALL * zeCommandListAppendHostFunction_fn)(
      ze_command_list_handle_t, void *, void *, void *, ze_event_handle_t,
      uint32_t, ze_event_handle_t *);

  zeCommandListAppendHostFunction_fn zeCommandListAppendHostFunction = nullptr;
  status = zeDriverGetExtensionFunctionAddress(
      ZeDriver, "zeCommandListAppendHostFunction",
      reinterpret_cast<void **>(&zeCommandListAppendHostFunction));
  if (status != ZE_RESULT_SUCCESS || !zeCommandListAppendHostFunction) {
    std::cout << "Failed to load zeCommandListAppendHostFunction: " << status
              << std::endl;
    return 1;
  }

  // Create queue with immediate command list property for native recording
  queue Queue{{property::queue::in_order{},
               ext::intel::property::queue::immediate_command_list{}}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  // Allocate shared memory (accessible from both device and host)
  const size_t N = 1024;
  uint32_t *DataShared = malloc_shared<uint32_t>(N, Queue);

  {
    // Create graph for native recording
    exp_ext::command_graph Graph{Context, Device};

    // Begin recording
    Graph.begin_recording(Queue);

    // 1. Record SYCL kernel - initialize data
    Queue.submit([&](handler &CGH) {
      CGH.parallel_for(range<1>{N},
                       [=](id<1> idx) { DataShared[idx] = idx + 10; });
    });

    // 2. Record SYCL kernel - multiply by 2
    Queue.submit([&](handler &CGH) {
      CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
        DataShared[idx] = DataShared[idx] * 2;
      });
    });

    // 3. Record L0 host function directly to the recording command list
    // The host function will run on the host but is part of the command list,
    // so it will execute after the kernels complete
    auto ZeQueueNative = get_native<backend::ext_oneapi_level_zero>(Queue);

    // For immediate command list queues, the native handle is a
    // ze_command_list_handle_t
    assert(std::holds_alternative<ze_command_list_handle_t>(ZeQueueNative));
    ze_command_list_handle_t ZeCommandList =
        std::get<ze_command_list_handle_t>(ZeQueueNative);

    // Append host function to recording command list
    // It depends on the previous kernel through implicit in-order queue
    // ordering
    status = zeCommandListAppendHostFunction(
        ZeCommandList, reinterpret_cast<void *>(HostFunction),
        static_cast<void *>(DataShared), nullptr, nullptr, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);

    // End recording
    Graph.end_recording(Queue);

    // Finalize and execute the graph
    auto ExecutableGraph = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
    Queue.wait();

    // Verify results
    // SYCL kernel: Data = (i + 10) * 2
    // Host function: Data = (i + 10) * 2 * 3
    for (size_t i = 0; i < N; i++) {
      uint32_t Expected = (i + 10) * 2 * 3;
      assert(check_value(i, Expected, DataShared[i], "DataShared"));
    }
  }

  // Cleanup after graph is destroyed
  free(DataShared, Queue);
  return 0;
}
