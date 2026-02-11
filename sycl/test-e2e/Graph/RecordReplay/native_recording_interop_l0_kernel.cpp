// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -lze_loader -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out %S/../Inputs/Kernels/saxpy.spv
// RUNx: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1
// %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/saxpy.spv 2>&1 |
// FileCheck %s --implicit-check-not=LEAK %}

// Test native recording with intermixed SYCL and Level-Zero kernels.
// Records 1 SYCL kernel and 1 L0 kernel directly to the recording command list
// over an in-order queue (no host_task).

#include "../graph_common.hpp"

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/properties/all_properties.hpp>
#include <variant>

std::vector<uint8_t> loadSpirvFromFile(std::string FileName) {
  std::ifstream SpvStream(FileName, std::ios::binary);
  SpvStream.seekg(0, std::ios::end);
  size_t sz = SpvStream.tellg();
  SpvStream.seekg(0);
  std::vector<uint8_t> Spv(sz);
  SpvStream.read(reinterpret_cast<char *>(Spv.data()), sz);
  return Spv;
}

int main(int, char **argv) {
  // Initialize Level Zero
  ze_result_t status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (status != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed with error code: " << status << std::endl;
    return 1;
  }

  // Create queue with immediate command list property for native recording
  queue Queue{{property::queue::in_order{},
               ext::intel::property::queue::immediate_command_list{}}};

  const sycl::context Context = Queue.get_context();
  const sycl::device Device = Queue.get_device();

  // Get native Level-Zero handles
  auto ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
  auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);

  // Load SPIR-V for L0 kernel
  std::vector<uint8_t> Spirv = loadSpirvFromFile(argv[1]);

  // Create L0 module and kernel BEFORE recording
  ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 Spirv.size(),
                                 Spirv.data(),
                                 nullptr,
                                 nullptr};
  ze_module_handle_t ZeModule;
  status = zeModuleCreate(ZeContext, ZeDevice, &moduleDesc, &ZeModule, nullptr);
  assert(status == ZE_RESULT_SUCCESS);

  ze_kernel_desc_t kernelDesc = {
      ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
      "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E5saxpy"};
  ze_kernel_handle_t ZeKernel;
  status = zeKernelCreate(ZeModule, &kernelDesc, &ZeKernel);
  assert(status == ZE_RESULT_SUCCESS);

  // Allocate device memory
  const size_t N = 1024;
  uint32_t *DataX = malloc_device<uint32_t>(N, Queue);
  uint32_t *DataZ = malloc_device<uint32_t>(N, Queue);

  // Initialize host data
  std::vector<uint32_t> HostX(N);
  std::vector<uint32_t> HostZ(N);
  for (size_t i = 0; i < N; i++) {
    HostX[i] = i + 10;
    HostZ[i] = i + 1;
  }

  // Copy to device
  Queue.memcpy(DataX, HostX.data(), N * sizeof(uint32_t)).wait();
  Queue.memcpy(DataZ, HostZ.data(), N * sizeof(uint32_t)).wait();
  {
    // Create graph for native recording
    exp_ext::command_graph Graph{Context, Device};

    // Begin recording
    Graph.begin_recording(Queue);

    // 1. Record SYCL kernel - multiply X by 2
    Queue.submit([&](handler &CGH) {
      CGH.parallel_for(range<1>{N},
                       [=](id<1> idx) { DataX[idx] = DataX[idx] * 2; });
    });

    // 2. Record L0 kernel directly to the recording command list
    // Get native command list handle from the queue (it's an immediate command
    // list)
    auto ZeQueueNative = get_native<backend::ext_oneapi_level_zero>(Queue);

    // For immediate command list queues, the native handle is a
    // ze_command_list_handle_t
    assert(std::holds_alternative<ze_command_list_handle_t>(ZeQueueNative));
    ze_command_list_handle_t ZeCommandList =
        std::get<ze_command_list_handle_t>(ZeQueueNative);

    // Set kernel arguments: saxpy computes Z = X * 2 + Z
    status = zeKernelSetArgumentValue(ZeKernel, 0, sizeof(DataZ), &DataZ);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeKernelSetArgumentValue(ZeKernel, 1, sizeof(DataX), &DataX);
    assert(status == ZE_RESULT_SUCCESS);

    // Set group size
    uint32_t GroupSizeX = 32;
    uint32_t GroupSizeY = 1;
    uint32_t GroupSizeZ = 1;
    status = zeKernelSuggestGroupSize(ZeKernel, N, 1, 1, &GroupSizeX,
                                      &GroupSizeY, &GroupSizeZ);
    assert(status == ZE_RESULT_SUCCESS);

    status = zeKernelSetGroupSize(ZeKernel, GroupSizeX, GroupSizeY, GroupSizeZ);
    assert(status == ZE_RESULT_SUCCESS);

    // Append kernel launch to recording command list
    ze_group_count_t ZeGroupCount{static_cast<uint32_t>(N) / GroupSizeX, 1, 1};
    status = zeCommandListAppendLaunchKernel(
        ZeCommandList, ZeKernel, &ZeGroupCount, nullptr, 0, nullptr);
    assert(status == ZE_RESULT_SUCCESS);

    // End recording
    Graph.end_recording(Queue);

    // Finalize and execute the graph
    auto ExecutableGraph = Graph.finalize();
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
    Queue.wait();

    // Copy results back to host
    Queue.memcpy(HostX.data(), DataX, N * sizeof(uint32_t)).wait();
    Queue.memcpy(HostZ.data(), DataZ, N * sizeof(uint32_t)).wait();

    // Verify results
    // SYCL kernel: X = (i + 10) * 2
    // L0 saxpy kernel: Z = X * 2 + Z = (i + 10) * 2 * 2 + (i + 1)
    for (size_t i = 0; i < N; i++) {
      uint32_t ExpectedX = (i + 10) * 2;
      uint32_t ExpectedZ = (i + 10) * 2 * 2 + (i + 1);
      assert(check_value(i, ExpectedX, HostX[i], "HostX"));
      assert(check_value(i, ExpectedZ, HostZ[i], "HostZ"));
    }
  }

  // Cleanup after graph is destroyed
  free(DataX, Queue);
  free(DataZ, Queue);

  status = zeKernelDestroy(ZeKernel);
  assert(status == ZE_RESULT_SUCCESS);

  status = zeModuleDestroy(ZeModule);
  assert(status == ZE_RESULT_SUCCESS);

  return 0;
}
