//==--------- graph_impl.hpp --- SYCL graph extension ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/handler.hpp>

#include <detail/kernel_impl.hpp>

#include <cstring>
#include <functional>
#include <list>
#include <set>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

struct node_impl {
  std::shared_ptr<graph_impl> MGraph;
  /// ID representing this node in the graph
  /// TODO this should be attached to an executable graph, rather than
  /// a modifiable graph
  pi_ext_sync_point MPiSyncPoint;

  // List of successors to this node.
  std::vector<std::shared_ptr<node_impl>> MSuccessors;
  // List of predecessors to this node. Using weak_ptr here to prevent circular
  // references between nodes.
  std::vector<std::weak_ptr<node_impl>> MPredecessors;

  /// Kernel to be executed by this node
  std::shared_ptr<sycl::detail::kernel_impl> MKernel;
  /// Description of the kernel global and local sizes as well as offset
  sycl::detail::NDRDescT MNDRDesc;
  /// Module handle for the kernel to be executed.
  sycl::detail::OSModuleHandle MOSModuleHandle =
      sycl::detail::OSUtil::ExeModuleHandle;
  /// Kernel name inside the module
  std::string MKernelName;
  std::vector<sycl::detail::AccessorImplPtr> MAccStorage;
  std::vector<sycl::detail::LocalAccessorImplPtr> MLocalAccStorage;
  std::vector<sycl::detail::AccessorImplHost *> MRequirements;

  /// Store arg descriptors for the kernel arguments
  std::vector<sycl::detail::ArgDesc> MArgs;
  // We need to store local copies of the values pointed to by MArgs since they
  // may go out of scope before execution.
  std::vector<std::vector<std::byte>> MArgStorage;

  void register_successor(const std::shared_ptr<node_impl> &Node,
                          const std::shared_ptr<node_impl> &Prev) {
    MSuccessors.push_back(Node);
    Node->register_predecessor(Prev);
  }

  void register_predecessor(const std::shared_ptr<node_impl> &Node) {
    MPredecessors.push_back(Node);
  }

  node_impl(const std::shared_ptr<graph_impl> &Graph)
      : MGraph(Graph) {}

  node_impl(
      const std::shared_ptr<graph_impl> &Graph,
      std::shared_ptr<sycl::detail::kernel_impl> Kernel,
      sycl::detail::NDRDescT NDRDesc,
      sycl::detail::OSModuleHandle OSModuleHandle, std::string KernelName,
      const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
      const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
      const std::vector<sycl::detail::AccessorImplHost *> &Requirements,
      const std::vector<sycl::detail::ArgDesc> &args)
      : MGraph(Graph), MKernel(Kernel), MNDRDesc(NDRDesc),
        MOSModuleHandle(OSModuleHandle), MKernelName(KernelName),
        MAccStorage(AccStorage), MLocalAccStorage(LocalAccStorage),
        MRequirements(Requirements), MArgs(args), MArgStorage() {

    // Need to copy the arg values to node local storage so that they don't go
    // out of scope before execution
    for (size_t i = 0; i < MArgs.size(); i++) {
      auto &CurrentArg = MArgs[i];
      MArgStorage.emplace_back(CurrentArg.MSize);
      auto StoragePtr = MArgStorage.back().data();
      if (CurrentArg.MPtr)
        std::memcpy(StoragePtr, CurrentArg.MPtr, CurrentArg.MSize);
      // Set the arg descriptor to point to the new storage
      CurrentArg.MPtr = StoragePtr;
    }
  }

  // Recursively adding nodes to execution stack:
  void topology_sort(std::shared_ptr<node_impl> NodeImpl,
                     std::list<std::shared_ptr<node_impl>> &Schedule) {
    for (auto Next : MSuccessors) {
      // Check if we've already scheduled this node
      if (std::find(Schedule.begin(), Schedule.end(), Next) == Schedule.end())
        Next->topology_sort(Next, Schedule);
    }
    if (MKernel != nullptr)
    Schedule.push_front(NodeImpl);
  }

  bool has_arg(const sycl::detail::ArgDesc &Arg) {
    for (auto &NodeArg : MArgs) {
      if (Arg.MType == NodeArg.MType && Arg.MSize == NodeArg.MSize) {
        // Args are actually void** so we need to dereference them to compare
        // actual values
        void *IncomingPtr = *static_cast<void **>(Arg.MPtr);
        void *ArgPtr = *static_cast<void **>(NodeArg.MPtr);
        if (IncomingPtr == ArgPtr) {
          return true;
        }
      }
    }
    return false;
  }
};

struct graph_impl {
  std::set<std::shared_ptr<node_impl>> MRoots;

  std::shared_ptr<graph_impl> MParent;

  void add_root(const std::shared_ptr<node_impl> &);
  void remove_root(const std::shared_ptr<node_impl> &);

  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      std::shared_ptr<sycl::detail::kernel_impl> Kernel,
      sycl::detail::NDRDescT NDRDesc,
      sycl::detail::OSModuleHandle OSModuleHandle, std::string KernelName,
      const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
      const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
      const std::vector<sycl::detail::AccessorImplHost *> &Requirements,
      const std::vector<sycl::detail::ArgDesc> &Args,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {},
      const std::vector<std::shared_ptr<sycl::detail::event_impl>> &DepEvents =
          {});

  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      std::function<void(handler &)> CGF,
      const std::vector<sycl::detail::ArgDesc> &Args,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  graph_impl() = default;

  /// Add a queue to the set of queues which are currently recording to this
  /// graph.
  void
  add_queue(const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.insert(RecordingQueue);
  }

  /// Remove a queue from the set of queues which are currently recording to
  /// this graph.
  void remove_queue(
      const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.erase(RecordingQueue);
  }

  /// Remove all queues which are recording to this graph, also sets all queues
  /// cleared back to the executing state. \return True if any queues were
  /// removed.
  bool clear_queues();

  void add_event_for_node(std::shared_ptr<sycl::detail::event_impl> EventImpl,
                          std::shared_ptr<node_impl> NodeImpl) {
    MEventsMap[EventImpl] = NodeImpl;
  }

private:
  std::set<std::shared_ptr<sycl::detail::queue_impl>> MRecordingQueues;
  // Map of events to their associated recorded nodes.
  std::unordered_map<std::shared_ptr<sycl::detail::event_impl>,
                     std::shared_ptr<node_impl>>
      MEventsMap;
};

class exec_graph_impl {
public:
  exec_graph_impl(sycl::context Context,
                  const std::shared_ptr<graph_impl> &GraphImpl)
      : MSchedule(), MGraphImpl(GraphImpl), MPiCommandBuffers(),
        MContext(Context) {}
  ~exec_graph_impl();
  /// Add nodes to MSchedule
  void schedule();
  /// Enqueues the backend objects for the graph to the parametrized queue
  sycl::event enqueue(const std::shared_ptr<sycl::detail::queue_impl> &);
  /// Called by handler::ext_oneapi_command_graph() to schedule graph for
  /// execution
  sycl::event exec(const std::shared_ptr<sycl::detail::queue_impl> &);
  /// Turns our internal graph representation into PI command-buffers for a
  /// device
  void create_pi_command_buffers(sycl::device D, const sycl::context &Ctx);

private:
  std::list<std::shared_ptr<node_impl>> MSchedule;
  // Pointer to the modifiable graph impl associated with this executable graph
  std::shared_ptr<graph_impl> MGraphImpl;
  // Map of devices to command buffers
  std::unordered_map<sycl::device, pi_ext_command_buffer> MPiCommandBuffers;
  // Context associated with this executable graph
  sycl::context MContext;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
