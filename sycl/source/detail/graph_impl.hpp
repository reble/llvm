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

class wrapper {
  using T = std::function<void(sycl::handler &)>;
  T MFunc;
  std::vector<sycl::event> MDeps;

public:
  wrapper(T Func, const std::vector<sycl::event> &Deps)
      : MFunc(Func), MDeps(Deps){};

  void operator()(sycl::handler &CGH) {
    CGH.depends_on(MDeps);
    std::invoke(MFunc, CGH);
  }
};

struct node_impl {
  bool MScheduled;

  std::shared_ptr<graph_impl> MGraph;
  sycl::event MEvent;

  std::vector<std::shared_ptr<node_impl>> MSuccessors;
  std::vector<std::shared_ptr<node_impl>> MPredecessors;

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
  // We need to store local copies of the values pointed to by MArgssince they
  // may go out of scope before execution.
  std::vector<std::vector<std::byte>> MArgStorage;

  void exec(const std::shared_ptr<sycl::detail::queue_impl> &Queue
                _CODELOCPARAM(&CodeLoc));

  void register_successor(const std::shared_ptr<node_impl> &Node) {
    MSuccessors.push_back(Node);
    Node->register_predecessor(std::shared_ptr<node_impl>(this));
  }

  void register_predecessor(const std::shared_ptr<node_impl> &Node) {
    MPredecessors.push_back(Node);
  }

  sycl::event get_event(void) const { return MEvent; }

  node_impl(
      const std::shared_ptr<graph_impl> &g, std::shared_ptr<sycl::detail::kernel_impl> Kernel,
      sycl::detail::NDRDescT NDRDesc,
      sycl::detail::OSModuleHandle OSModuleHandle, std::string KernelName,
      const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
      const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
      const std::vector<sycl::detail::AccessorImplHost *> &Requirements,
      const std::vector<sycl::detail::ArgDesc> &args)
      : MScheduled(false), MGraph(g), MKernel(Kernel), MNDRDesc(NDRDesc),
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
  void topology_sort(std::list<std::shared_ptr<node_impl>> &Schedule) {
    MScheduled = true;
    for (auto Next : MSuccessors) {
      if (!Next->MScheduled)
        Next->topology_sort(Schedule);
    }
    Schedule.push_front(std::shared_ptr<node_impl>(this));
  }


  bool has_arg(const sycl::detail::ArgDesc &arg) {
    for (auto &nodeArg : MArgs) {
      if (arg.MType == nodeArg.MType && arg.MSize == nodeArg.MSize) {
        // Args are actually void** so we need to dereference them to compare
        // actual values
        void *incomingPtr = *static_cast<void **>(arg.MPtr);
        void *argPtr = *static_cast<void **>(nodeArg.MPtr);
        if (incomingPtr == argPtr) {
          return true;
        }
      }
    }
    return false;
  }
};

struct graph_impl {
  std::set<std::shared_ptr<node_impl>> MRoots;
  std::list<std::shared_ptr<node_impl>> MSchedule;
  // TODO: Change one time initialization to per executable object
  bool MFirst;

  std::shared_ptr<graph_impl> MParent;

  void exec(const std::shared_ptr<sycl::detail::queue_impl> &);
  void exec_and_wait(const std::shared_ptr<sycl::detail::queue_impl> &);

  void add_root(const std::shared_ptr<node_impl> &);
  void remove_root(const std::shared_ptr<node_impl> &);

  std::shared_ptr<node_impl>

  add(const std::shared_ptr<graph_impl> &impl, std::shared_ptr<sycl::detail::kernel_impl> Kernel,
      sycl::detail::NDRDescT NDRDesc,
      sycl::detail::OSModuleHandle OSModuleHandle, std::string KernelName,
      const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
      const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
      const std::vector<sycl::detail::AccessorImplHost *> &Requirements,
      const std::vector<sycl::detail::ArgDesc> &args,
      const std::vector<std::shared_ptr<node_impl>> &dep = {});

  std::shared_ptr<node_impl> add(const std::shared_ptr<graph_impl> &impl, std::function<void(handler &)> cgf,
               const std::vector<sycl::detail::ArgDesc> &args,
               const std::vector<std::shared_ptr<node_impl>> &dep = {});

  graph_impl() : MFirst(true) {}

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

private:
  std::set<std::shared_ptr<sycl::detail::queue_impl>> MRecordingQueues;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
