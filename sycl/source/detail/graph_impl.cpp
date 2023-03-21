//==--------- graph_impl.cpp - SYCL graph extension -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/graph_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <sycl/queue.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

void graph_impl::exec(const std::shared_ptr<sycl::detail::queue_impl> &Queue) {
  if (MSchedule.empty()) {
    for (auto Node : MRoots) {
      Node->topology_sort(MSchedule);
    }
  }
  for (auto Node : MSchedule)
    Node->exec(Queue);
}

void graph_impl::exec_and_wait(
    const std::shared_ptr<sycl::detail::queue_impl> &Queue) {
  bool IsSubGraph = Queue->getIsGraphSubmitting();
  if (!IsSubGraph) {
    Queue->setIsGraphSubmitting(true);
  }
  if (MFirst) {
    exec(Queue);
    MFirst = false;
  }
  if (!IsSubGraph) {
    Queue->setIsGraphSubmitting(false);
    Queue->wait();
  }
}

void graph_impl::add_root(const std::shared_ptr<node_impl> &Root) {
  MRoots.insert(Root);
  for (auto Node : MSchedule)
    Node->MScheduled = false;
  MSchedule.clear();
}

void graph_impl::remove_root(const std::shared_ptr<node_impl> &Root) {
  MRoots.erase(Root);
  for (auto Node : MSchedule)
    Node->MScheduled = false;
  MSchedule.clear();
}

// Recursive check if a graph node or its successors contains a given kernel
// argument.
//
// @param[in] Arg The kernel argument to check for.
// @param[in] CurrentNode The current graph node being checked.
// @param[in,out] Deps The unique list of dependencies which have been
// identified for this arg.
//
// @returns True if a dependency was added in this node of any of its
// successors.
bool check_for_arg(const sycl::detail::ArgDesc &Arg,
                   const std::shared_ptr<node_impl> &CurrentNode,
                   std::set<std::shared_ptr<node_impl>> &Deps) {
  bool SuccessorAddedDep = false;
  for (auto &Successor : CurrentNode->MSuccessors) {
    SuccessorAddedDep |= check_for_arg(Arg, Successor, Deps);
  }

  if (Deps.find(CurrentNode) == Deps.end() && CurrentNode->has_arg(Arg) &&
      !SuccessorAddedDep) {
    Deps.insert(CurrentNode);
    return true;
  }
  return SuccessorAddedDep;
}

std::shared_ptr<node_impl>
graph_impl::add(const std::shared_ptr<graph_impl> &Impl,
                std::function<void(handler &)> CGF,
                const std::vector<sycl::detail::ArgDesc> &Args,
                const std::vector<std::shared_ptr<node_impl>> &Dep) {
  sycl::queue TempQueue{};
  auto QueueImpl = sycl::detail::getSyclObjImpl(TempQueue);
  QueueImpl->setCommandGraph(Impl);
  sycl::handler Handler{QueueImpl, false};
  CGF(Handler);

  return this->add(Impl, Handler.MKernel, Handler.MNDRDesc,
                   Handler.MOSModuleHandle, Handler.MKernelName,
                   Handler.MAccStorage, Handler.MLocalAccStorage,
                   Handler.MRequirements, Handler.MArgs, {});
}

std::shared_ptr<node_impl> graph_impl::add(
    const std::shared_ptr<graph_impl> &Impl,
    std::shared_ptr<sycl::detail::kernel_impl> Kernel,
    sycl::detail::NDRDescT NDRDesc, sycl::detail::OSModuleHandle OSModuleHandle,
    std::string KernelName,
    const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
    const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
    const std::vector<sycl::detail::AccessorImplHost *> &Requirements,
    const std::vector<sycl::detail::ArgDesc> &Args,
    const std::vector<std::shared_ptr<node_impl>> &Dep) {
  const std::shared_ptr<node_impl> &NodeImpl = std::make_shared<node_impl>(
      Impl, Kernel, NDRDesc, OSModuleHandle, KernelName, AccStorage,
      LocalAccStorage, Requirements, Args);
  // Copy deps so we can modify them
  auto Deps = Dep;
  // A unique set of dependencies obtained by checking kernel arguments
  // for accessors
  std::set<std::shared_ptr<node_impl>> UniqueDeps;
  for (auto &Arg : Args) {
    if (Arg.MType != sycl::detail::kernel_param_kind_t::kind_accessor) {
      continue;
    }
    // Look through the graph for nodes which share this argument
    for (auto NodePtr : MRoots) {
      check_for_arg(Arg, NodePtr, UniqueDeps);
    }
  }

  // Add any deps determined from accessor arguments into the dependency list
  Deps.insert(Deps.end(), UniqueDeps.begin(), UniqueDeps.end());
  if (!Deps.empty()) {
    for (auto N : Deps) {
      N->register_successor(NodeImpl); // register successor
      this->remove_root(NodeImpl);     // remove receiver from root node
                                       // list
    }
  } else {
    this->add_root(NodeImpl);
  }
  return NodeImpl;
}

bool graph_impl::clear_queues() {
  bool AnyQueuesCleared = false;
  for (auto &Queue : MRecordingQueues) {
    Queue->setCommandGraph(nullptr);
    AnyQueuesCleared = true;
  }
  MRecordingQueues.clear();

  return AnyQueuesCleared;
}

void node_impl::exec(const std::shared_ptr<sycl::detail::queue_impl> &Queue
                         _CODELOCPARAMDEF(&CodeLoc)) {
  std::vector<sycl::event> Deps;
  for (auto Sender : MPredecessors)
    Deps.push_back(Sender->get_event());

  // Enqueue kernel here instead of submit

  std::vector<pi_event> RawEvents;
  pi_event *OutEvent = nullptr;
  auto NewEvent = std::make_shared<sycl::detail::event_impl>(Queue);
  NewEvent->setContextImpl(Queue->getContextImplPtr());
  NewEvent->setStateIncomplete();
  OutEvent = &NewEvent->getHandleRef();
  pi_result Res =
      Queue->getPlugin().call_nocheck<sycl::detail::PiApiKind::piEventCreate>(
          sycl::detail::getSyclObjImpl(Queue->get_context())->getHandleRef(),
          OutEvent);
  if (Res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::event,
                          "Failed to create event for node submission");
  }

  pi_int32 Result = enqueueImpKernel(
      Queue, MNDRDesc, MArgs, /* KernelBundleImpPtr */ nullptr, MKernel,
      MKernelName, MOSModuleHandle, RawEvents, OutEvent, nullptr);
  if (Result != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::kernel, "Error enqueuing graph node kernel");
  }
  sycl::event QueueEvent =
      sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  Queue->addEvent(QueueEvent);
  MEvent = QueueEvent;
}
} // namespace detail

template <>
command_graph<graph_state::modifiable>::command_graph(
    const sycl::property_list &)
    : impl(std::make_shared<detail::graph_impl>()) {}

template <>
node command_graph<graph_state::modifiable>::add_impl(
    std::function<void(handler &)> CGF, const std::vector<node> &Deps) {
  std::vector<std::shared_ptr<detail::node_impl>> DepImpls;
  for (auto &D : Deps) {
    DepImpls.push_back(sycl::detail::getSyclObjImpl(D));
  }

  std::shared_ptr<detail::node_impl> NodeImpl =
      impl->add(impl, CGF, {}, DepImpls);
  return sycl::detail::createSyclObjFromImpl<node>(NodeImpl);
}

template <>
void command_graph<graph_state::modifiable>::make_edge(node Sender,
                                                       node Receiver) {
  std::shared_ptr<detail::node_impl> SenderImpl =
      sycl::detail::getSyclObjImpl(Sender);
  std::shared_ptr<detail::node_impl> ReceiverImpl =
      sycl::detail::getSyclObjImpl(Receiver);

  SenderImpl->register_successor(ReceiverImpl); // register successor
  impl->remove_root(ReceiverImpl); // remove receiver from root node list
}

template <>
command_graph<graph_state::executable>
command_graph<graph_state::modifiable>::finalize(
    const sycl::context &CTX, const sycl::property_list &) const {
  return command_graph<graph_state::executable>{this->impl, CTX};
}

template <>
bool command_graph<graph_state::modifiable>::begin_recording(
    queue RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl->getCommandGraph() == nullptr) {
    QueueImpl->setCommandGraph(impl);
    impl->add_queue(QueueImpl);
    return true;
  } else if (QueueImpl->getCommandGraph() != impl) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "begin_recording called for a queue which is already "
                          "recording to a different graph.");
  }

  // Queue was already recording to this graph.
  return false;
}

template <>
bool command_graph<graph_state::modifiable>::begin_recording(
    const std::vector<queue> &RecordingQueues) {
  bool QueueStateChanged = false;
  for (auto &Queue : RecordingQueues) {
    QueueStateChanged |= this->begin_recording(Queue);
  }
  return QueueStateChanged;
}

template <> bool command_graph<graph_state::modifiable>::end_recording() {
  return impl->clear_queues();
}

template <>
bool command_graph<graph_state::modifiable>::end_recording(
    queue RecordingQueue) {
  auto QueueImpl = sycl::detail::getSyclObjImpl(RecordingQueue);
  if (QueueImpl->getCommandGraph() == impl) {
    QueueImpl->setCommandGraph(nullptr);
    impl->remove_queue(QueueImpl);
    return true;
  } else if (QueueImpl->getCommandGraph() != nullptr) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "end_recording called for a queue which is recording "
                          "to a different graph.");
  }

  // Queue was not recording to a graph.
  return false;
}

template <>
bool command_graph<graph_state::modifiable>::end_recording(
    const std::vector<queue> &RecordingQueues) {
  bool QueueStateChanged = false;
  for (auto &Queue : RecordingQueues) {
    QueueStateChanged |= this->end_recording(Queue);
  }
  return QueueStateChanged;
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
