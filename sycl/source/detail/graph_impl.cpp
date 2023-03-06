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
// @param[in] arg The kernel argument to check for.
// @param[in] currentNode The current graph node being checked.
// @param[in,out] deps The unique list of dependencies which have been
// identified for this arg.
// @param[in] dereferencePtr if true arg comes direct from the handler in which
// case it will need to be deferenced to check actual value.
//
// @returns True if a dependency was added in this node of any of its
// successors.
bool check_for_arg(const sycl::detail::ArgDesc &arg,
                   const std::shared_ptr<node_impl> &currentNode,
                   std::set<std::shared_ptr<node_impl>> &deps) {
  bool successorAddedDep = false;
  for (auto &successor : currentNode->MSuccessors) {
    successorAddedDep |= check_for_arg(arg, successor, deps);
  }

  if (deps.find(currentNode) == deps.end() && currentNode->has_arg(arg) &&
      !successorAddedDep) {
    deps.insert(currentNode);
    return true;
  }
  return SuccessorAddedDep;
}

std::shared_ptr<node_impl>
graph_impl::add(const std::shared_ptr<graph_impl> &impl,
                std::function<void(handler &)> cgf,
                const std::vector<sycl::detail::ArgDesc> &args,
                const std::vector<std::shared_ptr<node_impl>> &dep) {
  sycl::queue TempQueue{};
  auto QueueImpl = sycl::detail::getSyclObjImpl(TempQueue);
  QueueImpl->setCommandGraph(impl);
  sycl::handler Handler{QueueImpl, false};
  cgf(Handler);

  return this->add(impl, Handler.MKernel, Handler.MNDRDesc,
                   Handler.MOSModuleHandle, Handler.MKernelName,
                   Handler.MAccStorage, Handler.MLocalAccStorage,
                   Handler.MRequirements, Handler.MArgs, {});
}

std::shared_ptr<node_impl> graph_impl::add(
    const std::shared_ptr<graph_impl> &impl,
    std::shared_ptr<sycl::detail::kernel_impl> Kernel,
    sycl::detail::NDRDescT NDRDesc, sycl::detail::OSModuleHandle OSModuleHandle,
    std::string KernelName,
    const std::vector<sycl::detail::AccessorImplPtr> &AccStorage,
    const std::vector<sycl::detail::LocalAccessorImplPtr> &LocalAccStorage,
    const std::vector<sycl::detail::AccessorImplHost *> &Requirements,
    const std::vector<sycl::detail::ArgDesc> &args,
    const std::vector<std::shared_ptr<node_impl>> &dep) {
  const std::shared_ptr<node_impl> & nodeImpl = std::make_shared<node_impl>(
      impl, Kernel, NDRDesc, OSModuleHandle, KernelName, AccStorage,
      LocalAccStorage, Requirements, args);
  // Copy deps so we can modify them
  auto deps = dep;
  // A unique set of dependencies obtained by checking kernel arguments
  std::set<std::shared_ptr<node_impl>> uniqueDeps;
  for (auto &arg : args) {
    if (arg.MType != sycl::detail::kernel_param_kind_t::kind_pointer) {
      continue;
    }
    // Look through the graph for nodes which share this argument
    for (auto nodePtr : MRoots) {
      check_for_arg(arg, nodePtr, uniqueDeps);
    }
  }

  // Add any deps determined from arguments into the dependency list
  deps.insert(deps.end(), uniqueDeps.begin(), uniqueDeps.end());
  if (!deps.empty()) {
    for (auto n : deps) {
      n->register_successor(nodeImpl); // register successor
      this->remove_root(nodeImpl);     // remove receiver from root node
                                       // list
    }
  } else {
    this->add_root(nodeImpl);
  }
  return nodeImpl;
}

bool graph_impl::clear_queues() {
  bool anyQueuesCleared = false;
  for (auto &q : MRecordingQueues) {
    q->setCommandGraph(nullptr);
    anyQueuesCleared = true;
  }
  MRecordingQueues.clear();

  return anyQueuesCleared;
}

void node_impl::exec(const std::shared_ptr<sycl::detail::queue_impl> &Queue
                         _CODELOCPARAMDEF(&CodeLoc)) {
  std::vector<sycl::event> Deps;
  for (auto Sender : MPredecessors)
    Deps.push_back(Sender->get_event());

  // Enqueue kernel here instead of submit

  std::vector<pi_event> RawEvents;
  pi_event *OutEvent = nullptr;
  auto NewEvent = std::make_shared<sycl::detail::event_impl>(q);
  NewEvent->setContextImpl(q->getContextImplPtr());
  NewEvent->setStateIncomplete();
  OutEvent = &NewEvent->getHandleRef();
  pi_result res =
      q->getPlugin().call_nocheck<sycl::detail::PiApiKind::piEventCreate>(
          sycl::detail::getSyclObjImpl(q->get_context())->getHandleRef(),
          OutEvent);
  if (res != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::event,
                          "Failed to create event for node submission");
  }

  pi_int32 Result = enqueueImpKernel(
      q, MNDRDesc, MArgs, /* KernelBundleImpPtr */ nullptr, MKernel,
      MKernelName, MOSModuleHandle, RawEvents, OutEvent, nullptr);
  if (Result != pi_result::PI_SUCCESS) {
    throw sycl::exception(errc::kernel, "Error enqueuing graph node kernel");
  }
  sycl::event QueueEvent =
      sycl::detail::createSyclObjFromImpl<sycl::event>(NewEvent);
  q->addEvent(QueueEvent);
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
