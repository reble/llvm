//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>

#include <sycl/detail/common.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/property_list.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

class handler;
class queue;
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {
struct node_impl;
struct graph_impl;
class exec_graph_impl;

} // namespace detail

enum class graph_state {
  modifiable,
  executable,
};

class __SYCL_EXPORT node {
private:
  node(const std::shared_ptr<detail::node_impl> &Impl) : impl(Impl) {}

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  std::shared_ptr<detail::node_impl> impl;
  std::shared_ptr<detail::graph_impl> MGraph;
};

template <graph_state State = graph_state::modifiable>
class __SYCL_EXPORT command_graph {
public:
  command_graph(const context &syclContext, const device &syclDevice,
                const property_list &propList = {});

  // Adding empty node with [0..n] predecessors:
  node add(const std::vector<node> &dep = {}) { return add_impl(dep); }

  // Adding device node:
  template <typename T> node add(T cgf, const std::vector<node> &dep = {}) {
    return add_impl(cgf, dep);
  }

  // Adding dependency between two nodes.
  void make_edge(node sender, node receiver);

  command_graph<graph_state::executable>
  finalize(const property_list &propList = {}) const;

  /// Change the state of a queue to be recording and associate this graph with
  /// it.
  /// @param recordingQueue The queue to change state on and associate this
  /// graph with.
  /// @return True if the queue had its state changed from executing to
  /// recording.
  bool begin_recording(queue recordingQueue);

  /// Change the state of multiple queues to be recording and associate this
  /// graph with each of them.
  /// @param recordingQueues The queues to change state on and associate this
  /// graph with.
  /// @return True if any queue had its state changed from executing to
  /// recording.
  bool begin_recording(const std::vector<queue> &recordingQueues);

  /// Set all queues currently recording to this graph to the executing state.
  /// @return True if any queue had its state changed from recording to
  /// executing.
  bool end_recording();

  /// Set a queues currently recording to this graph to the executing state.
  /// @param recordingQueue The queue to change state on.
  /// @return True if the queue had its state changed from recording to
  /// executing.
  bool end_recording(queue recordingQueue);

  /// Set multiple queues currently recording to this graph to the executing
  /// state.
  /// @param recordingQueue The queues to change state on.
  /// @return True if any queue had its state changed from recording to
  /// executing.
  bool end_recording(const std::vector<queue> &recordingQueues);

private:
  command_graph(const std::shared_ptr<detail::graph_impl> &Impl) : impl(Impl) {}

  // Template-less implementation of add()
  node add_impl(std::function<void(handler &)> cgf,
                const std::vector<node> &dep);

  node add_impl(const std::vector<node> &dep);

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  std::shared_ptr<detail::graph_impl> impl;
};

template <> class __SYCL_EXPORT command_graph<graph_state::executable> {
public:
  command_graph() = delete;

  command_graph(const std::shared_ptr<detail::graph_impl> &Graph,
                const sycl::context &Ctx);

private:
  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  // Creates a backend representation of the graph in impl
  void finalize_impl();

  int MTag;
  std::shared_ptr<detail::exec_graph_impl> impl;
};
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
