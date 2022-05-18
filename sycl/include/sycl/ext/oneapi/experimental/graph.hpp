//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <list>
#include <set>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

struct node_impl;

struct graph_impl;

using node_ptr = std::shared_ptr<node_impl>;

using graph_ptr = std::shared_ptr<graph_impl>;

class wrapper {
  using T = std::function<void(sycl::handler &)>;
  T my_func;
  std::vector<sycl::event> my_deps;

public:
  wrapper(T t, const std::vector<sycl::event> &deps)
      : my_func(t), my_deps(deps){};

  void operator()(sycl::handler &cgh) {
    cgh.depends_on(my_deps);
    std::invoke(my_func, cgh);
  }
};

struct node_impl {
  bool is_scheduled;
  bool is_empty;

  graph_ptr my_graph;
  sycl::event my_event;

  std::vector<node_ptr> my_successors;
  std::vector<node_ptr> my_predecessors;

  std::function<void(sycl::handler &)> my_body;

  void exec(sycl::queue q) {
    std::vector<sycl::event> __deps;
    std::vector<node_ptr> pred_nodes = my_predecessors;
    while (!pred_nodes.empty()) {
      node_ptr curr_node = pred_nodes.back();
      pred_nodes.pop_back();
      // Add predecessors to dependence list if node is empty
      if (curr_node->is_empty)
        for (auto i : curr_node->my_predecessors)
          pred_nodes.push_back(i);
      else
        __deps.push_back(curr_node->get_event());
    }
    if (my_body)
      my_event = q.submit(wrapper{my_body, __deps});
  }

  void register_successor(node_ptr n) {
    my_successors.push_back(n);
    n->register_predecessor(node_ptr(this));
  }

  void register_predecessor(node_ptr n) { my_predecessors.push_back(n); }

  sycl::event get_event(void) { return my_event; }

  node_impl() : is_scheduled(false), is_empty(true) {}

  node_impl(graph_ptr g) : is_scheduled(false), is_empty(true), my_graph(g) {}

  template <typename T>
  node_impl(graph_ptr g, T cgf)
      : is_scheduled(false), is_empty(false), my_graph(g), my_body(cgf) {}

  // Recursively adding nodes to execution stack:
  void topology_sort(std::list<node_ptr> &schedule) {
    is_scheduled = true;
    for (auto i : my_successors) {
      if (!i->is_scheduled)
        i->topology_sort(schedule);
    }
    schedule.push_front(node_ptr(this));
  }
};

struct graph_impl {
  std::set<node_ptr> my_roots;
  std::list<node_ptr> my_schedule;

  graph_ptr parent;

  void exec(sycl::queue q) {
    if (my_schedule.empty()) {
      for (auto n : my_roots) {
        n->topology_sort(my_schedule);
      }
    }
    for (auto n : my_schedule)
      n->exec(q);
  }

  void exec_and_wait(sycl::queue q) {
    exec(q);
    q.wait();
  }

  void add_root(node_ptr n) {
    my_roots.insert(n);
    for (auto n : my_schedule)
      n->is_scheduled = false;
    my_schedule.clear();
  }

  void remove_root(node_ptr n) {
    my_roots.erase(n);
    for (auto n : my_schedule)
      n->is_scheduled = false;
    my_schedule.clear();
  }

  graph_impl() {}
};

} // namespace detail

class node;

class graph;

class executable_graph;

struct node {
  // TODO: add properties to distinguish between empty, host, device nodes.
  detail::node_ptr my_node;
  detail::graph_ptr my_graph;

  node() : my_node(new detail::node_impl()) {}

  node(detail::graph_ptr g) : my_graph(g), my_node(new detail::node_impl(g)){};

  template <typename T>
  node(detail::graph_ptr g, T cgf)
      : my_graph(g), my_node(new detail::node_impl(g, cgf)){};
  void register_successor(node n) { my_node->register_successor(n.my_node); }
  void exec(sycl::queue q, sycl::event = sycl::event()) { my_node->exec(q); }

  void set_root() { my_graph->add_root(my_node); }

  // TODO: Add query functions: is_root, ...
};

class executable_graph {
public:
  int my_tag;
  sycl::queue my_queue;

  void exec_and_wait(); // { my_queue.wait(); }

  executable_graph(detail::graph_ptr g, sycl::queue q)
      : my_queue(q), my_tag(rand()) {
    g->exec(my_queue);
  }
};

class graph {
public:
  // Adds a node
  template <typename T> node submit(T cgf, const std::vector<node> &dep = {});

  template <typename T>
  void submit(node &Node, T cgf, const std::vector<node> &dep = {});

  // Adds an empty node
  node submit(const std::vector<node> &dep = {});

  // Updates a node
  void update(node &Node, const std::vector<node> &dep = {});
  template <typename T>
  void update(node &Node, T cgf, const std::vector<node> &dep = {});

  // Shortcuts to add graph nodes

  // Adds a fill node
  template <typename T>
  node fill(void *Ptr, const T &Pattern, size_t Count,
            const std::vector<node> &dep = {});
  template <typename T>
  void fill(node &Node, void *Ptr, const T &Pattern, size_t Count,
            const std::vector<node> &dep = {});

  // Adds a memset node
  node memset(void *Ptr, int Value, size_t Count,
              const std::vector<node> &dep = {});
  void memset(node &Node, void *Ptr, int Value, size_t Count,
              const std::vector<node> &dep = {});

  // Adds a memcpy node
  node memcpy(void *Dest, const void *Src, size_t Count,
              const std::vector<node> &dep = {});
  void memcpy(node &Node, void *Dest, const void *Src, size_t Count,
              const std::vector<node> &dep = {});

  // Adds a copy node
  template <typename T>
  node copy(const T *Src, T *Dest, size_t Count,
            const std::vector<node> &dep = {});
  template <typename T>
  void copy(node &Node, const T *Src, T *Dest, size_t Count,
            const std::vector<node> &dep = {});

  // Adds a mem_advise node
  node mem_advise(const void *Ptr, size_t Length, int Advice,
                  const std::vector<node> &dep = {});
  void mem_advise(node &Node, const void *Ptr, size_t Length, int Advice,
                  const std::vector<node> &dep = {});

  // Adds a prefetch node
  node prefetch(const void *Ptr, size_t Count,
                const std::vector<node> &dep = {});
  void prefetch(node &Node, const void *Ptr, size_t Count,
                const std::vector<node> &dep = {});

  // Adds a single_task node
  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  node single_task(const KernelType &(KernelFunc),
                   const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  void single_task(node &Node, const KernelType &(KernelFunc),
                   const std::vector<node> &dep = {});

  // Adds a parallel_for node
  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  node parallel_for(range<1> NumWorkItems, const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  void parallel_for(node &Node, range<1> NumWorkItems,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  node parallel_for(range<2> NumWorkItems, const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  void parallel_for(node &Node, range<2> NumWorkItems,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  node parallel_for(range<3> NumWorkItems, const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType>
  void parallel_for(node &Node, range<3> NumWorkItems,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims>
  node parallel_for(range<Dims> NumWorkItems, const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(node &Node, range<Dims> NumWorkItems,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims>
  node parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(node &Node, range<Dims> NumWorkItems,
                    id<Dims> WorkItemOffset, const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims>
  node parallel_for(nd_range<Dims> ExecutionRange,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims>
  void parallel_for(node &Node, nd_range<Dims> ExecutionRange,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  node parallel_for(range<Dims> NumWorkItems, Reduction Redu,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  void parallel_for(node &Node, range<Dims> NumWorkItems, Reduction Redu,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  node parallel_for(nd_range<Dims> ExecutionRange, Reduction Redu,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});
  template <typename KernelName = sycl::detail::auto_name, typename KernelType,
            int Dims, typename Reduction>
  void parallel_for(node &Node, nd_range<Dims> ExecutionRange, Reduction Redu,
                    const KernelType &(KernelFunc),
                    const std::vector<node> &dep = {});

  // Adds a dependency between two nodes.
  void make_edge(node sender, node receiver);

  // TODO: Extend queue to directly submit graph
  void exec_and_wait(sycl::queue q);

  executable_graph instantiate(sycl::queue q) {
    return executable_graph{my_graph, q};
  };

  graph() : my_graph(new detail::graph_impl()) {}

  // Creates a subgraph (with predecessors)
  graph(graph &parent, const std::vector<node> &dep = {}) {}

  bool is_subgraph();

private:
  detail::graph_ptr my_graph;
};

void executable_graph::exec_and_wait() { my_queue.wait(); }

/// Submits a node to the graph, in order to be executed upon graph execution.
///
/// \param cgf is a function object containing command group.
/// \param dep is a vector of graph nodes the to be submitted node depends on.
/// \return a graph node representing the command group operation.
template <typename T> node graph::submit(T cgf, const std::vector<node> &dep) {
  node _node(my_graph, cgf);
  if (!dep.empty()) {
    for (auto n : dep)
      this->make_edge(n, _node);
  } else {
    _node.set_root();
  }
  return _node;
}

/// Submits an empty node to the graph, in order to be executed upon graph
/// execution.
///
/// \param dep is a vector of graph nodes the to be submitted node depends on.
/// \return a graph node representing no operations but potentially node
/// dependencies.
node graph::submit(const std::vector<node> &dep) {
  node _node(my_graph);
  if (!dep.empty()) {
    for (auto n : dep)
      this->make_edge(n, _node);
  } else {
    _node.set_root();
  }
  return _node;
}

/// Submits a node to the graph, in order to be executed upon graph execution.
///
/// \param Node is the graph node to be used. This overwrites the node
/// parameters.
/// \param cgf is a function object containing command group.
/// \param dep is a vector of graph nodes the to be submitted node depends on.
template <typename T>
void graph::submit(node &Node, T cgf, const std::vector<node> &dep) {
  Node.my_graph = this->my_graph;
  Node.my_node->my_graph = this->my_graph;
  Node.my_node->my_body = cgf;
  Node.my_node->is_empty = false;
  if (!dep.empty()) {
    for (auto n : dep)
      this->make_edge(n, Node);
  } else {
    Node.set_root();
  }
}

/// Sets or updates a graph node by overwriting its dependencies.
///
/// \param Node is a graph node to be updated.
/// \param dep is a vector of graph nodes the to be updated node depends on.
void graph::update(node &Node, const std::vector<node> &dep) {
  Node.my_graph = this->my_graph;
  Node.my_node->my_graph = this->my_graph;
  Node.my_node->is_empty = true;
  if (!dep.empty()) {
    for (auto n : dep)
      this->make_edge(n, Node);
  } else {
    Node.set_root();
  }
}

/// Sets or updates a graph node by overwriting its parameters.
///
/// \param Node is a graph node to be updated.
/// \param cgf is a function object containing command group.
/// \param dep is a vector of graph nodes the to be updated node depends on.
template <typename T>
void graph::update(node &Node, T cgf, const std::vector<node> &dep) {
  Node.my_graph = this->my_graph;
  Node.my_node->my_graph = this->my_graph;
  Node.my_node->my_body = cgf;
  Node.my_node->is_empty = false;
  if (!dep.empty()) {
    for (auto n : dep)
      this->make_edge(n, Node);
  } else {
    Node.set_root();
  }
}

/// Fills the specified memory with the specified pattern.
///
/// \param Ptr is the pointer to the memory to fill.
/// \param Pattern is the pattern to fill into the memory.  T should be
/// trivially copyable.
/// \param Count is the number of times to fill Pattern into Ptr.
/// \param dep is a vector of graph nodes the fill depends on.
/// \return a graph node representing the fill operation.
template <typename T>
node graph::fill(void *Ptr, const T &Pattern, size_t Count,
                 const std::vector<node> &dep) {
  return graph::submit([=](sycl::handler &h) { h.fill(Ptr, Pattern, Count); },
                       dep);
}

/// Fills the specified memory with the specified pattern.
///
/// \param Node is the graph node to be used for the fill. This overwrites
/// the node parameters.
/// \param Ptr is the pointer to the memory to fill.
/// \param Pattern is the pattern to fill into the memory.  T should be
/// trivially copyable.
/// \param Count is the number of times to fill Pattern into Ptr.
/// \param dep is a vector of graph nodes the fill depends on.
template <typename T>
void graph::fill(node &Node, void *Ptr, const T &Pattern, size_t Count,
                 const std::vector<node> &dep) {
  graph::update(
      Node, [=](sycl::handler &h) { h.fill(Ptr, Pattern, Count); }, dep);
}

/// Copies data from one memory region to another, both pointed by
/// USM pointers.
/// No operations is done if \param Count is zero. An exception is thrown
/// if either \param Dest or \param Src is nullptr. The behavior is undefined
/// if any of the pointer parameters is invalid.
///
/// \param Dest is a USM pointer to the destination memory.
/// \param Src is a USM pointer to the source memory.
/// \param dep is a vector of graph nodes the memset depends on.
/// \return a graph node representing the memset operation.
node graph::memset(void *Ptr, int Value, size_t Count,
                   const std::vector<node> &dep) {
  return graph::submit([=](sycl::handler &h) { h.memset(Ptr, Value, Count); },
                       dep);
}

/// Copies data from one memory region to another, both pointed by
/// USM pointers.
/// No operations is done if \param Count is zero. An exception is thrown
/// if either \param Dest or \param Src is nullptr. The behavior is undefined
/// if any of the pointer parameters is invalid.
///
/// \param Node is the graph node to be used for the memset. This overwrites
/// the node parameters.
/// \param Dest is a USM pointer to the destination memory.
/// \param Src is a USM pointer to the source memory.
/// \param dep is a vector of graph nodes the memset depends on.
void graph::memset(node &Node, void *Ptr, int Value, size_t Count,
                   const std::vector<node> &dep) {
  graph::update(
      Node, [=](sycl::handler &h) { h.memset(Ptr, Value, Count); }, dep);
}

/// Copies data from one memory region to another, both pointed by
/// USM pointers.
/// No operations is done if \param Count is zero. An exception is thrown
/// if either \param Dest or \param Src is nullptr. The behavior is undefined
/// if any of the pointer parameters is invalid.
///
/// \param Dest is a USM pointer to the destination memory.
/// \param Src is a USM pointer to the source memory.
/// \param Count is a number of bytes to copy.
/// \param dep is a vector of graph nodes the memcpy depends on.
/// \return a graph node representing the memcpy operation.
node graph::memcpy(void *Dest, const void *Src, size_t Count,
                   const std::vector<node> &dep) {
  return graph::submit([=](sycl::handler &h) { h.memcpy(Dest, Src, Count); },
                       dep);
}

/// Copies data from one memory region to another, both pointed by
/// USM pointers.
/// No operations is done if \param Count is zero. An exception is thrown
/// if either \param Dest or \param Src is nullptr. The behavior is undefined
/// if any of the pointer parameters is invalid.
///
/// \param Node is the graph node to be used for the memcpy. This overwrites
/// the node parameters.
/// \param Dest is a USM pointer to the destination memory.
/// \param Src is a USM pointer to the source memory.
/// \param Count is a number of bytes to copy.
/// \param dep is a vector of graph nodes the memcpy depends on.
void graph::memcpy(node &Node, void *Dest, const void *Src, size_t Count,
                   const std::vector<node> &dep) {
  graph::update(
      Node, [=](sycl::handler &h) { h.memcpy(Dest, Src, Count); }, dep);
}

/// Copies data from one memory region to another, both pointed by
/// USM pointers.
/// No operations is done if \param Count is zero. An exception is thrown
/// if either \param Dest or \param Src is nullptr. The behavior is undefined
/// if any of the pointer parameters is invalid.
///
/// \param Src is a USM pointer to the source memory.
/// \param Dest is a USM pointer to the destination memory.
/// \param Count is a number of elements of type T to copy.
/// \param dep is a vector of graph nodes the copy depends on.
/// \return a graph node representing the copy operation.
template <typename T>
node graph::copy(const T *Src, T *Dest, size_t Count,
                 const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) { h.memcpy(Dest, Src, Count * sizeof(T)); }, dep);
}

/// Copies data from one memory region to another, both pointed by
/// USM pointers.
/// No operations is done if \param Count is zero. An exception is thrown
/// if either \param Dest or \param Src is nullptr. The behavior is undefined
/// if any of the pointer parameters is invalid.
///
/// \param Node is the graph node to be used for the copy. This overwrites
/// the node parameters.
/// \param Src is a USM pointer to the source memory.
/// \param Dest is a USM pointer to the destination memory.
/// \param Count is a number of elements of type T to copy.
/// \param dep is a vector of graph nodes the copy depends on.
template <typename T>
void graph::copy(node &Node, const T *Src, T *Dest, size_t Count,
                 const std::vector<node> &dep) {
  graph::update(
      Node, [=](sycl::handler &h) { h.memcpy(Dest, Src, Count * sizeof(T)); },
      dep);
}

/// Provides additional information to the underlying runtime about how
/// different allocations are used.
///
/// \param Ptr is a USM pointer to the allocation.
/// \param Length is a number of bytes in the allocation.
/// \param Advice is a device-defined advice for the specified allocation.
/// \param dep is a vector of graph nodes the mem_advise depends on.
/// \return a graph node representing the mem_advise operation.
node graph::mem_advise(const void *Ptr, size_t Length, int Advice,
                       const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) { h.mem_advise(Ptr, Length, Advice); }, dep);
}

/// Provides additional information to the underlying runtime about how
/// different allocations are used.
///
/// \param Node is the graph node to be used for the mem_advise. This overwrites
/// the node parameters.
/// \param Ptr is a USM pointer to the allocation.
/// \param Length is a number of bytes in the allocation.
/// \param Advice is a device-defined advice for the specified allocation.
/// \param dep is a vector of graph nodes the mem_advise depends on.
void graph::mem_advise(node &Node, const void *Ptr, size_t Length, int Advice,
                       const std::vector<node> &dep) {
  graph::update(
      Node, [=](sycl::handler &h) { h.mem_advise(Ptr, Length, Advice); }, dep);
}

/// Provides hints to the runtime library that data should be made available
/// on a device earlier than Unified Shared Memory would normally require it
/// to be available.
///
/// \param Ptr is a USM pointer to the memory to be prefetched to the device.
/// \param Count is a number of bytes to be prefetched.
/// \param dep is a vector of graph nodes the prefetch depends on.
/// \return a graph node representing the prefetch operation.
node graph::prefetch(const void *Ptr, size_t Count,
                     const std::vector<node> &dep) {
  return graph::submit([=](sycl::handler &h) { h.prefetch(Ptr, Count); }, dep);
}

/// Provides hints to the runtime library that data should be made available
/// on a device earlier than Unified Shared Memory would normally require it
/// to be available.
///
/// \param Node is the graph node to be used for the prefetch. This overwrites
/// the node parameters.
/// \param Ptr is a USM pointer to the memory to be prefetched to the device.
/// \param Count is a number of bytes to be prefetched.
/// \param dep is a vector of graph nodes the prefetch depends on.
void graph::prefetch(node &Node, const void *Ptr, size_t Count,
                     const std::vector<node> &dep) {
  graph::update(
      Node, [=](sycl::handler &h) { h.prefetch(Ptr, Count); }, dep);
}

/// single_task version with a kernel represented as a lambda.
///
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the single_task depends on.
/// \return a graph node representing the single_task operation.
template <typename KernelName, typename KernelType>
node graph::single_task(const KernelType &(KernelFunc),
                        const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template single_task<KernelName, KernelType>(KernelFunc);
      },
      dep);
}

/// single_task version with a kernel represented as a lambda.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the single_task depends on.
template <typename KernelName, typename KernelType>
void graph::single_task(node &Node, const KernelType &(KernelFunc),
                        const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template single_task<KernelName, KernelType>(KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType>
node graph::parallel_for(range<1> NumWorkItems, const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType>
void graph::parallel_for(node &Node, range<1> NumWorkItems,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType>
node graph::parallel_for(range<2> NumWorkItems, const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType>
void graph::parallel_for(node &Node, range<2> NumWorkItems,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType>
node graph::parallel_for(range<3> NumWorkItems, const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType>
void graph::parallel_for(node &Node, range<3> NumWorkItems,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType, int Dims>
node graph::parallel_for(range<Dims> NumWorkItems,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global size only.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType, int Dims>
void graph::parallel_for(node &Node, range<Dims> NumWorkItems,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(NumWorkItems,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range and
/// offset that specify global size and global offset correspondingly.
///
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param WorkItemOffset specifies the offset for each work item id
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType, int Dims>
node graph::parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(
            NumWorkItems, WorkItemOffset, KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range and
/// offset that specify global size and global offset correspondingly.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param WorkItemOffset specifies the offset for each work item id
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType, int Dims>
void graph::parallel_for(node &Node, range<Dims> NumWorkItems,
                         id<Dims> WorkItemOffset,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(
            NumWorkItems, WorkItemOffset, KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + nd_range that
/// specifies global, local sizes and offset.
///
/// \param ExecutionRange is a range that specifies the work space of the
/// kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType, int Dims>
node graph::parallel_for(nd_range<Dims> ExecutionRange,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(ExecutionRange,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + nd_range that
/// specifies global, local sizes and offset.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param ExecutionRange is a range that specifies the work space of the
/// kernel
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType, int Dims>
void graph::parallel_for(node &Node, nd_range<Dims> ExecutionRange,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType>(ExecutionRange,
                                                        KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global, local sizes and offset.
///
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param Redu is a reduction operation
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType, int Dims,
          typename Reduction>
node graph::parallel_for(range<Dims> NumWorkItems, Reduction Redu,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType, Dims, Reduction>(
            NumWorkItems, Redu, KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + range that
/// specifies global, local sizes and offset.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param NumWorkItems is a range that specifies the work space of the kernel
/// \param Redu is a reduction operation
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType, int Dims,
          typename Reduction>
void graph::parallel_for(node &Node, range<Dims> NumWorkItems, Reduction Redu,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType, Dims, Reduction>(
            NumWorkItems, Redu, KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + nd_range that
/// specifies global, local sizes and offset.
///
/// \param ExecutionRange is a range that specifies the work space of the
/// kernel
/// \param Redu is a reduction operation
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
/// \return a graph node representing the parallel_for operation.
template <typename KernelName, typename KernelType, int Dims,
          typename Reduction>
node graph::parallel_for(nd_range<Dims> ExecutionRange, Reduction Redu,
                         const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  return graph::submit(
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType, Dims, Reduction>(
            ExecutionRange, Redu, KernelFunc);
      },
      dep);
}

/// parallel_for version with a kernel represented as a lambda + nd_range that
/// specifies global, local sizes and offset.
///
/// \param Node is the graph node to be used for the single_task. This
/// overwrites the node parameters.
/// \param ExecutionRange is a range that specifies the work space of the
/// kernel
/// \param Redu is a reduction operation
/// \param KernelFunc is the Kernel functor or lambda
/// \param dep is a vector of graph nodes the parallel_for depends on
template <typename KernelName, typename KernelType, int Dims,
          typename Reduction>
void graph::parallel_for(node &Node, nd_range<Dims> ExecutionRange,
                         Reduction Redu, const KernelType &(KernelFunc),
                         const std::vector<node> &dep) {
  graph::update(
      Node,
      [=](sycl::handler &h) {
        h.template parallel_for<KernelName, KernelType, Dims, Reduction>(
            ExecutionRange, Redu, KernelFunc);
      },
      dep);
}

void graph::make_edge(node sender, node receiver) {
  sender.register_successor(receiver);     // register successor
  my_graph->remove_root(receiver.my_node); // remove receiver from root node
                                           // list
}

void graph::exec_and_wait(sycl::queue q) { my_graph->exec_and_wait(q); };

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
