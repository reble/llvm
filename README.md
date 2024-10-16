# SYCL Command Graph Extensions

**This fork is mostly stale since the main development has been moved to the** [Intel staging area for llvm.org contributions](https://github.com/intel/llvm).

This fork has been the collaboration space for the oneAPI vendor Command Graph extension for SYCL2020 until September 2023. 
SYCL Graph provides an API for defining a graph of operations and their dependencies once and submitting this graph repeatedly for execution.

### Specification

A draft of our Command Graph extension proposal can be found here:
[sycl_ext_oneapi_graph](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).

### Implementation

Our implementation can be found here:
[https://github.com/intel/llvm](https://github.com/intel/llvm).

#### Backends

An application can query the SYCL library for the level of support it
provides for using the extension with a device by using
`ext::oneapi::experimental::info::device::graph_support`, which returns one of:

* Native - Backend command-buffer construct is used to implement graphs.
* Emulated - Graphs support is emulated by reissuing commands to the backend.
* Unsupported - Extension is not supported on the device.

Currently the Level Zero backend is the only supported SYCL backend for the
`sycl_ext_oneapi_graph` extension. As the focus of the current prototype is good
Level Zero support to prove the value of the extension, rather than emulated
support for many backends. However, broadening the number of backends supported
is something we are interested in expanding on.

| Backend    | Implementation Support     |
| ---------- | -------------------------- |
| Level Zero | Native using command-lists |
| CUDA       | Native support is a work-in-progress |
| OpenCL     | Unsupported                |
| HIP        | Unsupported                |
| Others     | Unsupported                |

#### Implementation Status

| Feature                                                            | Implementation Status (- error type) |
| ------------------------------------------------------------------ | --------------------- |
| Adding a command-group node with `command_graph::add()`            | Implemented           |
| Begin & end queue recording to a graph to create nodes             | Implemented           |
| Edges created from buffer accessor dependencies                    | Implemented           |
| Edges created from `handler::depends_on` dependencies              | Implemented           |
| Edges created using `make_edge()`                                  | Implemented           |
| Edges created by passing a property list to `command_graph::add()` | Implemented           |
| Empty node                                                         | Implemented           |
| Queue `ext_oneapi_get_state()` query                               | Implemented           |
| Vendor test macro                                                  | Implemented           |
| Ability to add a graph as a node of another graph (Sub-graphs)     | Implemented           |
| Using all capabilities of USM in a graph node                      | Implemented           |
| Extending lifetime of buffers used in a graph, as defined by the "Storage Lifetimes" specification section  | Not implemented - Throws an exception that feature is not supported yet |
| Buffer taking a copy of underlying host data when buffer is used in a graph, as defined by the "Storage Lifetimes" specification section  | Not implemented - Throws an exception that feature is not supported yet |
| Executable graph `update()`                                        | Not implemented - Exception "Method not yet implemented"      |
| Recording an in-order queue preserves linear dependencies          | Implemented           |
| Using `handler::parallel_for` in a graph node                      | Implemented           |
| Using `handler::single_task` in a graph node                       | Implemented           |
| Using `handler::memcpy` in a graph node                            | Implemented           |
| Using `handler::copy` in a graph node                              | Implemented           |
| Using `handler::host_task` in a graph node                         | Not implemented - Assert: "getCGCopy() const: Assertion `false' failed."       |
| Using `handler::fill` in a graph node                              | Implemented for USM, not implemented for buffer accessors - Exception: "CG type not implemented for command buffers" |
| Using `handler::memset` in a graph node                            | Not implemented - Exception: "CG type not implemented for command buffers"       |
| Using `handler::prefetch` in a graph node                          | Not implemented - Exception: "CG type not implemented for command buffers"       |
| Using `handler::memadvise` in a graph node                         | Not implemented - Exception: "CG type not implemented for command buffers"       |
| Using specialization constants in a graph node                     | Not implemented - Throws an exception that feature is not supported yet |
| Using reductions in a graph node                                   | Not implemented - Throws an exception that feature is not supported yet |
| Using kernel bundles in a graph node                                   | Not implemented - Throws an exception that feature is not supported yet |
| Using sycl streams in a graph node                                 | Not implemented - Exception: "Failed to add kernel to PI command-buffer"      |
| Thread safety of new methods                                       | Implemented       |
| Profiling an event returned from graph submission with `event::get_profiling_info()`       | Not implemented - Throws an exception that feature is not supported yet |
| Querying the state of an event returned from graph submission with `event::get_info<info::event::command_execution_status>()`     | Implemented       |

### Other Material

This extension was presented at the oneAPI Technical Advisory board (Sept'22 meeting). Slides: [https://github.com/oneapi-src/oneAPI-tab/blob/main/language/presentations/2022-09-28-TAB-SYCL-Graph.pdf](https://github.com/oneapi-src/oneAPI-tab/blob/main/language/presentations/2022-09-28-TAB-SYCL-Graph.pdf).

Extension was presented at IWOCL 2023, and the [talk can be found on Youtube](https://www.youtube.com/watch?v=aOTAmyr04rM).

## Intel Project for LLVM\* technology

We target a contribution through the origin of this fork: [Intel staging area for llvm.org contributions](https://github.com/intel/llvm).

### How to use DPC++

#### Releases

TDB

#### Build from sources

See [Get Started Guide](./sycl/doc/GetStartedGuide.md).

### Report a problem

Submit an [issue](https://github.com/intel/llvm/issues) or initiate a 
[discussion](https://github.com/intel/llvm/discussions).

### How to contribute to DPC++

This project welcomes contributions from the community. Please refer to [CONTRIBUTING](/CONTRIBUTING.md) 
for general guidelines around contributing to this project. You can then see 
[ContributeToDPCPP](./sycl/doc/developer/ContributeToDPCPP.md) for DPC++ specific 
guidelines.

# License

See [LICENSE](./sycl/LICENSE.TXT) for details.

<sub>\*Other names and brands may be claimed as the property of others.</sub>
