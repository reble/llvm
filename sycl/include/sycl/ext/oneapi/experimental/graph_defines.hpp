//==--------- graph.hpp --- SYCL graph extension ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {
enum class graph_state {
  modifiable,
  executable,
};
}
} // namespace oneapi
} // namespace ext
} // namespace sycl