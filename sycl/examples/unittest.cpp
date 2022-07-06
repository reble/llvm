//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include <CL/sycl.hpp>

#include <gtest/gtest.h>

using namespace oneapi::dpl::execution;
using namespace sycl;


const int initial_value = 10;
const int array_size = 6;


// one parallel_for in the capture window 
TEST(CaptureGraph, one_parallel_for) {
  sycl::property_list properties{
    sycl::property::queue::in_order(),
    sycl::ext::oneapi::property::queue::capture_mode{}};
  
  sycl::default_selector device_selector;
  
  sycl::queue q(device_selector, properties);
    
  int* data = sycl::malloc_shared<int>(array_size, q);
  
  for (int i = 0; i < array_size; i++) {
    data[i] = initial_value;
  }
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Initialize Error: data[" << i << "] != " << initial_value;    
  }

  sycl::ext::oneapi::experimental::graph g;

  q.begin_capture(&g);
  
  auto e =
  q.parallel_for(
    range<1>{array_size},
    [=](id<1> idx) {
      data[idx] = data[idx] + idx;
    }
  );
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  q.end_capture();
  
  auto exec_graph = g.instantiate(q);

  q.submit(exec_graph);
    
  for (int i = 0; i < array_size; i++) {
    EXPECT_EQ(initial_value + i, data[i]) 
    << "Execution Error: data[" << i << "] != " << initial_value + i;
  }
  
  free(data, q);
}


// one parallel_for and wait in the capture window 
TEST(CaptureGraph, one_parallel_for_wait) {
  sycl::property_list properties{
    sycl::property::queue::in_order(),
    sycl::ext::oneapi::property::queue::capture_mode{}};
  
  sycl::default_selector device_selector;
  
  sycl::queue q(device_selector, properties);
    
  int* data = sycl::malloc_shared<int>(array_size, q);
  
  for (int i = 0; i < array_size; i++) {
    data[i] = initial_value;
  }
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Initialize Error: data[" << i << "] != " << initial_value;    
  }

  sycl::ext::oneapi::experimental::graph g;

  q.begin_capture(&g);
 
  auto e =  
  q.parallel_for(
    range<1>{array_size},
    [=](id<1> idx) {
      data[idx] = data[idx] + idx;
    }
  );

  e.wait();
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  q.end_capture();
  
  auto exec_graph = g.instantiate(q);

  q.submit(exec_graph);
    
  for (int i = 0; i < array_size; i++) {
    EXPECT_EQ(initial_value + i, data[i]) 
    << "Execution Error: data[" << i << "] != " << initial_value + i;
  }
  
  free(data, q);
}


// two parallel_for, two wait and one dependent in the capture window 
TEST(CaptureGraph, two_parallel_for_wait_dep) {
  sycl::property_list properties{
    sycl::property::queue::in_order(),
    sycl::ext::oneapi::property::queue::capture_mode{}};
  
  sycl::default_selector device_selector;
  
  sycl::queue q(device_selector, properties);
    
  int* data = sycl::malloc_shared<int>(array_size, q);
  
  for (int i = 0; i < array_size; i++) {
    data[i] = initial_value;
  }
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Initialize Error: data[" << i << "] != " << initial_value;    
  }

  sycl::ext::oneapi::experimental::graph g;

  q.begin_capture(&g);
 
  auto e1 =  
  q.parallel_for(
    range<1>{array_size},
    [=](id<1> idx) {
      data[idx] = data[idx] + idx;
    }
  );

  e1.wait();
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  auto e2 =  
  q.parallel_for(
    range<1>{array_size},
    {e1},
    [=](id<1> idx) {
      data[idx] = data[idx] * 2;
    }
  );

  e2.wait();
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  q.end_capture();
  
  auto exec_graph = g.instantiate(q);

  q.submit(exec_graph);
    
  for (int i = 0; i < array_size; i++) {
    EXPECT_EQ(2*(initial_value + i), data[i]) 
    << "Execution Error: data[" << i << "] != " << 2*(initial_value + i);
  }
  
  free(data, q);
}


// one submit in the capture window 
TEST(CaptureGraph, one_submit) {
  sycl::property_list properties{
    sycl::property::queue::in_order(),
    sycl::ext::oneapi::property::queue::capture_mode{}};
  
  sycl::default_selector device_selector;
  
  sycl::queue q(device_selector, properties);
    
  int* data = sycl::malloc_shared<int>(array_size, q);
  
  for (int i = 0; i < array_size; i++) {
    data[i] = initial_value;
  }
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Initialize Error: data[" << i << "] != " << initial_value;    
  }

  sycl::ext::oneapi::experimental::graph g;

  q.begin_capture(&g);
  
  auto e =
  q.submit([&](sycl::handler& h){
    h.parallel_for(
      range<1>{array_size},
      [=](id<1> idx) {
        data[idx] = data[idx] + idx;
      }
    );
  });
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  q.end_capture();
  
  auto exec_graph = g.instantiate(q);

  q.submit(exec_graph);
    
  for (int i = 0; i < array_size; i++) {
    EXPECT_EQ(initial_value + i, data[i]) 
    << "Execution Error: data[" << i << "] != " << initial_value + i;
  }
  
  free(data, q);
}



/*
 * The followings capture external libraries
 */

// one PSTL - for_each in the capture window
TEST(CaptureGraph, one_PSTL_for_each) {
  sycl::property_list properties{
    sycl::property::queue::in_order(),
    sycl::ext::oneapi::property::queue::capture_mode{}};
  
  sycl::default_selector device_selector;
  
  sycl::queue q(device_selector, properties);
    
  int* data = sycl::malloc_shared<int>(array_size, q);
  
  for (int i = 0; i < array_size; i++) {
    data[i] = initial_value;
  }
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Initialize Error: data[" << i << "] != " << initial_value;    
  }

  sycl::ext::oneapi::experimental::graph g;

  q.begin_capture(&g);
  
  std::for_each(make_device_policy(q), data, data + array_size, [](int& d){
    d = d + 1;
  });
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  q.end_capture();
  
  auto exec_graph = g.instantiate(q);

  q.submit(exec_graph);
    
  for (int i = 0; i < array_size; i++) {
    EXPECT_EQ(initial_value + 1, data[i]) 
    << "Execution Error: data[" << i << "] != " << initial_value+1;
  }
  
  free(data, q);
}


// two PSTLs - "for_each and fill" in the capture window
TEST(CaptureGraph, two_PSTLs) {
  sycl::property_list properties{
    sycl::property::queue::in_order(),
    sycl::ext::oneapi::property::queue::capture_mode{}};
  
  sycl::default_selector device_selector;
  
  sycl::queue q(device_selector, properties);
    
  int* data = sycl::malloc_shared<int>(array_size, q);
  
  for (int i = 0; i < array_size; i++) {
    data[i] = initial_value;
  }
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Initialize Error: data[" << i << "] != " << initial_value;    
  }

  sycl::ext::oneapi::experimental::graph g;

  q.begin_capture(&g);
  
  std::fill(make_device_policy(q), data, data + array_size, initial_value * 2);
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  std::for_each(make_device_policy(q), data, data + array_size, [](int& d){
    d = d + 1;
  });
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  q.end_capture();
  
  auto exec_graph = g.instantiate(q);

  q.submit(exec_graph);
    
  for (int i = 0; i < array_size; i++) {
    EXPECT_EQ(2*initial_value + 1, data[i]) 
    << "Execution Error: data[" << i << "] != " << 2*initial_value+1;
  }
  
  free(data, q);
}


// four PSTLs in the capture window
TEST(CaptureGraph, four_PSTLs) {
  sycl::property_list properties{
    sycl::property::queue::in_order(),
    sycl::ext::oneapi::property::queue::capture_mode{}};
  
  sycl::default_selector device_selector;
  
  sycl::queue q(device_selector, properties);
    
  int* data = sycl::malloc_shared<int>(array_size, q);
  
  for (int i = 0; i < array_size; i++) {
    data[i] = initial_value;
  }
  
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Initialize Error: data[" << i << "] != " << initial_value;    
  }

  sycl::ext::oneapi::experimental::graph g;
  
  q.begin_capture(&g);
  
  std::fill(make_device_policy(q), data, data + array_size, initial_value*2);
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  std::for_each(make_device_policy(q), data, data + array_size, [](int& d){
    d = d + 1;
  });
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  std::fill(make_device_policy(q), data, data + array_size, 10);
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  std::for_each(make_device_policy(q), data, data + array_size, [](int& d){
    d = d - 10;
  });
 
  for (int i = 0; i < array_size; ++i) {
    EXPECT_EQ(initial_value, data[i])
    << "Error: PSTL executes in the capture window.";    
  }
  
  q.end_capture();
  
  auto exec_graph = g.instantiate(q);

  q.submit(exec_graph);
    
  for (int i = 0; i < array_size; i++) {
    EXPECT_EQ(0, data[i]) 
    << "Execution Error: data[" << i << "] != " << "0";
  }
  
  free(data, q);
}
