# SYCL compiler

# Set to the path of the Intel LLVM build with the SYCL Graph extension.
#LLVMBUILDDIR = ${DPCPP_HOME}/llvm/build
#export PATH := ${LLVMBUILDDIR}/bin:${PATH}
#export LD_LIBRARY_PATH := ${LLVMBUILDDIR}/lib:${LD_LIBRARY_PATH}

CXX = clang++
#CPPFLAGS = -fsycl -std=c++17 -O3 -fsycl-device-code-split=off -fsycl-early-optimizations -fsycl-dead-args-optimization -Wall -Wpedantic -v
CPPFLAGS = -fsycl -std=c++17 -g -fsycl-unnamed-lambda
CPPFLAGSGRAPH = -D SYCL_EXT_ONEAPI_LAZY_QUEUE=1
CPPINCLUDEPATH = -I${DPCPP_HOME}/llvm/sycl/include -I${DPCPP_HOME}/llvm/include
# Enables AOT compilation. Specify the target device. Disable AOT and use JIT by setting DEVICE to an empty string.
CPPFLAGSAOT = -fsycl-targets=spir64_gen -Xs "-device Gen9"  # Intel Gen9 GPU
# CPPFLAGSAOT = -fsycl-targets=nvptx64-nvidia-cuda  # NVIDIA GPU

# Set the L0 command batch size to 20. This is a workaround with the current PoC since the dynamic adjusting of the command batch size does not work with the lazy queue.
#ENVGRAPH = SYCL_PI_LEVEL_ZERO_BATCH_SIZE=20


# CUDA compiler
NVCC = nvcc
# Adjust for the target architecture
NVFLAGS = -O3 -lineinfo -arch=sm_75


SRCS = $(wildcard *.cpp) $(wildcard *.hpp) $(wildcard *.cu)

#1DHEAT-OBJECTS = 1Dheat-sycl-buffer 1Dheat-sycl-usm 1Dheat-sycl-buffer-graph 1Dheat-sycl-usm-graph 1Dheat-sycl-usm-graph-capture 1Dheat-sycl-usm-graph-unroll

#ACCESSORS-OBJECTS = accessors-sycl-buffer accessors-sycl-usm accessors-sycl-buffer-graph accessors-sycl-usm-graph

#APSP-OBJECTS = apsp-sycl-buffer apsp-sycl-usm apsp-sycl-buffer-graph apsp-sycl-usm-graph apsp-sycl-usm-graph-capture apsp-sycl-usm-graph-unroll

#DOTP-OBJECTS = dotp-sycl-buffer dotp-sycl-usm dotp-sycl-buffer-graph dotp-sycl-usm-graph dotp-sycl-usm-graph-capture

#ISO2DFD-OBJECTS = iso2dfd-sycl-buffer iso2dfd-sycl-usm iso2dfd-sycl-buffer-graph iso2dfd-sycl-usm-graph iso2dfd-sycl-usm-graph-capture iso2dfd-sycl-usm-graph-unroll

#NN-OBJECTS = nn-sycl-usm nn-sycl-usm-graph nn-sycl-usm-graph-unroll nn-cuda nn-cuda-graphs nn-cuda-graphs-explicit

#SYCL-USM-OBJECTS = 1Dheat-sycl-usm accessors-sycl-usm apsp-sycl-usm dotp-sycl-usm iso2dfd-sycl-usm nn-sycl-usm

#SYCL-BUFFER-OBJECTS = 1Dheat-sycl-buffer accessors-sycl-buffer apsp-sycl-buffer dotp-sycl-buffer iso2dfd-sycl-buffer

#SYCL-USM-GRAPH-OBJECTS = 1Dheat-sycl-usm-graph 1Dheat-sycl-usm-graph-capture 1Dheat-sycl-usm-graph-unroll accessors-sycl-usm-graph apsp-sycl-usm-graph apsp-sycl-usm-graph-capture apsp-sycl-usm-graph-unroll dotp-sycl-usm-graph dotp-sycl-usm-graph-capture iso2dfd-sycl-usm-graph iso2dfd-sycl-usm-graph-capture iso2dfd-sycl-usm-graph-unroll nn-sycl-usm-graph nn-sycl-usm-graph-unroll
#SYCL-USM-GRAPH-OBJECTS = 1Dheat-sycl-usm-graph 1Dheat-sycl-usm-graph-capture iso2dfd-sycl-usm-graph accessors-sycl-usm-graph apsp-sycl-usm-graph dotp-sycl-usm-graph nn-sycl-usm-graph
SYCL-USM-GRAPH-OBJECTS = 1Dheat-sycl-usm-graph 1Dheat-sycl-usm-graph-capture

#SYCL-BUFFER-GRAPH-OBJECTS = 1Dheat-sycl-buffer-graph accessors-sycl-buffer-graph apsp-sycl-buffer-graph dotp-sycl-buffer-graph iso2dfd-sycl-buffer-graph

.PHONY: all 1Dheat accessors apsp dotp iso2dfd nn sycl-graph sycl-usm-graph sycl-buffer-graph run run-1Dheat run-accessors run-apsp run-dotp run-iso2dfd run-nn run-sycl-usm-graph run-sycl-buffer-graph format tidy clean

#all: 1Dheat accessors apsp dotp iso2dfd nn
#1Dheat: $(1DHEAT-OBJECTS)
#accessors: $(ACCESSORS-OBJECTS)
#apsp: $(APSP-OBJECTS)
#dotp: $(DOTP-OBJECTS)
#iso2dfd: $(ISO2DFD-OBJECTS)
#nn: $(NN-OBJECTS)
#
#sycl: sycl-usm sycl-buffer
#sycl-usm: $(SYCL-USM-OBJECTS)
#sycl-buffer: $(SYCL-BUFFER-OBJECTS)
#
#sycl-graph: sycl-usm-graph sycl-buffer-graph
sycl-usm-graph: $(SYCL-USM-GRAPH-OBJECTS)
#sycl-buffer-graph: $(SYCL-BUFFER-GRAPH-OBJECTS)
#
#run: run-1Dheat run-accessors run-apsp run-dotp run-iso2dfd run-nn
#run-1Dheat: $(addprefix run-,$(1DHEAT-OBJECTS))
#run-accessors: $(addprefix run-,$(ACCESSORS-OBJECTS))
#run-apsp: $(addprefix run-,$(APSP-OBJECTS))
#run-dotp: $(addprefix run-,$(DOTP-OBJECTS))
#run-iso2dfd: $(addprefix run-,$(ISO2DFD-OBJECTS))
#run-nn: $(addprefix run-,$(NN-OBJECTS))
#
#run-sycl: run-sycl-usm run-sycl-buffer
#run-sycl-usm: $(addprefix run-,$(SYCL-USM-OBJECTS))
#run-sycl-buffer: $(addprefix run-,$(SYCL-BUFFER-OBJECTS))
#
#run-sycl-graph: run-sycl-usm-graph run-sycl-buffer-graph
run-sycl-usm-graph: $(addprefix run-,$(SYCL-USM-GRAPH-OBJECTS))
#run-sycl-buffer-graph: $(addprefix run-,$(SYCL-BUFFER-GRAPH-OBJECTS))
#
#%-sycl-buffer: %-sycl-buffer.cpp
#	$(CXX) $(CPPFLAGS) $(CPPFLAGSAOT) $< -o $@
#
#%-sycl-usm: %-sycl-usm.cpp
#	$(CXX) $(CPPFLAGS) $(CPPFLAGSAOT) $< -o $@
#
%-sycl-usm-graph: %-sycl-usm-graph.cpp
	$(CXX) $(CPPFLAGS) $(CPPINCLUDEPATH) $< -o $@

%-sycl-usm-graph-capture: %-sycl-usm-graph-capture.cpp
	$(CXX) $(CPPFLAGS) $(CPPINCLUDEPATH) $< -o $@
#
#%-sycl-usm-graph-unroll: %-sycl-usm-graph-unroll.cpp
#	$(CXX) $(CPPFLAGS) $(CPPFLAGSAOT) $(CPPFLAGSGRAPH) $< -o $@
#
#run-1Dheat-sycl-buffer: 1Dheat-sycl-buffer
#	./$< 10000 1000
#
#run-1Dheat-sycl-usm: 1Dheat-sycl-usm
#	./$< 10000 1000
#
#run-1Dheat-sycl-buffer-graph: 1Dheat-sycl-buffer-graph
#	$(ENVGRAPH) ./$< 10000 1000
#
run-1Dheat-sycl-usm-graph: 1Dheat-sycl-usm-graph
	$(ENVGRAPH) ./$< 100 10
#	$(ENVGRAPH) ./$< 10000 1000

run-1Dheat-sycl-usm-graph-capture: 1Dheat-sycl-usm-graph-capture
	$(ENVGRAPH) ./$< 1000 1


#run-1Dheat-sycl-usm-graph-unroll: 1Dheat-sycl-usm-graph-unroll
#	$(ENVGRAPH) ./$< 10000 1000
#
#run-iso2dfd-sycl-buffer: iso2dfd-sycl-buffer
#	./$< 1024 1024 2000
#
#run-iso2dfd-sycl-usm: iso2dfd-sycl-usm
#	./$< 1024 1024 2000
#
#run-iso2dfd-sycl-buffer-graph: iso2dfd-sycl-buffer-graph
#	$(ENVGRAPH) ./$< 1024 1024 2000
#
run-iso2dfd-sycl-usm-graph: iso2dfd-sycl-usm-graph
	$(ENVGRAPH) ./$< 16 16 20
#	$(ENVGRAPH) ./$< 1024 1024 2000
#
#run-iso2dfd-sycl-usm-graph-capture: iso2dfd-sycl-usm-graph-capture
#	$(ENVGRAPH) ./$< 1024 1024 2000
#
#run-iso2dfd-sycl-usm-graph-unroll: iso2dfd-sycl-usm-graph-unroll
#	$(ENVGRAPH) ./$< 1024 1024 2000
#
#run-%-sycl-buffer: %-sycl-buffer
#	./$<
#
#run-%-sycl-usm: %-sycl-usm
#	./$<
#
#run-%-sycl-buffer-graph: %-sycl-buffer-graph
#	$(ENVGRAPH) ./$<
#
run-%-sycl-usm-graph: %-sycl-usm-graph
	$(ENVGRAPH) ./$<
#
#run-%-sycl-usm-graph-capture: %-sycl-usm-graph-capture
#	$(ENVGRAPH) ./$<
#
#run-%-sycl-usm-graph-unroll: %-sycl-usm-graph-unroll
#	$(ENVGRAPH) ./$<
#
#%-cuda: %-cuda.cu
#	$(NVCC) $(NVFLAGS) $< -o $@
#
#%-cuda-graphs: %-cuda-graphs.cu
#	$(NVCC) $(NVFLAGS) $< -o $@
#
#%-cuda-graphs-capture: %-cuda-graphs-capture.cu
#	$(NVCC) $(NVFLAGS) $< -o $@
#
#run-%-cuda: %-cuda
#	./$<
#
#run-%-cuda-graphs: %-cuda-graphs
#	./$<
#
#run-%-cuda-graphs-capture: %-cuda-graphs-capture
#	./$<

format:
	clang-format -i $(SRCS)

tidy:
	clang-tidy -header-filter=.* $(SRCS) -- -I${LLVMBUILDDIR}/include -I${LLVMBUILDDIR}/include/sycl

clean:
	rm -f $(1DHEAT-OBJECTS) $(ACCESSORS-OBJECTS) $(APSP-OBJECTS) $(DOTP-OBJECTS) $(ISO2DFD-OBJECTS) $(NN-OBJECTS) wavefield_snapshot.bin wavefield_snapshot_cpu.bin *_error_diff.txt

