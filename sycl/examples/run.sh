#!/bin/bash
#source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

export DPCPP_HOME=/home/u154255/sycl_graph/chchiu
export LLVMBUILDDIR=$DPCPP_HOME/llvm/build 

export PATH=$LLVMBUILDDIR/bin:$PATH
export LD_LIBRARY_PATH=$LLVMBUILDDIR/lib:$LD_LIBRARY_PATH

#fullname=$1
#filename="${fullname%%.*}"
#
#clang++ -g $1 -o $filename -I$DPCPP_HOME/sycl/include -I$DPCPP_HOME/include -std=c++17 -fsycl -fsycl-unnamed-lambda 
#export LD_LIBRARY_PATH=$LLVMBUILDDIR/lib:$LD_LIBRARY_PATH
#./$filename 1 1
#

# clean all the executable and output texts
make clean

## compile *-sycl-usm-graph source code
#make sycl-usm-graph
#
## execute *-sycl-usm-graph executables
#echo ""
#echo "------ Start  running executables ------"
#echo ""
#make run-sycl-usm-graph
#echo ""
#echo "------ Finish running executables ------"
#echo ""
 
##compile *dpl-usm-pointer source code
#make dpl-usm-pointer-graph-capture
#
## execute *-dpl-usm-pointer executables
#echo ""
#echo "------ Start  running executables ------"
#echo ""
#make run-dpl-usm-pointer-graph-capture
#echo ""
#echo "------ Finish running executables ------"
#echo ""

make test
./unittest

#export LD_LIBRARY_PATH=$LLVMBUILDDIR/lib:$LD_LIBRARY_PATH
#rm -rf ./a.out
#PATH=$LLVMBUILDDIR/bin:$PATH clang++ -g $1 -I $LLVMBUILDDIR/include/sycl -I $LLVMBUILDDIR/include -std=c++17 -fsycl -fsycl-unnamed-lambda 


