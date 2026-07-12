#! /bin/bash

#Step 1: Convert the CUDA files to HIP. Basically replaces cu or cuda with HIP in all CUDA standard functions, and
#replaces __CUDACC__ with __HIPCC__
shopt -s nullglob
prehips=( ../production_cuda/*.prehip ../production_c/*.prehip ../INCLUDE/*.prehip )
[[ -e ../main.c.prehip ]] && prehips+=( ../main.c.prehip )
for f in "${prehips[@]}"; do mv -- "$f" "${f%.prehip}"; done

hipify-perl --print-stats --inplace ../production_cuda/*.cu ../INCLUDE/*.h ../production_c/*.c ../main.c

#Step 2: Deal with some unsupported __managed__ tags
sed -i -e 's/__managed__ //g' ../production_c/*.c ../main.c
sed -i -e 's/__forceinline__ //g' ../production_cuda/*.cu ../INCLUDE/*.h*
sed -i -e 's/__constant__ //g' ../production_cuda/*.cu ../INCLUDE/*.h*
sed -i -e 's/__grid_constant__ //g' ../production_cuda/*.cu ../INCLUDE/*.h*

#Step 3: Remove some over-zealous headers from the pure C code paths
perl -ni -e 'print unless m{^\s*#include\s*[<"]hip/hip_runtime\.h[">]}' \
  ../production_c/*.c ../main.c ../INCLUDE/*.h 
