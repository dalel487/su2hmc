#! /bin/bash

#Step 1: If we've run this already, restore the CUDA files to their original configuration
shopt -s nullglob
prehips=( ../production_cuda/*.prehip ../production_c/*.prehip ../INCLUDE/*.prehip )
[[ -e ../main.c.prehip ]] && prehips+=( ../main.c.prehip )
#The syntax is a little confusing here. It is source (prehip) to dest (same file name with prehip removed).
for f in "${prehips[@]}"; do mv -- "$f" "${f%.prehip}"; done

#Step 2: Convert the CUDA files to HIP. Basically replaces cu or cuda with HIP in all CUDA standard functions, and
#replaces __CUDACC__ with __HIPCC__
hipify-perl -print-stats -inplace -whitelist="cudaDeviceSynchronise" ../production_cuda/*.cu ../INCLUDE/*.h ../production_c/*.c ../main.c

#Step 3: Deal with some unsupported CUDA tags
perl -i -pe 's/__managed__ //g' ../production_c/*.c ../main.c
perl -i -pe 's/(__forceinline__|__constant__|__grid_constant__) //g' ../production_cuda/*.cu ../INCLUDE/*.h*

#Step 4: Remove some over-zealous headers from the pure C code paths
perl -ni -e 'print unless m{^\s*#include\s*[<"]hip/hip_runtime\.h[">]}' \
  ../production_c/*.c ../main.c ../INCLUDE/*.h 
