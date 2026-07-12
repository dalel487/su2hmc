# Running on GPUs
If you have access to GPUs we strongly recommend running on them. All functions of this code are supported by GPUs and
have been tested on them (whereas the clover breaks with MPI). Additonally we have refactored the code to obtain
optimal performance from NVIDIA GPUs, which was the original target for the GPU port.

The primary bottlenecks for this particular code are memory bandwidth and register pressure. An @f$24^3\times 32@f$
lattice needs less than 2GB of GPU memory to run. As a stress test for new chips, we have also run @f$64^3\times 128@f$
on a NVIDIA H200. Running on server grade GPUs is almost always more performant, but for smaller runs at zero chemical
potential workstation GPUs are useable.

At present, multi-GPU is *not* supported and there is currently no plan to do so. The volumes used in dense QCD are
small enough to fit inside a single GPU and the extra communication costs with the current lattice layout are likely to
take longer than running on just one GPU.

## NVIDIA Notes

All GPU kernels can be found in `production_cuda`. There are a few quirks to be aware of though.

- CUDA is `C++` based. The plus side is we can use templating to avoid multiple copies of the same function at
  different precisions. The down side is that it cannot be called directly from `C` even without the templating.
  As a result the following design decisions have been made:
  - At the end of each `.cu` file there are calling wrappers. These are declared in the header files and called from the
    corressponding `C` function inside a `#USE_GPU` pre-processor macro.
  - CUDA kernels are *not* declared in the headers, as the only thing that needs to see them is the calling wrapper
    which is in the same file scope. Kernels are in the Kernels namespace to help doxygen to distinguish them from
    non-kernel functions with similar names.
  - Device code is laid out similarly at the top of the file.
- Out of laziness, we use a fixed grid and block size for all kernels, based on what is optimal for the most heavily
  used kernels.
- One may need to tweak the paths in the Makefile for CUDA headers and libraries.

## HIP Notes (EXPERIMENTAL)
We have made progress in getting the code ready for HIP. However it is not fully tested yet and we make no guarantees on
correctness at this point.

### HIP Conversion Progress
-   We can run on @f$8^3\times 8@f$ on a Radeon Pro W6400 which isn`t officially suppported. @f$24^3\times 24@f$ gives
    incorrect results. We suspect this is due to the lack of xnack on the testbed GPU but until we run on officially
    supported ones we cannot be certain.

-   Debug and profiling modes do not work fully yet as far as we can tell. We are unsure if this is an xnack issue but
    cannot test it at this time (20260712).

### Building for HIP
Building for HIP is currently a three to five step process

1.  To ensure the original cuda source code is uncorrupted, we recommend first checking out the latest master branch with
    no alterations. 
2.  (Optional) Make any necessary changes to the Makefile. Before reverting to the master branch you may want to back
    this up as `Makefile.bak` so it is easier to restore.
3.  Run `Hipify.sh` to produce HIP compatible code from the CUDA.
4.  (Optional)  Edit sizes.h to set the desired lattice size and number of OpenMP threads.
5.  Run make. We recommend using `make -B -f Makefile <target>`. The -B will forcibly rebuild any previous `.o` files
    that may break the build.
