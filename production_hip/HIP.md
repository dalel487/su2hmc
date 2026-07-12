HIP
===
We have made progress in getting the code ready for HIP. However it is not fully tested yet and we make no guarantees on
correctness at this point.

HIP Conversion Progress
-----------------------
-   We can run on $8^3\times 8$ on a Radeon Pro W6400 which isn't officially suppported. $24^3\times 24$ gives incorrect
    results. We suspect this is due to the lack of xnack on the testbed GPU but until we run on officially supported
    ones we cannot be certain.

-   Debug and profiling modes do not work fully yet as far as we can tell. We are unsure if this is an xnack issue but
    cannot test it at this time (20260712).

Building for HIP
================
Building for HIP is currently a three to five step process

1.  To ensure the original cuda source code is uncorrupted, we recommend first checking out the latest master branch with
    no alterations. 
2.  (Optional) Make any necessary changes to the Makefile. Before reverting to the master branch you may want to back
    this up as 'Makefile.bak' so it is easier to restore.
3.  Run 'Hipify.sh' to produce HIP compatible code from the CUDA.
4.  (Optional)  Edit sizes.h to set the desired lattice size and number of OpenMP threads.
5.  Run make. We recommend using 'make -B -f Makefile <target>'. The -B will remove any previous '.o' files that may
    break the build.
