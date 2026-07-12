HIP Conversion notes
====================
1.  Certain CUDA keywords are totally absent from HIP. Like '__managed__', '__forceinline__', '__grid_constant__' and
    '__constant__'. These need to be removed with sed or perl before compiling, or removed from the CUDA code.
2.  There is a bug with the reduction routine. Tis is currently only called by a debugging path though so for now I'm
    commenting out all calls, declarations and definitions of 'reduce_sum' until I can find a solution.
3.  HIP is clang based so a lot stricter about certain coding practices. For examplie there were '#pragma omp simd'
    calls before GetBilinear. The file isn't compilied with -fopenmp so CUDA ignored them completely. HIP freaks out. A
    lot of these are now fixed though.
