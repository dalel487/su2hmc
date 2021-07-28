#ifndef MULTIPLY
#define MULTIPLY 
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_complex.hpp>
#include <curand.h>
#endif
#include <par_mpi.h>
#include <su2hmc.h>

//Device code: Mainly the loops
//Making things up as I go along here
//Threads are grouped together to form warps of 32 threads
//best to keep the block dimension (ksizex*ksizey) multiples of 32,
//usually between 128 and 256
//Note that from Volta/Turing  each SM (group of processors)
//is smaller than on previous generations of GPUs
int Dslash(Complex *phi, Complex *r);
int Dslashd(Complex *phi, Complex *r);
int Hdslash(Complex *phi, Complex *r);
int Hdslashd(Complex *phi, Complex *r);
//Float version
int Hdslash_f(Complex_f *phi, Complex_f *r);
int Hdslashd_f(Complex_f *phi, Complex_f *r);
extern inline int Reunitarise();

#ifdef __CUDACC__
__global__ void cuDslash(Complex *phi, Complex *r);
__global__ void cuDslashd(Complex *phi, Complex *r);
__global__ void cuHdslash(Complex *phi, Complex *r);
__global__ void cuHdslashd(Complex *phi, Complex *r);
__global__ void cuHdslash_f(Complex_f *phi, Complex_f *r);
__global__ void cuHdslashd_f(Complex_f *phi, Complex_f *r);
__global__ inline void cuReunitarise();

//New Trial Fields
__global__ void New_trial(double dt);
#else
//New Trial Fields
int New_trial();
#ifdef DIAGNOSTIC
int Diagnostics(int istart);
#endif
#endif
#endif
