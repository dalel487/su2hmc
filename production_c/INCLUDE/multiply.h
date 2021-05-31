#ifndef MULTIPLY
#define MULTIPLY 
#ifdef __NVCC__
#include <cuda.h>
#include <cuComplex.h>
#include <curand.h>
#endif
#include <par_mpi.h>
#include <su2hmc.h>

//Device code: Mainly the loops
#ifdef __NVCC__
//Making things up as I go along here
//Threads are grouped together to form warps of 32 threads
//best to keep the block dimension (ksizex*ksizey) multiples of 32,
//usually between 128 and 256
//Note that from Volta/Turing  each SM (group of processors)
//is smaller than on previous generations of GPUs
int Dslash(cuDoubleComplex *phi, cuDoubleComplex *r);
int Dslashd(cuDoubleComplex *phi, cuDoubleComplex *r);
int Hdslash(cuDoubleComplex *phi, cuDoubleComplex *r);
int Hdslashd(cuDoubleComplex *phi, cuDoubleComplex *r);
__global__ void cuDslash(cuDoubleComplex *phi, cuDoubleComplex *r);
__global__ void cuDslashd(cuDoubleComplex *phi, cuDoubleComplex *r);
__global__ void cuHdslash(cuDoubleComplex *phi, cuDoubleComplex *r);
__global__ void cuHdslashd(cuDoubleComplex *phi, cuDoubleComplex *r);
__global__ void cuForce(double *dSdpi, cuDoubleComplex *X2);

//New Trial Fields
__global__ void New_trial(double dt);
#else
//D Slash Functions
//=================
int Dslash(complex *phi, complex *r);
int Dslashd(complex *phi, complex *r);
int Hdslash(complex *phi, complex *r);
int Hdslashd(complex *phi, complex *r);

//New Trial Fields
int New_trial();

#ifdef DIAGNOSTIC
int Diagnostics(int istart);
#endif
#endif
#endif
