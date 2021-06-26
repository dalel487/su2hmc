#ifndef MULTIPLY
#define MULTIPLY 
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_complex.hpp>
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
int Dslash(complex *phi, complex *r);
int Dslashd(complex *phi, complex *r);
int Hdslash(complex *phi, complex *r);
int Hdslashd(complex *phi, complex *r);
__global__ void cuDslash(complex *phi, complex *r);
__global__ void cuDslashd(complex *phi, complex *r);
__global__ void cuHdslash(complex *phi, complex *r);
__global__ void cuHdslashd(complex *phi, complex *r);
__global__ void cuForce(double *dSdpi, complex *X2);

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
