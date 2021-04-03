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
//best to keep the block dimension multiples of 32, usually
//between 128 and 256
//Note that from Volta on that each SM (group of processors)
//is smaller than on previous generations of GPUs
int Dslash(cuComplex *phi, cuComplex *r);
int Dslashd(cuComplex *phi, cuComplex *r);
int Hdslash(cuComplex *phi, cuComplex *r);
int Hdslashd(cuComplex *phi, cuComplex *r);
__global__ void cuDslash(cuComplex *phi, cuComplex *r);
__global__ void cuDslashd(cuComplex *phi, cuComplex *r);
__global__ void cuHdslash(cuComplex *phi, cuComplex *r);
__global__ void cuHdslashd(cuComplex *phi, cuComplex *r);
__global__ void cuForce(double *dSdpi, cuComplex *X2);

#else
//D Slash Functions
//=================
int Dslash(complex *phi, complex *r);
int Dslashd(complex *phi, complex *r);
int Hdslash(complex *phi, complex *r);
int Hdslashd(complex *phi, complex *r);
#endif
#endif
