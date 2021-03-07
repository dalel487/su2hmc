#ifndef SLASH
#define SLASH
#include <par_mpi.h>
#include <su2hmc.h>

//D Slash Functions
//=================
int Dslash(complex *phi, complex *r);
int Dslashd(complex *phi, complex *r);
int Hdslash(complex *phi, complex *r);
int Hdslashd(complex *phi, complex *r);

//Device code: Mainly the loops
#ifdef __NVCC__
#include <cuda.h>
//Making things up as I go along here
//Threads are grouped together to form warps of 32 threads
//best to keep the block dimension multiples of 32, usually
//between 128 and 256
//Note that from Volta on that each SM (group of processors)
//is smaller than on previous generations of GPUs
dim3 dimBlock(192,1,1);
dim3 dimGrid(kvol/dimBlock.x,1,1);
__global__ void cuDslash(complex *phi, complex *r);
__global__ void cuDslashd(complex *phi, complex *r);
__global__ void cuHdslash(complex *phi, complex *r);
__global__ void cuHdslashd(complex *phi, complex *r);
__global__ void cuForce(double *dSdpi, complex *X2);
#endif
#endif
