#ifndef MULTIPLY
#define MULTIPLY 
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_complex.hpp>
#include <curand.h>
extern "C"
{
#endif
#include <par_mpi.h>
#include <su2hmc.h>

	int Dslash(Complex *phi, Complex *r);
	int Dslashd(Complex *phi, Complex *r);
	int Hdslash(Complex *phi, Complex *r);
	int Hdslashd(Complex *phi, Complex *r);
	//Float version
	int Hdslash_f(Complex_f *phi, Complex_f *r);
	int Hdslashd_f(Complex_f *phi, Complex_f *r);
#ifdef __CUDACC__
}
__global__ void cuDslash(Complex *phi, Complex *r);
__global__ void cuDslashd(Complex *phi, Complex *r);
__global__ void cuHdslash(Complex *phi, Complex *r);
__global__ void cuHdslashd(Complex *phi, Complex *r);
__global__ void cuHdslash_f(Complex_f *phi, Complex_f *r);
__global__ void cuHdslashd_f(Complex_f *phi, Complex_f *r);
__global__ void cuNew_trial(double dt);
__global__ inline void cuReunitarise();

//New Trial Fields
extern "C"
{
#endif
	int Reunitarise();
	int New_trial(double dt);
#ifdef DIAGNOSTIC
	int Diagnostics(int istart);
#endif
#ifdef __CUDACC__
}
#endif
#endif
