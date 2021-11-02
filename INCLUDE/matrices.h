#ifndef MATRICES
#define MATRICES
#ifdef __NVCC__
#include <cuda.h>
#include <curand.h>
#endif
#include <par_mpi.h>
#include <su2hmc.h>
#ifndef __NVCC__
int Dslash(Complex *phi, Complex *r);
int Dslashd(Complex *phi, Complex *r);
int Hdslash(Complex *phi, Complex *r);
int Hdslashd(Complex *phi, Complex *r);
//Float version
int Dslash_f(Complex_f *phi, Complex_f *r);
int Dslashd_f(Complex_f *phi, Complex_f *r);
int Hdslash_f(Complex_f *phi, Complex_f *r);
int Hdslashd_f(Complex_f *phi, Complex_f *r);
#else
//Extern C flag required to keep CUDA happy
extern "C" int Dslash(Complex *phi, Complex *r);
extern "C" int Dslashd(Complex *phi, Complex *r);
extern "C" int Hdslash(Complex *phi, Complex *r);
extern "C" int Hdslashd(Complex *phi, Complex *r);
//Float version
extern "C" int Dslash_f(Complex_f *phi, Complex_f *r);
extern "C" int Dslashd_f(Complex_f *phi, Complex_f *r);
extern "C" int Hdslash_f(Complex_f *phi, Complex_f *r);
extern "C" int Hdslashd_f(Complex_f *phi, Complex_f *r);

__global__ void cuDslash(Complex *phi, Complex *r);
__global__ void cuDslashd(Complex *phi, Complex *r);
__global__ void cuHdslash(Complex *phi, Complex *r);
__global__ void cuHdslashd(Complex *phi, Complex *r);
__global__ void cuHdslash_f(Complex_f *phi, Complex_f *r);
__global__ void cuHdslashd_f(Complex_f *phi, Complex_f *r);
__global__ void cuNew_trial(double dt);
__global__ inline void cuReunitarise();

//New Trial Fields
#endif
int Reunitarise();
int New_trial(double dt);
#ifdef DIAGNOSTIC
int Diagnostics(int istart);
#endif
#endif
