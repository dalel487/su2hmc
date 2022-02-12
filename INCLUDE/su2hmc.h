#ifndef SU2HEAD
#define SU2HEAD
#ifdef __NVCC__
#include <cuda.h>
#include	<cublas_v2.h>
cublasHandle_t cublas_handle;
cublasHandle_t cublas_status;
//Get rid of that dirty yankee English
#define cudaDeviceSynchronise() cudaDeviceSynchronize()
#endif 
#include	<fields.h>
//ARM Based machines.) BLAS routines should work with other libraries, so we can set a compiler
//flag to sort them out. But the PRNG routines etc. are MKL exclusive
#ifdef	__INTEL_MKL__
#include	<mkl.h>
#elif defined USE_BLAS
#include	<cblas.h>
#endif
#include	<stdio.h>
#include	<stdlib.h>
#include	<sizes.h>
#include	<time.h>

//Definitions:
//###########
//Variables:
//#########
int ibound; 
//Arrays:
//------
//We have the four γ Matrices, and in the final index (labelled 4 in C) is γ_5)
#ifdef __NVCC__
__device__ extern Complex *gamval_d;
#else
__attribute__((aligned(AVX)))
#endif 
	extern Complex gamval[5][4];
#ifdef __NVCC__
	__device__ extern Complex_f *gamval_f_d;
#else
__attribute__((aligned(AVX)))
#endif 
	extern Complex_f gamval_f[5][4];
	//Seems a bit redundant looking
#ifdef __NVCC__
	__managed__ extern int *gamin_d;
	__managed__ 
#endif 
	extern int 
#ifndef __NVCC__ 
__attribute__((aligned(AVX)))
#endif
	gamin[4][4];

	//From common_trial_u11u12
	//complex *u11, *u12;
	//double pp[kvol+halo][nadj][ndim] __attribute__((aligned(AVX)));

	//Values:
	//------
	//The diquark
#ifdef __NVCC__
	__device__ extern Complex *jqq_d;
	__device__ extern Complex *jqq_f_d;
#endif 
	extern Complex jqq;
	extern Complex_f jqq_f;

	//Average # of congrad iter guidance and acceptance
	double ancg, ancgh;
#ifdef __NVCC__
	__device__ extern double *beta_d, *akappa_d;
#endif 
	extern double fmu, beta, akappa;
#ifdef __NVCC__
	__device__ extern float *akappa_f_d;
#endif 
	extern float akappa_f;

	//Function Declarations:
	//#####################
#ifdef __cplusplus
	extern "C"
{
#endif
#ifndef __NVCC__
	int Force(double *dSdpi, int iflag, double res1);
	int Gauge_force(double *dSdpi);
#else
	int Force(double *dSdpi, int iflag, double res1);
	int Gauge_force(double *dSdpi);
#endif
	int Init(int istart, int iread, double beta, double fmu, double akappa, Complex ajq);
	int Hamilton(double *h, double *s, double res2);
	int Congradq(int na, double res, Complex *smallPhi, int *itercg);
	int Congradp(int na, double res, Complex_f *xi_f, int *itercg);
	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg);
#ifndef __NVCC__
	int SU2plaq(double *hg, double *avplaqs, double *avplaqt, Complex *u11t, Complex *u12t, int *iu);
	double Polyakov(Complex * u11t, Complex *u12t);
#else
	int SU2plaq(double *hg, double *avplaqs, double *avplaqt);
	double Polyakov();
#endif
	//Inline Stuff
	extern int Z_gather(Complex*x, Complex *y, int n, unsigned int *table, unsigned int mu);
	extern int Fill_Small_Phi(int na, Complex *smallPhi);
#ifdef __cplusplus
}
#endif

//CUDA Declarations:
//#################
#ifdef __NVCC__
__global__ void cuForce(double *dSdpi, Complex *X2);
__global__ void Plus_staple(int mu, int nu, Complex *Sigma11, Complex *Sigma12);
__global__ void Minus_staple(int mu, int nu, Complex *Sigma11, Complex *Sigma12, Complex *u11sh, Complex *u12sh);
__global__ void cuGaugeForce(int mu, Complex *Sigma11, Complex *Sigma12, double * dSdpi);
__global__ void cuSU2plaq(int mu, int nu, double *hgs, double *hgt);
__global__ void cuPolyakov(Complex *Sigma11, Complex * Sigma12);
#endif
#endif
