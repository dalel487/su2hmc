#ifndef SU2HEAD
#define SU2HEAD
#ifdef __NVCC__
#include <cuda.h>
#include	<cublas_v2.h>
cublasHandle_t cublas_handle;
cublasHandle_t cublas_status;
#endif 
#ifdef __CUDACC__
#include <cuda_complex.hpp>
#undef	complex
#define	Complex_f	 complex<float>
#define	Complex	 complex<double>
#else
#include	<complex.h>
#define Complex_f	float	complex
#define Complex	complex
#endif
//MKL is powerful, but not guaranteed to be available (especially on AMD systems or future
//ARM Based machines.) BLAS routines should work with other libraries, so we can set a compiler
//flag to sort them out. But the PRNG routines etc. are MKL exclusive
#ifdef	USE_MKL
#include	<mkl.h>
#elif defined __NVCC__
#include	<cublas.h>
#elif defined USE_BLAS
#include	<cblas.h>
#endif
#include	<sizes.h>

//Definitions:
//###########
//Variables:
//#########
#ifdef __cplusplus
extern "C"
{
#endif
int ibound; 
//Arrays:
//------
//Seems a bit redundant looking
#ifdef __NVCC__
__managed__ int *gamin_d;
__managed__ 
#endif 
extern int gamin[4][4];
//We have the four γ Matrices, and in the final index (labelled 4 in C) is γ_5)
#ifdef __NVCC__
__device__ Complex *gamval_d;
#endif 
extern Complex gamval[5][4];
#ifdef __NVCC__
__device__ Complex_f *gamval_f_d;
#endif 
extern Complex_f gamval_f[5][4];

//From common_pseud
#ifdef __NVCC__
__managed__ 
#endif 
Complex *Phi, *R1, *X0, *X1, *xi;
//From common_mat
#ifdef __NVCC__
__managed__ 
#endif 
double *dk4m, *dk4p, *pp;
#ifdef __NVCC__
__managed__ 
#endif 
float	*dk4m_f, *dk4p_f;
//From common_trial_u11u12
//complex *u11, *u12;
//double pp[kvol+halo][nadj][ndim] __attribute__((aligned(AVX)));

//Values:
//------
//The diquark
#ifdef __NVCC__
__device__ Complex *jqq_d;
#endif 
extern Complex jqq;

//Average # of congrad iter guidance and acceptance
double ancg, ancgh;
#ifdef __NVCC__
__device__ double *beta_d, *akappa_d;
#endif 
extern double fmu, beta, akappa;
#ifdef __NVCC__
__device__ float *akappa_f_d;
#endif 
extern float akappa_f;

//Function Declarations:
//#####################
int Force(double *dSdpi, int iflag, double res1);
int Init(int istart);
int Gauge_force(double *dSdpi);
int Hamilton(double *h, double *s, double res2);
int Congradq(int na, double res, Complex *smallPhi, int *itercg);
int Congradp(int na, double res, int *itercg);
int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg);
int SU2plaq(double *hg, double *avplaqs, double *avplaqt);
double Polyakov();
//Inline Stuff
extern int Z_gather(Complex*x, Complex *y, int n, unsigned int *table, unsigned int mu);
extern int Fill_Small_Phi(int na, Complex *smallPhi);

//CUDA Declarations:
//#################
#ifdef __CUDACC__
}
__global__ void cuForce(double *dSdpi, Complex *X2);
__global__ void Plus_staple(int mu, int nu, Complex *Sigma11, Complex *Sigma12);
__global__ void Minus_staple(int mu, int nu, Complex *Sigma11, Complex *Sigma12, Complex *u11sh, Complex *u12sh);
__global__ void cuGaugeForce(int mu, Complex *Sigma11, Complex *Sigma12, double * dSdpi);
__global__ void cuSU2plaq(int mu, int nu, double *hgs, double *hgt);
__global__ void cuPolyakov(Complex *Sigma11, Complex * Sigma12);
#endif
#endif
