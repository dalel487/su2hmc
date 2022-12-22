#ifndef SU2HEAD
#define SU2HEAD
#ifdef __NVCC__
#include <cuda.h>
#include	<cublas_v2.h>
extern cublasHandle_t cublas_handle;
extern cublasStatus_t cublas_status;
//Get rid of that dirty yankee English
#define cudaDeviceSynchronise() cudaDeviceSynchronize()
#endif 
//ARM Based machines. BLAS routines should work with other libraries, so we can set a compiler
//flag to sort them out. But the PRNG routines etc. are MKL exclusive
#ifdef	__INTEL_MKL__
#define	USE_BLAS
#include	<mkl.h>
#elif defined GSL_BLAS
#define	USE_BLAS
#include <gsl/gsl_cblas.h>
#elif defined AMD_BLAS
#define	USE_BLAS
#include	<cblas.h>
#endif
#include	<sizes.h>
#ifdef __cplusplus
#include	<cstdio>
#include	<cstdlib>
#include	<ctime>
#else
#include	<stdio.h>
#include	<stdlib.h>
#include	<time.h>
#endif

//Definitions:
//###########
//Function Declarations:
//#####################
#if (defined __cplusplus)
extern "C"
{
#endif
	//	int Force(double *dSdpi, int iflag, double res1);
	int Force(double *dSdpi, int iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,Complex *u11t, Complex *u12t,\
			Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,Complex *gamval,Complex_f *gamval_f,\
			int *gamin,double *dk4m, double *dk4p, float *dk4m_f,float *dk4p_f,Complex_f jqq,\
			float akappa,float beta,double *ancg);
	//	int Gauge_force(double *dSdpi);
	int Gauge_force(double *dSdpi,Complex *u11t, Complex *u12t, unsigned int *iu, unsigned int *id, float beta);
	int Init(int istart, int ibound, int iread, float beta, float fmu, float akappa, Complex_f ajq,\
			Complex *u11, Complex *u12, Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
			Complex *gamval, Complex_f *gamval_f, int *gamin, double *dk4m, double *dk4p, float *dk4m_f, float *dk4p_f,\
			unsigned int *iu, unsigned int *id);
	int Hamilton(double *h, double *s, double res2, double *pp, Complex *X0, Complex *X1, Complex *Phi,\
			Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int * iu, unsigned int *id,\
			Complex_f *gamval_f, int *gamin, float *dk4m_f, float * dk4p_f, Complex_f jqq,\
			float akappa, float beta,double *ancgh);
	//	int Congradq(int na, double res, Complex *smallPhi, int *itercg);
	int Congradq(int na,double res,Complex *X1,Complex *r,Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin,float *dk4m_f,float *dk4p_f,Complex_f jqq,float akappa,int *itercg);
	//	int Congradp(int na, double res, Complex_f *xi_f, int *itercg);
	int Congradp(int na,double res,Complex *Phi,Complex *xi,Complex *u11t,Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval,int *gamin,double *dk4m,double *dk4p,Complex jqq,double akappa,int *itercg);
	//	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg);
	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg,\
			Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int *iu, unsigned int *id,\
			Complex *gamval, Complex_f *gamval_f,	int *gamin, double *dk4m, double *dk4p,\
			float *dk4m_f, float *dk4p_f, Complex jqq, double akappa,	Complex *Phi, Complex *R1);
	int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex *u11t, Complex *u12t, unsigned int *iu, float beta);
	double SU2plaq(Complex *u11t, Complex *u12t, unsigned int *iu, int i, int mu, int nu);
	double Polyakov(Complex *u11t, Complex *u12t);
	//Inline Stuff
	extern int Z_gather(Complex*x, Complex *y, int n, unsigned int *table, unsigned int mu);
	extern int Fill_Small_Phi(int na, Complex *smallPhi, Complex *Phi);

	//CUDA Declarations:
	//#################
#ifdef __NVCC__
	//Not a function. An array of concurrent GPU streams to keep it busy
	extern cudaStream_t streams[ndirac*ndim*nadj];
	//Calling Functions:
	//=================
	void cuAverage_Plaquette(double *hgs, double *hgt, Complex *u11t, Complex *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock);
	void cuPolyakov(Complex *Sigma11, Complex * Sigma12, Complex *u11t, Complex *u12t,dim3 dimGrid, dim3 dimBlock);
	void cuGauge_force(int mu,Complex *Sigma11, Complex *Sigma12, Complex *u11t,Complex *u12t,double *dSdpi,float beta,\
			dim3 dimGrid, dim3 dimBlock);
	void cuPlus_staple(int mu, int nu, unsigned int *iu, Complex *Sigma11, Complex *Sigma12, Complex *u11t, Complex *u12t,\
			dim3 dimGrid, dim3 dimBlock);
	void cuMinus_staple(int mu, int nu, unsigned int *iu, unsigned int *id, Complex *Sigma11, Complex *Sigma12,\
			Complex *u11sh, Complex *u12sh,Complex *u11t, Complex*u12t,	dim3 dimGrid, dim3 dimBlock);
	void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, \
			Complex *gamval,double *dk4m, double *dk4p,unsigned int *iu,int *gamin,\
			float akappa, dim3 dimGrid, dim3 dimBlock);
	//cuInit was taken already by CUDA (unsurprisingly)
	void	Init_CUDA(Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, Complex *gamval,\
			Complex_f *gamval_f, int *gamin,\
			double *dk4m, double *dk4p, float *dk4m_f, float *dk4p_f, unsigned int *iu, unsigned int *id);
	//			dim3 *dimBlock, dim3 *dimGrid);
#endif
#if (defined __cplusplus)
}
#endif
//CUDA Kernels:
//============
#ifdef __CUDACC__
//__global__ void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
//		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa);
__global__ void Plus_staple(int mu, int nu,unsigned int *iu, Complex *Sigma11, Complex *Sigma12, Complex *u11t, Complex *u12t);
__global__ void Minus_staple(int mu, int nu,unsigned int *iu,unsigned int *id, Complex *Sigma11, Complex *Sigma12,\
		Complex *u11sh, Complex *u12sh, Complex *u11t, Complex *u12t);
__global__ void cuGaugeForce(int mu, Complex *Sigma11, Complex *Sigma12,double*dSdpi,Complex *u11t, Complex *u12t, float beta);
__global__ void cuAverage_Plaquette(double *hgs, double *hgt, Complex *u11t, Complex *u12t, unsigned int *iu);
__global__ void cuPolyakov(Complex *Sigma11, Complex * Sigma12, Complex *u11t, Complex *u12t);
__device__ double SU2plaq(Complex *u11t, Complex *u12t, unsigned int *iu, int i, int mu, int nu);
//Force Kernels. We've taken each nadj index and the spatial/temporal components and created a separate kernel for each
//CPU code just has these as a huge blob that the vectoriser can't handle. May be worth splitting it there too?
//It might not be a bad idea to make a seperate header for all these kernels...
__global__ void cuForce_s0(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa, int idirac, int mu);
__global__ void cuForce_s1(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa, int idirac, int mu);
__global__ void cuForce_s2(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa, int idirac, int mu);
__global__ void cuForce_t0(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa, int idirac);
__global__ void cuForce_t1(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa, int idirac);
__global__ void cuForce_t2(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa, int idirac);
#endif
#endif
