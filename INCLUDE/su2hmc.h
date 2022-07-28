#ifndef SU2HEAD
#define SU2HEAD
#ifdef __NVCC__
#include <cuda.h>
#define	USE_BLAS
#include	<cublas_v2.h>
extern cublasHandle_t cublas_handle;
extern cublasHandle_t cublas_status;
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
#include	<stdio.h>
#include	<stdlib.h>
#include	<sizes.h>
#include	<time.h>

//Definitions:
//###########
//Function Declarations:
//#####################
#if (defined __NVCC__ || defined __cplusplus)
extern "C"
{
#endif
	//	int Force(double *dSdpi, int iflag, double res1);
	int Force(double *dSdpi, int iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,Complex *u11t, Complex *u12t,\
			Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,Complex gamval[5][4],Complex_f gamval_f[5][4],\
			int gamin[4][4],double *dk4m, double *dk4p, float *dk4m_f,float *dk4p_f,Complex_f jqq,\
			float akappa,float beta,double *ancg);
	//	int Gauge_force(double *dSdpi);
	int Gauge_force(double *dSdpi,Complex *u11t, Complex *u12t, unsigned int *iu, unsigned int *id, float beta);
	int Init(int istart, int ibound, int iread, float beta, float fmu, float akappa, Complex_f ajq,\
			Complex *u11, Complex *u12, Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
			double *dk4m, double *dk4p, float *dk4m_f, float *dk4p_f, unsigned int *iu, unsigned int *id);
	int Hamilton(double *h, double *s, double res2, double *pp, Complex *X0, Complex *X1, Complex *Phi,\
			Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int * iu, unsigned int *id,\
			Complex_f gamval_f[5][4], int gamin[4][4], float *dk4m_f, float * dk4p_f, Complex_f jqq,\
			float akappa, float beta,double *ancgh);
	//	int Congradq(int na, double res, Complex *smallPhi, int *itercg);
	int Congradq(int na,double res,Complex *X1,Complex *r,Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f gamval_f[5][4],int gamin[4][4],float *dk4m_f,float *dk4p_f,Complex_f jqq,float akappa,int *itercg);
	//	int Congradp(int na, double res, Complex_f *xi_f, int *itercg);
	int Congradp(int na,double res,Complex *Phi,Complex *xi,Complex *u11t,Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex gamval[5][4],int gamin[4][4],double *dk4m,double *dk4p,Complex jqq,double akappa,int *itercg);
	//	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg);
	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg,\
			Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int *iu, unsigned int *id,\
			Complex gamval[5][4], Complex_f gamval_f[5][4],	int gamin[4][4], double *dk4m, double *dk4p,\
			float *dk4m_f, float *dk4p_f, Complex jqq, double akappa,	Complex *Phi, Complex *R1);
	int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex *u11t, Complex *u12t, unsigned int *iu, float beta);
	double SU2plaq(Complex *u11t, Complex *u12t, unsigned int *iu, int i, int mu, int nu);
	double Polyakov(Complex *u11t, Complex *u12t);
	//Inline Stuff
	extern int Z_gather(Complex*x, Complex *y, int n, unsigned int *table, unsigned int mu);
	extern int Fill_Small_Phi(int na, Complex *smallPhi, Complex *Phi);
#if (defined __NVCC__ || defined __cplusplus)
}
#endif

//CUDA Declarations:
//#################
#ifdef __NVCC__
__global__ void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa);
__global__ void Plus_staple(int mu, int nu,unsigned int *iu, Complex *Sigma11, Complex *Sigma12, Complex *u11t, Complex *u12t);
__global__ void Minus_staple(int mu, int nu,unsigned int *iu,unsigned int *id, Complex *Sigma11, Complex *Sigma12,\
		Complex *u11sh, Complex *u12sh, Complex *u11t, Complex *u12t);
__global__ void cuGaugeForce(int mu, Complex *Sigma11, Complex *Sigma12,double*dSdpi,Complex *u11t, Complex *u12t, float beta);
//Calling Functions:
//=================
void SU2plaq(double *hgs, double *hgt, Complex *u11t, Complex *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock);
__global__ void cuSU2plaq(double *hgs, double *hgt, Complex *u11t, Complex *u12t, unsigned int *iu);
void Polyakov(Complex *Sigma11, Complex * Sigma12, Complex *u11t, Complex *u12t,dim3 dimGrid, dim3 dimBlock);
__global__ void cuPolyakov(Complex *Sigma11, Complex * Sigma12, Complex *u11t, Complex *u12t);
void Gauge_force(int mu,Complex *Sigma11, Complex *Sigma12, Complex *u11t,Complex *u12t,double *dSdpi,float beta,\
		dim3 dimGrid, dim3 dimBlock);
void Plus_staple(int mu, int nu, unsigned int *iu, Complex *Sigma11, Complex *Sigma12, Complex *u11t, Complex *u12t,\
		dim3 dimGrid, dim3 dimBlock);
void Minus_staple(int mu, int nu, unsigned int *iu, unsigned int *id, Complex *Sigma11, Complex *Sigma12,\
		Complex *u11sh, Complex *u12sh,Complex *u11t, Complex*u12t,	dim3 dimGrid, dim3 dimBlock);
void Force(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, \
		Complex gamval[5][4],double *dk4m, double *dk4p,unsigned int *iu,int gamin[4][4],\
		float akappa, dim3 dimGrid, dim3 dimBlock);
#endif
#endif
