#ifndef MATRICES
#define MATRICES
#ifdef __NVCC__
#include <curand.h>
#endif
#include <par_mpi.h>
#include <su2hmc.h>
#if (defined __cplusplus)
extern "C"
{
#endif
	int Dslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
			Complex gamval[5][4], int gamin[4][4],	double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	int Dslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex gamval[5][4], int gamin[4][4],	double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	int Hdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
			Complex gamval[5][4], int gamin[4][4],	double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	int Hdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex gamval[5][4], int gamin[4][4],double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	//Float version
	int Dslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f gamval_f[5][4],int gamin[4][4],	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f);
	int Dslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f gamval_f[5][4],int gamin[4][4],	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f);
	int Hdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f gamval_f[5][4],int gamin[4][4],	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f);
	int Hdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f gamval_f[5][4],int gamin[4][4],	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f);
	//New Trial Fields
	int Reunitarise(Complex *u11t, Complex *u12t);
	int New_trial(double dt, double *pp, Complex *u11t, Complex *u12t);
#ifdef DIAGNOSTIC
	int Diagnostics(int istart, Complex *u11, Complex *u12,Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
			unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk4m, double *dk4p,\
			float *dk4m_f, float *dk4p_f, int *gamin, Complex *gamval, Complex_f *gamval_f,\
			Complex_f jqq,float akappa, float beta, double ancg);
#endif
#ifdef __NVCC__
	//Calling Functions
	void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin,double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	//Float version
	void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	//New Trial Fields
	void cuReunitarise(Complex *u11t, Complex *u12t,dim3 dimGrid, dim3 dimBlock);
	void cuNew_trial(double dt, double *pp, Complex *u11t, Complex *u12t, dim3 dimGrid, dim3 dimBlock);
#endif
#if (defined __cplusplus)
}
#endif
#ifdef __CUDACC__
//Actual CUDA
__global__ void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa);
__global__ void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa);
__global__ void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa);
__global__ void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa);
//Float version
__global__ void cuDslash0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa);
__global__ void cuDslashd0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa);
__global__ void cuDslash1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa);
__global__ void cuDslashd1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa);

__global__ void cuHdslash0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa,int idirac);
__global__ void cuHdslashd0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa,int idirac);
__global__ void cuHdslash1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa,int idirac);
__global__ void cuHdslashd1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa,int idirac);
//New Trial Fields
__global__ void cuReunitarise(Complex *u11t, Complex *u12t);
__global__ void cuNew_trial(double dt, double *pp, Complex *u11t, Complex *u12t, int mu);
#endif
#endif
