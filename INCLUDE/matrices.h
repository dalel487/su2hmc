/**
 * @file matrices.h
 *
 * @brief Matrix multiplication and related declarations
 */
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
	/**
	 * @brief Evaluates @f(\Phi=M r@f) in double precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk4m:		
	 *	@param	dk4p:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in double precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk4m:		
	 *	@param	dk4p:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f(\Phi=M r@f) in double precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk4m:	
	 *	@param	dk4p:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa);
	/**
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in double precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk4m:	
	 *	@param	dk4p:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa);
	//Float version
	/**
	 * @brief Evaluates @f(\Phi=M r@f) in single precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk4m:		
	 *	@param	dk4p:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval,int *gamin, float *dk4m, float *dk4p, Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in single precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk4m:		
	 *	@param	dk4p:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval,int *gamin, float *dk4m, float *dk4p, Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f(\Phi=M r@f) in single precision.
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk4m:	
	 *	@param	dk4p:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, float akappa);
	/**
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in single precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk4m:	
	 *	@param	dk4p:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval,int *gamin, float *dk4m, float *dk4p, float akappa);
	/**
	 * @brief Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * @see cuReunitarise (CUDA Wrapper)
	 *
	 * @param u11t, u12t Trial fields to be reunitarised
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Reunitarise(Complex *u11t, Complex *u12t);
	/**
	 * @brief Generates new trial fields
	 *
	 * @see cuNew_trial (CUDA Wrapper)
	 * 
	 * @param	dt:		Half lattice spacing
	 * @param	pp:		Momentum field
	 * @param	u11t:		First colour field
	 * @param	u12t:		Second colour field
	 *
	 * @returns	Zero on success, integer error code otherwise
	 */
	int New_trial(double dt, double *pp, Complex *u11t, Complex *u12t);
#ifdef DIAGNOSTIC
	int Diagnostics(int istart, Complex *u11, Complex *u12,Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
			unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk4m, double *dk4p,\
			float *dk4m_f, float *dk4p_f, int *gamin, Complex *gamval, Complex_f *gamval_f,\
			Complex_f jqq, float akappa, float beta, double ancg);
#endif
#ifdef __NVCC__
	//Calling Functions
	void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa,dim3 dimGrid, dim3 dimBlock);
	void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa, dim3 dimGrid, dim3 dimBlock);
	//Float version
	void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f, int *gamin, float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin, float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin, float *dk4m_f, float *dk4p_f, float akappa_f,dim3 dimGrid, dim3 dimBlock);
	void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin, float *dk4m_f, float *dk4p_f, float akappa_f, dim3 dimGrid, dim3 dimBlock);
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
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
__global__ void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
__global__ void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa);
__global__ void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa);
//Float version
__global__ void cuDslash0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, Complex_f jqq, float akappa);
__global__ void cuDslashd0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, Complex_f jqq, float akappa);
__global__ void cuDslash1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, Complex_f jqq, float akappa);
__global__ void cuDslashd1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, Complex_f jqq, float akappa);

__global__ void cuHdslash0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, float akappa,int idirac);
__global__ void cuHdslashd0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, float akappa,int idirac);
__global__ void cuHdslash1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, float akappa,int idirac);
__global__ void cuHdslashd1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, float akappa,int idirac);
//New Trial Fields
__global__ void cuReunitarise(Complex *u11t, Complex *u12t);
__global__ void cuNew_trial(double dt, double *pp, Complex *u11t, Complex *u12t, int mu);
#endif
#endif
