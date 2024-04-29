/**
 * @file		su2hmc.h
 * @brief	Function declarations for most of the routines
 */
#ifndef SU2HEAD
#define SU2HEAD
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
#ifdef _DEBUGCG
#define _DEBUG
#endif
//Function Declarations:
//#####################
#if (defined __cplusplus)
extern "C"
{
#endif
	/**
	 *	@brief Calculates the force @f$\frac{dS}{d\pi}@f$ at each intermediate time
	 *	
	 *	@param	dSdpi:			The force
	 *	@param	iflag:			Invert before evaluating the force?	
	 *	@param	res1:				Conjugate gradient residule
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 *	@param	u11t,u12t		Double precision colour fields
	 *	@param	u11t_f,u12t_f:	Single precision colour fields
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices
	 *	@param	gamval_f:		Single precision gamma matrices
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk4m_f:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ float
	 * @param	dk4p_f:			@f$\left(1-\gamma_0\right)e^\mu@f$ float
	 * @param 	jqq:				Diquark source
	 *	@param	akappa:			Hopping parameter
	 *	@param	beta:				Inverse gauge coupling
	 *	@param	ancg:				Counter for conjugate gradient iterations
	 *
	 *	@return Zero on success, integer error code otherwise
	 */
	int Force(double *dSdpi, int iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,Complex *u11t, Complex *u12t,\
			Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,Complex *gamval,Complex_f *gamval_f,\
			int *gamin,double *dk4m, double *dk4p, float *dk4m_f,float *dk4p_f,Complex_f jqq,\
			float akappa,float beta,double *ancg);
	/**
	 * @brief	Calculates the gauge force due to the Wilson Action at each intermediate time
	 *
	 * @param	dSdpi:		The force
	 *	@param	u11t,u12t:	Gauge fields
	 * @param	iu,id:		Lattice indices 
	 * @param	beta:			Inverse gauge coupling
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Gauge_force(double *dSdpi,Complex_f *u11t, Complex_f *u12t, unsigned int *iu, unsigned int *id, float beta);
	/**
	 * @brief Initialises the system
	 *
	 * @param	istart:				Zero for cold, >1 for hot, <1 for none
	 * @param	ibound:				Periodic boundary conditions
	 * @param	iread:				Read configuration from file
	 * @param	beta:					Inverse gauge coupling
	 * @param	fmu:					Chemical potential
	 * @param	akappa:				Hopping parameter
	 * @param	ajq:					Diquark source
	 * @param	u11, u12:			Gauge fields
	 * @param	u11t, u12t:			Trial gauge field
	 * @param	u11t_f, u12t_f:	Trial gauge field (single precision)
	 * @param	dk4m					@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:					@f$\left(1-\gamma_0\right)^\mu@f$
	 * @param	dk4m_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ float 	
	 * @param	dk4p_f:				@f$\left(1-\gamma_0\right)e^\mu@f$ float 	
	 * @param	iu,id:				Up halo indices
	 * @param	gamin:				Gamma matrix indices
	 * @param	gamval:				Gamma Matrices
	 * @param	gamval_f:			Float Gamma matrices:
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Init(int istart, int ibound, int iread, float beta, float fmu, float akappa, Complex_f ajq,\
			Complex *u11, Complex *u12, Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
			Complex *gamval, Complex_f *gamval_f, int *gamin, double *dk4m, double *dk4p, float *dk4m_f, float *dk4p_f,\
			unsigned int *iu, unsigned int *id);
	/**
	 * @brief Calculate the Hamiltonian
	 *
	 * @param	h:						Hamiltonian
	 * @param	s:						Action
	 * @param	res2:					Limit for conjugate gradient
	 * @param	pp:					Momentum field
	 *	@param	X0:					Up/down partitioned pseudofermion field
	 *	@param	X1:					Holder for the partitioned fermion field, then the conjugate gradient output
	 * @param	Phi:					Pseudofermion field
	 * @param	u11t,u12t:			Gauge fields
	 * @param	u11t_f,u12t_f:		Gauge fields (single precision)
	 * @param	iu,id:				Lattice indices
	 * @param	gamval_f:			Gamma matrices
	 * @param	gamin:				Gamma indices
	 * @param	dk4m_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ float
	 * @param	dk4p_f:				@f$\left(1-\gamma_0\right)e^\mu@f$ float
	 * @param	jqq:					Diquark source
	 * @param	akappa:				Hopping parameter
	 * @param	beta:					Inverse gauge coupling
	 * @param	ancgh:				Conjugate gradient iterations counter 
	 * @param	traj:					Calling trajectory for error reporting
	 *
	 * @return	Zero on success. Integer Error code otherwise.
	 */	
	int Hamilton(double *h, double *s, double res2, double *pp, Complex *X0, Complex *X1, Complex *Phi,\
			Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int * iu, unsigned int *id,\
			Complex_f *gamval_f, int *gamin, float *dk4m_f, float * dk4p_f, Complex_f jqq,\
			float akappa, float beta,double *ancgh,int traj);
	/**
	 * @brief Matrix Inversion via Conjugate Gradient (up/down flavour partitioning).
	 * Solves @f$(M^\dagger)Mx=\Phi@f$
	 * Implements up/down partitioning
	 * The matrix multiplication step is done at single precision, while the update is done at double
	 *
	 * @param	na:			Flavour index
	 * @param	res:			Limit for conjugate gradient
	 * @param	X1:			Pseudofermion field @f$\Phi@f$ initially, returned as @f$(M^\dagger M)^{-1} \Phi@f$
	 * @param	r:				Partition of @f$\Phi@f$ being used. Gets recycled as the residual vector
	 * @param	u11t_f:		First colour's trial field
	 * @param	u12t_f:		Second colour's trial field
	 * @param	iu:			Upper halo indices
	 * @param	id:			Lower halo indices
	 * @param	gamval_f:	Gamma matrices
	 * @param	gamin:		Dirac indices
	 * @param	dk4m_f:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p_f:		@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	jqq:			Diquark source
	 * @param	akappa:		Hopping Parameter
	 * @param	itercg:		Counts the iterations of the conjugate gradient
	 *
	 * @return 0 on success, integer error code otherwise
	 */
	int Congradq(int na,double res,Complex *X1,Complex *r,Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin,float *dk4m_f,float *dk4p_f,Complex_f jqq,float akappa,int *itercg);
	/**
	 * @brief Matrix Inversion via Conjugate Gradient (no up/down flavour partitioning).
	 * Solves @f$(M^\dagger)Mx=\Phi@f$
	 * The matrix multiplication step is done at single precision, while the update is done at double
	 *
	 * @param 	na:			Flavour index
	 * @param 	res:			Limit for conjugate gradient
	 * @param 	Phi:			Pseudofermion field.
	 * @param 	xi:			Returned as @f$(M^\dagger M)^{-1} \Phi@f$
	 * @param 	u11t:			First colour's trial field
	 * @param 	u12t:			Second colour's trial field
	 * @param 	iu:			Upper halo indices
	 * @param 	id:			Lower halo indices
	 * @param 	gamval:		Gamma matrices
	 * @param 	gamin:		Dirac indices
	 * @param	dk4m:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:			@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param 	jqq:			Diquark source
	 * @param 	akappa:		Hopping Parameter
	 * @param 	itercg:		Counts the iterations of the conjugate gradient
	 * 
	 * @return 0 on success, integer error code otherwise
	 */
	int Congradp(int na,double res,Complex *Phi,Complex *xi,Complex_f *u11t,Complex_f *u12t,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval,int *gamin,float *dk4m,float *dk4p,Complex_f jqq,float akappa,int *itercg);
	/**
	 * @brief	Calculate fermion expectation values via a noisy estimator
	 * 
	 * Matrix inversion via conjugate gradient algorithm
	 * Solves @f$MX=X_1@f$
	 * (Numerical Recipes section 2.10 pp.70-73)   
	 * uses NEW lookup tables **
	 * Implemented in Congradp()
	 *
	 * @param	pbp:				@f$\langle\bar{\Psi}\Psi\rangle@f$
	 *	@param	endenf:			Energy density
	 *	@param	denf:				Number Density
	 *	@param	qq:				Diquark condensate
	 *	@param	qbqb:				Antidiquark condensate
	 *	@param	res:				Conjugate Gradient Residue
	 *	@param	itercg:			Iterations of Conjugate Gradient
	 * @param	u11t,u12t		Double precisiongauge field
	 * @param	u11t_f,u12t_f:	Single precision gauge fields
	 *	@param	iu,id				Lattice indices
	 *	@param	gamval:			Gamma matrices
	 *	@param	gamval_f:		Gamma matrices (float)
	 *	@param	gamin:			Indices for Dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk4m_f:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p_f:			@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	jqq:				Diquark source
	 *	@param	akappa:			Hopping parameter
	 *	@param	Phi:				Pseudofermion field	
	 *	@param	R1:				A useful array for holding things that was already assigned in main.
	 *									In particular, we'll be using it to catch the output of
	 *									@f$ M^\dagger\Xi@f$ before the inversion, then used to store the
	 *									output of the inversion
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg,\
			Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int *iu, unsigned int *id,\
			Complex *gamval, Complex_f *gamval_f,	int *gamin, double *dk4m, double *dk4p,\
			float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa,	Complex *Phi, Complex *R1);
	/** 
	 * @brief	Calculates the gauge action using new (how new?) lookup table
	 * @brief	Follows a routine called qedplaq in some QED3 code
	 *
	 * @param	hg				Gauge component of Hamilton
	 * @param	avplaqs		Average spacial Plaquette
	 * @param	avplaqt		Average Temporal Plaquette
	 * @param	u11t,u12t	The trial fields
	 * @param	iu				Upper halo indices
	 * @param	beta			Inverse gauge coupling
	 *
	 * @see Par_dsum
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex_f *u11t, Complex_f *u12t,\
			unsigned int *iu, float beta);
#if (!defined __NVCC__ && !defined __HIPCC__)
	/**
	 * @brief Calculates the plaquette at site i in the @f$\mu--\nu@f$ direction
	 *
	 * @param	u11t, u12t:	Trial fields
	 * @param	Sigma11, Sigma12:	Plaquette components
	 * @param	i:				Lattice site
	 * @param	iu:			Upper halo indices
	 * @param 	mu, nu:		Plaquette direction. Note that mu and nu can be negative
	 * 							to facilitate calculating plaquettes for Clover terms. No
	 * 							sanity checks are conducted on them in this routine.
	 *
	 * @return double corresponding to the plaquette value
	 *
	 */
	int SU2plaq(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12, unsigned int *iu, int i, int mu, int nu);
#endif
	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 * 
	 * @param	u11t, u12t	The gauge fields
	 *
	 * @see Par_tmul, Par_dsum
	 * 
	 * @return Double corresponding to the polyakov loop
	 */
	double Polyakov(Complex_f *u11t, Complex_f *u12t);
	//Inline functions
	/**
	 * @brief	Extracts all the single precision gauge links in the @f$\mu@f$ direction only
	 *
	 * @param	x:			The output 
	 * @param	y:			The gauge field for a particular colour
	 * @param	n:			Number of sites in the gauge field. This is typically kvol
	 * @param	table:	Table containing information on nearest neighbours. Usually id or iu
	 * @param	mu:		Direciton we're interested in extractng	
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int C_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu);
	/**
	 * @brief	Extracts all the double precision gauge links in the @f$\mu@f$ direction only
	 *
	 * @param	x:			The output 
	 * @param	y:			The gauge field for a particular colour
	 * @param	n:			Number of sites in the gauge field. This is typically kvol
	 * @param	table:	Table containing information on nearest neighbours. Usually id or iu
	 * @param	mu:		Direciton we're interested in extractng	
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Z_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu);
	/**
	 * Copies necessary (2*4*kvol) elements of Phi into a vector variable
	 *
	 * @param	na: 				flavour index
	 * @param	smallPhi:		The partitioned output
	 * @param	Phi:				The pseudofermion field
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Fill_Small_Phi(int na, Complex *smallPhi, Complex *Phi);
	/*
	 *	@brief Up/Down partitioning of the pseudofermion field
	 *
	 *	@param	na:	Flavour index
	 *	@param	X0:	Partitioned field
	 *	@param	R1:	Full pseudofermion field
	 *
	 *	@return	Zero on success, integer error code otherwise	
	 */
	int UpDownPart(const int na, Complex *X0, Complex *R1);

	//CUDA Declarations:
	//#################
#ifdef __NVCC__
	//Not a function. An array of concurrent GPU streams to keep it busy
	extern cudaStream_t streams[ndirac*ndim*nadj];
	//Calling Functions:
	//=================
	void cuAverage_Plaquette(double *hgs, double *hgt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock);
	void cuPolyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f *u11t, Complex_f *u12t,dim3 dimGrid, dim3 dimBlock);
	void cuGauge_force(int mu,Complex_f *Sigma11, Complex_f *Sigma12, Complex_f *u11t,Complex_f *u12t,double *dSdpi,float beta,\
			dim3 dimGrid, dim3 dimBlock);
	void cuPlus_staple(int mu, int nu, unsigned int *iu, Complex_f *Sigma11, Complex_f *Sigma12, Complex_f *u11t, Complex_f *u12t,\
			dim3 dimGrid, dim3 dimBlock);
	void cuMinus_staple(int mu, int nu, unsigned int *iu, unsigned int *id, Complex_f *Sigma11, Complex_f *Sigma12,\
			Complex_f *u11sh, Complex_f *u12sh,Complex_f *u11t, Complex_f*u12t,	dim3 dimGrid, dim3 dimBlock);
	void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, \
			Complex *gamval,double *dk4m, double *dk4p,unsigned int *iu,int *gamin,\
			float akappa, dim3 dimGrid, dim3 dimBlock);
	//cuInit was taken already by CUDA (unsurprisingly)
	void Init_CUDA(Complex *u11t, Complex *u12t,Complex *gamval, Complex_f *gamval_f, int *gamin, double*dk4m,\
			double *dk4p, unsigned int *iu, unsigned int *id);
	void cuFill_Small_Phi(int na, Complex *smallPhi, Complex *Phi,dim3 dimBlock, dim3 dimGrid);
	void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu,dim3 dimBlock, dim3 dimGrid);
	void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu,dim3 dimBlock, dim3 dimGrid);
	void cuComplex_convert(Complex_f *a, Complex *b, int len, bool ftod, dim3 dimBlock, dim3 dimGrid);
	void cuReal_convert(float *a, double *b, int len, bool ftod, dim3 dimBlock, dim3 dimGrid);
	void cuUpDownPart(int na, Complex *X0, Complex *R1,dim3 dimBlock, dim3 dimGrid);
#endif
#if (defined __cplusplus)
}
#endif
//CUDA Kernels:
//============
#ifdef __CUDACC__
//__global__ void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
//		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa);
__global__ void Plus_staple(int mu, int nu,unsigned int *iu, Complex_f *Sigma11, Complex_f *Sigma12,\
		Complex_f *u11t, Complex_f *u12t);
__global__ void Minus_staple(int mu, int nu,unsigned int *iu,unsigned int *id, Complex_f *Sigma11, Complex_f *Sigma12,\
		Complex_f *u11sh, Complex_f *u12sh, Complex_f *u11t, Complex_f *u12t);
__global__ void cuGaugeForce(int mu, Complex_f *Sigma11, Complex_f *Sigma12,double* dSdpi,Complex_f *u11t, Complex_f *u12t,\
		float beta);
__global__ void cuAverage_Plaquette(float *hgs_d, float *hgt_d, Complex_f *u11t, Complex_f *u12t, unsigned int *iu);
__global__ void cuPolyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f *u11t, Complex_f *u12t);
__device__ void SU2plaq(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12, unsigned int *iu, int i, int mu, int nu);
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
__global__ void cuFill_Small_Phi(int na, Complex *smallPhi, Complex *Phi);
__global__ void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu);
__global__ void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu);
__global__ void cuComplex_convert(Complex_f *a, Complex *b, int len, bool dtof);
__global__ void cuReal_convert(float *a, double *b, int len, bool dtof);
__global__ void cuUpDownPart(int na, Complex *X0, Complex *R1);
#endif
#endif
