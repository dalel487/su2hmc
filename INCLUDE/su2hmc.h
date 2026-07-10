/**
 * @file		su2hmc.h
 * @brief	Function declarations for most of the routines
 */
#pragma once
//ARM Based machines. BLAS routines should work with other libraries, so we can set a compiler
//flag to sort them out. But the PRNG routines etc. are MKL exclusive
#include <integrate.h>
#ifdef __cplusplus
#include	<cstdio>
#include	<cstdlib>
#include	<ctime>
#else
#include	<time.h>
#endif

//Definitions:
//###########
#ifdef _DEBUGCG
#define _DEBUG 1
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
	 *	@param[in,out]	dSdpi:				The force
	 *	@param[in]	ut:					Float precision colour fields
	 *	@param[in]	X1:					Inverted field
	 *	@param[in]	X2:					@f$MX_1@f$
	 *	@param[in]	gamval:				Gamma matrices rescaled by @f$\kappa@f$
	 *	@param[in]	iu:					Lattice indices
	 *	@param[in]	gamin:				Gamma indices
	 *	@param[in]	akappa:				Hopping parameter
	 *	@param[in]	mu:					Force direction
	 *
	 *	@post	Force added to @p dSdpi 
	 */
	void Force_s(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, Complex_f gamval[20],\
			unsigned int *iu, const unsigned short gamin[16],const float akappa, const unsigned short mu);
	/**
	 *	@brief Calculates the force @f$\frac{dS}{d\pi}@f$ at each intermediate time
	 *	
	 *	@param[in,out]	dSdpi:				The force
	 *	@param[in]	ut:					Float precision colour fields
	 *	@param[in]	X1:					Inverted field
	 *	@param[in]	X2:					@f$MX_1@f$
	 *	@param[in]	gamval:				Gamma matrices rescaled by @f$\kappa@f$
	 * @param[in]	dk:					@f$e^{-\mu}@f$ and @f$e^\mu@f$
	 *	@param[in]	iu:					Lattice indices
	 *	@param[in]	gamin:				Gamma indices
	 *	@param[in]	akappa:				Hopping parameter
	 *
	 *	@post	Force added to @p dSdpi 
	 */
	void Force_t(double *dSdpi, Complex_f *ut[2],Complex_f *X1, Complex_f *X2, Complex_f gamval[20],\
			float *dk[2], unsigned int *iu, const unsigned short gamin[16],float akappa);
	/**
	 *	@brief Calculates the force @f$\frac{dS}{d\pi}@f$ at each intermediate time
	 *
	 *	@param[in,out]	dSdpi:				The force
	 *	@param[in]	iflag:				Invert before evaluating the force. 0 to invert, one not to. Blame FORTRAN...	
	 *	@param[in]	res1:					Conjugate gradient residue
	 *	@param[in]	X0:					Up/down partitioned pseudofermion field
	 *	@param[in]	X1:					Holder for the partitioned fermion field, then the inverted dield
	 *	@param[in]	Phi:					Pseudofermion field
	 *	@param[in]	ut,ut_f:				Double/float precision colour fields
	 *	@param[in]	iu,id:				Lattice indices
	 *	@param[in]	gamval,gamval_f:	Double/float precision gamma matrices rescaled by @f$\kappa@f$
	 *	@param[in]	gamin:				Gamma indices
	 *	@param[in]	sigval,sigval_f:	Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}{2}@f$
	 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	dk,dk_f:				Double/float @f$e^{-\mu}@f$ and @f$e^\mu@f$
	 * @param[in] 	jqq:					Diquark source
	 *	@param[in]	akappa:				Hopping parameter
	 *	@param[in]	beta:					Inverse gauge coupling
	 *	@param[in]	c_sw:					Clover coefficient. If non-zero calculate the clover contribution
	 *	@param[in]	ancg:					Counter for conjugate gradient iterations
	 *
	 *	@return Zero on success, integer error code otherwise
	 *	@post	Force added to @p dSdpi 
	 */
	int Force(double *dSdpi, const bool iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,\
			Complex *ut[2], Complex_f *ut_f[2],unsigned int *iu,unsigned int *id,\
			Complex gamval[20],Complex_f gamval_f[20],const unsigned short gamin[16],Complex *sigval,Complex_f *sigval_f, unsigned short *sigin,\
			double *dk[2], float *dk_f[2],const Complex_f jqq, const float akappa,const float beta,const float c_sw,double *ancg);
	/**
	 * @brief	Calculates the gauge force due to the Wilson Action at each intermediate time
	 *
	 * @param[out]	dSdpi:		The force
	 *	@param[in]	ut:			Gauge fields
	 * @param[in]	iu,id:		Lattice indices 
	 * @param[in]	beta:			Inverse gauge coupling
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post	Contents of @p dSdpi replaced by gauge force
	 */
	int Gauge_force(double *dSdpi, Complex_f *ut[2],unsigned int *iu,unsigned int *id, float beta);
	/**
	 * @brief Initialises the system
	 *
	 * @param[in]	istart:				Zero for cold, >1 for hot, <1 for none
	 * @param[in]	ibound:				Periodic boundary conditions
	 * @param[in]	iread:				Read configuration from file
	 * @param[in]	beta:					Inverse gauge coupling
	 * @param[in]	fmu:					Chemical potential
	 * @param[in]	akappa:				Hopping parameter
	 * @param[in]	ajq:					Diquark source
	 * @param[in]	c_sw:					Clover coefficient
	 * @param[out]	u:						Gauge fields
	 * @param[out]	ut,ut_f:				Double/float Trial gauge field
	 * @param[out]	dk,dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)^\mu@f$
	 * @param[out]	iu,id:				Up halo indices
	 *	@param[out]	gamval,gamval_f:	Double/float precision gamma matrices rescaled by kappa
	 * @param[out]	gamin:				Gamma matrix indices
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post	Contents of all out arguments overwritten
	 */
	int Init(const int istart, const int ibound, const int iread, const float beta, const float fmu, const float akappa,\
			const Complex_f ajq, const float c_sw, Complex *u[2], Complex *ut[2], Complex_f *ut_f[2], Complex gamval[20],\
			Complex_f gamval_f[20], unsigned short gamin[16], double *dk[2], float *dk_f[2],\
			unsigned int *iu, unsigned int *id);
	/**
	 * @brief Calculate the Hamiltonian
	 *
	 * @param[out]	h:				Hamiltonian
	 * @param[out]	s:				Action
	 * @param[in]	res2:			Limit for conjugate gradient
	 * @param[in]	pp:			Momentum field
	 *	@param[in]	X0:			Up/down partitioned pseudofermion field
	 *	@param[in]	X1:			Holder for the partitioned fermion field, then the conjugate gradient output
	 * @param[in]	Phi:			Pseudofermion field
	 * @param[in]	ut,ud:		Gauge fields (single/double precision)
	 * @param[in]	iu,id:		Lattice indices
	 *	@param[in]	gamval,gamval_f:	Gamma matrices rescaled by kappa
	 * @param[in]	gamin:		Gamma indices
	 *	@param[in]	sigval,sigval_f:	Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/3@f$
	 * @param[in]	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	dk,dk_f:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 * @param[in]	jqq:			Diquark source
	 * @param[in]	akappa:		Hopping parameter
	 * @param[in]	beta:			Inverse gauge coupling
	 * @param[in]	c_sw:			Clover coefficient. If non-zero calculate the clover contribution
	 * @param[in]	ancgh:		Conjugate gradient iterations counter 
	 * @param[in]	traj:			Calling trajectory for error reporting
	 *
	 * @return	Zero on success. Integer Error code otherwise.
	 * @post	@p h and @p s overwritten with output
	 */	
	int Hamilton(double *h,double *s,double res2,double *pp,Complex *X0,Complex *X1,Complex *Phi, Complex *ud[2],Complex_f *ut[2],
			unsigned int *iu,unsigned int *id, Complex gamval[20], Complex_f gamval_f[20],const unsigned short gamin[16], Complex *sigval, Complex_f *sigval_f,
			unsigned short *sigin, double *dk[2],float *dk_f[2],Complex_f jqq,float akappa,float beta,float c_sw, double *ancgh,
			int traj);
	/**
	 * @brief Matrix Inversion via Conjugate Gradient (up/down flavour partitioning).
	 * Solves @f$(M^\dagger)Mx=\Phi@f$
	 * Implements up/down partitioning
	 * The matrix multiplication step is done at mixed precision, while the update is done at double
	 *
	 * @param[in]	na:					Flavour index
	 * @param[in]	res:					Limit for conjugate gradient
	 * @param[in,out]	X1:					Pseudofermion field @f$\Phi@f$ initially, returned as @f$(M^\dagger M)^{-1} \Phi@f$
	 * @param[in]	r:						Partition of @f$\Phi@f$ being used. Gets recycled as the residual vector
	 * @param[in]	ud,ut:				Double/float Trial colour fields
	 * @param[in]	iu,id:				Upper/lower halo indices
	 *	@param[in]	gamval,gamval_f:	Double/float gamma matrices rescaled by kappa
	 * @param[in]	gamin:				What element of the spinor is multiplied by row idirac each gamma matrix?
	 *	@param[in]	clover_f:			Array of clover fields
	 *	@param[in]	sigval,sigval_f:	Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
	 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	dk,dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param[in]	jqq:					Diquark source
	 * @param[in]	akappa:				Hopping Parameter
	 * @param[in]	c_sw:					Clover coefficient. If non-zero calculate the clover contribution
	 * @param[in]	itercg:				Counts the iterations of the conjugate gradient
	 *
	 * @return 0 on success, integer error code otherwise
	 * @post	Contents of @p X1 and @p r overwritten
	 */
	int Congradq(int na,double res,Complex *X1,Complex *r,Complex *ud[2], Complex_f *ut[2],Complex_f *clover_f[nc],
			unsigned int *iu, unsigned int *id, Complex gamval[20], Complex_f gamval_f[20],const unsigned short gamin[16],
			Complex *sigval, Complex_f *sigval_f,unsigned short *sigin, double *dk[2], float *dk_f[2],
			Complex_f jqq,float akappa,float c_sw,int *itercg);
	/**
	 * @brief Matrix Inversion via Conjugate Gradient (no up/down flavour partitioning).
	 * Solves @f$(M^\dagger)Mx=\Phi@f$
	 * The matrix multiplication step is done at single precision, while the update is done at double
	 *
	 * @param[in] 	na:						Flavour index
	 * @param[in] 	res:						Limit for conjugate gradient
	 * @param[in] 	Phi:						Pseudofermion field.
	 * @param[in,out] 	xi:						Returned as @f$(M^\dagger M)^{-1} \Phi@f$
	 * @param[in] 	ut,ud:					Double/float Gauge fields
	 * @param[in] 	iu,id:					Upper/Lower halo indices
	 *	@param[in]	gamval,gamval_f:		double float Gamma matrices rescaled by kappa
	 * @param[in] 	gamin:					Dirac indices
	 *	@param[in]	clover_f:				Array of clover fields
	 *	@param[in]	sigval,sigval_f:		Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
	 * @param[in]	sigin:					What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	dk,dk_f:					Double/float @f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param[in] 	jqq:						Diquark source
	 * @param[in] 	akappa:					Hopping Parameter
	 * @param[in] 	c_sw:						Clover coefficient.
	 * @param[in] 	itercg:					Counts the iterations of the conjugate gradient
	 * 
	 * @return 0 on success, integer error code otherwise
	 * @post	Contents of @p Phi overwritten
	 */
	int Congradp(int na, double res, Complex *Phi, Complex *xi, Complex *ud[2], Complex_f *ut[2], Complex_f *clover_f[nc],
			unsigned int *iu, unsigned int *id, Complex gamval[20], Complex_f gamval_f[20], const unsigned short gamin[16],
			Complex *sigval, Complex_f *sigval_f,unsigned short *sigin, double *dk[2], float *dk_f[2],
			Complex_f jqq,float akappa,float c_sw,int *itercg);
	/**
	 * @brief	Calculate fermion expectation values via a noisy estimator
	 * 
	 * Matrix inversion via conjugate gradient algorithm
	 * Solves @f$MX=X_1@f$
	 * (Numerical Recipes section 2.10 pp.70-73)   
	 * uses NEW lookup tables **
	 * Implemented in Congradp()
	 *
	 * @param[out]	pbp:							@f$\langle\bar{\Psi}\Psi\rangle@f$
	 *	@param[out]	endenf:						Energy density
	 *	@param[out]	denf:							Number Density
	 *	@param[out]	qq:							Diquark condensate
	 *	@param[out]	qbqb:							Antidiquark condensate
	 *	@param[in]	res:							Conjugate Gradient Residue
	 *	@param[in]	itercg:						Iterations of Conjugate Gradient
	 * @param[in]	ut,ut_f:						Double/float precision gauge field
	 *	@param[in]	iu,id							Up/down Lattice indices
	 *	@param[in]	gamval,gamval_f:			Double/float precision gamma matrices rescaled by kappa
	 *	@param[in]	gamin:						Indices for Dirac terms
	 *	@param[in]	sigval,sigval_f:			Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
	 * @param[in]	sigin:						What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	dk,dk_f:						Double/float @f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ 
	 *	@param[in]	jqq:							Diquark source
	 *	@param[in]	akappa:						Hopping parameter
	 *	@param[in]	c_sw:							Clover parameter
	 *	@param[in]	Phi:							Pseudofermion field	
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post The values of @p Phi are not used. Since the memory is allocated already it is instead overwritten with the
	 * noisy estimator.
	 */
	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg,\
			Complex *ut[2], Complex_f *ut_f[2], unsigned int *iu, unsigned int *id,\
			Complex gamval[20], Complex_f gamval_f[20],	const unsigned short gamin[16],\
			Complex *sigval,Complex_f *sigval_f, unsigned short *sigin, double *dk[2],float *dk_f[2],\
			Complex_f jqq, float akappa,	float c_sw,Complex *Phi);
	/** 
	 * @brief	Calculates the gauge action using new (how new?) lookup table
	 * 			Follows a routine called qedplaq in some QED3 code
	 *
	 * @param[out]	hg				Gauge component of Hamilton
	 * @param[out]	avplaqs		Average spacial Plaquette
	 * @param[out]	avplaqt		Average Temporal Plaquette
	 * @param[in]	ut:			The trial fields
	 * @param[in]	iu				Upper halo indices
	 * @param[in]	beta			Inverse gauge coupling
	 *
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post	Contents of @p hg, @p avplaqs and @p avplaqt replaced with output
	 */
	int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex_f *ut[2],unsigned int *iu, float beta);
	/**
	 * @brief Calculates the plaquette at site i in the @f$\mu--\nu@f$ direction
	 *
	 * @param[in]	ut:			Trial fields
	 * @param[out]	Sigma:		Plaquette components
	 * @param[in]	i:				Lattice site
	 * @param[in]	iu:			Upper halo indices
	 * @param[in] 	mu, nu:		Plaquette direction. Note that mu and nu can be negative
	 * 							to facilitate calculating plaquettes for Clover terms. No
	 * 							sanity checks are conducted on them in this routine.
	 *
	 *	@return	Zero on success, integer error code otherwise
	 * @post	Plaquettes written into @p Sigma
	 */
	int SU2plaq(Complex_f *ut[2], Complex_f Sigma[2], unsigned int *iu, int i, int mu, int nu);

	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 * 
	 * @param[in]	ut:	The gauge fields
	 * 
	 * @return Double corresponding to the polyakov loop
	 */
	double Polyakov(Complex_f *ut[2]);
	//Inline functions
	/**
	 * @brief	Extracts all the single precision gauge links in the @f$\mu@f$ direction only
	 *
	 * @param[out]	x:			The output 
	 * @param[in]	y:			The gauge field for a particular colour
	 * @param[in]	n:			Number of sites in the gauge field. This is typically kvol
	 * @param[in]	table:	Table containing information on nearest neighbours. Usually id or iu
	 * @param[in]	mu:		Direction we're interested in extracting	
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post	Contents of @p x replaced with output
	 */
	int C_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu);
	/**
	 * @brief	Extracts all the double precision gauge links in the @f$\mu@f$ direction only
	 *
	 * @param[out]	x:			The output 
	 * @param[in]	y:			The gauge field for a particular colour
	 * @param[in]	n:			Number of sites in the gauge field. This is typically kvol
	 * @param[in]	table:	Table containing information on nearest neighbours. Usually id or iu
	 * @param[in]	mu:		Direciton we're interested in extracting	
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post	Contents of @p x replaced with output
	 */
	int Z_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu);
	/**
	 * @brief Copies necessary (2*4*kvol) elements of Phi into a vector variable
	 *
	 * @param[in]	na: 				flavour index
	 * @param[out]	smallPhi:		The partitioned output
	 * @param[in]	Phi:				The pseudofermion field
	 *
	 * @return Zero on success, integer error code otherwise
	 *	@post	Result written into @p smallPhi
	 */
	int Fill_Small_Phi(int na, Complex *smallPhi, Complex *Phi);
	/**
	 *	@brief Up/Down partitioning of the pseudofermion field
	 *
	 *	@param[in]	na:	Flavour index
	 *	@param[out]	X0:	Partitioned field
	 *	@param[in]	R1:	Full pseudofermion field
	 *
	 *	@return	Zero on success, integer error code otherwise	
	 *	@post	Result written into @p X0
	 */
	int UpDownPart(const unsigned int na, Complex *X0, Complex *R1);
	/**
	 * @brief Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * @param[out,in] ut:	 Trial fields to be reunitarised
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post	@p ut replaced with reunitarised gauge fields
	 */
	int Reunitarise(Complex *ut[2]);
	/**
	 * @brief takes an array of complex float and double precision numbers and converts the precision
	 *
	 * @param[out,in]	a:				Float array
	 * @param[out,in]	b:				Double array
	 * @param[in]	len:			Number of elements to convert per stride. Striding needed to handle halo terms
	 * @param[in]	dtof:			If true, convert double to float. Otherwise convert float to double
	 * @param[in]	stride:		For terms with a halo, we need to convert in blocks of len separated by (len+halo)
	 *
	 * @return Zero on success, integer error code otherwise
	 * @post Depending on the value of @p dtof, either the contents of @p a or @p b are overwritten with those of the
	 * other array in the opposite precision.
	 */
	int ComplexConvert(Complex_f *a, Complex *b, const unsigned int len, const bool dtof, const unsigned short stride);

#ifdef DIAGNOSTIC
	int Diagnostics(int istart, Complex *u[2], Complex *ut[2],Complex_f *ut_f[2],\
			unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk[2], float *dk_f[2],\
			const unsigned short gamin[16], const Complex gamval[20], const Complex_f gamval_f[20],\
			const Complex *sigval, const Complex_f *sigval_f, const unsigned short *sigin,
			Complex_f jqq,float akappa,float beta, float c_sw, double ancg);
#endif
	//CUDA Declarations:
	//#################
#ifdef __NVCC__
	/// @brief	An array of concurrent GPU streams to keep it busy
	extern cudaStream_t streams[ndirac*ndim*nadj];
	//Calling Functions:
	//=================
	/** 
	 * @brief	Calculates the gauge action using new (how new?) lookup table
	 * 		Follows a routine called qedplaq in some QED3 code
	 *
	 * @param[out]	hgs,hgt			Gauge component of Hamilton
	 * @param[in]	u11t,u12t		Gauge fields
	 * @param[in]	iu					Upper halo indices
	 * @param[in]	dimGrid			CUDA grid dimensions
	 * @param[in]	dimBlock			CUDA block dimensions
	 *
	 * @post	Contents of @p hgs and @p hgt replaced with the results
	 */
	void cuAverage_Plaquette(double *hgs, double *hgt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * @param[out]	Sigma		Components of the Polyakov loop
	 * @param[in]	ut:		The gauge fields
	 * @param[in]	dimGrid	CUDA grid dimensions
	 * @param[in]	dimBlock	CUDA block dimensions
	 * 
	 * @post	Contents of @p Sigma replaced with the Polyakov loop values
	 */
	void cuPolyakov(Complex_f *Sigma[2], Complex_f *ut[2],dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief Calculate the gauge contribution to the force
	 * 
	 * @param[in] ut:						Gauge fields
	 * @param[out] dSdpi:					Force
	 * @param[in] beta:					Inverse gauge coupling
	 * @param[in] iu,id:					Upper/lower indices
	 * @param[in] dimGrid,dimBlock:	CUDA grid/block size
	 *
	 * @post	Contents of @p dSdpi overwritten
	 */
	void cuGauge_force(Complex_f *ut[2],double *dSdpi,float beta,unsigned int *iu,unsigned int *id,dim3 dimGrid, dim3 dimBlock);
	/**
	 *	@brief Calculates the force @f$\frac{dS}{d\pi}@f$ at each intermediate time
	 *	
	 *	@param[in,out]	dSdpi:				The force
	 *	@param[in]	ut:					Gauge fields
	 *	@param[in]	X1:					Inverted field
	 *	@param[in]	X2:					@f$MX_1@f$
	 *	@param[in]	gamval:				Double/float precision gamma matrices rescaled by @f$\kappa@f$
	 *	@param[in]	gamin:				Gamma indices
	 *	@param[in]	iu:					Lattice indices
	 * @param[in]	dk:					@f$e^{-\mu}@f$ and @f$e^\mu@f$
	 *	@param[in]	akappa:				Hopping parameter
	 * @param[in] 	dimGrid,dimBlock:	CUDA grid/block size
	 *
	 * @post	Fermion force added onto @p dSdpi
	 */
	void cuForce(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, \
			Complex_f gamval[20],float *dk[2],unsigned int *iu,const unsigned short gamin[16],\
			float akappa, dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief  Initialise CUDA cuInit was taken already by CUDA (unsurprisingly)
	 * 
	 * @param[in]	u11t,u12t:			Trial gauge fields
	 * @param[in]	gamval,gamval_f:	Double/float precision gamma matrices rescaled by kappa
	 * @param[in]	gamin:				Gamma matrix indices
	 * @param[in]	dk4m,dk4p:			@f$e^{-\mu}@f$ and @f$e^\mu@f$
	 * @param[in]	iu,id:				Up/lower halo indices
	 *
	 * @todo CUDA 13 changed how setting devices work, so it's not just an integer any more. Those lines are commented out for now.
	 * 		They are not critical. Only hints for unified memory management.
	 */
	void Init_CUDA(Complex *u11t, Complex *u12t,Complex gamval[20], Complex_f gamval_f[20], unsigned short gamin[16], double*dk4m,\
			double *dk4p, unsigned int *iu, unsigned int *id);
	/**
	 * @brief Copies necessary (2*4*kvol) elements of Phi into a vector variable
	 *
	 * @param[in]	na:					flavour index
	 * @param[out]	smallPhi:			The partitioned output
	 * @param[in]	Phi:					The pseudofermion field
	 * @param[in] 	dimGrid,dimBlock:	CUDA grid/block size
	 * 
	 */
	void cuFill_Small_Phi(const unsigned int na, Complex *smallPhi, Complex *Phi,dim3 dimBlock, dim3 dimGrid);
	/**
	 * @brief takes an array of complex float and double precision numbers and converts the precision
	 *
	 * @param[out,in]	a:						Float array
	 * @param[out,in]	b:						Double array
	 * @param[in]	len:					Number of elements to convert
	 * @param[in]	dtof:					If true, convert double to float. Otherwise convert float to double
	 * @param[in] 	dimGrid,dimBlock:	CUDA grid/block size
	 *
	 * @post Depending on the value of @p dtof, either the contents of @p a or @p b are overwritten with those of the
	 * other array in the opposite precision.
	 */
	void cuComplex_convert(Complex_f *a, Complex *b, const unsigned int len,  const bool dtof, dim3 dimBlock, dim3 dimGrid);
	/**
	 * @brief takes an array of real-valued float and double precision numbers and converts the precision
	 *
	 * @param[out,in]	a:						Float array
	 * @param[out,in]	b:						Double array
	 * @param[in]	len:					Number of elements to convert
	 * @param[in]	dtof:					If true, convert double to float. Otherwise convert float to double
	 * @param[in] 	dimGrid,dimBlock:	CUDA grid/block size
	 *
	 * @post Depending on the value of @p dtof, either the contents of @p a or @p b are overwritten with those of the
	 * other array in the opposite precision.
	 */
	void cuReal_convert(float *a, double *b, const unsigned int len, const bool dtof, dim3 dimBlock, dim3 dimGrid);
	/**
	 *	@brief Up/Down partitioning of the pseudofermion field
	 *
	 *	@param[in]	na:	Flavour index
	 *	@param[out]	X0:	Partitioned field
	 *	@param[in]	R1:	Full pseudofermion field
	 * @param[in] 	dimGrid,dimBlock:	CUDA grid/block size
	 *
	 *	@post	Result written into @p X0
	 */
	void cuUpDownPart(const unsigned int na, Complex *X0, Complex *R1,dim3 dimBlock, dim3 dimGrid);
	/**
	 * @brief Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * @param[out,in] ut:						Trial fields to be reunitarised
	 * @param[in] dimGrid,dimBlock:	CUDA grid/block size
	 *
	 * @post	@p ut replaced with reunitarised gauge fields
	 */
	void cuReunitarise(Complex *ut[2],dim3 dimGrid, dim3 dimBlock);
	/**	
	 * @brief Initialises the CUDA grid and block size for a given lattice
	 *
	 * @param[in]	x,y,z,t:				Lattice dimensions
	 * @param[out] 	dimGrid,dimBlock:	CUDA grid/block size
	 *
	 * @post	@p dimGrid and @p dimBlock initialised
	 */
	void blockInit(int x, int y, int z, int t, dim3 *dimBlock, dim3 *dimGrid);
#endif
#if (defined __cplusplus)
}
#endif
