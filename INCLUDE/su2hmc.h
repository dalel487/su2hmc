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
	 *	@param[in]	X1:					Holder for the partitioned fermion field, then the inverted dield
	 *	@param[in]	X2:					Pseudofermion field
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
	 *	@param[in]	X1:					Holder for the partitioned fermion field, then the inverted dield
	 *	@param[in]	X2:					Pseudofermion field
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
	 *	@param[in]	sigval,sigval_f:	Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
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
	 * @param	dSdpi:		The force
	 *	@param	ut:			Gauge fields
	 * @param	iu,id:		Lattice indices 
	 * @param	beta:			Inverse gauge coupling
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Gauge_force(double *dSdpi, Complex_f *ut[2],unsigned int *iu,unsigned int *id, float beta);
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
	 * @param	c_sw:					Clover coefficient
	 * @param	u:						Gauge fields
	 * @param	ut,ut_f:				Double/float Trial gauge field
	 * @param	dk,dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)^\mu@f$
	 * @param	iu,id:				Up halo indices
	 *	@param	gamval,gamval_f:	Double/float precision gamma matrices rescaled by kappa
	 * @param	gamin:				Gamma matrix indices
	 *	@param	sigval,sigval_f:	@f$ \sigma_{\mu\nu}=\frac{1}{2i}[\gamma_\mu,\gamma_\nu]@f$ in double and single
	 *										precision
	 *	@param	sigin:				Which column does row idirac of @f$(\sigma_{\mu\nu}@f$ act on
	 *
	 * @return Zero on success, integer error code otherwise
	 */
int Init(const int istart, const int ibound, const int iread, const float beta, const float fmu, const float akappa,\
			const Complex_f ajq, const float c_sw, Complex *u[2], Complex *ut[2], Complex_f *ut_f[2], Complex gamval[20],\
			Complex_f gamval_f[20], unsigned short gamin[16], double *dk[2], float *dk_f[2],\
			unsigned int *iu, unsigned int *id);
	/**
	 * @brief Calculate the Hamiltonian
	 *
	 * @param	h:				Hamiltonian
	 * @param	s:				Action
	 * @param	res2:			Limit for conjugate gradient
	 * @param	pp:			Momentum field
	 *	@param	X0:			Up/down partitioned pseudofermion field
	 *	@param	X1:			Holder for the partitioned fermion field, then the conjugate gradient output
	 * @param	Phi:			Pseudofermion field
	 * @param	ut:			Gauge fields (single precision)
	 * @param	iu,id:		Lattice indices
	 *	@param	gamval_f:	Single precision gamma matrices rescaled by kappa
	 * @param	gamin:		Gamma indices
	 *	@param	sigval_f:	Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
	 * @param	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 * @param	jqq:			Diquark source
	 * @param	akappa:		Hopping parameter
	 * @param	beta:			Inverse gauge coupling
	 * @param	c_sw:			Clover coefficient. If non-zero calculate the clover contribution
	 * @param	ancgh:		Conjugate gradient iterations counter 
	 * @param	traj:			Calling trajectory for error reporting
	 *
	 * @return	Zero on success. Integer Error code otherwise.
	 */	
	int Hamilton(double *h,double *s,double res2,double *pp,Complex *X0,Complex *X1,Complex *Phi, Complex *ud[2],Complex_f *ut[2],
			unsigned int *iu,unsigned int *id, Complex gamval[20], Complex_f gamval_f[20],const unsigned short gamin[16], Complex *sigval, Complex_f *sigval_f,
			unsigned short *sigin, double *dk[2],float *dk_f[2],Complex_f jqq,float akappa,float beta,float c_sw, double *ancgh,
			int traj);
	/**
	 * @brief Matrix Inversion via Conjugate Gradient (up/down flavour partitioning).
	 * Solves @f$(M^\dagger)Mx=\Phi@f$
	 * Implements up/down partitioning
	 * The matrix multiplication step is done at single precision, while the update is done at double
	 *
	 * @param	na:					Flavour index
	 * @param	res:					Limit for conjugate gradient
	 * @param	X1:					Pseudofermion field @f$\Phi@f$ initially, returned as @f$(M^\dagger M)^{-1} \Phi@f$
	 * @param	r:						Partition of @f$\Phi@f$ being used. Gets recycled as the residual vector
	 * @param	ud,ut:				Double/float Trial colour fields
	 * @param	iu,id:				Upper/lower halo indices
	 *	@param	gamval,gamval_f:	Double/float gamma matrices rescaled by kappa
	 * @param	gamin:				What element of the spinor is multiplied by row idirac each gamma matrix?
	 *	@param	clover_f:			Array of clover fields
	 *	@param	sigval,sigval_f:	Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
	 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param	dk,dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	jqq:					Diquark source
	 * @param	akappa:				Hopping Parameter
	 * @param	c_sw:					Clover coefficient. If non-zero calculate the clover contribution
	 * @param	itercg:				Counts the iterations of the conjugate gradient
	 *
	 * @return 0 on success, integer error code otherwise
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
	 * @param 	na:						Flavour index
	 * @param 	res:						Limit for conjugate gradient
	 * @param 	Phi:						Pseudofermion field.
	 * @param 	xi:						Returned as @f$(M^\dagger M)^{-1} \Phi@f$
	 * @param 	ut,ud:					Double/float Gauge fields
	 * @param 	iu,id:					Upper/Lower halo indices
	 *	@param	gamval,gamval_f:		double float Gamma matrices rescaled by kappa
	 * @param 	gamin:					Dirac indices
	 *	@param	clover_f:				Array of clover fields
	 *	@param	sigval,sigval_f:		Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
	 * @param	sigin:					What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param	dk,dk_f:					Double/float @f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param 	jqq:						Diquark source
	 * @param 	akappa:					Hopping Parameter
	 * @param 	c_sw:						Clover coefficient.
	 * @param 	itercg:					Counts the iterations of the conjugate gradient
	 * 
	 * @return 0 on success, integer error code otherwise
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
	 * @param	pbp:							@f$\langle\bar{\Psi}\Psi\rangle@f$
	 *	@param	endenf:						Energy density
	 *	@param	denf:							Number Density
	 *	@param	qq:							Diquark condensate
	 *	@param	qbqb:							Antidiquark condensate
	 *	@param	res:							Conjugate Gradient Residue
	 *	@param	itercg:						Iterations of Conjugate Gradient
	 * @param	ut,ut_f:						Double/float precision gauge field
	 *	@param	iu,id							Up/down Lattice indices
	 *	@param	gamval/gamval_f:			Double/float precision gamma matrices rescaled by kappa
	 *	@param	gamin:						Indices for Dirac terms
	 *	@param	sigval,sigval_f:			Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}/2@f$
	 * @param	sigin:						What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param	dk,dk_f:						Double/float @f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ 
	 *	@param	jqq:							Diquark source
	 *	@param	akappa:						Hopping parameter
	 *	@param	c_sw:							Clover parameter
	 *	@param	Phi:							Pseudofermion field	
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg,\
			Complex *ut[2], Complex_f *ut_f[2], unsigned int *iu, unsigned int *id,\
			Complex gamval[20], Complex_f gamval_f[20],	const unsigned short gamin[16],\
			Complex *sigval,Complex_f *sigval_f, unsigned short *sigin, double *dk[2],float *dk_f[2],\
			Complex_f jqq, float akappa,	float c_sw,Complex *Phi);
	/** 
	 * @brief	Calculates the gauge action using new (how new?) lookup table
	 * @brief	Follows a routine called qedplaq in some QED3 code
	 *
	 * @param	hg				Gauge component of Hamilton
	 * @param	avplaqs		Average spacial Plaquette
	 * @param	avplaqt		Average Temporal Plaquette
	 * @param	ut:			The trial fields
	 * @param	iu				Upper halo indices
	 * @param	beta			Inverse gauge coupling
	 *
	 * @see Par_dsum
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex_f *ut[2],unsigned int *iu, float beta);
	/**
	 * @brief Calculates the plaquette at site i in the @f$\mu--\nu@f$ direction
	 *
	 * @param	ut:			Trial fields
	 * @param	Sigma:		Plaquette components
	 * @param	i:				Lattice site
	 * @param	iu:			Upper halo indices
	 * @param 	mu, nu:		Plaquette direction. Note that mu and nu can be negative
	 * 							to facilitate calculating plaquettes for Clover terms. No
	 * 							sanity checks are conducted on them in this routine.
	 *
	 * @return double corresponding to the plaquette value
	 *
	 */
	int SU2plaq(Complex_f *ut[2], Complex_f Sigma[2], unsigned int *iu, int i, int mu, int nu);

	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 * 
	 * @param	ut:	The gauge fields
	 *
	 * @see Par_tmul, Par_dsum
	 * 
	 * @return Double corresponding to the polyakov loop
	 */
	double Polyakov(Complex_f *ut[2]);
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
	/**
	 *	@brief Up/Down partitioning of the pseudofermion field
	 *
	 *	@param	na:	Flavour index
	 *	@param	X0:	Partitioned field
	 *	@param	R1:	Full pseudofermion field
	 *
	 *	@return	Zero on success, integer error code otherwise	
	 */
	int UpDownPart(const unsigned int na, Complex *X0, Complex *R1);
	/**
	 * @brief Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * @param ut:	 Trial fields to be reunitarised
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Reunitarise(Complex *ut[2]);
	/**
	 * @brief takes an array of complex float and double precision numbers and converts the precision
	 *
	 * @param	a:				Float array
	 * @param	b:				Double array
	 * @param	len:			Number of elements to convert per stride. Striding needed to handle halo terms
	 * @param	dtof:			If true, convert double to float. Otherwise convert float to double
	 * @param	stride:		For terms with a halo, we need to convert in blocks of len seperated by (len+halo)
	 *
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
	 * @brief	Follows a routine called qedplaq in some QED3 code
	 *
	 * @param	hgs,hgt			Gauge component of Hamilton
	 * @param	u11t,u12t		Gauge fields
	 * @param	iu					Upper halo indices
	 * @param	dimGrid			CUDA grid dimensions
	 * @param	dimBlock			CUDA block dimensions
	 */
	void cuAverage_Plaquette(double *hgs, double *hgt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * @param	Sigma		Components of the Polyakov loop
	 * @param	ut:		The gauge fields
	 * @param	dimGrid	CUDA grid dimensions
	 * @param	dimBlock	CUDA block dimensions
	 * 
	 */
	void cuPolyakov(Complex_f *Sigma[2], Complex_f *ut[2],dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief Calculate the gauge contribution to the force
	 * 
	 * @param ut:						Gauge fields
	 * @param dSdpi:					Force
	 * @param beta:					Inverse gauge coupling
	 * @param iu,id:					Upper/lower indices
	 * @param dimGrid,dimBlock:	CUDA grid/block size
	 */
	void cuGauge_force(Complex_f *ut[2],double *dSdpi,float beta,unsigned int *iu,unsigned int *id,dim3 dimGrid, dim3 dimBlock);
	/**
	 *	@brief Calculates the force @f$\frac{dS}{d\pi}@f$ at each intermediate time
	 *	
	 *	@param	dSdpi:				The force
	 *	@param	ut:					Gauge fields
	 *	@param	X0:					Up/down partitioned pseudofermion field
	 *	@param	X1:					Inverted field
	 *	@param	gamval,gamval_f:	Double/float precision gamma matrices rescaled by @f$\kappa@f$
	 *	@param	gamin:				Gamma indices
	 *	@param	iu:					Lattice indices
	 * @param	dk:					@f$e^{-\mu}@f$ and @f$e^\mu@f$
	 *	@param	akappa:				Hopping parameter
	 * @param 	dimGrid,dimBlock:	CUDA grid/block size
	 *
	 */
	void cuForce(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, \
			Complex_f gamval[20],float *dk[2],unsigned int *iu,const unsigned short gamin[16],\
			float akappa, dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief  Initialise CUDA cuInit was taken already by CUDA (unsurprisingly)
	 * 
	 * @param	u11t,u12t:			Trial gauge fields
	 * @param	gamval,gamval_f:	Double/float precision gamma matrices rescaled by kappa
	 * @param	gamin:				Gamma matrix indices
	 * @param	dk4m,dk4p:			@f$e^{-\mu}@f$ and @f$e^\mu@f$
	 * @param	iu,id:				Up/lower halo indices
	 *
	 * @todo CUDA 13 changed how setting devices work, so it's not just an integer any more. Those lines are commented out for now.
	 * 		They are not critical. Only hints for unified memory management.
	 */
	void Init_CUDA(Complex *u11t, Complex *u12t,Complex gamval[20], Complex_f gamval_f[20], unsigned short gamin[16], double*dk4m,\
			double *dk4p, unsigned int *iu, unsigned int *id);
	/**
	 * Copies necessary (2*4*kvol) elements of Phi into a vector variable
	 *
	 * @param	na:					flavour index
	 * @param	smallPhi:			The partitioned output
	 * @param	Phi:					The pseudofermion field
	 * @param 	dimGrid,dimBlock:	CUDA grid/block size
	 * 
	 */
	void cuFill_Small_Phi(const unsigned int na, Complex *smallPhi, Complex *Phi,dim3 dimBlock, dim3 dimGrid);
	/**
	 * @brief takes an array of complex float and double precision numbers and converts the precision
	 *
	 * @param	a:						Float array
	 * @param	b:						Double array
	 * @param	len:					Number of elements to convert
	 * @param	dtof:					If true, convert double to float. Otherwise convert float to double
	 * @param 	dimGrid,dimBlock:	CUDA grid/block size
	 */
	void cuComplex_convert(Complex_f *a, Complex *b, const unsigned int len,  const bool dtof, dim3 dimBlock, dim3 dimGrid);
	/**
	 * @brief takes an array of real-valued float and double precision numbers and converts the precision
	 *
	 * @param	a:						Float array
	 * @param	b:						Double array
	 * @param	len:					Number of elements to convert
	 * @param	dtof:					If true, convert double to float. Otherwise convert float to double
	 * @param 	dimGrid,dimBlock:	CUDA grid/block size
	 */
	void cuReal_convert(float *a, double *b, const unsigned int len, const bool dtof, dim3 dimBlock, dim3 dimGrid);
	/**
	 *	@brief Up/Down partitioning of the pseudofermion field
	 *
	 *	@param	na:	Flavour index
	 *	@param	X0:	Partitioned field
	 *	@param	R1:	Full pseudofermion field
	 * @param 	dimGrid,dimBlock:	CUDA grid/block size
	 */
	void cuUpDownPart(const unsigned int na, Complex *X0, Complex *R1,dim3 dimBlock, dim3 dimGrid);
	/**
	 * @brief Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * @param ut:						Trial fields to be reunitarised
	 * @param dimGrid,dimBlock:	CUDA grid/block size
	 *
	 */
	void cuReunitarise(Complex *ut[2],dim3 dimGrid, dim3 dimBlock);
	/**	
	 * @brief Initialises the CUDA grid and block size for a given lattice
	 *
	 * @param	x,y,z,t:				Lattice dimensions
	 * @param 	dimGrid,dimBlock:	CUDA grid/block size
	 */
	void blockInit(int x, int y, int z, int t, dim3 *dimBlock, dim3 *dimGrid);
#endif
#if (defined __cplusplus)
}
#endif
