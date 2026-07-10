/**
 *	@file		clover.h
 *
 *	@brief	Routines needed for Clover improved wilson fermions
 *
 *	@author 	D. Lawlor
 *	@todo	Multiple MPI Ranks are not currently supported for the clover action. This is due to the corner halo terms
 *				needed to compute the clover force not being implemented.
 *	@defgroup Clover
 *	Clover related functions
 *
 *	@defgroup Clover_Force
 *	Clover Force related functions
 *	@ingroup Clover
 *	@ingroup MD
 *
 *	@defgroup Clover_Prod
 *	Clover Multiplication routines
 *	@ingroup Clover
 *	@ingroup Dslashes
 */
#pragma once
#include <su2hmc.h>

/**
 * @brief	Structure of arrays for Hermitian bilinear @f$X_{\mu\nu}@f$ in memory.
 *	@ingroup Clover_Force
 */
typedef struct{
 ///	Real valued diagonal terms.
	float *diag;
 ///	Complex valued off-diagonal terms. We only need to store one of these to get the other in @f$SU(2)@f$.
	Complex_f *offd;
}Bilinear_a;
/**
 * @brief	Hermitian bilinear @f$X_{\mu\nu}@f$ on the local stack.
 *	@ingroup Clover_Force
 */
typedef struct{
 ///	Real valued diagonal terms.
	float diag[2];
 ///	Complex valued off-diagonal terms. We only need to store one of these to get the other in @f$SU(2)@f$.
	Complex_f offd;
}Bilinear;

/**
 * @brief Multiply leaf (or part of one) by generator from left
 * @ingroup Clover
 *
 *	The leaves contributing to each force term need to be scaled by the generator, but the generator appears at
 *	different points in each leaf.  This routine multiples by the generator from the left side.
 *
 *	@param[in,out]	a:		The leaf or partial leaf
 *	@param[in]	gen:	What generator are we multiplying by?
 *
 *	@post		Product stored in @p a
 */
void ByGenLeft(Complex_f a[nc],const unsigned short gen);
/**
 * @brief Multiply leaf (or part of one) by generator from right
 * @ingroup Clover
 *
 *	The leaves contributing to each force term need to be scaled by the generator, but the generator appears at
 *	different points in each leaf.  This routine multiples by the generator from the right side.
 *
 *	@param[in,out]	a:		The leaf or partial leaf
 *	@param[in]	gen:	What generator are we multiplying by?
 *
 *	@post		Product stored in @p a
 */
void ByGenRight(Complex_f a[nc],const unsigned short gen);

/**
 * @brief Calculates the SU2 plaquette at site i in the @f$\mu--\nu@f$ direction
 * @ingroup Clover
 *
 * @param[in] ut:		Trial fields
 * @param[out] Leaves:	Trial fields
 * @param[in] iu:		Upper halo indices
 * @param[in] i:		site index
 * @param[in] mu, nu:	Plaquette direction. Note that mu and nu can be negative
 * 					to facilitate calculating plaquettes for Clover terms. No
 * 					sanity checks are conducted on them in this routine.
 *	@post	Leaves overwritten by plaquette values
 */
void Clover_SU2plaq(Complex_f *ut[nc], Complex_f Leaves[nc], unsigned int *iu,  int i, int mu, int nu);
/**
 *	@brief Calculates the products of the first two links in a plaquette
 * @ingroup Clover
 *
 *	@param[out]	hLeaves:		Product of first two links
 *	@param[in]	ut:			Gauge fields
 *	@param[in]	iu,id:		Upper and lower indices
 *	@param[in]	mu,nu:		Clover direction
 *
 *	@post	Product of first two links stored in @p hLeaves
 */
void Half_Leaves(Complex_f *hLeaves[2],Complex_f *ut[2], unsigned int *iu,unsigned int *id,\
		const unsigned short mu,const unsigned short nu);
/**
 *	@brief	Calculates a leaf for a clover term.
 * @ingroup Clover
 *
 *	@param[out]	Leaves:	Array of leaves
 *	@param[in]	ut:		Gauge fields
 *	@param[in]	iu,id:	Upper and lower site indices
 *	@param[in]	i:			Lattice index of the clover in question
 *	@param[in]	mu,nu:	Direction in which we're evaluating the leaf
 *	@param[in]	leaf:		Which leaf of the clover is being calculated
 *	
 *	@post		Clover leaf stored in @p Leaves
 */
void Leaf(Complex_f Leaves[nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id, unsigned int i,\
		const unsigned short mu, const unsigned short nu,const unsigned short leaf);
/**
 *	@brief Calculates the clovers in all directions at all sites
 *	@f[ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f]
 *	@ingroup Clover
 *
 *	@param[out]	clover:	Array of clovers
 *	@param[in]	ut:		Gauge fields
 *	@param[in]	iu,id:	Upper and lower indices
 *
 *	@post		Clover stored in @p clover
 */
void Clover(Complex_f *clover[2], Complex_f *ut[2], unsigned int *iu, unsigned int *id);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours similar to Dslash and Dslash_d
 *	@ingroup Clover_Prod
 *
 *	@param[out,in]	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:				Array of clovers
 *	@param[in]	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:				Hopping Parameter
 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:					Daggered output has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void ByClover(Complex *phi, Complex *r, Complex *clover[2], Complex *sigval, const float akappa, unsigned short *sigin, bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours similar to Dslash and Dslash_d
 *	@ingroup Clover_Prod
 *
 *	@param[out,in]	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:				Array of clovers
 *	@param[in]	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:				Hopping Parameter
 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:					Daggered output has no MPI halo, but undaggered does.
 * 
 * @post		Result added to @p phi
 */
void ByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[2], Complex_f *sigval, const float akappa, unsigned short *sigin, bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours similar to Dslash and Dslash_d
 *	@ingroup Clover_Prod
 *
 *	@param[out,in]	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:				Array of clovers
 *	@param[in]	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:				Hopping Parameter
 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:					Daggered output has no MPI halo, but undaggered does.
 * 
 * @post		Result added to @p phi
 */
void HbyClover(Complex *phi, Complex *r, Complex *clover[2],Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours similar to Dslash and Dslash_d
 *	@ingroup Clover_Prod
 *
 *	@param[out,in]	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:				Array of clovers
 *	@param[in]	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:				Hopping Parameter
 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:					Daggered output has no MPI halo, but undaggered does.
 * 
 * @post		Result added to @p phi
 */
void HbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[2],Complex_f *sigval, const float akappa, unsigned short *sigin,bool dag);

/**
 *	@brief	Gets @f$X_{\mu\nu}@f$ for the clover force
 *	@ingroup Clover_Force
 *
 *	@param[out]	Xmunu:	All Xmunu values
 *	@param[in]	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param[in]	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param[in]	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param[in]	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param[in]	mu,nu:	Lattice directions
 *
 *	@post	Bilinears written to @p Xmunu
 */
void CalcXmunu(Bilinear_a Xmunu, Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
					const unsigned short *sigin,const unsigned short mu, const unsigned short nu);
/**
 *	@brief Gets the clover contribution to the force
 *	@ingroup Clover_Force
 *
 *	@param[in,out]	dSdpi:	Force
 *	@param[in]	ut:		Gauge fields
 *	@param[in]	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param[in]	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param[in]	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param[in]	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param[in]	iu,id:	Neighbouring sites
 *	@param[in]	akappa:	Hopping parameter
 *	
 *	@post		Force contribution added to @p dSdpi
 */
void Clov_Force(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, const Complex_f *sigval, const unsigned short *sigin,\
						unsigned int *iu, unsigned int *id, const float akappa);
/**
 *	@brief	Initialise values needed for the clover terms
 *	@ingroup Clover
 *
 *	@param[out]	sigval,sigval_f:	@f$ \sigma_{\mu\nu}=\frac{1}{2i}[\gamma_\mu,\gamma_\nu]@f$ in double and single precision
 *										scaled by @f$c_{sw}@f$
 *	@param[out]	sigin:				Which column does row idirac of @f$\sigma_{\mu\nu}@f$ act on
 *	@param[in]	c_sw:					Clover coefficient
 *
 *	@return	Zero on success, integer error code otherwise
 *	@post		@p sigval and @p sigval_f initialised with matrix entries. @p sigin initialised with index of non-zero
 *				entries
 */
int Init_clover(Complex **sigval, Complex_f **sigval_f,unsigned short **sigin, float c_sw);
/**
 *	@brief	Free's memory used for clover terms and leaves
 *	@ingroup Clover
 *
 *	@param[in,out]	clover:	Clovers
 *	
 *	@post		@p clover memory freed
 */
void Clover_free(Complex_f *clover[nc]);

#ifdef __NVCC__
#ifdef __cplusplus
extern "C"
{
#endif

/**
 *	@brief CUDA wrapper for calculating the clovers in all directions at all sites
 *			@f$ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f$
 *	@ingroup Clover
 *
 *	@param[out]	clover:	Array of clovers
 *	@param[in]	ut:		Gauge fields
 *	@param[in]	iu,id:	Upper and lower indices
 *
 *	@return	Zero on success, integer error code otherwise
 *	@post		Clover stored in @p clover
 */
int cuClover(Complex_f *clover[nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id);
/**
 *	@brief CUDA wrapper for ByClover
 *	@ingroup Clover_Prod
 *
 *	@param[in,out]	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:	Array of clovers
 *	@param[in]	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:	Hopping Parameter
 * @param[in]	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuByClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief CUDA wrapper for HbyClover
 *	@ingroup Clover_Prod
 *
 *	@param[in,out]	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:	Array of clovers
 *	@param[in]	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:	Hopping Parameter
 * @param[in]	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuHbyClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief CUDA wrapper for ByClover_f
 *	@ingroup Clover_Prod
 *
 *	@param[in,out]	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:	Array of clovers
 *	@param[in]	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:	Hopping Parameter
 * @param[in]	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float akappa,unsigned short *sigin,bool dag);
/**
 *	@brief CUDA wrapper for HbyClover_f
 *	@ingroup Clover_Prod
 *
 *	@param[in,out]	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param[in]	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param[in]	clover:	Array of clovers
 *	@param[in]	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param[in]	akappa:	Hopping Parameter
 * @param[in]	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuHbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float akappa,unsigned short *sigin,bool dag);
/**
 *	@brief	CUDA wrapper for CalcXmunu. Only called during testing to be honest
 *	@ingroup Clover_Force
 *
 *	@param[out]	Xmunu:	All Xmunu values
 *	@param[in]	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param[in]	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param[in]	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param[in]	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param[in]	mu,nu:	Lattice directions
 *
 *	@post	Bilinears written to @p Xmunu
 */
void cuCalcXmunu(Bilinear_a Xmunu, Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
		const unsigned short *sigin,const unsigned short mu, const unsigned short nu);
/**
 *	@brief	CUDA wrapper for Clover_Force
 *	@ingroup Clover_Force
 *
 *	@param[in,out]	dSdpi:	Force
 *	@param[in]	ut:			Gauge fields
 *	@param[in]	X1:			@f$\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param[in]	X2:			@f$M\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param[in]	sigval:		@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$c_sw@f$
 * @param[in]	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param[in]	iu,id:		Up/down indices
 * @param[in]	akappa:		Hopping parameter
 *	
 *	@return	Zero on success, integer error code otherwise
 *	@post		Force contribution added to @p dSdpi
 */
int cuClov_Force(double *dSdpi, Complex_f *ut[nc], Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
		const unsigned short *sigin, const unsigned int *iu, const unsigned int *id, const float akappa);
#ifdef __cplusplus
}
#endif
#endif
