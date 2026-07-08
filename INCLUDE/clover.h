/**
 *	@file		clover.h
 *
 *	@brief	Routines needed for Clover imporved wilson fermions
 *
 *	@author 	D. Lawlor
 */
#pragma once
#include <su2hmc.h>


/**
 * @brief Multiply leaf (or part of one) by generator from left
 *
 *	The leaves contributing to each force term need to be scaled by the generator, but the generator appears at
 *	different points in each leaf.  This routine multiples by the generator from the left side.
 *
 *	@param	a:		The leaf or partial leaf
 *	@param	gen:	What generator are we multiplying by?
 *
 *	@post		Product stored in @p a
 */
void ByGenLeft(Complex_f a[nc],const unsigned short gen);
/**
 * @brief Multiply leaf (or part of one) by generator from right
 *
 *	The leaves contributing to each force term need to be scaled by the generator, but the generator appears at
 *	different points in each leaf.  This routine multiples by the generator from the right side.
 *
 *	@param	a:		The leaf or partial leaf
 *	@param	gen:	What generator are we multiplying by?
 *
 *	@post		Product stored in @p a
 */
void ByGenRight(Complex_f a[nc],const unsigned short gen);

/**
 * @brief Calculates the SU2 plaquette at site i in the @f$\mu--\nu@f$ direction
 *
 * @param ut:		Trial fields
 * @param Leaves:	Trial fields
 * @param iu:		Upper halo indices
 * @param i:		site index
 * @param mu, nu:	Plaquette direction. Note that mu and nu can be negative
 * 					to facilitate calculating plaquettes for Clover terms. No
 * 					sanity checks are conducted on them in this routine.
 *
 */
int Clover_SU2plaq(Complex_f *ut[nc], Complex_f Leaves[nc], unsigned int *iu,  int i, int mu, int nu);
/**
 *	@brief Calculates the products of the first two links in a plaquette
 *
 *	@param	hleaves:		Product of first two links
 *	@param	ut:			Gauge fields
 *	@param	iu,id:		Upper and lower indices
 *	@param	mu,nu:		Clover direction
 *
 *	@post	Product of first two links stored in @p hLeaves
 */
void Half_Leaves(Complex_f *hLeaves[2],Complex_f *ut[2], unsigned int *iu,unsigned int *id,\
		const unsigned short mu,const unsigned short nu);
/**
 *	@brief	Calculates a leaf for a clover term.
 *
 *	@param	Leaves:	Array of leaves
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower site indices
 *	@param	i:			Lattice index of the clover in question
 *	@param	mu,nu:	Direction in which we're evaluating the leaf
 *	@param	leaf:		Which leaf of the clover is being calculated
 *	
 *	@post		Clover leaf stored in @p Leaves
 */
int Leaf(Complex_f Leaves[nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id, unsigned int i,\
		const unsigned short mu, const unsigned short nu,const unsigned short leaf);
/**
 *	@brief	Calculates the clover in the forward direction and the leaves. Subtracting the conjugate of this yields the
 *	full clover
 *
 *	@param	clover:	Clover array
 *	@param	Leaves:	Array of leaves
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower site indices
 *	@param	i:			Lattice index of the clover in question
 *	@param	mu,nu:	Direction of the clover
 *
 *	@post		Half clover stored in @p clover
 */
int Half_Clover(Complex_f *clover[nc],	Complex_f *ut[nc], unsigned int *iu, unsigned int *id, int i, int mu, int nu,short clov);
/**
 *	@brief Calculates the clovers in all directions at all sites
 *	@f[ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f]
 *
 *	@param	clover:	Array of clovers
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower indices
 *
 *	@post		Clover stored in @p clover
 */
void Clover(Complex_f *clover[2], Complex_f *ut[2], unsigned int *iu, unsigned int *id);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void ByClover(Complex *phi, Complex *r, Complex *clover[2], Complex *sigval, const float akappa, unsigned short *sigin, bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 * 
 * @post		Result added to @p phi
 */
void ByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[2], Complex_f *sigval, const float akappa, unsigned short *sigin, bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 * 
 * @post		Result added to @p phi
 */
void HbyClover(Complex *phi, Complex *r, Complex *clover[2],Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 * 
 * @post		Result added to @p phi
 */
void HbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[2],Complex_f *sigval, const float akappa, unsigned short *sigin,bool dag);

/**
 *	@brief	Gets @f$X_{\mu\nu}@f$ for the clover force
 *
 *	@param	Xmunu:	All Xmunu values
 *	@param	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param	mu,nu:	Lattice directions
 *
 *	@post	Bilinears written to @p Xmunu
 */
void CalcXmunu(Complex_f *Xmunu, Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
					const unsigned short *sigin,const unsigned short mu, const unsigned short nu);
/**
 *	@brief Gets the clover contribution to the force
 *
 *	@param	dSdpi:	Force
 *	@param	ut:		Gauge fields
 *	@param	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param	iu,id:	Neighbouring sites
 *	
 *	@post		Force contribution added to @p dSdpi
 */
void Clov_Force(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, const Complex_f *sigval, const unsigned short *sigin,\
						unsigned int *iu, unsigned int *id, const float akappa);
/**
 *	@brief	Initialise values needed for the clover terms
 *
 *	@param	sigval,sigval_f:	@f$ \sigma_{\mu\nu}=\frac{1}{2i}[\gamma_\mu,\gamma_\nu]@f$ in double and single precision
 *										scaled by @f$c_{sw}@f$
 *	@param	sigin:				Which column does row idirac of @f$\sigma_{\mu\nu}@f$ act on
 *	@param	c_sw:					Clover coefficient
 *
 *	@post		@p sigval and @p sigval_f initialised with matrix entries. @p sigin initialised with index of non-zero
 *				entries
 */
int Init_clover(Complex **sigval, Complex_f **sigval_f,unsigned short **sigin, float c_sw);
/**
 *	@brief	Free's memory used for clover terms and leaves
 *
 *	@param	clover:	Clovers
 *	
 *	@post		@p clover memory freed
 */
int Clover_free(Complex_f *clover[nc]);

#ifdef __NVCC__
#ifdef __cplusplus
extern "C"
{
#endif

/**
 *	@brief CUDA wrapper for calculating the clovers in all directions at all sites
 *			@f$ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f$
 *
 *	@param	clover:	Array of clovers
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower indices
 *
 *	@post		Clover stored in @p clover
 */
int cuClover(Complex_f *clover[nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id);
/**
 *	@brief CUDA wrapper for ByClover
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:	Hopping Parameter
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuByClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief CUDA wrapper for HbyClover
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:	Hopping Parameter
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuHbyClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief CUDA wrapper for ByClover_f
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float akappa,unsigned short *sigin,bool dag);
/**
 *	@brief CUDA wrapper for HbyClover_f
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:	Hopping Parameter
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:		Daggered has no MPI halo, but undaggered does.
 *
 * @post		Result added to @p phi
 */
void cuHbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float akappa,unsigned short *sigin,bool dag);
/**
 *	@brief	CUDA wrapper for CalcXmunu. Only called during testing to be honest
 *
 *	@param	Xmunu:	All Xmunu values
 *	@param	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param	mu,nu:	Lattice directions
 *
 *	@post	Bilinears written to @p Xmunu
 */
void cuCalcXmunu(Complex_f *Xmunu, Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
		const unsigned short *sigin,const unsigned short mu, const unsigned short nu);
/**
 *	@brief	CUDA wrapper for Clover_Force
 *
 *	@param	dSdpi:		Force
 *	@param	u11t,u12t:	Gauge fields
 *	@param	X1:			@f$\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	X2:			@f$M\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	sigval:		@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$c_sw@f$
 * @param	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	iu,id:		Up/down indices
 * @param	kappa:		Hopping parameter
 *	
 *	@post		Force contribution added to @p dSdpi
 */
int cuClov_Force(double *dSdpi, Complex_f *ut[nc], Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
		const unsigned short *sigin, const unsigned int *iu, const unsigned int *id, const float akappa);
#ifdef __cplusplus
}
#endif
#endif
