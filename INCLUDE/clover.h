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
 * @brief Calculates the SU2 plaquette at site i in the @f$\mu--\nu@f$ direction
 *
 * @param ut:		Trial fields
 * @param Leaves:	Trial fields
 * @param iu:		Upper halo indices
 * @param i:		site index
 * @param mu, nu:	Plaquette direction. Note that mu and nu can be negative
 * 					to facilitate calculating plaquettes for Clover terms. No
 * 					sanity checks are conducted on them in this routine.
 */
int Clover_SU2plaq(Complex_f *ut[nc], Complex_f Leaves[nc], unsigned int *iu,  int i, int mu, int nu);
/**
 *	@brief	Calculates a leaf for a clover term.
 *
 *	@param	ut:		Gauge fields
 *	@param	Leaves:	Array of leaves
 *	@param	iu,id:	Upper and lower site indices
 *	@param	i:			Lattice index of the clover in question
 *	@param	mu,nu:	Direction in which we're evaluating the leaf
 *	@param	leaf:		Which leaf of the clover is being calculated
 *	
 */
int Leaf(Complex_f *ut[nc], Complex_f Leaves[nc], unsigned int *iu, unsigned int *id, int i, int mu, int nu, short leaf);
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
 */
int Half_Clover(Complex_f *clover[nc],	Complex_f *ut[nc], unsigned int *iu, unsigned int *id, int i, int mu, int nu,short clov);
/**
 *	@brief Calculates the clovers in all directions at all sites
 *	@f$ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f$
 *
 *	@param	clover:	Array of clovers
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower indices
 */
int Clover(Complex_f *clover[nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:	Hopping Parameter
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
 */
int ByClover(Complex *phi, Complex *r, Complex *clover[nc], Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:	Hopping Parameter
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
 */
int ByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc], Complex_f *sigval,const float akappa,  unsigned short *sigin,bool dag);
/**
 *	@brief Clover analogue of the Dslash operation. The H in front is for half, as we only act on the fermions of flavour
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:	Hopping Parameter
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
 */
int HbyClover(Complex *phi, Complex *r, Complex *clover[nc], Complex *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 *	@brief Clover analogue of the Dslashd operation. The H in front is for half, as we only act on the fermions of flavour
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:	Hopping Parameter
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
 */
int HbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc], Complex_f *sigval, const float akappa, unsigned short *sigin,bool dag);
/**
 * @brief	Extracts the leaves required for the clover force term and adds them correctly
 *
 * @param	fleaf:	The summed leaves
 * @param	Leaves:	The individual leaves of a particular clover. The clover itself is chosen in Clover_Force
 * @param	i:			Lattice site index
 */
void Fleaf(Complex_f fleaf[nc], Complex_f *Leaves[nc], const unsigned int i);
/**
 *	@brief	CUDA wrapper for Clover_Force
 *
 *	@param	dSdpi:		Force
 *	@param	ut:		Gauge fields
 *	@param	X1:			@f$\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	X2:			@f$M\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	sigval:		@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$c_sw@f$
 * @param	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	iu:			Up indices
 * @param	id:			Down indices
 * @param	kappa:		Hopping parameter
 */
int Clover_Force(double *dSdpi, Complex_f *ut[nc],Complex_f *X1, Complex_f *X2, Complex_f *sigval,\
		unsigned short *sigin, unsigned int *iu, unsigned int *id, const float kappa);
/**
 *	@brief	Scales a clover leaf by the relevant SU(2) generator
 *
 *	@param	Fleaf:	Array of scaled leaves. Name comes from Force-leaf as thats where we use them
 *	@param	Leaves:	Array of clover leaves being scaled
 *	@param	i:			Site index
 *	@param	leaf:		Which leaf are we scaling
 *	@param	adj:		Which generator. Since we're zero indexed subtract one from the usual textbook label
 *	@param	pm:		Are we adding or subtracting this contribution from Fleaf? The force only needs the sum of the
 *							Fleaf terms so I've done it here.
 */
void GenLeaf(Complex_f Fleaf[nc],const unsigned short adj);
/**
 *	@brief	Initialise values needed for the clover terms
 *
 *	@param	sigval,sigval_f:	@f$ \sigma_{\mu\nu}=\frac{1}{2i}[\gamma_\mu,\gamma_\nu]@f$ in double and single precision	scaled by @f$c_{sw}@f$
 *	@param	sigin:				Which column does row idirac of @f$\sigma_{\mu\nu}@f$ act on
 *	@param	c_sw:					Clover coefficient
 */
int Init_clover(Complex **sigval, Complex_f **sigval_f,unsigned short **sigin, float c_sw);
/**
 *	@brief	Free's memory used for clover terms and leaves
 *
 *	@param	clover:	Clovers
 *	@param	Leaves:	Leaves
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
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
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
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
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
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
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
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
 */
void cuHbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float akappa,unsigned short *sigin,bool dag);
/**
 *	@brief	CUDA wrapper for Clover_Force
 *
 *	@param	dSdpi:		Force
 *	@param	u11t,u12t:	Gauge fields
 *	@param	X1:			@f$\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	X2:			@f$M\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	sigval:		@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$c_sw@f$
 * @param	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	iu:			Up indices
 * @param	id:			Down indices
 * @param	kappa:		Hopping parameter
 */
int cuClover_Force(double *dSdpi, Complex_f *ut[nc], Complex_f *X1, Complex_f *X2, Complex_f *sigval,\
		unsigned short *sigin, unsigned int *iu, unsigned int *id, const float kappa);
#ifdef __cplusplus
}
#endif
#endif
