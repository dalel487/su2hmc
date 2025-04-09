#pragma once
#include <coord.h>
#include	<errorcodes.h>
#include	<stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/**
 *	@file		clover.h
 *
 *	@brief	Routines needed for Clover imporved wilson fermions
 *
 *	@author 	D. Lawlor
 */

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
int Clover_SU2plaq(Complex_f *ut[2], Complex_f *Leaves[2], unsigned int *iu,  int i, int mu, int nu);
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
int Leaf(Complex_f *ut[2], Complex_f *Leaves[2], unsigned int *iu, unsigned int *id, int i, int mu, int nu, short leaf);
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
int Half_Clover(Complex_f *clover[2],	Complex_f *Leaves[2], Complex_f *ut[2], unsigned int *iu, unsigned int *id, int i, int mu, int nu);
/**
 *	@brief Calculates the clovers in all directions at all sites
 *	@f$ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f$
 *
 *	@param	clover:	Array of clovers
 *	@param	Leaves:	Array of clover leaves
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower indices
 */
int Clover(Complex_f *clover[6][2],Complex_f *Leaves[6][2],Complex_f *ut[2], unsigned int *iu, unsigned int *id);
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by c_sw
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 */
int ByClover(Complex_f *phi, Complex_f *r, Complex_f *clover[6][2], Complex_f *sigval, unsigned short *sigin);
/**
 *	@brief Clover analogue of the Dslashd operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by c_sw
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 */
int ByCloverd(Complex_f *phi, Complex_f *r, Complex_f *clover[6][2], Complex_f *sigval, unsigned short *sigin);
/**
 *	@brief Clover analogue of the Dslash operation. The H in front is for half, as we only act on the fermions of flavour
 *	1
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by c_sw
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 */
int HbyClover(Complex_f *phi, Complex_f *r, Complex_f *clover[6][2], Complex_f *sigval, unsigned short *sigin);
/**
 *	@brief Clover analogue of the Dslashd operation. The H in front is for half, as we only act on the fermions of flavour
 *	1
 *
 *	@param	phi:		Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:			Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:	Array of clovers
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by c_sw
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 */
int HbyCloverd(Complex_f *phi, Complex_f *r, Complex_f *clover[6][2], Complex_f *sigval, unsigned short *sigin);
/**
 *	@brief	Clover contribution to the Molecular Dynamics force
 *
 *	@param	dSdpi:	Force
 *	@param	Leaves:	Clover leaves. We don't need the full clover for the force as most get killed in the derivative
 *							We do however need the individual leaves making up the clover
 *	@param	sigval:	@f$ \sigma_{\mu\nu}@f$ entries scaled by c_sw
 * @param	sigin:	What element of the spinor is multiplied by row idirac each sigma matrix?
 */
int Clover_Force(double *dSdpi, Complex_f *Leaves[6][2], Complex_f *X1, Complex_f *X2, Complex_f *sigval,unsigned short *sigin);
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
int GenLeaf(Complex_f Fleaf[2], Complex_f *Leaves[2],const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm);
/**
 *	@brief	Scales the hermitian conjugate of a clover leaf by the relevant SU(2) generator
 *
 *	@param	Fleaf:	Array of scaled leaves. Name comes from Force-leaf as thats where we use them
 *	@param	Leaves:	Array of clover leaves being scaled
 *	@param	i:			Site index
 *	@param	leaf:		Which leaf are we scaling
 *	@param	adj:		Which generator. Since we're zero indexed subtract one from the usual textbook label
 *	@param	pm:		Are we adding or subtracting this contribution from Fleaf? The force only needs the sum of the
 *							Fleaf terms so I've done it here.
 */
int GenLeafd(Complex_f Fleaf[2], Complex_f *Leaves[2],const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm);
/**
 *	@brief	Initialise values needed for the clover terms
 *
 *	@param	sigval,sigval_f:	@f$ \sigma_{\mu\nu}=\frac{1}{2i}[\gamma_\mu,\gamma_\nu]@f$ in double and single precision
 *	@param	sigin:				Which column does row idirac of @f$(\sigma_{\mu\nu}@f$ act on
 *	@param	c_sw:					Clover coefficient
 */
int Init_clover(Complex *sigval, Complex_f *sigval_f,unsigned short *sigin, float c_sw);
/**
 *	@brief	Free's memory used for clover terms and leaves
 *
 *	@param	clover:	Clovers
 *	@param		Leaves:	Leaves
 */
int Clover_free(Complex_f *clover[6][2],Complex_f *Leaves[6][2]);
