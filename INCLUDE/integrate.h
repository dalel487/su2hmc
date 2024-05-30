/**
 * @file		integrate.h
 * @brief	Integrators for the HMC
 */
#ifndef INTEGRATE_H
#define INTEGRATE_H
#include <stdbool.h>
#include <random.h>
#include <sizes.h>

	/**
	 *	@brief	Leapfrog integrator. Each trajectory step takes the form of p->p+dt/2,u->u+dt,p->p+dt/2
	 *				In practice this is implemented for the entire trajectory as
	 *				p->p+dt/2,u->u+dt,p->p+dt,u->u+dt,p->p+dt,...p->p+dt/2,u->u+dt,p->p+dt/2
	 *	
	 *	@param	u11t,u12t		Double precision colour fields
	 *	@param	u11t_f,u12t_f:	Single precision colour fields
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk4m_f:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ float
	 * @param	dk4p_f:			@f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param	dSdpi:			The force
	 *	@param	pp:				Momentum field
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices
	 *	@param	gamval_f:		Single precision gamma matrices
	 * @param 	jqq:				Diquark source
	 *	@param	akappa:			Hopping parameter
	 *	@param	beta:				Inverse gauge coupling
	 *	@param	stepl:			Steps per trajectory
	 *	@param	dt:				Step size
	 *	@param	ancg:				Counter for average conjugate gradient iterations
	 *	@param   itot:				Total average conjugate gradient iterations
	 *	@param	proby:			Termination probability for random trajectory length
	 *
	 *	@return Zero on success, integer error code otherwise
	 */
int Leapfrog(Complex *u11t,Complex *u12t,Complex_f *u11t_f,Complex_f *u12t_f,Complex *X0,Complex *X1,
					Complex *Phi,double *dk4m,double *dk4p,float *dk4m_f,float *dk4p_f,double *dSdpi,double *pp,
					int *iu,int *id, Complex *gamval, Complex_f *gamval_f, int *gamin, Complex jqq,
					float beta, float akappa, int stepl, float dt, double *ancg, int *itot, float proby);
	/**
	 *	@brief	OMF integrator. Each trajectory step takes the form of p->p+dt/2,u->u+dt,p->p+dt/2
	 *				In practice this is implemented for the entire trajectory as
	 *				p->p+dt/2,u->u+dt,p->p+dt,u->u+dt,p->p+dt,...p->p+dt/2,u->u+dt,p->p+dt/2
	 *	
	 *	@param	u11t,u12t		Double precision colour fields
	 *	@param	u11t_f,u12t_f:	Single precision colour fields
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk4m_f:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ float
	 * @param	dk4p_f:			@f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param	dSdpi:			The force
	 *	@param	pp:				Momentum field
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices
	 *	@param	gamval_f:		Single precision gamma matrices
	 * @param 	jqq:				Diquark source
	 *	@param	akappa:			Hopping parameter
	 *	@param	beta:				Inverse gauge coupling
	 *	@param	stepl:			Steps per trajectory
	 *	@param	dt:				Step size
	 *	@param	ancg:				Counter for average conjugate gradient iterations
	 *	@param   itot:				Total average conjugate gradient iterations
	 *	@param	proby:			Termination probability for random trajectory length
	 *	@param	alpha:			Tunes the size of the momentum updates. If equal to .5 should in theory give the leapfrog
	 *
	 *	@return Zero on success, integer error code otherwise
	 */
int OMF2(Complex *u11t,Complex *u12t,Complex_f *u11t_f,Complex_f *u12t_f,Complex *X0,Complex *X1,
					Complex *Phi,double *dk4m,double *dk4p,float *dk4m_f,float *dk4p_f,double *dSdpi,double *pp,
					int *iu,int *id, Complex *gamval, Complex_f *gamval_f, int *gamin, Complex jqq,
					float beta, float akappa, int stepl, float dt, double *ancg, int *itot, float proby, float alpha);
#endif
