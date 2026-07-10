/**
 * @file		integrate.h
 * @brief	Integrators for the HMC
 * @author D.Lawlor
 *
 *	@defgroup MD Molecular dynamics
 */
#ifndef INTEGRATE_H
#define INTEGRATE_H
#include <random.h>

#if (defined __cplusplus)
extern "C"
{
#endif
	/**
	 * @brief Gauge update for the integration step of the HMC
	 *	@ingroup MD
	 *
	 * @param[in] d:		Gauge step size
	 * @param[in] pp:		Momentum field
	 * @param[in,out] ut:		Double precision gauge fields
	 * @param[in,out] ut_f:	Single precision gauge fields
	 *
	 * @return Zero on success, integer error code otherwise.
	 * @post @p ut and @p ut_f updated in place
	 */
	int Gauge_Update(const double d, double *pp, Complex *ut[2],Complex_f *ut_f[2]);
	/**
	 * @brief Wrapper for the momentum update during the integration step of the HMC
	 *	@ingroup MD
	 *
	 * @param[in] d:		Step size
	 * @param[in,out] pp:		Momentum field
	 * @param[in] dSdpi:	Force field
	 *
	 * @return Zero on success, integer error code otherwise.
	 * @post @p pp update in place
	 */
	int Momentum_Update(const double d,const double *dSdpi, double *pp);
	/**
	 *	@brief	Leapfrog integrator. Each trajectory step takes the form of p->p+dt/2,u->u+dt,p->p+dt/2
	 *				In practice this is implemented for the entire trajectory as
	 *				p->p+dt/2,u->u+dt,p->p+dt,u->u+dt,p->p+dt,...p->p+dt/2,u->u+dt,p->p+dt/2
	 *	@ingroup MD
	 *	
	 *	@param[in,out]	ut					Double precision colour fields
	 *	@param[in,out]	ut_f:				Single precision colour fields
	 *	@param[in]	X0:				Up/down partitioned pseudofermion field
	 *	@param[in]	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param[in]	Phi:				Pseudofermion field
	 * @param[in]	dk:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param[in]	dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param[in,out]	dSdpi:			The force
	 *	@param[in,out]	pp:				Momentum field
	 *	@param[in]	iu,id:			Lattice indices
	 *	@param[in]	gamin:			Gamma indices
	 *	@param[in]	gamval:			Double precision gamma matrices rescaled by kappa
	 *	@param[in]	gamval_f:		Single precision gamma matrices rescaled by kappa
	 *	@param[in]	sigval,sigval_f:	Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}{2}@f$
	 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	c_sw:				Clover coefficient
	 * @param[in] 	jqq:				Diquark source
	 *	@param[in]	akappa:			Hopping parameter
	 *	@param[in]	beta:				Inverse gauge coupling
	 *	@param[in]	stepl:			Steps per trajectory
	 *	@param[in]	dt:				Step size
	 *	@param[in]	ancg:				Counter for average conjugate gradient iterations
	 *	@param[in]   itot:				Total average conjugate gradient iterations
	 *	@param[in]	proby:			Termination probability for random trajectory length
	 *
	 *	@return Zero on success, integer error code otherwise
	 *	@post	@p ut, @p ut_f, @p pp and @p dSdpi updated in place throughout the integration
	 */
	int Leapfrog(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
			double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex gamval[20], Complex_f gamval_f[20], const unsigned short gamin[16],
			Complex *sigval, Complex_f *sigval_f, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
			const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby);
	/**
	 *	@brief	OMF second order five step integrator.
	 *	@ingroup MD
	 *	
	 *	@param[in,out]	ut					Double precision colour fields
	 *	@param[in,out]	ut_f:				Single precision colour fields
	 *	@param[in]	X0:				Up/down partitioned pseudofermion field
	 *	@param[in]	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param[in]	Phi:				Pseudofermion field
	 * @param[in]	dk:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param[in]	dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param[in,out]	dSdpi:			The force
	 *	@param[in,out]	pp:				Momentum field
	 *	@param[in]	iu,id:			Lattice indices
	 *	@param[in]	gamin:			Gamma indices
	 *	@param[in]	gamval:			Double precision gamma matrices rescaled by kappa
	 *	@param[in]	gamval_f:		Single precision gamma matrices rescaled by kappa
	 *	@param[in]	sigval,sigval_f:	Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}{2}@f$
	 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	c_sw:				Clover coefficient
	 * @param[in] 	jqq:				Diquark source
	 *	@param[in]	akappa:			Hopping parameter
	 *	@param[in]	beta:				Inverse gauge coupling
	 *	@param[in]	stepl:			Steps per trajectory
	 *	@param[in]	dt:				Step size
	 *	@param[in]	ancg:				Counter for average conjugate gradient iterations
	 *	@param[in]   itot:				Total average conjugate gradient iterations
	 *	@param[in]	proby:			Termination probability for random trajectory length
	 *
	 *	@return Zero on success, integer error code otherwise
	 *	@post	@p ut, @p ut_f, @p pp and @p dSdpi updated in place throughout the integration
	 */
	int OMF2(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
			double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex gamval[20], Complex_f gamval_f[20], const unsigned short gamin[16],
			Complex *sigval, Complex_f *sigval_f, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
			const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby);
	/**
	 *	@brief	OMF fourth order eleven step integrator.
	 *	@ingroup MD
	 *	
	 *	@param[in,out]	ut					Double precision colour fields
	 *	@param[in,out]	ut_f:				Single precision colour fields
	 *	@param[in]	X0:				Up/down partitioned pseudofermion field
	 *	@param[in]	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param[in]	Phi:				Pseudofermion field
	 * @param[in]	dk:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param[in]	dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param[in,out]	dSdpi:			The force
	 *	@param[in,out]	pp:				Momentum field
	 *	@param[in]	iu,id:			Lattice indices
	 *	@param[in]	gamin:			Gamma indices
	 *	@param[in]	gamval:			Double precision gamma matrices rescaled by kappa
	 *	@param[in]	gamval_f:		Single precision gamma matrices rescaled by kappa
	 *	@param[in]	sigval,sigval_f:	Double/float Commutators of gamma matrices scaled by @f$\frac{c_\text{SW}}{2}@f$
	 * @param[in]	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
	 * @param[in]	c_sw:				Clover coefficient
	 * @param[in] 	jqq:				Diquark source
	 *	@param[in]	akappa:			Hopping parameter
	 *	@param[in]	beta:				Inverse gauge coupling
	 *	@param[in]	stepl:			Steps per trajectory
	 *	@param[in]	dt:				Step size
	 *	@param[in]	ancg:				Counter for average conjugate gradient iterations
	 *	@param[in,out]   itot:				Total average conjugate gradient iterations
	 *	@param[in]	proby:			Termination probability for random trajectory length
	 *
	 *	@return Zero on success, integer error code otherwise
	 *	@post	@p ut, @p ut_f, @p pp and @p dSdpi updated in place throughout the integration
	 */
	int OMF4(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
			double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex gamval[20], Complex_f gamval_f[20], const unsigned short gamin[16],
			Complex *sigval, Complex_f *sigval_f, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
			const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby);
	//CUDA Calling functions
#ifdef __NVCC__
	/**
	 * @brief CUDA wrapper for the gauge update during the integration step of the HMC
	 *	@ingroup MD
	 *
	 * @param[in] d:						Gauge step size
	 * @param[in] pp:						Momentum field
	 * @param[in,out] ut		:				Double precision gauge fields
	 * @param[in] dimGrid,dimBlock:	CUDA Grid/Block dimensions
	 *
	 * @post @p ut updated in place
	 */
	void cuGauge_Update(const double d, double *pp, Complex *ut[2], dim3 dimGrid, dim3 dimBlock);
#endif

#if (defined __cplusplus)
}
#endif
#endif
