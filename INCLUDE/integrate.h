/**
 * @file		integrate.h
 * @brief	Integrators for the HMC
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
	 *
	 * @param d:		Gauge step size
	 * @param pp:		Momentum field
	 * @param ut:		Double precision gauge fields
	 * @param ut_f:	Single precision gauge fields
	 *
	 * @return Zero on success, integer error code otherwise.
	 */
	int Gauge_Update(const double d, double *pp, Complex *ut[2],Complex_f *ut_f[2]);
	/**
	 * @brief Wrapper for the momentum update during the integration step of the HMC
	 *
	 * @param d:		Step size
	 * @param pp:		Momentum field
	 * @param dSdpi:	Force field
	 *
	 * @return Zero on success, integer error code otherwise.
	 */
	int Momentum_Update(const double d,const double *dSdpi, double *pp);
	/**
	 *	@brief	Leapfrog integrator. Each trajectory step takes the form of p->p+dt/2,u->u+dt,p->p+dt/2
	 *				In practice this is implemented for the entire trajectory as
	 *				p->p+dt/2,u->u+dt,p->p+dt,u->u+dt,p->p+dt,...p->p+dt/2,u->u+dt,p->p+dt/2
	 *	
	 *	@param	ut					Double precision colour fields
	 *	@param	ut_f:				Single precision colour fields
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 * @param	dk:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param	dSdpi:			The force
	 *	@param	pp:				Momentum field
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices rescaled by kappa
	 *	@param	gamval_f:		Single precision gamma matrices rescaled by kappa
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
int Leapfrog(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
			double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex *gamval, Complex_f *gamval_f, int *gamin,
			Complex_f *sigval, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
			const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby);
	/**
	 *	@brief	OMF second order five step integrator.
	 *	
	 *	@param	ut					Double precision colour fields
	 *	@param	ut_f:				Single precision colour fields
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 * @param	dk:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param	dSdpi:			The force
	 *	@param	pp:				Momentum field
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices rescaled by kappa
	 *	@param	gamval_f:		Single precision gamma matrices rescaled by kappa
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
int OMF2(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
			double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex *gamval, Complex_f *gamval_f, int *gamin,
			Complex_f *sigval, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
			const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby);
	/**
	 *	@brief	OMF fourth order eleven step integrator.
	 *	
	 *	@param	ut					Double precision colour fields
	 *	@param	ut_f:				Single precision colour fields
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 * @param	dk:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk_f:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1-\gamma_0\right)e^\mu@f$ float
	 *	@param	dSdpi:			The force
	 *	@param	pp:				Momentum field
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices rescaled by kappa
	 *	@param	gamval_f:		Single precision gamma matrices rescaled by kappa
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
int OMF4(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
			double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex *gamval, Complex_f *gamval_f, int *gamin,
			Complex_f *sigval, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
			const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby);
	//CUDA Calling functions
#ifdef __NVCC__
	void cuGauge_Update(const double d, double *pp, Complex *u11t, Complex *u12t, dim3 dimGrid, dim3 dimBlock);
#endif

#if (defined __cplusplus)
}
#endif
//Actual CUDA
#ifdef __CUDACC__
//Update Gauge fields
__global__ void cuGauge_Update(const double d, double *pp, Complex *u11t, Complex *u12t, int mu);
#endif
#endif
