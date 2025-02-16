/** @file
 *
 * @brief Code for bosonic observables
 * 
 * @brief Code for bosonic observables, Basically polyakov loop and Plaquette routines.
 *
 * @author S J Hands (Original Fortran, March 2005)
 * @author P. Giudice (Hybrid Code, May 2013)
 * @author D. Lawlor (C version March 2021, CUDA/Mixed Precision/Clover Feb 2024 and beyond...)
 */
#include	<par_mpi.h>
#include	<su2hmc.h>


int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex_f *ut[2], unsigned int *iu, float beta){
	const char *funcname = "Average_Plaquette";
	/*There was a halo exchange here but moved it outside
	  The FORTRAN code used several consecutive loops to get the plaquette
	  Instead we'll just make the arrays variables and do everything in one loop
	  Should work since in the FORTRAN Sigma11[i] only depends on i components  for example
	  Since the \nu loop doesn't get called for \mu=0 we'll start at \mu=1
	  */
#ifdef __NVCC__
	__managed__ double hgs = 0; __managed__ double hgt = 0;
	cuAverage_Plaquette(&hgs, &hgt, ut[0], ut[1], iu,dimGrid,dimBlock);
#else
	double hgs = 0; double hgt = 0;
	for(int mu=1;mu<ndim;mu++)
		for(int nu=0;nu<mu;nu++)
			//Don't merge into a single loop. Makes vectorisation easier?
			//Or merge into a single loop and dispense with the a arrays?
#pragma omp parallel for simd reduction(+:hgs,hgt)
			for(int i=0;i<kvol;i++){
				Complex_f Sigma[2];
				SU2plaq(ut,Sigma,iu,i,mu,nu);
				switch(mu){
					//Time component
					case(ndim-1):	hgt -= creal(Sigma[0]);
										break;
										//Space component
					default:	hgs -= creal(Sigma[0]);
								break;
				}
			}
#endif
#if(nproc>1)
	Par_dsum(&hgs); Par_dsum(&hgt);
#endif
	*avplaqs=-hgs/(3.0*gvol); *avplaqt=-hgt/(gvol*3.0);
	*hg=(hgs+hgt)*beta;
#ifdef _DEBUG
	if(!rank)
		printf("hgs=%e  hgt=%e  hg=%e\n", hgs, hgt, *hg);
#endif
	return 0;
}
#ifndef __NVCC__
#pragma omp declare simd
inline int SU2plaq(Complex_f *ut[2], Complex_f Sigma[2], unsigned int *iu,  int i, int mu, int nu){
	const char *funcname = "SU2plaq";
	int uidm = iu[mu+ndim*i]; 
	/***
	 *	Let's take a quick moment to compare this to the analysis code.
	 *	The analysis code stores the gauge field as a 4 component real valued vector, whereas the produciton code
	 *	used two complex numbers.
	 *
	 *	Analysis code: u=(Re(u11),Im(u12),Re(u12),Im(u11))
	 *	Production code: u11=u[0]+I*u[3]	u12=u[2]+I*u[1]
	 *
	 *	This applies to the Sigmas and a's below too
	 */

	Sigma[0]=ut[0][i*ndim+mu]*ut[0][uidm*ndim+nu]-ut[1][i*ndim+mu]*conj(ut[1][uidm*ndim+nu]);
	Sigma[1]=ut[0][i*ndim+mu]*ut[1][uidm*ndim+nu]+ut[1][i*ndim+mu]*conj(ut[0][uidm*ndim+nu]);

	int uidn = iu[nu+ndim*i]; 
	Complex_f a11=Sigma[0]*conj(ut[0][uidn*ndim+mu])+Sigma[1]*conj(ut[1][uidn*ndim+mu]);
	Complex_f a12=-Sigma[0]*ut[1][uidn*ndim+mu]+Sigma[1]*ut[0][uidn*ndim+mu];

	Sigma[0]=a11*conj(ut[0][i*ndim+nu])+a12*conj(ut[1][i*ndim+nu]);
	Sigma[1]=-a11*ut[1][i*ndim+nu]+a12*ut[0][i*ndim+mu];
	return 0;
}
#endif
double Polyakov(Complex_f *ut[2]){
	const char *funcname = "Polyakov";
	double poly = 0;
	Complex_f *Sigma[2];
#ifdef __NVCC__
	cuPolyakov(Sigma,ut,dimGrid,dimBlock);
#else
	Sigma[0] = (Complex_f *)aligned_alloc(AVX,kvol3*sizeof(Complex_f));
	Sigma[1] = (Complex_f *)aligned_alloc(AVX,kvol3*sizeof(Complex_f));

	//Extract the time component from each site and save in corresponding Sigma
#ifdef USE_BLAS
	cblas_ccopy(kvol3, ut[0]+3, ndim, Sigma[0], 1);
	cblas_ccopy(kvol3, ut[1]+3, ndim, Sigma[1], 1);
#else
	for(int i=0; i<kvol3; i++){
		Sigma[0][i]=ut[0][i*ndim+3];
		Sigma[1][i]=ut[1][i*ndim+3];
	}
#endif
	/*	Some Fortran commentary
		Changed this routine.
		ut[0] and ut[1] now defined as normal ie (kvol+halo,4).
		Copy of Sigma[0] and Sigma[1] is changed so that it copies
		in blocks of ksizet.
		Variable indexu also used to select correct element of ut[0] and ut[1] 
		in loop 10 below.

		Change the order of multiplication so that it can
		be done in parallel. Start at t=1 and go up to t=T:
		previously started at t+T and looped back to 1, 2, ... T-1
		Buffers
		There is a dependency. Can only parallelise the inner loop
		*/
#pragma unroll
	for(int it=1;it<ksizet;it++)
#pragma omp parallel for simd
		for(int i=0;i<kvol3;i++){
			//Seems a bit more efficient to increment indexu instead of reassigning
			//it every single loop
			int indexu=it*kvol3+i;
			Complex_f	a11=Sigma[0][i]*ut[0][indexu*ndim+3]-Sigma[1][i]*conj(ut[1][indexu*ndim+3]);
			//Instead of having to store a second buffer just assign it directly
			Sigma[1][i]=Sigma[0][i]*ut[1][indexu*ndim+3]+Sigma[1][i]*conj(ut[0][indexu*ndim+3]);
			Sigma[0][i]=a11;
		}
	free(Sigma[1]);
#endif

	//Multiply this partial loop with the contributions of the other cores in the
	//Time-like dimension
	//
	//Par_tmul does nothing if there is only a single processor in the time direction. So we only compile
	//its call if it is required
#if (npt>1)
#ifdef __NVCC_
#error Par_tmul is not yet implimented in CUDA as Sigma[1] is device only memory
#endif
#ifdef _DEBUG
	printf("Multiplying with MPI\n");
#endif
	Par_tmul(Sigma[0], Sigma[1]);
	//end of #if(npt>1)
#endif
	/*Now all cores have the value for the complete Polyakov line at all spacial sites
	  We need to globally sum over spacial processors but not across time as these
	  are duplicates. So we zero the value for all but t=0
	  This is (according to the FORTRAN code) a bit of a hack
	  I will expand on this hack and completely avoid any work
	  for this case rather than calculating everything just to set it to zero
	  */
	if(!pcoord[3+rank*ndim])
#pragma omp parallel for simd reduction(+:poly)
		for(int i=0;i<kvol3;i++)
			poly+=creal(Sigma[0][i]);
#ifdef __NVCC__
	cudaFree(Sigma[0]);
#else
	free(Sigma[0]); 
#endif

#if(nproc>1)
	Par_dsum(&poly);
#endif
	poly/=gvol3;
	return poly;	
}
