/**
 * @file bosonic.c
 *
 * @brief Code for bosonic observables, Basically polyakov loop and Plaquette routines.
 */
#include	<par_mpi.h>
#include	<su2hmc.h>

/** @file
 *
 * @brief Code for bosonic observables
 * 
 * Routines for polyakov loop, plaquettes and clovers.
 *
 * @author S J Hands (Original Fortran, March 2005)
 * @author P. Giudice (Hybrid Code, May 2013)
 * @author D. Lawlor (C version March 2021, CUDA/Mixed Precision/Clover Feb 2024 and beyond...)
 */

int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex_f *ut[2], unsigned int *iu, float beta){
	/* 
	 *	Calculates the gauge action using new (how new?) lookup table
	 *	Follows a routine called qedplaq in some QED3 code
	 * 
	 * Parameters:
	 * =========
	 * hg				Gauge component of Hamilton
	 * avplaqs		Average spacial Plaquette
	 * avplaqt		Average Temporal Plaquette
	 * u11t,u12t	The trial fields
	 * iu				Upper halo indices
	 * beta			Inverse gauge coupling
	 *
	 * Calls:
	 * ======
	 * Par_dsum()
	 *
	 * Return:
	 * ======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Average_Plaquette";
	/*There was a halo exchange here but moved it outside
	  The FORTRAN code used several consecutive loops to get the plaquette
	  Instead we'll just make the arrays variables and do everything in one loop
	  Should work since in the FORTRAN Sigma11[i] only depends on i components  for example
	  Since the ν loop doesn't get called for μ=0 we'll start at μ=1
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
	/**
	 * @brief Calculates the trace of the plaquette at site i in the μ-ν direction
	 *
	 * @param ut[0], ut[1]:			Trial fields
	 * @param Sigma11, Sigma12:	Trial fields
	 * @param *iu:						Upper halo indices
	 * @param i:						site index
	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
	 * 									to facilitate calculating plaquettes for Clover terms. No
	 * 									sanity checks are conducted on them in this routine.
	 *
	 * Return:
	 * =======
	 * double corresponding to the plaquette value
	 *
	 */
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
	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * @param ut[0], ut[1]	The trial fields
	 * 
	 * Calls:
	 * ======
	 * Par_tmul(), Par_dsum()
	 * 
	 * @return Double corresponding to the polyakov loop
	 */
	const char *funcname = "Polyakov";
	double poly = 0;
	Complex_f *Sigma[2];
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	cudaMallocManaged((void **)&Sigma[0],kvol3*sizeof(Complex_f),cudaMemAttachGlobal);
#ifdef _DEBUG
	cudaMallocManaged((void **)&Sigma[1],kvol3*sizeof(Complex_f),cudaMemAttachGlobal);
#else
	cudaMallocAsync((void **)&Sigma[1],kvol3*sizeof(Complex_f),streams[0]);
#endif
#else
	Sigma[0] = (Complex_f *)aligned_alloc(AVX,kvol3*sizeof(Complex_f));
	Sigma[1] = (Complex_f *)aligned_alloc(AVX,kvol3*sizeof(Complex_f));
#endif

	//Extract the time component from each site and save in corresponding Sigma
#ifdef __NVCC__
	cublasCcopy(cublas_handle,kvol3, (cuComplex *)(ut[0])+3*kvol, 1, (cuComplex *)Sigma[0], 1);
	cublasCcopy(cublas_handle,kvol3, (cuComplex *)(ut[1])+3*kvol, 1, (cuComplex *)Sigma[1], 1);
#elif defined USE_BLAS
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
#ifdef __NVCC__
	cudaDeviceSynchronise();
	cuPolyakov(Sigma[0],Sigma[1],ut[0],ut[1],dimGrid,dimBlock);
	cudaDeviceSynchronise();
	cudaMemPrefetchAsync(Sigma[0],kvol3*sizeof(Complex_f),cudaCpuDeviceId,NULL);
#else
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
	//Multiply this partial loop with the contributions of the other cores in the
	//Time-like dimension
#endif
	//End of CUDA-CPU pre-processor for evaluating Polyakov
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
#ifdef _DEBUG
	cudaFree(Sigma[1]);
#else
	cudaFreeAsync(Sigma[1],streams[0]);
#endif
#else
	free(Sigma[0]); free(Sigma[1]);
#endif

#if(nproc>1)
	Par_dsum(&poly);
#endif
	poly/=gvol3;
	return poly;	
}
#ifdef _CLOVER
int Leaf(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12,
		unsigned int *iu, unsigned int *id, int i, int mu, int nu, short leaf){
	/** @brief Evaluates the required clover leaf
	 *
	 * @param u11t, u12t:			Trial fields
	 * @param Sigma11, Sigma12:	Plaquette terms
	 * @param iu, id:					Upper/lower halo indices
	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
	 *										to facilitate calculating plaquettes for Clover terms. No
	 *										sanity checks are conducted on them in this routine.
	 *	@param i:						Centre of plaquette
	 * @param leaf:					Which leaf of the halo are we looking for. Based on the
	 * 									signs of μ and ν
	 *
	 * Calls:
	 * ======
	 * SU2plaq()
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	char *funcname="Leaf";
	Complex_f a11,a12;
	unsigned int didm,didn,uidn,uidm;
	switch(leaf){
		case(0):
			//Both positive is just a standard plaquette
			SU2plaq(u11t,u12t,Sigma11,Sigma12,iu,i,mu,nu);
			return 0;
		case(1):
			//μ<0 and ν>=0
			didm = id[mu+ndim*i]; uidn = iu[nu+ndim*i]; 
			//U_ν(x)*U_-μ(x+ν)=U_ν(x)*U^† _μ(x-μ+ν)
			//Awkward index here unfortunately. Seems safer than trying to find -μ
			int uin_didm=id[mu+ndim*uidn];
			*Sigma11=u11t[i*ndim+nu]*conj(u11t[uin_didm*ndim+mu])+u12t[i*ndim+nu]*conj(u12t[uin_didm*ndim+mu]);
			*Sigma12=-u11t[i*ndim+nu]*conj(u12t[uin_didm*ndim+mu])+u12t[i*ndim+nu]*u11t[uin_didm*ndim+mu];

			//(U_ν(x)*U_-μ(x+ν))*U_-ν(x-μ+ν)=(U_ν(x)*U^† _μ(x-μ+ν))*U^†_ν(x-μ)
			a11=*Sigma11*conj(u11t[didm*ndim+nu])+*Sigma12*conj(u12t[didm*ndim+nu]);
			a12=-*Sigma11*u12t[didm*ndim+nu]+*Sigma12*u11t[didm*ndim+nu];

			//((U_ν(x)*U_-μ(x+ν))*U_-ν(x-μ+ν))*U_μ(x-μ)=((U_ν(x)*U^† _μ(x-μ_ν))*U^† _ν(x-μ))*U_μ(x-μ)
			*Sigma11=a11*u11t[didm*ndim+mu]-a12*conj(u12t[didm*ndim+mu]);
			*Sigma12=a11*u12t[didm*ndim+mu]+a12*conj(u11t[didm*ndim+mu]);
			return 0;
		case(2):
			//μ>=0 and ν<0
			//TODO: Figure out down site index
			uidm = iu[mu+ndim*i]; didn = id[nu+ndim*i]; 
			//U_-ν(x)*U_μ(x-ν)=U^†_ν(x-ν)*U_μ(x-ν)
			*Sigma11=conj(u11t[didn*ndim+nu])*u11t[didn*ndim+mu]+conj(u12t[didn*ndim+nu])*u12t[didn*ndim+mu];
			*Sigma12=conj(u11t[didn*ndim+nu])*u12t[didn*ndim+mu]-u12t[didn*ndim+mu]*conj(u11t[didn*ndim+nu]);

			//(U_-ν(x)*U_μ(x-ν))*U_ν(x+μ-ν)=(U^†_ν(x-ν)*U_μ(x-ν))*U_ν(x+μ-ν)
			//Another awkward index
			int uim_didn=id[nu+ndim*uidm];
			a11=*Sigma11*u11t[uim_didn*ndim+nu]-*Sigma12*conj(u12t[uim_didn*ndim+nu]);
			a12=*Sigma11*u12t[uim_didn*ndim+nu]+*Sigma12*conj(u11t[uim_didn*ndim+nu]);

			//((U_-ν(x)*U_μ(x-ν))*U_ν(x+μ-ν))*U_-μ(x+μ)=(U^†_ν(x-ν)*U_μ(x-ν))*U_ν(x+μ-ν)
			*Sigma11=a11*conj(u11t[i*ndim+mu])+a12*conj(u12t[i*ndim+mu]);
			*Sigma12=-a11*u12t[i*ndim+mu]+a12*u11t[i*ndim+mu];
			return 0;
		case(3):
			//μ<0 and ν<0
			didm = id[mu+ndim*i]; didn = id[nu+ndim*i]; 
			//U_-μ(x)*U_-ν(x-μ)=U^†_μ(x-μ)*U^†_ν(x-μ-ν)
			int dim_didn=id[nu+ndim*didm];
			*Sigma11=conj(u11t[didm*ndim+mu])*conj(u11t[dim_didn*ndim+nu])+conj(u12t[didm*ndim+mu])*conj(u12t[dim_didn*ndim+nu]);

			Sigma11=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
			//				Sigma12[i]=-a11[i]*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
			//				Not needed in final result as it traces out
			return creal(Sigma11);
	}
#endif
