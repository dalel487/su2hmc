/*
 * Code for bosonic observables
 * Basically polyakov loop and Plaquette routines
 */
#include	<par_mpi.h>
#include	<su2hmc.h>
int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu, float beta){
	/* 
	 * Calculates the gauge action using new (how new?) lookup table
	 * Follows a routine called qedplaq in some QED3 code
	 *
	 * Globals:
	 * =======
	 * rank
	 *
	 * Parameters:
	 * ===========
	 * double	hg				Gauge component of Hamilton
	 * double	avplaqs		Average spacial Plaquette
	 * double	avplaqt		Average Temporal Plaquette
	 * Complex*	u11t,u12t	The trial fields
	 * int*		iu				Upper halo indices
	 * double	beta
	 *
	 * Calls:
	 * =====
	 * Par_dsum
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Average_Plaquette";
	/*Was a halo exchange here but moved it outside
	  The FORTRAN code used several consecutive loops to get the plaquette
	  Instead we'll just make the arrays variables and do everything in one loop
	  Should work since in the FORTRAN Sigma11[i] only depends on i components  for example
	  Since the ν loop doesn't get called for μ=0 we'll start at μ=1
	 */
#ifdef __NVCC__
	__managed__ double hgs = 0; __managed__ double hgt = 0;
	cuAverage_Plaquette(&hgs, &hgt, u11t, u12t, iu,dimGrid,dimBlock);
#else
	double hgs = 0; double hgt = 0;
	for(int mu=1;mu<ndim;mu++)
		for(int nu=0;nu<mu;nu++)
			//Don't merge into a single loop. Makes vectorisation easier?
			//Or merge into a single loop and dispense with the a arrays?
#pragma omp parallel for simd aligned(u11t,u12t,iu:AVX) reduction(+:hgs,hgt)
			for(int i=0;i<kvol;i++){
				//Save us from typing iu[mu+ndim*i] everywhere
				switch(mu){
					//Time component
					case(ndim-1):	hgt -= SU2plaq(u11t,u12t,iu,i,mu,nu);
										break;
										//Space component
					default:	hgs -= SU2plaq(u11t,u12t,iu,i,mu,nu);
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
#pragma omp declare simd
inline float SU2plaq(Complex_f *u11t, Complex_f *u12t, unsigned int *iu, int i, int mu, int nu){
	/*
	 * Calculates the plaquette at site i in the μ-ν direction
	 *
	 * Parameters:
	 * ==========
	 * Complex u11t, u12t:	Trial fields
	 * unsignedi int *iu:	Upper halo indices
	 * int mu, nu:				Plaquette direction. Note that mu and nu can be negative
	 * 							to facilitate calculating plaquettes for Clover terms. No
	 * 							sanity checks are conducted on them in this routine.
	 *
	 * Returns:
	 * ========
	 * double corresponding to the plaquette value
	 *
	 */
	const char *funcname = "SU2plaq";
	int uidm = iu[mu+ndim*i]; 

	Complex_f Sigma11=u11t[i*ndim+mu]*u11t[uidm*ndim+nu]-u12t[i*ndim+mu]*conj(u12t[uidm*ndim+nu]);
	Complex_f Sigma12=u11t[i*ndim+mu]*u12t[uidm*ndim+nu]+u12t[i*ndim+mu]*conj(u11t[uidm*ndim+nu]);

	int uidn = iu[nu+ndim*i]; 
	Complex_f a11=Sigma11*conj(u11t[uidn*ndim+mu])+Sigma12*conj(u12t[uidn*ndim+mu]);
	Complex_f a12=-Sigma11*u12t[uidn*ndim+mu]+Sigma12*u11t[uidn*ndim+mu];

	Sigma11=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
	//				Sigma12[i]=-a11[i]*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
	//				Not needed in final result as it traces out
	return creal(Sigma11);
}
double Polyakov(Complex_f *u11t, Complex_f *u12t){
	/*
	 * Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * Calls:
	 * ======
	 * Par_tmul, Par_dsum
	 * 
	 * Parameters:
	 * ==========
	 * Complex*	u11t, u12t	The trial fields
	 * 
	 * Returns:
	 * =======
	 * Double corresponding to the polyakov loop
	 */
	const char *funcname = "Polyakov";
	double poly = 0;
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	Complex_f *Sigma11,*Sigma12;
	cudaMallocManaged((void **)&Sigma11,kvol3*sizeof(Complex_f),cudaMemAttachGlobal);
#ifdef _DEBUG
	cudaMallocManaged((void **)&Sigma12,kvol3*sizeof(Complex_f),cudaMemAttachGlobal);
#else
	cudaMalloc((void **)&Sigma12,kvol3*sizeof(Complex_f));
#endif
#else
	Complex_f *Sigma11 = aligned_alloc(AVX,kvol3*sizeof(Complex_f));
	Complex_f *Sigma12 = aligned_alloc(AVX,kvol3*sizeof(Complex_f));
#endif

	//Extract the time component from each site and save in corresponding Sigma
#ifdef __NVCC__
	cublasCcopy(cublas_handle,kvol3, (cuComplex *)(u11t+3), ndim, (cuComplex *)Sigma11, 1);
	cublasCcopy(cublas_handle,kvol3, (cuComplex *)(u12t+3), ndim, (cuComplex *)Sigma12, 1);
#elif defined USE_BLAS
	cblas_ccopy(kvol3, u11t+3, ndim, Sigma11, 1);
	cblas_ccopy(kvol3, u12t+3, ndim, Sigma12, 1);
#else
	for(int i=0; i<kvol3; i++){
		Sigma11[i]=u11t[i*ndim+3];
		Sigma12[i]=u12t[i*ndim+3];
	}
#endif
	/*	Some Fortran commentary
		Changed this routine.
		u11t and u12t now defined as normal ie (kvol+halo,4).
		Copy of Sigma11 and Sigma12 is changed so that it copies
		in blocks of ksizet.
		Variable indexu also used to select correct element of u11t and u12t 
		in loop 10 below.

		Change the order of multiplication so that it can
		be done in parallel. Start at t=1 and go up to t=T:
		previously started at t+T and looped back to 1, 2, ... T-1
		Buffers
		There is a dependency. Can only parallelise the inner loop
	 */
#ifdef __NVCC__
	cuPolyakov(Sigma11,Sigma12,u11t,u12t,dimGrid,dimBlock);
	cudaMemPrefetchAsync(Sigma11,kvol3*sizeof(Complex_f),cudaCpuDeviceId,NULL);
#else
#pragma unroll
	for(int it=1;it<ksizet;it++)
#pragma omp parallel for simd aligned(u11t,u12t,Sigma11,Sigma12:AVX)
		for(int i=0;i<kvol3;i++){
			//Seems a bit more efficient to increment indexu instead of reassigning
			//it every single loop
			int indexu=it*kvol3+i;
			Complex_f	a11=Sigma11[i]*u11t[indexu*ndim+3]-Sigma12[i]*conj(u12t[indexu*ndim+3]);
			//Instead of having to store a second buffer just assign it directly
			Sigma12[i]=Sigma11[i]*u12t[indexu*ndim+3]+Sigma12[i]*conj(u11t[indexu*ndim+3]);
			Sigma11[i]=a11;
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
#error Par_tmul is not yet implimented in CUDA as Sigma12 is device only memory
#endif
#ifdef _DEBUG
	printf("Multiplying with MPI\n");
#endif
	Par_tmul(Sigma11, Sigma12);
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
#ifdef __NVCC__
#pragma omp parallel for simd reduction(+:poly)
#else
#pragma omp parallel for simd reduction(+:poly) aligned(Sigma11:AVX)
#endif
		for(int i=0;i<kvol3;i++)
			poly+=creal(Sigma11[i]);
#ifdef __NVCC__
	cudaFree(Sigma11); cudaFree(Sigma12);
#else
	free(Sigma11); free(Sigma12);
#endif

#if(nproc>1)
	Par_dsum(&poly);
#endif
	poly/=gvol3;
	return poly;	
}
