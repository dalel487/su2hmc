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

int Average_Plaquette(double *hg, double *avplaqs, double *avplaqt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu, float beta){
	/**
	 * @brief Calculates the gauge action using new (how new?) lookup table
	 * Follows a routine called qedplaq in some QED3 code
	 *
	 * Globals:
	 * =======
	 * rank
	 *
	 * @param double	hg					Gauge component of Hamilton
	 * @param double	avplaqs			Average spacial Plaquette
	 * @param double	avplaqt			Average Temporal Plaquette
	 * @param Complex_f*	u11t,u12t	The trial fields
	 * @param int*		iu					Upper halo indices
	 * @param double	beta				Inverse gauge coupling
	 *
	 * Calls:
	 * ======
	 * Par_dsum()
	 *
	 * @return: Zero on success, integer error code otherwise
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
				Complex_f Sigma11, Sigma12;
				SU2plaq(u11t,u12t,&Sigma11,&Sigma12,iu,i,mu,nu);
				switch(mu){
					//Time component
					case(ndim-1):	hgt -= creal(Sigma11);
										break;
										//Space component
					default:	hgs -= creal(Sigma11);
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
inline int SU2plaq(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12, unsigned int *iu,  int i, int mu, int nu){
	/**
	 * @brief Calculates the trace of the plaquette at site i in the μ-ν direction
	 *
	 * @param u11t, u12t:			Trial fields
	 * @param Sigma11, Sigma12:	Trial fields
	 * @param *iu:						Upper halo indices
	 * @param i:						site index
	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
	 * 									to facilitate calculating plaquettes for Clover terms. No
	 * 									sanity checks are conducted on them in this routine.
	 *
	 * @return Zero on success, integer error code otherwise
	 *
	 */
	const char *funcname = "SU2plaq";
	/*
	 *	Let's take a quick moment to compare this to the analysis code.
	 *	The analysis code stores the gauge field as a 4 component real valued vector, whereas the produciton code
	 *	used two complex numbers.
	 *
	 *	Analysis code: u=(Re(u11),Im(u12),Re(u12),Im(u11))
	 *	Production code: u11=u[0]+I*u[3]	u12=u[2]+I*u[1]
	 *
	 *	This applies to the Sigmas and a's below too
	 */

	//Save us from typing iu[mu+ndim*i] everywhere
	int uidm = iu[mu+ndim*i]; int uidn = iu[nu+ndim*i]; 
	//U_μ(x)*U_ν(x+μ)
	*Sigma11=u11t[i*ndim+mu]*u11t[uidm*ndim+nu]-u12t[i*ndim+mu]*conj(u12t[uidm*ndim+nu]);
	*Sigma12=u11t[i*ndim+mu]*u12t[uidm*ndim+nu]+u12t[i*ndim+mu]*conj(u11t[uidm*ndim+nu]);

	//(U_μ(x)*U_ν(x+μ))*(U_-μ(x+μ+ν))
	Complex_f a11=*Sigma11*conj(u11t[uidn*ndim+mu])+*Sigma12*conj(u12t[uidn*ndim+mu]);
	Complex_f a12=-*Sigma11*u12t[uidn*ndim+mu]+*Sigma12*u11t[uidn*ndim+mu];

	//[(U_μ(x)*U_ν(x+μ))*(U_-μ(x+μ+ν))]*U_-ν(x+ν)
	*Sigma11=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
	*Sigma12=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
	return 0;
}
inline int Leaf(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12,
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
	switch(leaf){
		case(0):
			//Both positive is just a standard plaquette
			return SU2plaq(u11t,u12t,&Sigma11,&Sigma12,iu,i,mu,nu);
		case(1):
			//μ<0 and ν>=0
			int didm = id[mu+ndim*i]; int uidn = iu[nu+ndim*i]; 
			//U_ν(x)*U_-μ(x+ν)=U_ν(x)*U^† _μ(x-μ+ν)
			//Awkward index here unfortunately. Seems safer than trying to find -μ
			int uin_didm=id[mu+ndim*uidn];
			*Sigma11=u11t[i*ndim+nu]*conj(u11t[uin_didm*ndim+mu])+u12t[i*ndim+nu]*conj(u12t[uin_didm*ndim+mu]);
			*Sigma12=-u11t[i*ndim+nu]*conj(u12t[uin_didm*ndim+mu])+u12t[i*ndim+nu]*u11t[uin_didm*ndim+mu];

			//(U_ν(x)*U_-μ(x+ν))*U_-ν(x-μ+ν)=(U_ν(x)*U^† _μ(x-μ+ν))*U^†_ν(x-μ)
			Complex_f a11=*Sigma11*conj(u11t[didm*ndim+nu])+*Sigma12*conj(u12t[didm*ndim+nu]);
			Complex_f a12=-*Sigma11*u12t[didm*ndim+nu]+*Sigma12*u11t[didm*ndim+nu];

			//((U_ν(x)*U_-μ(x+ν))*U_-ν(x-μ+ν))*U_μ(x-μ)=((U_ν(x)*U^† _μ(x-μ_ν))*U^† _ν(x-μ))*U_μ(x-μ)
			*Sigma11=a11*u11t[didm*ndim+mu]-a12*conj(u12t[didm*ndim+mu]);
			*Sigma12=a11*u12t[didm*ndim+mu]+a12*conj(u11t[didm*ndim+mu]);
			return 0;
		case(2):
			//μ>=0 and ν<0
			//TODO: Figure out down site index
			int uidm = iu[mu+ndim*i]; int didn = id[nu+ndim*i]; 
			//U_-ν(x)*U_μ(x-ν)=U^†_ν(x-ν)*U_μ(x-ν)
			*Sigma11=conj(u11t[didn*ndim+nu])*u11t[didn*ndim+mu]+conj(u12t[didn*ndim+nu])*u12t[didn*ndim+mu];
			*Sigma12=conj(u11t[didn*ndim+nu])*u12t[didn*ndim+mu]-u12t[didn*ndim+mu]*conj(u11t[didn*ndim+nu]);

			//(U_-ν(x)*U_μ(x-ν))*U_ν(x+μ-ν)=(U^†_ν(x-ν)*U_μ(x-ν))*U_ν(x+μ-ν)
			//Another awkward index
			int uim_didn=id[nu+ndim*uidm];
			Complex_f a11=*Sigma11*u11t[uim_didn*ndim+nu]-*Sigma12*conj(u12t[uim_didn*ndim+nu]);
			Complex_f a12=*Sigma11*u12t[uim_didn*ndim+nu]+*Sigma12*conj(u11t[uim_didn*ndim+nu]);

			//((U_-ν(x)*U_μ(x-ν))*U_ν(x+μ-ν))*U_-μ(x+μ)=(U^†_ν(x-ν)*U_μ(x-ν))*U_ν(x+μ-ν)
			*Sigma11=a11*conj(u11t[i*ndim+mu])+a12*conj(u12t[i*ndim+mu]);
			*Sigma12=-a11*u12t[i*ndim+mu]+a12*u11t[i*ndim+mu];
			return 0;
		case(3):
			//μ<0 and ν<0
			int didm = id[mu+ndim*i]; int didn = id[nu+ndim*i]; 
			//U_-μ(x)*U_-ν(x-μ)=U^†_μ(x-μ)*U^†_ν(x-μ-ν)
			int dim_didn=id[nu+ndim*didm];
			*Sigma11=conj(u11t[didm*ndim+mu])*conj(u11t[didm_didn*ndim+nu])+conj(u12t[didm*ndim+mu])*conj(u12t[dim_didn*ndim+nu]);

			//(U_-μ(x)*U_-ν(x-μ))*(U_μ(x-μ-ν))
			Complex_f a11=*Sigma11*u11t[dim_didn*ndim+mu]-*Sigma12*conj(u12t[dim_didn*ndim+mu]);
			Complex_f a12=*Sigma11*u12t[dim_didn*ndim+mu]+*Sigma12*conj(u11t[dim_didn*ndim+mu]);

			//[(U_-μ(x)*U_-ν(x-μ))*(U_μ(x-μ-ν))]*U_ν(x-ν)
			*Sigma11=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
			*Sigma12=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+mu]);
			return 0;
	}
}
inline int Half_Clover(Complex_f *u11t, Complex_f *u12t, Complex_f *clover11, Complex_f *clover12,
		unsigned int *iu, unsigned int *id, int i, int mu, int nu){
	/** @brief Calculate one clover leaf \f(Q_{μν}\f), which is half the full clover term
	 *
	 * @param u11t, u12t:			Trial fields
	 * @param clover11, clover12:	Clover fields
	 * @param *iu, *id:				Upper/lower halo indices
	 *	@param i:						Centre of plaquette
	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
	 * 									to facilitate calculating plaquettes for Clover terms. No
	 * 									sanity checks are conducted on them in this routine.
	 *
	 * Calls:
	 * ======
	 * Leaf()
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	char *funcname ="Half_Clover";
#pragma omp simd reduction(+:clover11,clover12)
	for(int i=0;i<ndim;i++)
	{
		Complex_f Sigma11, Sigma12;
		Leaf(u11t,u12t,Sigma11,Sigma12,iu,id,i,mu,nu);
		*clover11+=Sigma11; *clover12+=Sigma12;
	}
	return 0;
}
inline int Clover(Complex_f *u11t, Complex_f *u12t, Complex_f *clover11, Complex_f *clover12,
		unsigned int *iu, unsigned int *id, int i, int mu, int nu){
	/** @brief Calculate the clover term in the μ-ν direction
	 *	\f$F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-{Q_{\nu\mu}(n)\right)\f$
	 *	
	 * @param u11t, u12t:			Trial fields
	 * @param clover11, clover12:	Clover fields
	 * @param *iu, *id:				Upper/lower halo indices
	 *	@param i:						Centre of plaquette
	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
	 * 									to facilitate calculating plaquettes for Clover terms. No
	 * 									sanity checks are conducted on them in this routine.
	 *
	 * Calls:
	 * =====
	 * Half_Clover()
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	char *funcname="Clover";
	Half_Clover(u11t,u12t,clover11,clover12,iu,id,i,mu,nu);	
	//Hmm, creal(Clover11) drops out then?
	clover11-=conj(clover11); clover12-=-clover12;
	clover11*=(-I/8.0); clover12*=(-I/8.0);
}
double Polyakov(Complex_f *u11t, Complex_f *u12t){
	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * @param u11t, u12t	The trial fields
	 * 
	 * Calls:
	 * ======
	 * Par_tmul(), Par_dsum()
	 * 
	 * @return Double corresponding to the polyakov loop
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
	cudaMallocAsync((void **)&Sigma12,kvol3*sizeof(Complex_f),streams[0]);
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
	cudaDeviceSynchronise();
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
		cudaDeviceSynchronise();
#pragma omp parallel for simd reduction(+:poly)
#else
#pragma omp parallel for simd reduction(+:poly) aligned(Sigma11:AVX)
#endif
	for(int i=0;i<kvol3;i++)
		poly+=creal(Sigma11[i]);
#ifdef __NVCC__
	cudaFree(Sigma11);
#ifdef _DEBUG
	cudaFree(Sigma12);
#else
	cudaFreeAsync(Sigma12,streams[0]);
#endif
#else
	free(Sigma11); free(Sigma12);
#endif

#if(nproc>1)
	Par_dsum(&poly);
#endif
	poly/=gvol3;
	return poly;	
}
