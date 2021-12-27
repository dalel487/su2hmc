/*
 * Code for bosonic observables
 * Basically polyakov loop and Plaquette routines
 */
#include	<par_mpi.h>
#include	<su2hmc.h>
int SU2plaq(double *hg, double *avplaqs, double *avplaqt){
	/* 
	 * Calculates the gauge action using new (how new?) lookup table
	 * Follows a routine called qedplaq in some QED3 code
	 *
	 * Globals:
	 * =======
	 * u11t,u12t,rank,beta
	 *
	 * Parameters:
	 * ===========
	 * double hg
	 * double avplaqs
	 * double avplaqt
	 *
	 * Calls:
	 * =====
	 * Par_dsum
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "SU2plaq";
	//Was a halo exchange here but moved it outside
	//	The fortran code used several consecutive loops to get the plaquette
	//	Instead we'll just make the arrays variables and do everything in one loop
	//	Should work since in the fortran Sigma11[i] only depends on i components  for example
	double hgs = 0; double hgt = 0;
	//Since the ν loop doesn't get called for μ=0 we'll start at μ=1
	for(int mu=1;mu<ndim;mu++)
		for(int nu=0;nu<mu;nu++)
			//Don't merge into a single loop. Makes vectorisation easier?
			//Or merge into a single loop and dispense with the a arrays?
//#ifdef __clang__
//#pragma omp target teams distribute parallel for simd aligned(u11t,u12t,iu:AVX)\
			reduction(+:hgs,hgt)	map(tofrom:hgs,hgt)
//#else
#pragma omp parallel for simd aligned(u11t,u12t,iu:AVX) reduction(+:hgs,hgt)
//#endif
			for(int i=0;i<kvol;i++){
				//Save us from typing iu[mu+ndim*i] everywhere
				int uidm = iu[mu+ndim*i]; 

				complex Sigma11=u11t[i*ndim+mu]*u11t[uidm*ndim+nu]-u12t[i*ndim+mu]*conj(u12t[uidm*ndim+nu]);
				complex Sigma12=u11t[i*ndim+mu]*u12t[uidm*ndim+nu]+u12t[i*ndim+mu]*conj(u11t[uidm*ndim+nu]);

				int uidn = iu[nu+ndim*i]; 
				complex a11=Sigma11*conj(u11t[uidn*ndim+mu])+Sigma12*conj(u12t[uidn*ndim+mu]);
				complex a12=-Sigma11*u12t[uidn*ndim+mu]+Sigma12*u11t[uidn*ndim+mu];

				Sigma11=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
				//				Sigma12[i]=-a11[i]*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
				//				Not needed in final result as it traces out

				switch(mu){
					//Time component
					case(ndim-1):	hgt -= creal(Sigma11);
									break;
									//Space component
					default:	hgs -= creal(Sigma11);
								break;
				}
			}
	Par_dsum(&hgs); Par_dsum(&hgt);
	*avplaqs=-hgs/(3.0*gvol); *avplaqt=-hgt/(gvol*3.0);
	*hg=(hgs+hgt)*beta;
#ifdef _DEBUG
	if(!rank)
		printf("hgs=%e  hgt=%e  hg=%e\n", hgs, hgt, *hg);
#endif
	return 0;
}
double Polyakov(){
	/*
	 * Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * Globals:
	 * =======
	 * u11t, u12t, u11t, u12t
	 *
	 * Calls:
	 * ======
	 * Par_tmul, Par_dsum
	 * 
	 * Parameters:
	 * ==========
	 * double *poly The Polyakov Loop value
	 * 
	 * Returns:
	 * =======
	 * Double corresponding to the polyakov loop
	 */
	const char *funcname = "Polyakov";
	double poly = 0;
	//Originally at the very end before Par_dsum
	//Now all cores have the value for the complete Polyakov line at all spacial sites
	//We need to globally sum over spacial processores but not across time as these
	//are duplicates. So we zero the value for all but t=0
	//This is (according to the FORTRAN code) a bit of a hack
	//I will expand on this hack and completely avoid any work
	//for this case rather than calculating everything just to set it to zero
#ifdef __NVCC__
	Complex *Sigma11,*Sigma12;
	cudaMallocManaged(&Sigma11,kvol3*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Sigma12,kvol3*sizeof(Complex),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
	Complex *Sigma11 = (Complex *)mkl_malloc(kvol3*sizeof(Complex),AVX);
	Complex *Sigma12 = (Complex *)mkl_malloc(kvol3*sizeof(Complex),AVX);
#else
	Complex *Sigma11 = aligned_alloc(AVX,kvol3*sizeof(Complex));
	Complex *Sigma12 = aligned_alloc(AVX,kvol3*sizeof(Complex));
#endif
#ifdef __NVCC__
	cublasZcopy(cublas_handle,kvol3, &u11t[3], ndim, Sigma11, 1);
	cublasZcopy(cublas_handle,kvol3, &u12t[3], ndim, Sigma12, 1);
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
	cblas_zcopy(kvol3, &u11t[3], ndim, Sigma11, 1);
	cblas_zcopy(kvol3, &u12t[3], ndim, Sigma12, 1);
#else
	for(int i=0; i<kvol3; i++){
		Sigma11[i]=u11t[i*ndim+3];
		Sigma12[i]=u12t[i*ndim+3];
	}
#endif
	//	Some Fortran commentary
	//	Changed this routine.
	//	u11t and u12t now defined as normal ie (kvol+halo,4).
	//	Copy of Sigma11 and Sigma12 is changed so that it copies
	//	in blocks of ksizet.
	//	Variable indexu also used to select correct element of u11t and u12t 
	//	in loop 10 below.
	//
	//	Change the order of multiplication so that it can
	//	be done in parallel. Start at t=1 and go up to t=T:
	//	previously started at t+T and looped back to 1, 2, ... T-1
	//Buffers
	//There is a dependency. Can only parallelise the inner loop
#pragma omp target enter data map(to:Sigma11[0:kvol3],Sigma12[0:kvol3])
#pragma unroll
		for(int it=1;it<ksizet;it++)
			//will be faster for parallel code
#ifdef __clang__
#pragma omp target teams distribute parallel for simd aligned(u11t,u12t,Sigma11,Sigma12:AVX)
#else
#pragma omp parallel for simd aligned(u11t,u12t,Sigma11,Sigma12:AVX)
#endif
			for(int i=0;i<kvol3;i++){
				//Seems a bit more efficient to increment indexu instead of reassigning
				//it every single loop
				int indexu=it*kvol3+i;
				Complex	a11=Sigma11[i]*u11t[indexu*ndim+3]-Sigma12[i]*conj(u12t[indexu*ndim+3]);
				//Instead of having to store a second buffer just assign it directly
				Sigma12[i]=Sigma11[i]*u12t[indexu*ndim+3]+Sigma12[i]*conj(u11t[indexu*ndim+3]);
				Sigma11[i]=a11;
			}
#pragma omp target update from(Sigma11[0:kvol3],Sigma12[0:kvol3])
	//Multiply this partial loop with the contributions of the other cores in the
	//timelike dimension
#if (npt>1)
#ifdef _DEBUG
	printf("Multiplying with MPI\n");
#endif
	//Par_tmul does nothing if there is only a single processor in the time direction. So we only compile
	//its call if it is required
	Par_tmul(Sigma11, Sigma12);
#endif
#pragma omp parallel for simd reduction(+:poly) aligned(Sigma11:AVX)
	for(int i=0;i<kvol3;i++)
		poly+=creal(Sigma11[i]);
#pragma omp target exit data map(from:Sigma11[0:kvol3],Sigma12[0:kvol3])
#ifdef __NVCC__
	cudaFree(Sigma11); cudaFree(sigma12);
#elif defined __INTEL_MKL__
	mkl_free(Sigma11); mkl_free(Sigma12);
#else
	free(Sigma11); free(Sigma12);
#endif

	if(pcoord[3+rank*ndim]) poly = 0;
	Par_dsum(&poly);
	poly/=gvol3;
	return poly;	
}
