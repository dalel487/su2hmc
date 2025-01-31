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
	/**
	 * @brief Calculates the trace of the plaquette at site i in the \mu-ν direction
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
#ifdef _CLOVER
inline int Clover_SU2plaq(Complex_f *ut[2], Complex_f *Leaves[2], unsigned int *iu,  int i, int mu, int nu){
	/**
	 * @brief Calculates the trace of the plaquette at site i in the \mu-ν direction
	 *
	 * @param ut[0], ut[1]:			Trial fields
	 * @param Leaves11, Leaves12:	Trial fields
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
	 *	This applies to the Leavess and a's below too
	 */

	//TODO: Figure out how we want to label the leaves. 12 clovers in total. Each with 4 leaves. The below should work for
	//the plaquette as the first leaf
	Leaves[0][i*ndim]=ut[0][i*ndim+mu]*ut[0][uidm*ndim+nu]-ut[1][i*ndim+mu]*conj(ut[1][uidm*ndim+nu]);
	Leaves[1][i*ndim]=ut[0][i*ndim+mu]*ut[1][uidm*ndim+nu]+ut[1][i*ndim+mu]*conj(ut[0][uidm*ndim+nu]);

	int uidn = iu[nu+ndim*i]; 
	Complex_f a11=Leaves[0][i*ndim]*conj(ut[0][uidn*ndim+mu])+Leaves[1][i*ndim]*conj(ut[1][uidn*ndim+mu]);
	Complex_f a12=-Leaves[0][i*ndim]*ut[1][uidn*ndim+mu]+Leaves[1][i*ndim]*ut[0][uidn*ndim+mu];

	Leaves[0][i*ndim]=a11*conj(ut[0][i*ndim+nu])+a12*conj(ut[1][i*ndim+nu]);
	Leaves[1][i*ndim]=-a11*ut[1][i*ndim+nu]+a12*ut[0][i*ndim+mu];
	return 0;
}
int Leaf(Complex_f *ut[2], Complex_f *Leaves[2], unsigned int *iu, unsigned int *id, int i, int mu, int nu, short leaf){
	/** @brief Evaluates the required clover leaf
	 *
	 * @param ut:			Trial fields
	 * @param Leaves:		Plaquette terms
	 * @param iu, id:		Upper/lower halo indices
	 * @param mu, nu:		Plaquette direction. Note that mu and nu can be negative
	 *					  		to facilitate calculating plaquettes for Clover terms. No
	 *					  		sanity checks are conducted on them in this routine.
	 *	@param i:	  		Centre of plaquette
	 * @param leaf:  		Which leaf of the halo are we looking for. Based on the
	 * 				  		signs of \mu and ν
	 *
	 * Calls:
	 * ======
	 * SU2plaq()
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	char *funcname="Leaf";
	Complex_f a[2];
	unsigned int didm,didn,uidn,uidm;
	switch(leaf){
		case(0):
			//Both positive is just a standard plaquette
			return SU2plaq(ut,Leaves,iu,i,mu,nu);
		case(1):
			//\mu<0 and \nu>=0
			didm = id[mu+ndim*i]; uidn = iu[nu+ndim*i]; 
			/// @f$U_\nu(x)*U_-\mu(x+\nu)=U_\nu(x)*U^\dagger _\mu(x-\mu+\nu)@f$
			//Awkward index here unfortunately. Seems safer than trying to find -\mu
			int uin_didm=id[mu+ndim*uidn];
			Leaves[0][i*ndim+leaf]=ut[0][i*ndim+nu]*conj(ut[0][uin_didm*ndim+mu])+conj(ut[1][i*ndim+nu])*ut[1][uin_didm*ndim+mu];
			Leaves[1][i*ndim+leaf]=ut[1][i*ndim+nu]*conj(ut[0][uin_didm*ndim+mu])-ut[0][i*ndim+nu]*ut[1][uin_didm*ndim+mu];

			/// @f$(U_\nu(x)*U_-\mu(x+\nu))*U_-\nu(x-\mu+\nu)=(U_\nu(x)*U^\dagger _\mu(x-\mu+\nu))*U^\dagger_\nu(x-\mu)@f$
			a[0]=Leaves[0][i*ndim+leaf]*conj(ut[0][didm*ndim+nu])+conj(Leaves[1][i*ndim+leaf])*conj(ut[1][didm*ndim+nu]);
			a[1]=Leaves[1][i*ndim+leaf]*conj(ut[0][didm*ndim+nu])-Leaves[0][i*ndim+leaf]*ut[1][didm*ndim+nu]+;

			/// @f$((U_\nu(x)*U_-\mu(x+\nu))*U_-\nu(x-\mu+\nu))*U_\mu(x-\mu)=((U_\nu(x)*U^\dagger _\mu(x-\mu_\nu))*U^\dagger _\nu(x-\mu))*U_\mu(x-\mu)@f$
			Leaves[0][i*ndim+leaf]=a[0]*ut[0][didm*ndim+mu]-conj(a[1])*ut[1][didm*ndim+mu];
			Leaves[1][i*ndim+leaf]=a[1]*ut[0][didm*ndim+mu]-a[0]*ut[1][didm*ndim+mu];
			return 0;
		case(2):
			//\mu>=0 and \nu<0
			//TODO: Figure out down site index
			uidm = iu[mu+ndim*i]; didn = id[nu+ndim*i]; 
			/// @f$U_-\nu(x)*U_\mu(x-\nu)=U^\dagger_\nu(x-\nu)*U_\mu(x-\nu)@f$
			Leaves[0][i*ndim+leaf]=conj(ut[0][didn*ndim+nu])*ut[0][didn*ndim+mu]+conj(ut[1][didn*ndim+nu])*ut[1][didn*ndim+mu];
			Leaves[1][i*ndim+leaf]=-ut[1][didn*ndim+mu]*conj(ut[0][didn*ndim+nu])+ut[0][didn*ndim+nu]*ut[1][didn*ndim+mu];

			/// @f$(U_-\nu(x)*U_\mu(x-\nu))*U_\nu(x+\mu-\nu)=(U^\dagger_\nu(x-\nu)*U_\mu(x-\nu))*U_\nu(x+\mu-\nu)@f$
			//Another awkward index
			int uim_didn=id[nu+ndim*uidm];
			a[0]=Leaves[0][i*ndim+leaf]*ut[0][uim_didn*ndim+nu]-conj(Leaves[1][i*ndim+leaf])*ut[1][uim_didn*ndim+nu];
			a[1]=-Leaves[1][i*ndim+leaf]*ut[0][uim_didn*ndim+nu]+Leaves[0][i*ndim+leaf]*ut[1][uim_didn*ndim+nu];

			/// @f$((U_-\nu(x)*U_\mu(x-\nu))*U_ν(x+\mu-\nu))*U_-\mu(x+\mu)=((U^\dagger_\nu(x-\nu)*U_\mu(x-ν))*U_\nu(x+\mu-\nu))*U^\dagger_\mu(x)@f$
			Leaves[0][i*ndim+leaf]=a[0]*ut[0][i*ndim+mu]-conj(a[1])*ut[1][i*ndim+mu];
			Leaves[1][i*ndim+leaf]=a[1]*ut[0][i*ndim+mu]+a[0]*ut[1][i*ndim+mu];
			return 0;
		case(3):
			//\mu<0 and \nu<0
			didm = id[mu+ndim*i]; didn = id[nu+ndim*i]; 
			/// @f$U_-\mu(x)*U_-\nu(x-\mu)=U^\dagger_\mu(x-\mu)*U^\dagger_\nu(x-\mu-\nu)@f$
			int dim_didn=id[nu+ndim*didm];
			Leaves[0][i*ndim+leaf]=conj(ut[0][didm*ndim+mu])*conj(ut[0][dim_didn*ndim+nu])-conj(ut[1][didm*ndim+mu])*ut[1][dim_didn*ndim+nu];
			Leaves[0][i*ndim+leaf]=-ut[1][didm*ndim+mu]*conj(ut[0][dim_didn*ndim+nu])-ut[0][didm*ndim+mu]*ut[1][dim_didn*ndim+nu];

			/// @f$(U_-\mu(x)*U_-\nu(x-\mu))U_\mu(x-\mu-\nu)=(U^\dagger_\mu(x-\mu)*U^\dagger_\nu(x-\mu-\nu))U_\mu(x-\mu-\nu)@f$
			a[0]=Leaves[0][i*ndim+leaf]*ut[0][uim_didn*ndim+mu]-\conj(Leaves[1][i*ndim+leaf])*ut[1][uim_didn*ndim+mu];
			a[1]=-Leaves[1][i*ndim+leaf]*ut[0][uim_didn*ndim+mu]+-Leaves[0][i*ndim+leaf]*ut[0][uim_didn*ndim+mu];

			/// @f$((U_-\mu(x)*U_-\nu(x-\mu))U_\mu(x-\mu-\nu))U_\nu(x-\nu)=((U^\dagger_\mu(x-\mu)*U^\dagger_\nu(x-\mu-\nu))U_\mu(x-\mu-\nu))U_\nu(x-\nu)@f$
			Leaves[0][i*ndim+leaf]=a[0]*ut[0][didn*ndim+nu]-conj(a[1])*ut[1][didn*ndim+nu];
			Leaves[1][i*ndim+leaf]=-a[1]*ut[0][didn*ndim+nu]+a[0]*ut[1][didn*ndim+nu];
			return 0;
	}
}
inline int Half_Clover(Complex_f *clover[2],	Complex_f *Leaves[2], Complex_f *ut[2], unsigned int *iu, unsigned int *id, int i, int mu, int nu){
	/** @brief Calculate one clover leaf \f(Q_{μν}\f), which is half the full clover term
	 *
	 * @param u11t, u12t:			Trial fields
	 * @param clover11, clover12:	Clover fields
	 * @param *iu, *id:				Upper/lower halo indices
	 *	@param i:						Centre of plaquette
	 * @param mu, nu:					Plaquette direction. 
	 *
	 * Calls:
	 * ======
	 * Leaf()
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char funcname[] ="Half_Clover";
	for(short leaf=0;i<ndim;leaf++)
	{
		Leaf(ut,Leaves,iu,id,i,mu,nu,leaf);
		//TODO: Site indices for leaf!
		clover[0][i]+=Leaves[0]; clover[1][i]+=Leaves[1];
	}
	return 0;
}

inline int Clover(Complex_f *clover[6][2],Complex_f *Leaves[6][2],Complex_f *ut[2], unsigned int *iu, unsigned int *id){
	/** @brief Calculate the clover term in the μ-ν direction
	 *	\f$F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-{Q_{\nu\mu}(n)\right)\f$
	 *	
	 * @param u11t, u12t:			Trial fields
	 * @param clover11, clover12:	Clover fields
	 * @param *iu, *id:				Upper/lower halo indices
	 *	@param i:						Centre of plaquette
	 * @param mu, nu:					Plaquette direction. 
	 *
	 * Calls:
	 * =====
	 * Half_Clover()
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char funcname[]="Clover";
	for(unsigned int mu=0;mu<ndim-1;mu++)
		for(unsigned int nu=mu+1;nu<ndim;nu++){
			if(mu!=nu){
				//Clover index
				unsigned int clov = (mu==0) ? nu :mu+nu;
#pragma omp parallel for
				for(unsigned int i=0;i<kvol;i++)
				{
					Half_Clover(clover[clov],Leaves[clov],ut,iu,id,i,mu,nu);	
					//Hmm, creal(clover[0]) drops out. And clover[1] just gets doubled (as does cimag(clover[1])
					clover[clov][0][i]-=conj(clover[clov][0][i]);	clover[clov][1][i]+=clover[clov][1][i];
					clover[clov][0][i]*=(-I/8.0); 				clover[clov][1][i]*=(-I/8.0);
				}
			}
		}
	return 0;
}
#endif
