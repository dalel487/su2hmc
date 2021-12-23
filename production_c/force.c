/*
 * Code for force calculations.
 * Requires multiply.cu to work
 */
#include	<matrices.h>
#include	<par_mpi.h>
#include	<su2hmc.h>
int Gauge_force(double *dSdpi){
	/*
	 * Calculates dSdpi due to the Wilson Action at each intermediate time
	 *
	 * Globals:
	 * =======
	 * u11t, u12t, u11, u12, iu, id, beta
	 * Calls:
	 * =====
	 * Z_Halo_swap_all, Z_gather, Z_Halo_swap_dir
	 */
	const char *funcname = "Gauge_force";

	//We define zero halos for debugging
	//	#ifdef _DEBUG
	//		memset(u11t[kvol], 0, ndim*halo*sizeof(complex));	
	//		memset(u12t[kvol], 0, ndim*halo*sizeof(complex));	
	//	#endif
	//Was a trial field halo exchange here at one point.
#ifdef __INTEL_MKL__
	complex *Sigma11 = mkl_malloc(kvol*sizeof(complex),AVX); 
	complex *Sigma12= mkl_malloc(kvol*sizeof(complex),AVX); 
	complex *u11sh = mkl_malloc((kvol+halo)*sizeof(complex),AVX); 
	complex *u12sh = mkl_malloc((kvol+halo)*sizeof(complex),AVX); 
#else
	complex *Sigma11 = aligned_alloc(AVX,kvol*sizeof(complex)); 
	complex *Sigma12= aligned_alloc(AVX,kvol*sizeof(complex)); 
	complex *u11sh = aligned_alloc(AVX,(kvol+halo)*sizeof(complex)); 
	complex *u12sh = aligned_alloc(AVX,(kvol+halo)*sizeof(complex)); 
#endif
	//Assign local arrays to accelerator
#ifdef __clang__
#pragma omp target enter data map(to:u11sh[0:kvol+halo],u12sh[0:kvol+halo],\
		Sigma11[0:kvol],Sigma12[0:kvol])
#endif
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
		memset(Sigma11,0, kvol*sizeof(complex));
		memset(Sigma12,0, kvol*sizeof(complex));
		//Send the Sigmas to the accelerator.
#pragma omp target update to(Sigma11[0:kvol],Sigma12[0:kvol])
		for(int nu=0; nu<ndim; nu++){
			if(nu!=mu){
				//The +ν Staple
#ifdef __clang__
#pragma omp target teams distribute parallel for simd\
				aligned(u11t,u12t,Sigma11,Sigma12,iu:AVX)
#endif
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int uidn = iu[nu+ndim*i];
					complex	a11=u11t[uidm*ndim+nu]*conj(u11t[uidn*ndim+mu])+\
									 u12t[uidm*ndim+nu]*conj(u12t[uidn*ndim+mu]);
					complex	a12=-u11t[uidm*ndim+nu]*u12t[uidn*ndim+mu]+\
									 u12t[uidm*ndim+nu]*u11t[uidn*ndim+mu];
					Sigma11[i]+=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
					Sigma12[i]+=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+nu];
				}
				Z_gather(u11sh, u11t, kvol, id, nu);
				ZHalo_swap_dir(u11sh, 1, mu, DOWN);
				//Send u11sh to the accelerator while u12sh is getting sorted
#pragma omp target update to(u11sh[0:kvol+halo])
				Z_gather(u12sh, u12t, kvol, id, nu);
				ZHalo_swap_dir(u12sh, 1, mu, DOWN);
#ifdef __clang__
#pragma omp target update to(u12sh[0:kvol+halo])
				//Next up, the -ν staple
#pragma omp target teams distribute parallel for simd\
				aligned(u11t,u12t,u11sh,u12sh,Sigma11,Sigma12,iu,id:AVX)
#endif
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int didn = id[nu+ndim*i];
					//uidm is correct here
					complex a11=conj(u11sh[uidm])*conj(u11t[didn*ndim+mu])-\
									u12sh[uidm]*conj(u12t[didn*ndim+mu]);
					complex a12=-conj(u11sh[uidm])*u12t[didn*ndim+mu]-\
									u12sh[uidm]*u11t[didn*ndim+mu];
					Sigma11[i]+=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
					Sigma12[i]+=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+nu]);
				}
			}
		}
#ifdef __clang__
#pragma omp target teams distribute parallel for simd\
		aligned(u11t,u12t,dSdpi,Sigma11,Sigma12:AVX) 
#else
#pragma omp parallel for simd aligned(u11t,u12t,Sigma11,Sigma12,dSdpi:AVX)
#endif
		for(int i=0;i<kvol;i++){
			complex a11 = u11t[i*ndim+mu]*Sigma12[i]+u12t[i*ndim+mu]*conj(Sigma11[i]);
			complex a12 = u11t[i*ndim+mu]*Sigma11[i]+conj(u12t[i*ndim+mu])*Sigma12[i];

			dSdpi[(i*nadj)*ndim+mu]=beta*cimag(a11);
			dSdpi[(i*nadj+1)*ndim+mu]=beta*creal(a11);
			dSdpi[(i*nadj+2)*ndim+mu]=beta*cimag(a12);
		}
	}
#pragma omp target update from(dSdpi[0:kmom])
#ifdef __clang__
#pragma omp target exit data map(delete:Sigma11[0:kvol],Sigma12[0:kvol],\
		u11sh[0:kvol+halo],u12sh[0:kvol+halo])
#endif
#ifdef __INTEL_MKL__
	mkl_free(u11sh); mkl_free(u12sh); mkl_free(Sigma11); mkl_free(Sigma12);
#else
	free(u11sh); free(u12sh); free(Sigma11); free(Sigma12);
#endif
	return 0;
}
int Force(double *dSdpi, int iflag, double res1){
	/*
	 *	Calculates dSds at each intermediate time
	 *	
	 *	Calls:
	 *	=====
	 *
	 *	Globals:
	 *	=======
	 *	u11t, u12t, X1, Phi
	 *
	 *	This X1 is the one being referred to in the common/vector/ statement in the original FORTRAN
	 *	code. There may subroutines with a different X1 (or use a different common block definition
	 *	for this X1) so keep your wits about you
	 *
	 *	Parameters:
	 *	===========
	 *	double dSdpi[3][kvol]
	 *	int	iflag
	 *	double	res1;
	 *
	 *	Returns:
	 *	=======
	 *	Zero on success, integer error code otherwise
	 */
	const char *funcname = "Force";
#ifndef NO_GAUGE
	Gauge_force(dSdpi);
#endif
	//X1=(M†M)^{1} Phi
	int itercg;
#if defined __INTEL_MKL__
	complex *X2= mkl_malloc(kferm2Halo*sizeof(complex), AVX);
	complex *smallPhi =mkl_malloc(kferm2Halo*sizeof(complex), AVX); 
#else
	complex *X2= aligned_alloc(AVX,kferm2Halo*sizeof(complex));
	complex *smallPhi = aligned_alloc(AVX,kferm2Halo*sizeof(complex)); 
#endif
	for(int na = 0; na<nf; na++){
		memcpy(X1, X0+na*kferm2Halo, nc*ndirac*kvol*sizeof(complex));
		//FORTRAN's logic is backwards due to the implied goto method
		//If iflag is zero we do some initalisation stuff 
		if(!iflag){
			Congradq(na, res1,smallPhi, &itercg );
			ancg+=itercg;
			//This is not a general BLAS Routine, just an MKL one
#if (defined __INTEL_MKL__ || defined USE_BLAS)
			complex blasa=2.0; complex blasb=-1.0;
			cblas_zaxpby(kvol*ndirac*nc, &blasa, X1, 1, &blasb, X0+na*kferm2Halo, 1); 
#else
			for(int i=0;i<kvol;i++){
#pragma unroll
				for(int idirac=0;idirac<ndirac;idirac++){
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc]=
						2*X1[(i*ndirac+idirac)*nc]-X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc];
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1]=
						2*X1[(i*ndirac+idirac)*nc+1]-X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1];
				}
			}
#endif
		}
		Hdslash(X2,X1);
#if (defined __INTEL_MKL__ || defined USE_BLAS)
		double blasd=2.0;
		cblas_zdscal(kferm2, blasd, X2, 1);
#else
#pragma unroll
		for(int i=0;i<kferm2;i++)
			X2[i]*=2;
#endif
#pragma unroll
		for(int mu=0;mu<4;mu++){
			ZHalo_swap_dir(X1,8,mu,DOWN);
			ZHalo_swap_dir(X2,8,mu,DOWN);
		}

		//	The original FORTRAN Comment:
		//    dSdpi=dSdpi-Re(X1*(d(Mdagger)dp)*X2) -- Yikes!
		//   we're gonna need drugs for this one......
		//
		//  Makes references to X1(.,.,iu(i,mu)) AND X2(.,.,iu(i,mu))
		//  as a result, need to swap the DOWN halos in all dirs for
		//  both these arrays, each of which has 8 cpts
		//
#ifdef __clang__
#pragma omp target teams distribute parallel for\
		map(to:X2[0:kferm2Halo],X1[0:kferm2Halo],akappa)
#else
#pragma omp parallel for
#endif
		for(int i=0;i<kvol;i++)
			for(int idirac=0;idirac<ndirac;idirac++){
				int mu, uid, igork1;
#ifndef NO_SPACE
#pragma omp simd aligned(dSdpi,X1,X2,u11t,u12t,iu:AVX)
				for(mu=0; mu<3; mu++){
					//Long term ambition. I used the diff command on the different
					//spacial components of dSdpi and saw a lot of the values required
					//for them are duplicates (u11(i,mu)*X2(1,idirac,i) is used again with
					//a minus in front for example. Why not evaluate them first /and then plug 
					//them into the equation? Reduce the number of evaluations needed and look
					//a bit neater (although harder to follow as a consequence).

					//Up indices
					uid = iu[mu+ndim*i];
					igork1 = gamin[mu][idirac];	
					dSdpi[(i*nadj)*ndim+mu]+=akappa*creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
							  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 ( u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
								-conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
							  +u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
							  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])));
					dSdpi[(i*nadj)*ndim+mu]+=creal(I*gamval[mu][idirac]*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
							  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
							  +u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])));

					dSdpi[(i*nadj+1)*ndim+mu]+=akappa*creal(
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
							  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
							  -conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (-u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
							  -u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
							  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])));
					dSdpi[(i*nadj+1)*ndim+mu]+=creal(gamval[mu][idirac]*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
							  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (-u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
							  -u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])));

					dSdpi[(i*nadj+2)*ndim+mu]+=akappa*creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
							  +u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
							  -u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
							  -conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
							  +u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])));
					dSdpi[(i*nadj+2)*ndim+mu]+=creal(I*gamval[mu][idirac]*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
							  +u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
							  +u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
							  -conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
							  -u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1])));

				}
#endif
				//We're not done tripping yet!! Time like term is different. dk4? shows up
				//For consistency we'll leave mu in instead of hard coding.
				mu=3;
				uid = iu[mu+ndim*i];
				//We are mutiplying terms by dk4?[i] Also there is no akappa or gamval factor in the time direction	
				//for the "gamval" terms the sign of d4kp flips
#ifndef NO_TIME
				dSdpi[(i*nadj)*ndim+mu]+=creal(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
									  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (dk4p[i]*      (+u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
											  -conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
												+u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (dk4p[i]*      (-u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
											  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))))
					+creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
										  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-dk4p[i]*       (u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													 -conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
													+u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-dk4p[i]*      (-u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													-conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))));

				dSdpi[(i*nadj+1)*ndim+mu]+=creal(
						conj(X1[(i*ndirac+idirac)*nc])*
						(dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
									 +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
						+conj(X1[(uid*ndirac+idirac)*nc])*
						(dk4p[i]*      (-u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
											 -conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))
						+conj(X1[(i*ndirac+idirac)*nc+1])*
						(dk4m[i]*      (-u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
											 -u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
						+conj(X1[(uid*ndirac+idirac)*nc+1])*
						(dk4p[i]*      ( u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
											  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])))
					+creal(
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
										  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-dk4p[i]*      (-u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													-conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (dk4m[i]*      (-u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
												  -u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-dk4p[i]*       (u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													 -conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))));

				dSdpi[(i*nadj+2)*ndim+mu]+=creal(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
												+u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (dk4p[i]*(-conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
									  -u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1]))
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (dk4m[i]* (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
										-conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (dk4p[i]*(-conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
									  +u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1]))))
					+creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
													+u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-dk4p[i]*(-conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
											-u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1]))
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (dk4m[i]* (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
											-conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-dk4p[i]*(-conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
											+u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1]))));

#endif
			}
	}
#pragma omp target update from(dSdpi[0:kmom])
#if defined __INTEL_MKL__
	mkl_free(X2); mkl_free(smallPhi);
#else
	free(X2); free(smallPhi);
#endif
	return 0;
}
