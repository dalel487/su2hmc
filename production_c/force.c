/*
 * Code for force calculations.
 * Requires multiply.cu to work
 */
#include	<matrices.h>
int Gauge_force(double *dSdpi,Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id, float beta){
	/*
	 * Calculates dSdpi due to the Wilson Action at each intermediate time
	 *
	 * Calls:
	 * =====
	 * Z_Halo_swap_all, Z_gather, Z_Halo_swap_dir
	 *
	 * Parameters:
	 * =======
	 * double			*dSdpi
	 * Complex 			*u11t
	 * Complex			*u12t
	 * unsigned int	*iu 
	 * unsigned int	*id 
	 * float				beta
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Gauge_force";

	//We define zero halos for debugging
	//	#ifdef _DEBUG
	//		memset(u11t[kvol], 0, ndim*halo*sizeof(Complex));	
	//		memset(u12t[kvol], 0, ndim*halo*sizeof(Complex));	
	//	#endif
	//Was a trial field halo exchange here at one point.
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	Complex *Sigma11, *Sigma12, *u11sh, *u12sh;
	cudaMallocManaged((Complex**)&Sigma11,kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged((Complex**)&Sigma12,kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged((Complex**)&u11sh,(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged((Complex**)&u12sh,(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
	Complex *Sigma11 = (Complex *)mkl_malloc(kvol*sizeof(Complex),AVX); 
	Complex *Sigma12= (Complex *)mkl_malloc(kvol*sizeof(Complex),AVX); 
	Complex *u11sh = (Complex *)mkl_malloc((kvol+halo)*sizeof(Complex),AVX); 
	Complex *u12sh = (Complex *)mkl_malloc((kvol+halo)*sizeof(Complex),AVX); 
#else
	Complex *Sigma11 = (Complex *)aligned_alloc(AVX,kvol*sizeof(Complex)); 
	Complex *Sigma12= (Complex *)aligned_alloc(AVX,kvol*sizeof(Complex)); 
	Complex *u11sh = (Complex *)aligned_alloc(AVX,(kvol+halo)*sizeof(Complex)); 
	Complex *u12sh = (Complex *)aligned_alloc(AVX,(kvol+halo)*sizeof(Complex)); 
#endif
#pragma acc enter data create(Sigma11[0:kvol],Sigma12[0:kvol],u11sh[0:kvol+halo],u12sh[0:kvol+halo])
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
		memset(Sigma11,0, kvol*sizeof(Complex));
		memset(Sigma12,0, kvol*sizeof(Complex));
#ifdef __NVCC__
		cudaMemPrefetchAsync(Sigma11,kvol*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(Sigma12,kvol*sizeof(Complex),device,NULL);
#endif
		for(int nu=0; nu<ndim; nu++)
			if(nu!=mu){
				//The +ν Staple
#ifdef __NVCC__
				cuPlus_staple(mu,nu,iu,Sigma11,Sigma12,u11t,u12t,dimGrid,dimBlock);
#else
#ifdef _OPENACC
#pragma acc parallel loop
#else
#pragma omp parallel for simd aligned(u11t,u12t,Sigma11,Sigma12,iu:AVX)
#endif
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int uidn = iu[nu+ndim*i];
					Complex	a11=u11t[uidm*ndim+nu]*conj(u11t[uidn*ndim+mu])+\
									 u12t[uidm*ndim+nu]*conj(u12t[uidn*ndim+mu]);
					Complex	a12=-u11t[uidm*ndim+nu]*u12t[uidn*ndim+mu]+\
									 u12t[uidm*ndim+nu]*u11t[uidn*ndim+mu];
					Sigma11[i]+=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
					Sigma12[i]+=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+nu];
				}
#endif
				Z_gather(u11sh, u11t, kvol, id, nu);
#if(nproc>1)
				ZHalo_swap_dir(u11sh, 1, mu, DOWN);
#endif
#ifdef __NVCC__
				cudaMemPrefetchAsync(u11sh, (kvol+halo)*sizeof(Complex),device,NULL);
#endif
#pragma acc update device(u11sh[0:kvol+halo])
				Z_gather(u12sh, u12t, kvol, id, nu);
#ifdef __NVCC__
				cudaMemPrefetchAsync(u12sh, (kvol+halo)*sizeof(Complex),device,NULL);
#endif
#if(nproc>1)
				ZHalo_swap_dir(u12sh, 1, mu, DOWN);
#endif
#pragma acc update device(u12sh[0:kvol+halo])
				//Next up, the -ν staple
#ifdef __NVCC__
				cuMinus_staple(mu,nu,iu,id,Sigma11,Sigma12,u11sh,u12sh,u11t,u12t,dimGrid,dimBlock);
#else
#ifdef _OPENACC
#pragma acc parallel loop
#else
#pragma omp parallel for simd aligned(u11t,u12t,u11sh,u12sh,Sigma11,Sigma12,iu,id:AVX)
#endif
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int didn = id[nu+ndim*i];
					//uidm is correct here
					Complex a11=conj(u11sh[uidm])*conj(u11t[didn*ndim+mu])-\
									u12sh[uidm]*conj(u12t[didn*ndim+mu]);
					Complex a12=-conj(u11sh[uidm])*u12t[didn*ndim+mu]-\
									u12sh[uidm]*u11t[didn*ndim+mu];
					Sigma11[i]+=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
					Sigma12[i]+=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+nu]);
				}
#endif
			}
#ifdef __NVCC__
		cuGauge_force(mu,Sigma11,Sigma12,u11t,u12t,dSdpi,beta,dimGrid,dimBlock);
#else
#ifdef _OPENACC
#pragma acc parallel loop
#else
#pragma omp parallel for simd aligned(u11t,u12t,Sigma11,Sigma12,dSdpi:AVX)
#endif
		for(int i=0;i<kvol;i++){
			Complex a11 = u11t[i*ndim+mu]*Sigma12[i]+u12t[i*ndim+mu]*conj(Sigma11[i]);
			Complex a12 = u11t[i*ndim+mu]*Sigma11[i]+conj(u12t[i*ndim+mu])*Sigma12[i];

			dSdpi[(i*nadj)*ndim+mu]=beta*cimag(a11);
			dSdpi[(i*nadj+1)*ndim+mu]=beta*creal(a11);
			dSdpi[(i*nadj+2)*ndim+mu]=beta*cimag(a12);
		}
#endif
	}
#pragma acc exit data delete(Sigma11[0:kvol],Sigma12[0:kvol],u11sh[0:kvol+halo],u12sh[0:kvol+halo])
#ifdef __NVCC__
	cudaFree(Sigma11); cudaFree(Sigma12); cudaFree(u11sh); cudaFree(u12sh);
#elif defined __INTEL_MKL__
	mkl_free(u11sh); mkl_free(u12sh); mkl_free(Sigma11); mkl_free(Sigma12);
#else
	free(u11sh); free(u12sh); free(Sigma11); free(Sigma12);
#endif
	return 0;
}
int Force(double *dSdpi, int iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,Complex *u11t, Complex *u12t,\
		Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,Complex gamval[5][4],Complex_f gamval_f[5][4],\
		int gamin[4][4],double *dk4m, double *dk4p, float *dk4m_f,float *dk4p_f,Complex_f jqq,\
		float akappa,float beta,double *ancg){
	/*
	 *	Calculates dSdpi at each intermediate time
	 *	
	 *	Calls:
	 *	=====
	 *	Gauge_force, Fill_Small_Phi, Congradq, Hdslash, ZHalo_swap_dir
	 *
	 *	Parameters:
	 *	===========
	 *	double			*dSdpi
	 *	int				iflag
	 *	double			res1:
	 *	Complex			*X0:
	 *	Complex			*X1:
	 *	Complex			*Phi:
	 *	Complex			*u11t:
	 *	Complex			*u12t:
	 *	Complex_f		*u11t_f:
	 *	Complex_f		*u12t_f:
	 *	unsigned int	*iu:
	 *	unsigned int	*id:
	 *	Complex			gamval[5][4]:
	 *	Complex_f		gamval_f[5][4]:
	 *	float				akappa:
	 *	float				beta:
	 *	double			*ancg:
	 *
	 *	Returns:
	 *	=======
	 *	Zero on success, integer error code otherwise
	 */
	const char *funcname = "Force";
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(u11t,(kvol+halo)*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(u12t,(kvol+halo)*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(X0,nf*kfermHalo*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(Phi,nf*kfermHalo*sizeof(Complex),device,NULL);
#endif
#pragma acc update device(dSdpi[0:kmom])
#ifndef NO_GAUGE
	Gauge_force(dSdpi,u11t,u12t,iu,id,beta);
#endif
	//X1=(M†M)^{1} Phi
	int itercg=1;
#ifdef __NVCC__
	Complex *X2;
	cudaMallocManaged(&X2,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
	Complex *X2= (Complex *)mkl_malloc(kferm2Halo*sizeof(Complex), AVX);
#else
	Complex *X2= (Complex *)aligned_alloc(AVX,kferm2Halo*sizeof(Complex));
#endif
	for(int na = 0; na<nf; na++){
		memcpy(X1, X0+na*kferm2, kferm2*sizeof(Complex));
		if(!iflag){
#ifdef __NVCC__
			Complex *smallPhi;
			cudaMallocManaged(&smallPhi,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
			Complex *smallPhi =(Complex *)mkl_malloc(kferm2Halo*sizeof(Complex), AVX); 
#else
			Complex *smallPhi = (Complex *)aligned_alloc(AVX,kferm2Halo*sizeof(Complex)); 
#endif
			Fill_Small_Phi(na, smallPhi, Phi);
			//	Congradq(na, res1,smallPhi, &itercg );
			Congradq(na,res1,X1,smallPhi,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa,&itercg);
#ifdef __NVCC__
			cudaFree(smallPhi);
#elif defined __INTEL_MKL__
			mkl_free(smallPhi);
#else
			free(smallPhi);
#endif
			*ancg+=itercg;
			//This is not a general BLAS Routine. BLIS and MKl support it
			//CUDA and GSL does not support it
#if (defined __INTEL_MKL__ || defined AMD_BLAS)
			Complex blasa=2.0; Complex blasb=-1.0;
			cblas_zaxpby(kferm2, &blasa, X1, 1, &blasb, X0+na*kferm2, 1); 
#else
#ifdef _OPENACC
#pragma acc update self(Z1[0:kferm2])
#pragma acc loop copy(X0[0:kferm2])
#else
#pragma omp parallel for simd collapse(2)
#endif
			for(int i=0;i<kvol;i++)
				for(int idirac=0;idirac<ndirac;idirac++){
					X0[((na*kvol+i)*ndirac+idirac)*nc]=
						2*X1[(i*ndirac+idirac)*nc]-X0[((na*kvol+i)*ndirac+idirac)*nc];
					X0[((na*kvol+i)*ndirac+idirac)*nc+1]=
						2*X1[(i*ndirac+idirac)*nc+1]-X0[((na*kvol+i)*ndirac+idirac)*nc+1];
				}
#endif
		}
		Hdslash(X2,X1,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
#ifdef __NVCC__
		double blasd=2.0;
		cublasZdscal(cublas_handle,kferm2, &blasd, (cuDoubleComplex *)X2, 1);
#elif defined USE_BLAS
		double blasd=2.0;
		cblas_zdscal(kferm2, blasd, X2, 1);
#else
#pragma unroll
		for(int i=0;i<kferm2;i++)
			X2[i]*=2;
#endif
#if(npx>1)
			ZHalo_swap_dir(X1,8,0,DOWN);
			ZHalo_swap_dir(X2,8,0,DOWN);
#endif
#if(npy>1)
			ZHalo_swap_dir(X1,8,1,DOWN);
			ZHalo_swap_dir(X2,8,1,DOWN);
#endif
#if(npz>1)
			ZHalo_swap_dir(X1,8,2,DOWN);
			ZHalo_swap_dir(X2,8,2,DOWN);
#endif
#if(npt>1)
			ZHalo_swap_dir(X1,8,3,DOWN);
			ZHalo_swap_dir(X2,8,3,DOWN);
#endif
#pragma acc update device(X1[kferm2:kferm2Halo])

		//	The original FORTRAN Comment:
		//    dSdpi=dSdpi-Re(X1*(d(Mdagger)dp)*X2) -- Yikes!
		//   we're gonna need drugs for this one......
		//
		//  Makes references to X1(.,.,iu(i,mu)) AND X2(.,.,iu(i,mu))
		//  as a result, need to swap the DOWN halos in all dirs for
		//  both these arrays, each of which has 8 cpts
		//
#ifdef __NVCC__
		cuForce(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa,dimGrid,dimBlock);
#else
#ifdef _OPENACC
#pragma acc parallel loop copyin(X2[na*kferm2Halo:(na+1)*kferm2Halo])
#else
		//#pragma omp target teams distribute parallel for map(to:X1[0:kferm2Halo],X2[0:kferm2Halo]) map(tofrom:dSdpi[0:kmom])
#pragma omp parallel for
#endif
		for(int i=0;i<kvol;i++)
			for(int idirac=0;idirac<ndirac;idirac++){
				int mu, uid, igork1;
#ifndef NO_SPACE
#ifndef _OPENACC
#pragma omp simd aligned(dSdpi,X1,X2,u11t,u12t,iu:AVX)
#endif
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
				igork1 = gamin[mu][idirac];	
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
#endif
	}
#pragma acc update self(dSdpi[0:kmom])
#ifdef __NVCC__
	cudaFree(X2);
#elif defined __INTEL_MKL__
	mkl_free(X2);
#else
	free(X2); 
#endif
	return 0;
}
