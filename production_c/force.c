/**
 * @file force.c
 * @brief Code for force calculations.
 */
#include	<matrices.h>
int Gauge_force(double *dSdpi, Complex_f *ut[2],unsigned int *iu,unsigned int *id, float beta){
	/*
	 * Calculates dSdpi due to the Wilson Action at each intermediate time
	 *
	 * Calls:
	 * =====
	 * C_Halo_swap_all, C_gather, C_Halo_swap_dir
	 *
	 * Parameters:
	 * =======
	 * double			*dSdpi
	 * Complex_f 			*ut[0]
	 * Complex_f			*ut[1]
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
	//		memset(ut[0][kvol], 0, ndim*halo*sizeof(Complex_f));	
	//		memset(ut[1][kvol], 0, ndim*halo*sizeof(Complex_f));	
	//	#endif
	//Was a trial field halo exchange here at one point.
#ifdef __NVCC__
	cuGauge_force(ut,dSdpi,beta,iu,id,dimGrid,dimBlock);
#else
	Complex_f *Sigma[2], *ush[2];
	Sigma[0] = (Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f)); 
	Sigma[1]= (Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f)); 
	ush[0] = (Complex_f *)aligned_alloc(AVX,(kvol+halo)*sizeof(Complex_f)); 
	ush[1] = (Complex_f *)aligned_alloc(AVX,(kvol+halo)*sizeof(Complex_f)); 
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
		memset(Sigma[0],0, kvol*sizeof(Complex_f));
		memset(Sigma[1],0, kvol*sizeof(Complex_f));
		for(int nu=0; nu<ndim; nu++)
			if(nu!=mu){
				//The +ν Staple
#pragma omp parallel for simd //aligned(ut[0],ut[1],Sigma[0],Sigma[1],iu:AVX)
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int uidn = iu[nu+ndim*i];
					Complex_f	a11=ut[0][uidm*ndim+nu]*conj(ut[0][uidn*ndim+mu])+\
										 ut[1][uidm*ndim+nu]*conj(ut[1][uidn*ndim+mu]);
					Complex_f	a12=-ut[0][uidm*ndim+nu]*ut[1][uidn*ndim+mu]+\
										 ut[1][uidm*ndim+nu]*ut[0][uidn*ndim+mu];
					Sigma[0][i]+=a11*conj(ut[0][i*ndim+nu])+a12*conj(ut[1][i*ndim+nu]);
					Sigma[1][i]+=-a11*ut[1][i*ndim+nu]+a12*ut[0][i*ndim+nu];
				}
				C_gather(ush[0], ut[0], kvol, id, nu);
				C_gather(ush[1], ut[1], kvol, id, nu);
#if(nproc>1)
				CHalo_swap_dir(ush[0], 1, mu, DOWN); CHalo_swap_dir(ush[1], 1, mu, DOWN);
#endif
				//Next up, the -ν staple
#pragma omp parallel for simd //aligned(ut[0],ut[1],ush[0],ush[1],Sigma[0],Sigma[1],iu,id:AVX)
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int didn = id[nu+ndim*i];
					//uidm is correct here
					Complex_f a11=conj(ush[0][uidm])*conj(ut[0][didn*ndim+mu])-\
									  ush[1][uidm]*conj(ut[1][didn*ndim+mu]);
					Complex_f a12=-conj(ush[0][uidm])*ut[1][didn*ndim+mu]-\
									  ush[1][uidm]*ut[0][didn*ndim+mu];
					Sigma[0][i]+=a11*ut[0][didn*ndim+nu]-a12*conj(ut[1][didn*ndim+nu]);
					Sigma[1][i]+=a11*ut[1][didn*ndim+nu]+a12*conj(ut[0][didn*ndim+nu]);
				}
			}
#pragma omp parallel for simd //aligned(ut[0],ut[1],Sigma[0],Sigma[1],dSdpi:AVX)
		for(int i=0;i<kvol;i++){
			Complex_f a11 = ut[0][i*ndim+mu]*Sigma[1][i]+ut[1][i*ndim+mu]*conj(Sigma[0][i]);
			Complex_f a12 = ut[0][i*ndim+mu]*Sigma[0][i]+conj(ut[1][i*ndim+mu])*Sigma[1][i];

			dSdpi[(i*nadj)*ndim+mu]=(double)(beta*cimag(a11));
			dSdpi[(i*nadj+1)*ndim+mu]=(double)(beta*creal(a11));
			dSdpi[(i*nadj+2)*ndim+mu]=(double)(beta*cimag(a12));
		}
	}
	free(ush[0]); free(ush[1]); free(Sigma[0]); free(Sigma[1]);
#endif
	return 0;
}
int Force(double *dSdpi, int iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,Complex *ut[2],\
		Complex_f *ut_f[2],unsigned int *iu,unsigned int *id,Complex *gamval,Complex_f *gamval_f,\
		int *gamin,double *dk[2], float *dk_f[2],Complex_f jqq,float akappa,float beta,double *ancg){
	/*
	 *	@brief Calculates the force @f$\frac{dS}{d\pi}@f$ at each intermediate time
	 *	
	 *	@param	dSdpi:			The force
	 *	@param	iflag:			Invert before evaluating the force?	
	 *	@param	res1:				Conjugate gradient residule
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 *	@param	ut[0],ut[1]		Double precision colour fields
	 *	@param	ut_f[0],ut_f[1]:	Single precision colour fields
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices
	 *	@param	gamval_f:		Single precision gamma matrices
	 * @param	dk[0]:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk[1]:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk_f[0]:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ float
	 * @param	dk_f[1]:			@f$\left(1-\gamma_0\right)e^\mu@f$ float
	 * @param 	jqq:				Diquark source
	 *	@param	akappa:			Hopping parameter
	 *	@param	beta:				Inverse gauge coupling
	 *	@param	ancg:				Counter for conjugate gradient iterations
	 *
	 *	@return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Force";
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
#endif
#ifndef NO_GAUGE
	Gauge_force(dSdpi,ut_f,iu,id,beta);
#endif
	if(!akappa)
		return 0;
	//X1=(M†M)^{1} Phi
	int itercg=1;
#ifdef __NVCC__
	Complex_f *X1_f, *X2_f;
	cudaMallocAsync((void **)&X2_f,kferm2*sizeof(Complex_f),streams[0]);
	cudaMallocAsync((void **)&X1_f,kferm2*sizeof(Complex_f),NULL);
	cuComplex_convert(X1_f,X1,kferm2,true,dimBlock,dimGrid);
#else
	Complex *X2= (Complex *)aligned_alloc(AVX,kferm2Halo*sizeof(Complex));
#endif
	for(int na = 0; na<nf; na++){
#ifdef __NVCC__
		cudaMemcpyAsync(X1,X0+na*kferm2,kferm2*sizeof(Complex),cudaMemcpyDeviceToDevice,NULL);
#else
		memcpy(X1,X0+na*kferm2,kferm2*sizeof(Complex));
#endif
		if(!iflag){
#ifdef __NVCC__
			Complex *smallPhi;
			cudaMallocAsync((void **)&smallPhi,kferm2*sizeof(Complex),streams[0]);
#else
			Complex *smallPhi = (Complex *)aligned_alloc(AVX,kferm2*sizeof(Complex)); 
#endif
			Fill_Small_Phi(na, smallPhi, Phi);
			//	Congradq(na, res1,smallPhi, &itercg );
			Congradq(na,res1,X1,smallPhi,ut_f,iu,id,gamval_f,gamin,dk_f,jqq,akappa,&itercg);
#ifdef __NVCC__
			cudaFreeAsync(smallPhi,streams[0]);
#else
			free(smallPhi);
#endif
			*ancg+=itercg;
#ifdef __NVCC__
			Complex blasa=2.0; double blasb=-1.0;
			cublasZdscal(cublas_handle,kferm2,&blasb,(cuDoubleComplex *)(X0+na*kferm2),1);
			cublasZaxpy(cublas_handle,kferm2,(cuDoubleComplex *)&blasa,(cuDoubleComplex *)X1,1,(cuDoubleComplex *)(X0+na*kferm2),1);
			cuComplex_convert(X1_f,X1,kferm2,true,dimBlock,dimGrid);
			//HDslash launches a different stream so we need a barrieer
			cudaDeviceSynchronise();
#elif (defined __INTEL_MKL__)
			Complex blasa=2.0; Complex blasb=-1.0;
			//This is not a general BLAS Routine. BLIS and MKl support it
			//CUDA and GSL does not support it
			cblas_zaxpby(kferm2, &blasa, X1, 1, &blasb, X0+na*kferm2, 1); 
#elif defined USE_BLAS
			Complex blasa=2.0; double blasb=-1.0;
			cblas_zdscal(kferm2,blasb,X0+na*kferm2,1);
			cblas_zaxpy(kferm2,&blasa,X1,1,X0+na*kferm2,1);
#else
#pragma omp parallel for simd collapse(2)
			for(int i=0;i<kvol;i++)
				for(int idirac=0;idirac<ndirac;idirac++){
					X0[((na*kvol+i)*ndirac+idirac)*nc]=
						2*X1[(i*ndirac+idirac)*nc]-X0[((na*kvol+i)*ndirac+idirac)*nc];
					X0[((na*kvol+i)*ndirac+idirac)*nc+1]=
						2*X1[(i*ndirac+idirac)*nc+1]-X0[((na*kvol+i)*ndirac+idirac)*nc+1];
				}
#endif
		}
		#ifdef __NVCC__
		Hdslash_f(X2_f,X1_f,ut_f,iu,id,gamval_f,gamin,dk_f,akappa);
		#else
		Hdslash(X2,X1,ut,iu,id,gamval,gamin,dk,akappa);
		#endif
#ifdef __NVCC__
		float blasd=2.0;
		cudaDeviceSynchronise();
		cublasCsscal(cublas_handle,kferm2, &blasd, (cuComplex *)X2_f, 1);
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

		//	The original FORTRAN Comment:
		//    dSdpi=dSdpi-Re(X1*(d(Mdagger)dp)*X2) -- Yikes!
		//   we're gonna need drugs for this one......
		//
		//  Makes references to X1(.,.,iu(i,mu)) AND X2(.,.,iu(i,mu))
		//  as a result, need to swap the DOWN halos in all dirs for
		//  both these arrays, each of which has 8 cpts
		//
#ifdef __NVCC__
		cuForce(dSdpi,ut_f,X1_f,X2_f,gamval_f,dk_f,iu,gamin,akappa,dimGrid,dimBlock);
		cudaDeviceSynchronise();
#else
#pragma omp parallel for
		for(int i=0;i<kvol;i++)
			for(int idirac=0;idirac<ndirac;idirac++){
				int mu, uid, igork1;
#ifndef NO_SPACE
#pragma omp simd //aligned(dSdpi,X1,X2,ut[0],ut[1],iu:AVX)
				for(mu=0; mu<3; mu++){
					//Long term ambition. I used the diff command on the different
					//spacial components of dSdpi and saw a lot of the values required
					//for them are duplicates (u11(i,mu)*X2(1,idirac,i) is used again with
					//a minus in front for example. Why not evaluate them first /and then plug 
					//them into the equation? Reduce the number of evaluations needed and look
					//a bit neater (although harder to follow as a consequence).

					//Up indices
					uid = iu[mu+ndim*i];
					igork1 = gamin[mu*ndirac+idirac];	

					//REMINDER. Gamma is already scaled by kappa when we defined them. So if yer trying to rederive this from
					//Montvay and Munster and notice a missing kappa in the code, that is why.
					dSdpi[(i*nadj)*ndim+mu]+=akappa*creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
							  +conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 ( ut[1][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
								-conj(ut[0][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (ut[0][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
							  +ut[1][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-ut[0][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
							  -conj(ut[1][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])));
					dSdpi[(i*nadj)*ndim+mu]+=creal(I*gamval[mu*ndirac+idirac]*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
							  +conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-ut[1][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(ut[0][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (ut[0][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
							  +ut[1][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (ut[0][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(ut[1][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])));

					dSdpi[(i*nadj+1)*ndim+mu]+=akappa*creal(
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
							  +conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-ut[1][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
							  -conj(ut[0][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (-ut[0][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
							  -ut[1][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (ut[0][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
							  -conj(ut[1][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])));
					dSdpi[(i*nadj+1)*ndim+mu]+=creal(gamval[mu*ndirac+idirac]*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
							  +conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (ut[1][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(ut[0][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (-ut[0][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
							  -ut[1][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-ut[0][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
							  +conj(ut[1][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])));

					dSdpi[(i*nadj+2)*ndim+mu]+=akappa*creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (ut[0][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
							  +ut[1][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-conj(ut[0][i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
							  -ut[1][i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
							  -conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-conj(ut[1][i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
							  +ut[0][i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])));
					dSdpi[(i*nadj+2)*ndim+mu]+=creal(I*gamval[mu*ndirac+idirac]*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (ut[0][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
							  +ut[1][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (conj(ut[0][i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
							  +ut[1][i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1])
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
							  -conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (conj(ut[1][i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
							  -ut[0][i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1])));

				}
#endif
				//We're not done tripping yet!! Time like term is different. dk4? shows up
				//For consistency we'll leave mu in instead of hard coding.
				mu=3;
				uid = iu[mu+ndim*i];
				igork1 = gamin[mu*ndirac+idirac];	
#ifndef NO_TIME
				dSdpi[(i*nadj)*ndim+mu]+=creal(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (dk[0][i]*(-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
										+conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (dk[1][i]*      (+ut[1][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
												-conj(ut[0][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (dk[0][i]*       (ut[0][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
												 +ut[1][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (dk[1][i]*      (-ut[0][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
												-conj(ut[1][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))))
					+creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (dk[0][i]*(-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
											+conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-dk[1][i]*       (ut[1][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													  -conj(ut[0][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (dk[0][i]*       (ut[0][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
													 +ut[1][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-dk[1][i]*      (-ut[0][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													 -conj(ut[1][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))));

				dSdpi[(i*nadj+1)*ndim+mu]+=creal(
						conj(X1[(i*ndirac+idirac)*nc])*
						(dk[0][i]*(-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
									  +conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
						+conj(X1[(uid*ndirac+idirac)*nc])*
						(dk[1][i]*      (-ut[1][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
											  -conj(ut[0][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))
						+conj(X1[(i*ndirac+idirac)*nc+1])*
						(dk[0][i]*      (-ut[0][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
											  -ut[1][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
						+conj(X1[(uid*ndirac+idirac)*nc+1])*
						(dk[1][i]*      ( ut[0][i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
												-conj(ut[1][i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])))
					+creal(
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (dk[0][i]*(-conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
											+conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-dk[1][i]*      (-ut[1][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													 -conj(ut[0][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (dk[0][i]*      (-ut[0][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
													-ut[1][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-dk[1][i]*       (ut[0][i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
													  -conj(ut[1][i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))));

				dSdpi[(i*nadj+2)*ndim+mu]+=creal(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (dk[0][i]*       (ut[0][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
												 +ut[1][i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (dk[1][i]*(-conj(ut[0][i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
										-ut[1][i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1]))
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (dk[0][i]* (conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
										 -conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (dk[1][i]*(-conj(ut[1][i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
										+ut[0][i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1]))))
					+creal(I*
							(conj(X1[(i*ndirac+idirac)*nc])*
							 (dk[0][i]*       (ut[0][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
													 +ut[1][i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc])*
							 (-dk[1][i]*(-conj(ut[0][i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
											 -ut[1][i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1]))
							 +conj(X1[(i*ndirac+idirac)*nc+1])*
							 (dk[0][i]* (conj(ut[1][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
											 -conj(ut[0][i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
							 +conj(X1[(uid*ndirac+idirac)*nc+1])*
							 (-dk[1][i]*(-conj(ut[1][i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
											 +ut[0][i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1]))));

#endif
			}
#endif
	}
#ifdef __NVCC__
	cudaFreeAsync(X1_f,streams[0]); cudaFreeAsync(X2_f,streams[1]);
#else
	free(X2); 
#endif
	return 0;
}
