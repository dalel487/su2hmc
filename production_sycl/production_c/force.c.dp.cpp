#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 * @file force.c
 * @brief Code for force calculations.
 */
#include	<matrices.h>
int Gauge_force(double *dSdpi, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id, float beta){
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
	 * Complex_f 			*u11t
	 * Complex_f			*u12t
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
	//		memset(u11t[kvol], 0, ndim*halo*sizeof(Complex_f));	
	//		memset(u12t[kvol], 0, ndim*halo*sizeof(Complex_f));	
	//	#endif
	//Was a trial field halo exchange here at one point.
#ifdef DPCT_COMPATIBILITY_TEMP
        int device=-1;
        device = dpct::dev_mgr::instance().current_device_id();
        Complex_f *Sigma11, *Sigma12, *u11sh, *u12sh;
	cudaMallocAsync((void **)&Sigma11,kvol*sizeof(Complex_f),streams[0]);
	cudaMallocAsync((void **)&Sigma12,kvol*sizeof(Complex_f),streams[1]);
        cudaMallocManaged((void **)&u11sh, (kvol + halo) * sizeof(Complex_f), 0);
        cudaMallocManaged((void **)&u12sh, (kvol + halo) * sizeof(Complex_f), 0);
#else
	Complex_f *Sigma11 = (Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f)); 
	Complex_f *Sigma12= (Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f)); 
	Complex_f *u11sh = (Complex_f *)aligned_alloc(AVX,(kvol+halo)*sizeof(Complex_f)); 
	Complex_f *u12sh = (Complex_f *)aligned_alloc(AVX,(kvol+halo)*sizeof(Complex_f)); 
#endif
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
#ifdef DPCT_COMPATIBILITY_TEMP
                cudaMemset(Sigma11,0, kvol*sizeof(Complex_f));
		cudaMemset(Sigma12,0, kvol*sizeof(Complex_f));
#else
		memset(Sigma11,0, kvol*sizeof(Complex_f));
		memset(Sigma12,0, kvol*sizeof(Complex_f));
#endif
		for(int nu=0; nu<ndim; nu++)
			if(nu!=mu){
				//The +ν Staple
#ifdef DPCT_COMPATIBILITY_TEMP
                                cuPlus_staple(mu,nu,iu,Sigma11,Sigma12,u11t,u12t,dimGrid,dimBlock);
#else
#pragma omp parallel for simd aligned(u11t,u12t,Sigma11,Sigma12,iu:AVX)
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int uidn = iu[nu+ndim*i];
					Complex_f	a11=u11t[uidm*ndim+nu]*conj(u11t[uidn*ndim+mu])+\
										 u12t[uidm*ndim+nu]*conj(u12t[uidn*ndim+mu]);
					Complex_f	a12=-u11t[uidm*ndim+nu]*u12t[uidn*ndim+mu]+\
										 u12t[uidm*ndim+nu]*u11t[uidn*ndim+mu];
					Sigma11[i]+=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
					Sigma12[i]+=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+nu];
				}
#endif
				C_gather(u11sh, u11t, kvol, id, nu);
				C_gather(u12sh, u12t, kvol, id, nu);
#if(nproc>1)
#ifdef __NVCC__
				//Prefetch to the CPU for until we get NCCL working
				cudaMemPrefetchAsync(u11sh, kvol*sizeof(Complex_f),cudaCpuDeviceId,streams[0]);
				cudaMemPrefetchAsync(u12sh, kvol*sizeof(Complex_f),cudaCpuDeviceId,streams[1]);
#endif
				CHalo_swap_dir(u11sh, 1, mu, DOWN); CHalo_swap_dir(u12sh, 1, mu, DOWN);
#ifdef __NVCC__
				cudaMemPrefetchAsync(u11sh+kvol, halo*sizeof(Complex_f),device,streams[0]);
				cudaMemPrefetchAsync(u12sh+kvol, halo*sizeof(Complex_f),device,streams[1]);
#endif
#endif
				//Next up, the -ν staple
#ifdef DPCT_COMPATIBILITY_TEMP
                                cudaDeviceSynchronise();
				cuMinus_staple(mu,nu,iu,id,Sigma11,Sigma12,u11sh,u12sh,u11t,u12t,dimGrid,dimBlock);
#else
#pragma omp parallel for simd aligned(u11t,u12t,u11sh,u12sh,Sigma11,Sigma12,iu,id:AVX)
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int didn = id[nu+ndim*i];
					//uidm is correct here
					Complex_f a11=conj(u11sh[uidm])*conj(u11t[didn*ndim+mu])-\
									  u12sh[uidm]*conj(u12t[didn*ndim+mu]);
					Complex_f a12=-conj(u11sh[uidm])*u12t[didn*ndim+mu]-\
									  u12sh[uidm]*u11t[didn*ndim+mu];
					Sigma11[i]+=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
					Sigma12[i]+=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+nu]);
				}
#endif
			}
#ifdef DPCT_COMPATIBILITY_TEMP
                cuGauge_force(mu,Sigma11,Sigma12,u11t,u12t,dSdpi,beta,dimGrid,dimBlock);
#else
#pragma omp parallel for simd aligned(u11t,u12t,Sigma11,Sigma12,dSdpi:AVX)
		for(int i=0;i<kvol;i++){
			Complex_f a11 = u11t[i*ndim+mu]*Sigma12[i]+u12t[i*ndim+mu]*conj(Sigma11[i]);
			Complex_f a12 = u11t[i*ndim+mu]*Sigma11[i]+conj(u12t[i*ndim+mu])*Sigma12[i];

			dSdpi[(i*nadj)*ndim+mu]=(double)(beta*cimag(a11));
			dSdpi[(i*nadj+1)*ndim+mu]=(double)(beta*creal(a11));
			dSdpi[(i*nadj+2)*ndim+mu]=(double)(beta*cimag(a12));
		}
#endif
	}
#ifdef DPCT_COMPATIBILITY_TEMP
        cudaDeviceSynchronise();
	cudaFreeAsync(Sigma11,streams[0]); cudaFreeAsync(Sigma12,streams[1]); cudaFree(u11sh); cudaFree(u12sh);
#else
	free(u11sh); free(u12sh); free(Sigma11); free(Sigma12);
#endif
	return 0;
}
int Force(double *dSdpi, int iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,Complex *u11t, Complex *u12t,\
		Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,Complex *gamval,Complex_f *gamval_f,\
		int *gamin,double *dk4m, double *dk4p, float *dk4m_f,float *dk4p_f,Complex_f jqq,\
		float akappa,float beta,double *ancg){
	/*
	 *	@brief Calculates the force @f$\frac{dS}{d\pi}@f$ at each intermediate time
	 *	
	 *	@param	dSdpi:			The force
	 *	@param	iflag:			Invert before evaluating the force?	
	 *	@param	res1:				Conjugate gradient residule
	 *	@param	X0:				Up/down partitioned pseudofermion field
	 *	@param	X1:				Holder for the partitioned fermion field, then the conjugate gradient output
	 *	@param	Phi:				Pseudofermion field
	 *	@param	u11t,u12t		Double precision colour fields
	 *	@param	u11t_f,u12t_f:	Single precision colour fields
	 *	@param	iu,id:			Lattice indices
	 *	@param	gamin:			Gamma indices
	 *	@param	gamval:			Double precision gamma matrices
	 *	@param	gamval_f:		Single precision gamma matrices
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 * @param	dk4m_f:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ float
	 * @param	dk4p_f:			@f$\left(1-\gamma_0\right)e^\mu@f$ float
	 * @param 	jqq:				Diquark source
	 *	@param	akappa:			Hopping parameter
	 *	@param	beta:				Inverse gauge coupling
	 *	@param	ancg:				Counter for conjugate gradient iterations
	 *
	 *	@return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Force";
#ifdef DPCT_COMPATIBILITY_TEMP
        int device=-1;
        device = dpct::dev_mgr::instance().current_device_id();
#endif
#ifndef NO_GAUGE
	Gauge_force(dSdpi,u11t_f,u12t_f,iu,id,beta);
#endif
	//X1=(M†M)^{1} Phi
	int itercg=1;
#ifdef DPCT_COMPATIBILITY_TEMP
        Complex *X2;
        cudaMallocManaged((void **)&X2, kferm2Halo * sizeof(Complex), 0);
#else
	Complex *X2= (Complex *)aligned_alloc(AVX,kferm2Halo*sizeof(Complex));
#endif
	for(int na = 0; na<nf; na++){
#ifdef DPCT_COMPATIBILITY_TEMP
                cudaMemcpyAsync(X1,X0+na*kferm2,kferm2*sizeof(Complex),cudaMemcpyDeviceToDevice,NULL);
#else
		memcpy(X1,X0+na*kferm2,kferm2*sizeof(Complex));
#endif
		if(!iflag){
#ifdef DPCT_COMPATIBILITY_TEMP
                        Complex *smallPhi;
			cudaMallocAsync((void **)&smallPhi,kferm2*sizeof(Complex),streams[0]);
#else
			Complex *smallPhi = (Complex *)aligned_alloc(AVX,kferm2*sizeof(Complex)); 
#endif
			Fill_Small_Phi(na, smallPhi, Phi);
			//	Congradq(na, res1,smallPhi, &itercg );
			Congradq(na,res1,X1,smallPhi,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa,&itercg);
#ifdef DPCT_COMPATIBILITY_TEMP
                        cudaFreeAsync(smallPhi,streams[0]);
#else
			free(smallPhi);
#endif
			*ancg+=itercg;
#ifdef DPCT_COMPATIBILITY_TEMP
                        Complex blasa=2.0; double blasb=-1.0;
			cublasZdscal(cublas_handle,kferm2,&blasb,(cuDoubleComplex *)(X0+na*kferm2),1);
			cublasZaxpy(cublas_handle,kferm2,(cuDoubleComplex *)&blasa,(cuDoubleComplex *)X1,1,(cuDoubleComplex *)(X0+na*kferm2),1);
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
		Hdslash(X2,X1,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
#ifdef DPCT_COMPATIBILITY_TEMP
                double blasd=2.0;
		cudaDeviceSynchronise();
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

		//	The original FORTRAN Comment:
		//    dSdpi=dSdpi-Re(X1*(d(Mdagger)dp)*X2) -- Yikes!
		//   we're gonna need drugs for this one......
		//
		//  Makes references to X1(.,.,iu(i,mu)) AND X2(.,.,iu(i,mu))
		//  as a result, need to swap the DOWN halos in all dirs for
		//  both these arrays, each of which has 8 cpts
		//
#ifdef DPCT_COMPATIBILITY_TEMP
                cudaDeviceSynchronise();
		cuForce(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa,dimGrid,dimBlock);
#else
#pragma omp parallel for
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
					igork1 = gamin[mu*ndirac+idirac];	
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
					dSdpi[(i*nadj)*ndim+mu]+=creal(I*gamval[mu*ndirac+idirac]*
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
					dSdpi[(i*nadj+1)*ndim+mu]+=creal(gamval[mu*ndirac+idirac]*
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
					dSdpi[(i*nadj+2)*ndim+mu]+=creal(I*gamval[mu*ndirac+idirac]*
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
				igork1 = gamin[mu*ndirac+idirac];	
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
#ifdef DPCT_COMPATIBILITY_TEMP
        dpct::dpct_free(X2, dpct::get_in_order_queue());
#else
	free(X2); 
#endif
	return 0;
}
