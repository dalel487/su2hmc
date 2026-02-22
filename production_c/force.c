/**
 * @file 	force.c
 *
 * @brief 	Code for force calculations.
 *
 * @author	D. Lawlor
 */
#include	<matrices.h>
#include	<clover.h>

int Gauge_force(double *dSdpi, Complex_f *ut[2],unsigned int *iu,unsigned int *id, float beta){
	const char funcname[] = "Gauge_force";

	//We define zero halos for debugging
	//	#ifdef _DEBUG
	//		memset(ut[0][kvol], 0, ndim*halo*sizeof(Complex_f));	
	//		memset(ut[1][kvol], 0, ndim*halo*sizeof(Complex_f));	
	//	#endif
	//Was a trial field halo exchange here at one point.
#ifdef __NVCC__
	cuGauge_force(ut,dSdpi,beta,iu,id,dimGrid,dimBlock);
	cudaDeviceSynchronise();
#else
	Complex_f *Sigma[2], *ush[2];
	Sigma[0] = (Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f)); 
	Sigma[1]= (Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f)); 
	ush[0] = (Complex_f *)aligned_alloc(AVX,kvolHalo*sizeof(Complex_f)); 
	ush[1] = (Complex_f *)aligned_alloc(AVX,kvolHalo*sizeof(Complex_f)); 
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
		memset(Sigma[0],0, kvol*sizeof(Complex_f));
		memset(Sigma[1],0, kvol*sizeof(Complex_f));
		for(int nu=0; nu<ndim; nu++)
			if(nu!=mu){
				//The +ν Staple
#pragma omp parallel for simd //aligned(ut[0],ut[1],Sigma[0],Sigma[1],iu:AVX)
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu*kvol+i];
					int uidn = iu[nu*kvol+i];
					Complex_f	a11=ut[0][uidm+kvolHalo*nu]*conj(ut[0][uidn+kvolHalo*mu])+\
										 ut[1][uidm+kvolHalo*nu]*conj(ut[1][uidn+kvolHalo*mu]);
					Complex_f	a12=-ut[0][uidm+kvolHalo*nu]*ut[1][uidn+kvolHalo*mu]+\
										 ut[1][uidm+kvolHalo*nu]*ut[0][uidn+kvolHalo*mu];
					Sigma[0][i]+=a11*conj(ut[0][i+kvolHalo*nu])+a12*conj(ut[1][i+kvolHalo*nu]);
					Sigma[1][i]+=-a11*ut[1][i+kvolHalo*nu]+a12*ut[0][i+kvolHalo*nu];
				}
				C_gather(ush[0], ut[0], kvol, id, nu);
				C_gather(ush[1], ut[1], kvol, id, nu);
#if(nproc>1)
				CHalo_swap_dir(ush[0], 1, mu, DOWN); CHalo_swap_dir(ush[1], 1, mu, DOWN);
#endif
				//Next up, the -ν staple
#pragma omp parallel for simd //aligned(ut[0],ut[1],ush[0],ush[1],Sigma[0],Sigma[1],iu,id:AVX)
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu*kvol+i];
					int didn = id[nu*kvol+i];
					//uidm is correct here
					Complex_f a11=conj(ush[0][uidm])*conj(ut[0][didn+kvolHalo*mu])-\
									  ush[1][uidm]*conj(ut[1][didn+kvolHalo*mu]);
					Complex_f a12=-conj(ush[0][uidm])*ut[1][didn+kvolHalo*mu]-\
									  ush[1][uidm]*ut[0][didn+kvolHalo*mu];
					Sigma[0][i]+=a11*ut[0][didn+kvolHalo*nu]-a12*conj(ut[1][didn+kvolHalo*nu]);
					Sigma[1][i]+=a11*ut[1][didn+kvolHalo*nu]+a12*conj(ut[0][didn+kvolHalo*nu]);
				}
			}
#pragma omp parallel for simd //aligned(ut[0],ut[1],Sigma[0],Sigma[1],dSdpi:AVX)
		for(int i=0;i<kvol;i++){
			const unsigned int ind = i+kvolHalo*mu;
			Complex_f a11 = ut[0][ind]*Sigma[1][i]+ut[1][ind]*conj(Sigma[0][i]);
			Complex_f a12 = ut[0][ind]*Sigma[0][i]+conj(ut[1][ind])*Sigma[1][i];

			dSdpi[i+kvol*mu]=(double)(beta*cimag(a11));
			dSdpi[i+kvol*(1*ndim+mu)]=(double)(beta*creal(a11));
			dSdpi[i+kvol*(2*ndim+mu)]=(double)(beta*cimag(a12));
		}
	}
	free(ush[0]); free(ush[1]); free(Sigma[0]); free(Sigma[1]);
#endif
	return 0;
}
void Force_s(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, Complex_f gamval[20],\
		unsigned int *iu, const unsigned short gamin[16],const float akappa, const unsigned short mu){

#pragma omp parallel for simd
	for(unsigned int i=0;i<kvol;i++){
		const unsigned int ind=i+kvolHalo*mu;
		const Complex_f u11s=ut[0][ind]; const Complex_f u12s=ut[1][ind];
		const unsigned int uid = iu[i+kvol*mu];
		//Similarly to Hdslash we always see idirac*nc so we do that here too.
		for(unsigned short idirac=0;idirac<nc*ndirac;idirac+=nc){
			Complex_f X1s[nc];	 Complex_f X1su[nc];
			Complex_f X2s[nc];	 Complex_f X2su[nc];

			X1s[0]=X1[i+kvolHalo*(idirac)]; X1s[1]=X1[i+kvolHalo*(1+idirac)];
			X1su[0]=X1[uid+kvolHalo*(idirac)]; X1su[1]=X1[uid+kvolHalo*(1+idirac)];
			X2s[0]=X2[i+kvolHalo*(idirac)]; X2s[1]=X2[i+kvolHalo*(1+idirac)];
			X2su[0]=X2[uid+kvolHalo*(idirac)]; X2su[1]=X2[uid+kvolHalo*(1+idirac)];

			float dSdpis[3];
			dSdpis[0]=dSdpi[i+kvol*mu];
			//Multiplying by i and taking the real component is the same as taking the negative imaginary component
			//The positions of u11 and u12 might look a bit funky here. That's just because we've multiplied by the
			//generators by hand
			dSdpis[0]+=-akappa*cimag(
					conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					+conj(X1s[1])*(u11s*X2su[0]+u12s*X2su[1])
					+conj(X1su[0])*(u12s*X2s[0]-conj(u11s)*X2s[1])
					+conj(X1su[1])*(-u11s*X2s[0]-conj(u12s)*X2s[1]));

			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=akappa*creal(
					(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					 +conj(X1s[1])*(-u11s*X2su[0]-u12s*X2su[1])
					 +conj(X1su[0])*(-u12s*X2s[0]-conj(u11s)*X2s[1])
					 +conj(X1su[1])*(u11s*X2s[0]-conj(u12s)*X2s[1])));

			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-akappa*cimag(
					conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					+conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					+conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
					+conj(X1su[1])*(-conj(u12s)*X2s[0]+u11s *X2s[1]));

			const unsigned short gindex=mu*ndirac+(idirac>>1);
			const Complex_f gamval_c=gamval[gindex];
			//Rescaling gind by nc
			const unsigned short gind = gamin[gindex]<<1;	
			X2s[0]=X2[i+kvolHalo*(gind)]; X2s[1]=X2[i+kvolHalo*(1+gind)];
			X2su[0]=X2[uid+kvolHalo*(gind)]; X2su[1]=X2[uid+kvolHalo*(1+gind)];

			//If you are asked to rederive the force from Montvay and Munster you'll notice that it should be kappa*gamma
			//but below is only gamma. We rescaled gamma by kappa already when we defined it so that's where it has gone
			dSdpis[0]+=-cimag(gamval_c*
					(conj(X1s[0])* (-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					 +conj(X1s[1])* (u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1su[0])* (-u12s *X2s[0] +conj(u11s)*X2s[1])
					 +conj(X1su[1])*(u11s *X2s[0] +conj(u12s)*X2s[1])));
			dSdpi[i+kvol*mu]=dSdpis[0];

			dSdpis[1]+=creal(gamval_c*
					(conj(X1s[0])* (-conj(u12s)*X2su[0] +conj(u11s)*X2su[1])
					 +conj(X1s[1])*(-u11s *X2su[0]-u12s *X2su[1])
					 +conj(X1su[0])* (u12s *X2s[0]+conj(u11s)*X2s[1])
					 +conj(X1su[1])* (-u11s *X2s[0]+conj(u12s)*X2s[1])));
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-cimag(gamval_c*
					(conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					 +conj(X1su[0])*(conj(u11s)*X2s[0]+u12s *X2s[1])
					 +conj(X1su[1])*(conj(u12s)*X2s[0]-u11s *X2s[1])));
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
	return;
}
void Force_t(double *dSdpi, Complex_f *ut[2],Complex_f *X1, Complex_f *X2, Complex_f gamval[20],\
		float *dk[2], unsigned int *iu, const unsigned short gamin[16],float akappa){

	const unsigned short mu=3;
#pragma omp parallel for simd
	for(unsigned int i=0;i<kvol;i++){
		const unsigned int ind=i+kvolHalo*mu;
		const Complex_f u11s=ut[0][ind];	const Complex_f u12s=ut[1][ind];
		//TODO: The only diffrence with these is that the sign flips for the temporal components
		//			Can we figure out a way of doing this without having to read in a large array. 
		//			Will result in a conditional inside a CUDA loop. If i>kvol3
		const float dks[2] = {dk[0][i],dk[1][i]};
		//Up indices
		const unsigned int uid = iu[i+kvol*mu];
		//Similarly to Hdslash we always see idirac*nc so we do that here too.
		for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
			Complex_f X1s[nc];	 Complex_f X1su[nc];
			Complex_f X2s[nc];	 Complex_f X2su[nc];

			X1s[0]=X1[i+kvolHalo*(idirac)]; X1s[1]=X1[i+kvolHalo*(1+idirac)];
			X1su[0]=X1[uid+kvolHalo*(idirac)]; X1su[1]=X1[uid+kvolHalo*(1+idirac)];
			X2s[0]=X2[i+kvolHalo*(idirac)]; X2s[1]=X2[i+kvolHalo*(1+idirac)];
			X2su[0]=X2[uid+kvolHalo*(idirac)]; X2su[1]=X2[uid+kvolHalo*(1+idirac)];

			float dSdpis[3];
			dSdpis[0]=dSdpi[i+kvol*mu];
			//Multiplying by i and taking the real component is the same as taking the negative imaginary component
			//The positions of u11 and u12 might look a bit funky here. That's just because we've multiplied by the
			//generators by hand
			dSdpis[0]+=-cimag(dks[0]*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(u11s *X2su[0]+u12s *X2su[1]))
					+dks[1]*(conj(X1su[0])*(+u12s*X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(-u11s*X2s[0]-conj(u12s)*X2s[1])));

			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=creal(dks[0]*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(-u11s *X2su[0]-u12s *X2su[1]))
					+dks[1]*(conj(X1su[0])*(-u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*( u11s *X2s[0]-conj(u12s)*X2s[1])));

			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-cimag(dks[0]* (conj(X1s[0])* (u11s *X2su[0]+u12s *X2su[1])
						+conj(X1s[1])* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					+dks[1]*(conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
						+conj(X1su[1])* (-conj(u12s)*X2s[0]+u11s *X2s[1])));

			const unsigned short gindex=mu*ndirac+(idirac>>1);
			//Rescaling gind by nc
			const unsigned short gind = gamin[gindex]<<1;	
			X2s[0]=X2[i+kvolHalo*(gind)]; X2s[1]=X2[i+kvolHalo*(1+gind)];
			X2su[0]=X2[uid+kvolHalo*(gind)]; X2su[1]=X2[uid+kvolHalo*(1+gind)];

			dSdpis[0]+=-cimag(dks[0]*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(u11s *X2su[0]+u12s *X2su[1]))
					-dks[1]*(conj(X1su[0])* (u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(-u11s *X2s[0]-conj(u12s)*X2s[1])));
			dSdpi[i+kvol*mu]=dSdpis[0];

			dSdpis[1]+=creal(dks[0]*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(-u11s*X2su[0]-u12s *X2su[1]))
					-dks[1]*(conj(X1su[0])*(-u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(u11s*X2s[0]-conj(u12s)*X2s[1])));
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-cimag(dks[0]*(conj(X1s[0])*(u11s*X2su[0] +u12s *X2su[1])
						+conj(X1s[1])* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					-dks[1]*(conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
						+conj(X1su[1])*(-conj(u12s)*X2s[0]+u11s *X2s[1])));
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
}
int Force(double *dSdpi, const bool iflag, double res1, Complex *X0, Complex *X1, Complex *Phi,\
		Complex *ut[2], Complex_f *ut_f[2],unsigned int *iu,unsigned int *id,\
		Complex gamval[20],Complex_f gamval_f[20],const unsigned short gamin[16],Complex *sigval,Complex_f *sigval_f, unsigned short *sigin,\
		double *dk[2], float *dk_f[2],const Complex_f jqq, const float akappa,const float beta,const float c_sw,double *ancg){
	const char funcname[] = "Force";
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
	Complex_f *clover[2];
#ifdef __NVCC__
	Complex_f *X1_f, *X2_f;
	cudaMallocAsync((void **)&X1_f,kferm2Halo*sizeof(Complex_f),streams[1]);
	cudaMallocAsync((void **)&X2_f,kferm2Halo*sizeof(Complex_f),streams[0]);
#else
	Complex_f *X1_f= (Complex_f *)aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
	Complex_f *X2_f= (Complex_f *)aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
#endif
	if(c_sw)
		Clover(clover,ut_f,iu,id);

	for(int na = 0; na<nf; na++){
#ifdef __NVCC__
#if(nproc>1) //Strided
		for(unsigned short j=0;j<nc*idirac;j++)
			cudaMemcpyAsync(X1+j*kvolHalo,X0+na*kferm2+j*kvol,kvol*sizeof(Complex),cudaMemcpyDeviceToDevice,streams[j]);
#else
		cudaMemcpyAsync(X1,X0+na*kferm2,kferm2*sizeof(Complex),cudaMemcpyDeviceToDevice,NULL);
#endif
#else
		for(unsigned short j=0;j<nc*ndirac;j++)
			memcpy(X1+j*kvolHalo,X0+na*kferm2+j*kvol,kvol*sizeof(Complex));
#endif
		if(!iflag){
			int itercg=1;
#ifdef __NVCC__
			Complex *smallPhi;
			cudaMallocAsync((void **)&smallPhi,kferm2*sizeof(Complex),streams[0]);
#else
			Complex *smallPhi = (Complex *)aligned_alloc(AVX,kferm2*sizeof(Complex)); 
#endif
			Fill_Small_Phi(na, smallPhi, Phi);
			///@f$(X1=(M\dagger M)^{-1} \Phi@f$
			Congradq(na,res1,X1,smallPhi,ut,ut_f,clover,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,\
					jqq,akappa,c_sw,&itercg);
#ifdef __NVCC__
			cudaFreeAsync(smallPhi,streams[0]);
#else
			free(smallPhi);
#endif
			*ancg+=itercg;
#ifdef __NVCC__
			alignas(16) const Complex blasa=2.0; alignas(16) const double blasb=-1.0;
			cublasZdscal(cublas_handle,kferm2,&blasb,(cuDoubleComplex *)(X0+na*kferm2),1);
#if(nproc>1) //strided
			for(unsigned short j=0;j<nc*ndirac;j++)
				cublasZaxpy(cublas_handle,kvol,(cuDoubleComplex *)&blasa,(cuDoubleComplex *)X1+j*kvolHalo,1,(cuDoubleComplex *)X0+na*kferm2+j*kvol,1);
#else
			cublasZaxpy(cublas_handle,kferm2,(cuDoubleComplex *)&blasa,(cuDoubleComplex *)X1,1,(cuDoubleComplex *)(X0+na*kferm2),1);
#endif
#elifdef __USE_MKL__
			const Complex blasa=2.0; const Complex blasb=-1.0;
			//This is not a general BLAS Routine. BLIS and MKl support it
			//CUDA and GSL does not support it
			cblas_zaxpby(kferm2, &blasa, X1, 1, &blasb, X0+na*kferm2, 1); 
#elifdef USE_BLAS
			const Complex blasa=2.0; const double blasb=-1.0;
			cblas_zdscal(kferm2,blasb,X0+na*kferm2,1);
			for(unsigned short j=0;j<nc*ndirac;j++)
				cblas_zaxpy(kvol,&blasa,X1+j*kvolHalo,1,X0+na*kferm2+j*kvol,1);
#else
#pragma omp parallel for simd collapse(2) aligned(X0,X1:AVX)
			for(int idirac=0;idirac<ndirac;idirac++){
				for(int i=0;i<kvol;i++)
					X0[i+kvol*(0+nc*(idirac+ndirac*na))]=
						2*X1[i+kvolHalo*(0+idirac*c)]-X0[i+kvol*(0+nc*(idirac+ndirac*na))];
				X0[i+kvol*(1+nc*(idirac+ndirac*na))]=
					2*X1[i+kvolHalo*(1+idirac*c)]-X0[i+kvol*(1+nc*(idirac+ndirac*na))];
			}
#endif
		}
		//Convert X1 to single precision
#ifdef __NVCC__
		cudaDeviceSynchronise();
#endif
		//Since it has to be stridded in MPI, we have to pass kvol and nc*ndirac instead of kferm2
		ComplexConvert(X1_f,X1,kvol,true,nc*ndirac);
		Hdslash_f(X2_f,X1_f,ut_f,iu,id,gamval_f,gamin,dk_f,akappa);
		//TODO: Clover product also needed here?
		if(c_sw)
			HbyClover_f(X2_f,X1_f,clover,sigval_f,akappa,sigin,false);
		//TODO: Get a single precision force update on CPU. It'll make things easier I' sure
		alignas(8) const float blasd=2.0;
#ifdef __NVCC__
		cudaDeviceSynchronise();
#if(nproc>1)
		for(unsigned short j=0;j<nc*ndirac;j++)
			cublasCsscal(cublas_handle,kvol, &blasd, (cuComplex *)X2_f+j*kvolHalo, 1);
#else
		cublasCsscal(cublas_handle,kferm2, &blasd, (cuComplex *)X2_f, 1);
#endif
#elif defined USE_BLAS
		for(unsigned short j=0;j<nc*ndirac;j++)
			cblas_csscal(kvol, blasd, X2_f+j*kvolHalo, 1);
#else
#pragma unroll
#pragma omp parallel for simd collapse(2) aligned(X2_f:AVX)
		for(unsigned short j=0;j<nc*ndirac;j++)
			for(unsigned int i=0;i<kvol;i++)
				X2_f[i+j*kvolHalo]*=2;
#endif
#if(npx>1)
		CHalo_swap_dir(X1_f,nc*ndirac,0,DOWN);
		CHalo_swap_dir(X2_f,nc*ndirac,0,DOWN);
#endif
#if(npy>1)
		CHalo_swap_dir(X1_f,nc*ndirac,1,DOWN);
		CHalo_swap_dir(X2_f,nc*ndirac,1,DOWN);
#endif
#if(npz>1)
		CHalo_swap_dir(X1_f,nc*ndirac,2,DOWN);
		CHalo_swap_dir(X2_f,nc*ndirac,2,DOWN);
#endif
#if(npt>1)
		CHalo_swap_dir(X1_f,nc*ndirac,3,DOWN);
		CHalo_swap_dir(X2_f,nc*ndirac,3,DOWN);
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
		//Thankfully the CUDA version is much neater so we're using that style going forwards
		for(unsigned short mu=0;mu<ndim-1;mu++)
			Force_s(dSdpi,ut_f,X1_f,X2_f,gamval_f,iu,gamin,akappa,mu);
		Force_t(dSdpi,ut_f,X1_f,X2_f,gamval_f,dk_f,iu,gamin,akappa);
#endif
		if(c_sw){
			Clover_Force(dSdpi,ut_f,X1_f,X2_f,sigval_f,sigin,iu,id,akappa);
		}
	}
	if(c_sw)
		Clover_free(clover);
#ifdef __NVCC__
	cudaFreeAsync(X1_f,streams[0]); cudaFreeAsync(X2_f,streams[1]);
#else
	free(X1_f); free(X2_f);
#endif
	return 0;
}
