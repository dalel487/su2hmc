/**
 *	@file fermionic.c
 *	@brief Code for fermionic observables
 */
#include	<matrices.h>
#include <clover.h>
int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg,\
		Complex *ut[2], Complex_f *ut_f[2], unsigned int *iu, unsigned int *id,\
		Complex gamval[20], Complex_f gamval_f[20],	const unsigned short gamin[16],\
		Complex *sigval,Complex_f *sigval_f, unsigned short *sigin, double *dk[2],float *dk_f[2],\
		Complex_f jqq, float akappa,	float c_sw,Complex *Phi){
	/*
	 * @brief	Calculate fermion expectation values via a noisy estimator
	 * 
	 * Matrix inversion via conjugate gradient algorithm
	 * Solves @f(Mx=x_1@f)
	 * (Numerical Recipes section 2.10 pp.70-73)   
	 * uses NEW lookup tables **
	 * Implemented in Congradq
	 *
	 * @param	pbp:				@f(\langle\bar{\Psi}\Psi\rangle@f)
	 *	@param	endenf:			Energy density
	 *	@param	denf:				Number Density
	 *	@param	qq:				Diquark condensate
	 *	@param	qbqb:				Antidiquark condensate
	 *	@param	res:				Conjugate Gradient Residue
	 *	@param	itercg:			Iterations of Conjugate Gradient
	 * @param	u11t,u12t		Double precisiongauge field
	 * @param	u11t_f,u12t_f:	Single precision gauge fields
	 *	@param	iu,id				Lattice indices
	 *	@param	gamval_f:		Gamma matrices
	 *	@param	gamin:			Indices for Dirac terms
	 * @param	dk4m_f:			$exp(-\mu)$ float
	 * @param	dk4p_f:			$exp(\mu)$ float
	 *	@param	jqq:				Diquark source
	 *	@param	akappa:			Hopping parameter
	 *	@param	Phi:				Pseudofermion field	
	 *	@param	R1:				A useful array for holding things that was already assigned in main.
	 *									In particular, we'll be using it to catch the output of
	 *									@f$ M^\dagger\Xi@f$ before the inversion, then used to store the
	 *									output of the inversion
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Measure";
	//This x is just a storage container

#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	Complex	*x, *xi; Complex_f *xi_f, *R1_f, *R1, *clover[nc];
#ifdef _DEBUG
	cudaMallocManaged((void **)&R1,kfermHalo*sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged((void **)&R1_f,kferm*sizeof(Complex_f), cudaMemAttachGlobal);
	if(c_sw){
		cudaMallocManaged((void **)&clover[0], 6*kvol*sizeof(Complex),cudaMemAttachGlobal);
		cudaMallocManaged((void **)&clover[1], 6*kvol*sizeof(Complex),cudaMemAttachGlobal);
	}
#else
	cudaMallocAsync((void **)&R1_f,kferm*sizeof(Complex_f),streams[0]);
	if(c_sw){
		cudaMallocAsync((void **)&clover[0], 6*kvol*sizeof(Complex),streams[1]);
		cudaMallocAsync((void **)&clover[1], 6*kvol*sizeof(Complex),streams[2]);
	}
#endif
	cudaMallocManaged((void **)&x,kfermHalo*sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged((void **)&xi,kferm*sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged((void **)&xi_f,kfermHalo*sizeof(Complex_f), cudaMemAttachGlobal);
#else
	Complex_f *clover[nc];
	if(c_sw){
		clover[0]=(Complex_f *)aligned_alloc(AVX,6*kvol*sizeof(Complex_f));
		clover[1]=(Complex_f *)aligned_alloc(AVX,6*kvol*sizeof(Complex_f));
	}
	Complex *x =(Complex *)aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex *xi =(Complex *)aligned_alloc(AVX,kferm*sizeof(Complex));
	Complex_f *xi_f =(Complex_f *)aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *R1_f = (Complex_f *)aligned_alloc(AVX,kferm*sizeof(Complex_f));
#endif
	//Setting up noise. Again need that annoying stride 
	for(unsigned short j=0;j<nc*ngorkov;j++)
		Gauss_c(xi_f+j*kvolHalo, kvol, 0, (float)(1/sqrt(2)));
#ifdef __NVCC__
	//cudaMemPrefetchAsync(xi_f,kferm*sizeof(Complex_f),device,streams[0]);
	for(unsigned short j=0;j<nc*ngorkov;j++){
		cuComplex_convert(xi_f+j*kvol,xi+j*kvolHalo,kvol,false,dimBlock,dimGrid);
		//Flip all the gauge fields around so memory is coalesced
		cudaMemcpyAsync(x+j*kvolHalo, xi+j*kvol, kvol*sizeof(Complex),cudaMemcpyDefault,0);
	}
#else
#pragma omp parallel for simd collapse(2) aligned(xi,xi_f:AVX)
	for(unsigned short j=0;j<nc*ngorkov;j++){
		for(unsigned int i=0;i<kvol;i++)
			xi[i+j*kvol]=(Complex)xi_f[i+j*kvolHalo];
		memcpy(x+j*kvolHalo, xi+j*kvol, kvol*sizeof(Complex));
	}
#endif
	//R_1= @f$M^\dagger\Xi@f$
	//global
	Dslashd_f(R1_f,xi_f,ut_f,iu,id,gamval_f,gamin,dk_f,jqq,akappa);
	if(c_sw){
		Clover(clover,ut_f,iu,id);
		ByClover_f(R1_f,xi_f,clover,sigval_f,akappa,sigin);
	}
#ifdef __NVCC__
	cudaDeviceSynchronise();
	cudaFree(xi_f);	
	for(unsigned short j=0;j<ngorkov*nc;j++){
		cuComplex_convert(R1_f+j*kvol,R1+j*kvolHalo,kvol,false,dimBlock,dimGrid);
		//Phi has no halo
		cudaMemcpy(Phi+j*kvol, R1+j*kvolHalo, kvol*sizeof(Complex),cudaMemcpyDefault);
	}
#ifdef _DEBUG
	cudaFree(R1_f);
#else
	cudaFreeAsync(R1_f,streams[0]);
	#endif
#else
#pragma omp parallel for simd aligned(R1,R1_f:AVX)
	for(unsigned short j=0;j<ngorkov*nc;j++){
		for(int i=0;i<kvol;i++)
			R1[i+j*kvolHalo]=(Complex)R1_f[i+j*kvol];
		//Copying R1 to the first (zeroth) flavour index of Phi
		//This should be safe with memcpy since the pointer name
		//references the first block of memory for that pointer
		memcpy(Phi+j*kvol, R1+j*kvolHalo, kvol*sizeof(Complex));
	}
	free(R1_f);
#endif
	///Evaluate xi = (M^† M)^-1 R_1 
	if(Congradp(0, res, Phi,R1,ut,ut_f,clover,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,c_sw,itercg)==ITERLIM)
		return ITERLIM;
#ifdef __NVCC__
	for(unsigned short j=0;j<ngorkov*nc;j++)
		cudaMemcpyAsync(xi+j*kvol,R1+j*kvolHalo,kvol*sizeof(Complex),cudaMemcpyDefault,streams[j]);
#ifdef _DEBUG
	if(c_sw){
		cudaFree(clover[0]); cudaFree(clover[1]);
	}
	cudaFree(R1);
#else
	if(c_sw){
		cudaFreeAsync(clover[0],streams[1]); cudaFreeAsync(clover[1],streams[2]);
	}
	cudaFreeAsync(R1,streams[0]);
#endif
#else
	for(unsigned short j=0;j<ngorkov*nc;j++)
		memcpy(xi+j*kvol,R1+j*kvolHalo,kvol*sizeof(Complex));
	free(xi_f);
	if(c_sw){
		free(clover[0]); free(clover[1]);
	}
	free(R1);
#endif
#ifdef USE_BLAS
	Complex buff;
#ifdef __NVCC__
	cublasZdotc(cublas_handle,kferm,(cuDoubleComplex *)x,1,(cuDoubleComplex *)xi,1,(cuDoubleComplex *)&buff);
	cudaDeviceSynchronise();
#elif defined USE_BLAS
	cblas_zdotc_sub(kferm, x, 1, xi,  1, &buff);
#endif
	*pbp=creal(buff);
#else
	*pbp = 0;
#pragma unroll
	for(int i=0;i<kferm;i++)
		*pbp+=creal(conj(x[i])*xi[i]);
#endif
#if(nproc>1)
	Par_dsum(pbp);
#endif
	*pbp/=4*gvol;

	*qbqb=*qq=0;
#if defined USE_BLAS
	for(int idirac = 0; idirac<ndirac; idirac++){
		int igork=idirac+4;
		//Unrolling the colour indices, Then its just (γ_5*x)*Ξ or (γ_5*Ξ)*x 
#pragma unroll
		for(int ic = 0; ic<nc; ic++){
			Complex dot;
			//Because we have kvol on the outer index and are summing over it, we set the
			//step for BLAS to be ngorkov*nc=16. 
			//Does this make sense to do on the GPU?
#ifdef __NVCC__
			cublasZdotc(cublas_handle,kvol,(cuDoubleComplex *)(x+idirac*nc+ic),ngorkov*nc,(cuDoubleComplex *)(xi+igork*nc+ic), ngorkov*nc,(cuDoubleComplex *)&dot);
#else
			cblas_zdotc_sub(kvol, &x[idirac*nc+ic], ngorkov*nc, &xi[igork*nc+ic], ngorkov*nc, &dot);
#endif
			*qbqb+=gamval[4*ndirac+idirac]*dot;
#ifdef __NVCC__
			cublasZdotc(cublas_handle,kvol,(cuDoubleComplex *)(x+igork*nc+ic),ngorkov*nc,(cuDoubleComplex *)(xi+idirac*nc+ic), ngorkov*nc,(cuDoubleComplex *)&dot);
#else
			cblas_zdotc_sub(kvol, &x[igork*nc+ic], ngorkov*nc, &xi[idirac*nc+ic], ngorkov*nc, &dot);
#endif
			*qq-=gamval[4*ndirac+idirac]*dot;
		}
	}
#else
#pragma unroll(2)
	for(int i=0; i<kvol; i++)
		//What is the optimal order to evaluate these in?
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork=idirac+4;
			*qbqb+=gamval[4*ndirac+idirac]*conj(x[(i*ngorkov+idirac)*nc])*xi[(i*ngorkov+igork)*nc];
			*qq-=gamval[4*ndirac+idirac]*conj(x[(i*ngorkov+igork)*nc])*xi[(i*ngorkov+idirac)*nc];
			*qbqb+=gamval[4*ndirac+idirac]*conj(x[(i*ngorkov+idirac)*nc+1])*xi[(i*ngorkov+igork)*nc+1];
			*qq-=gamval[4*ndirac+idirac]*conj(x[(i*ngorkov+igork)*nc+1])*xi[(i*ngorkov+idirac)*nc+1];
		}
#endif
	//In the FORTRAN Code dsum was used instead despite qq and qbqb being complex
	//Since we only care about the real part this shouldn't cause (m)any serious issues
#if(nproc>1)
	Par_dsum((double *)qq); Par_dsum((double *)qbqb);
#endif
	*qq=(*qq+*qbqb)/(2*gvol);
	Complex xu, xd, xuu, xdd;
	xu=xd=xuu=xdd=0;

	//Halos
#if(npt>1)
	ZHalo_swap_dir(x,16,3,DOWN);		ZHalo_swap_dir(x,16,3,UP);
#endif
	//Pesky halo exchange indices again
	//The halo exchange for the trial fields was done already at the end of the trajectory
	//No point doing it again

	//Instead of typing id[i+kvol*3] a lot, we'll just assign them to variables.
	//Idea. One loop instead of two loops but for xuu and xdd just use ngorkov-(igorkov+1) instead
	//Dirty CUDA work around since it won't convert thrust<complex> to double
	//TODO: get a reduction routine ready for CUDA
#ifdef __NVCC__
	//Swapping back the gauge fields to SoA since the rest of the code is running on CPU and hasn't been ported
	//	Transpose_z(ut[0],kvol,ndim);
	//	Transpose_z(ut[1],kvol,ndim);
	//Set up  index arrays for CPU
	//Transpose_U(iu,kvol,ndim);
	//Transpose_U(id,kvol,ndim);
	//	cudaDeviceSynchronise();
#else
#pragma omp parallel for reduction(+:xd,xu,xdd,xuu) 
#endif
	for(int i = 0; i<kvol; i++){
		int did=id[3*kvol+i];
		int uid=iu[3*kvol+i];
		for(int igorkov=0; igorkov<4; igorkov++){
			int igork1=gamin[3*ndirac+igorkov];
			//For the C Version I'll try and factorise where possible
			xu+=dk[1][did]*(conj(x[(did*ngorkov+igorkov)*nc])*(\
						ut[0][did+kvol*3]*(xi[(i*ngorkov+igork1)*nc]-xi[(i*ngorkov+igorkov)*nc])+\
						ut[1][did+kvol*3]*(xi[(i*ngorkov+igork1)*nc+1]-xi[(i*ngorkov+igorkov)*nc+1]) )+\
					conj(x[(did*ngorkov+igorkov)*nc+1])*(\
						conj(ut[0][did+kvol*3])*(xi[(i*ngorkov+igork1)*nc+1]-xi[(i*ngorkov+igorkov)*nc+1])+\
						conj(ut[1][did+kvol*3])*(xi[(i*ngorkov+igorkov)*nc]-xi[(i*ngorkov+igork1)*nc])));
		}
		for(int igorkov=0; igorkov<4; igorkov++){
			int igork1=gamin[3*ndirac+igorkov];
			xd+=dk[0][i]*(conj(x[(uid*ngorkov+igorkov)*nc])*(\
						conj(ut[0][i+kvol*3])*(xi[(i*ngorkov+igork1)*nc]+xi[(i*ngorkov+igorkov)*nc])-\
						ut[1][i+kvol*3]*(xi[(i*ngorkov+igork1)*nc+1]+xi[(i*ngorkov+igorkov)*nc+1]) )+\
					conj(x[(uid*ngorkov+igorkov)*nc+1])*(\
						ut[0][i+kvol*3]*(xi[(i*ngorkov+igork1)*nc+1]+xi[(i*ngorkov+igorkov)*nc+1])+\
						conj(ut[1][i+kvol*3])*(xi[(i*ngorkov+igorkov)*nc]+xi[(i*ngorkov+igork1)*nc]) ) );
		}
		for(int igorkovPP=4; igorkovPP<8; igorkovPP++){
			int igork1PP=4+gamin[3*ndirac+igorkovPP-4];
			xuu-=dk[0][did]*(conj(x[(did*ngorkov+igorkovPP)*nc])*(\
						ut[0][did+kvol*3]*(xi[(i*ngorkov+igork1PP)*nc]-xi[(i*ngorkov+igorkovPP)*nc])+\
						ut[1][did+kvol*3]*(xi[(i*ngorkov+igork1PP)*nc+1]-xi[(i*ngorkov+igorkovPP)*nc+1]) )+\
					conj(x[(did*ngorkov+igorkovPP)*nc+1])*(\
						conj(ut[0][did+kvol*3])*(xi[(i*ngorkov+igork1PP)*nc+1]-xi[(i*ngorkov+igorkovPP)*nc+1])+\
						conj(ut[1][did+kvol*3])*(xi[(i*ngorkov+igorkovPP)*nc]-xi[(i*ngorkov+igork1PP)*nc]) ) );
		}
		for(int igorkovPP=4; igorkovPP<8; igorkovPP++){
			int igork1PP=4+gamin[3*ndirac+igorkovPP-4];
			xdd-=dk[1][i]*(conj(x[(uid*ngorkov+igorkovPP)*nc])*(\
						conj(ut[0][i+kvol*3])*(xi[(i*ngorkov+igork1PP)*nc]+xi[(i*ngorkov+igorkovPP)*nc])-\
						ut[1][i+kvol*3]*(xi[(i*ngorkov+igork1PP)*nc+1]+xi[(i*ngorkov+igorkovPP)*nc+1]) )+\
					conj(x[(uid*ngorkov+igorkovPP)*nc+1])*(\
						ut[0][i+kvol*3]*(xi[(i*ngorkov+igork1PP)*nc+1]+xi[(i*ngorkov+igorkovPP)*nc+1])+\
						conj(ut[1][i+kvol*3])*(xi[(i*ngorkov+igorkovPP)*nc]+xi[(i*ngorkov+igork1PP)*nc]) ) );
		}
	}
	*endenf=creal(xu-xd-xuu+xdd);
	*denf=creal(xu+xd+xuu+xdd);

#if(nproc>1)
	Par_dsum(endenf); Par_dsum(denf);
#endif
	*endenf/=2*gvol; *denf/=2*gvol;
	//Future task. Chiral susceptibility measurements
#ifdef __NVCC__
	cudaFree(x); cudaFree(xi);
	//Revert index and gauge arrays
	//	Transpose_z(ut[0],ndim,kvol);
	//	Transpose_z(ut[1],ndim,kvol);
	//Transpose_U(iu,ndim,kvol);
	//Transpose_U(id,ndim,kvol);
#else
	free(x); free(xi);
#endif
	return 0;
}
