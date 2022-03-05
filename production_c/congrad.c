#include	<matrices.h>
#include	<par_mpi.h>
#include	<su2hmc.h>
int Congradq(int na,double res,Complex *X1,Complex *r,Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
		Complex_f gamval_f[5][4],int gamin[4][4],float *dk4m_f,float *dk4p_f,Complex_f jqq,float akappa,int *itercg){
	/*
	 * Matrix Inversion via Mixed Precision Conjugate Gradient
	 * Solves (M^†)Mx=Phi
	 * Implements up/down partitioning
	 * The matrix multiplication step is done at single precision, while the update is done at double
	 * 
	 * Calls:
	 * =====
	 * Hdslash_f, Hdslashd_f, Par_fsum, Par_dsum
	 *
	 * Parameters:
	 * ==========
	 * int			na:			Flavour index
	 * double		res:			Limit for conjugate gradient
	 * Complex		*X1:			Phi initially, returned as (M†M)^{1} Phi
	 * Complex		*r:			Partition of Phi being used. Gets recycled as the residual vector
	 * Complex		*u11t_f:		First colour's trial field
	 * Complex		*u12t_f:		Second colour's trial field
	 * int			*iu:			Upper halo indices
	 * int			*id:			Lower halo indices
	 * Complex_f	*gamval_f:	Gamma matrices
	 * int			*gamin:		Dirac indices
	 * float			*dk4m_f:
	 * float			*dk4p_f:
	 * Complex_f	jqq:			Diquark source
	 * float			akappa:		Hopping Parameter
	 * int 			*itercg:		Counts the iterations of the conjugate gradient
	 *
	 * Returns:
	 * =======
	 * 0 on success, integer error code otherwise
	 */
	const char *funcname = "Congradq";
	int ret_val=0;
	const double resid = kferm2*res*res;
	//The κ^2 factor is needed to normalise the fields correctly
	//jqq is the diquark codensate and is global scope.
	Complex_f fac_f = conj(jqq)*jqq*akappa*akappa;
	//These were evaluated only in the first loop of niterx so we'll just do it ouside of the loop.
	double alphan;
	//The alpha and beta terms should be double, but that causes issues with BLAS pointers. Instead we declare
	//them complex and work with the real part (especially for α_d)
	//Give initial values Will be overwritten if niterx>0
	double betad = 1.0; Complex_f alphad=0; Complex alpha = 1;
	//Because we're dealing with flattened arrays here we can call cblas safely without the halo
#ifdef __NVCC__
	Complex *p, *x2;
	Complex_f *p_f, *x1_f, *x2_f;
	int device=-1; cudaGetDevice(&device);
	cudaMallocManaged(&p, kferm2*sizeof(Complex),cudaMemAttachGlobal);
	cudaMemAdvise(p,kferm2*sizeof(Complex),cudaMemAdviseSetPreferredLocation,device);

	cudaMallocManaged(&p_f, kferm2Halo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMemAdvise(p_f,kferm2Halo*sizeof(Complex_f),cudaMemAdviseSetPreferredLocation,device);

	cudaMalloc(&x1_f, kferm2Halo*sizeof(Complex_f));
	cudaMalloc(&x2_f, kferm2Halo*sizeof(Complex_f));

	cudaMallocManaged(&x2, kferm2*sizeof(Complex),cudaMemAttachGlobal);
	cudaMemAdvise(x2,kferm2*sizeof(Complex),cudaMemAdviseSetPreferredLocation,device);
	cudaMemPrefetchAsync(x2,kferm2*sizeof(Complex),device,NULL);
#elif defined __INTEL_MKL__
	Complex *p  = mkl_calloc(kferm2,sizeof(Complex),AVX);
	Complex *x2=mkl_calloc(kferm2, sizeof(Complex), AVX);

	Complex_f *p_f  = mkl_calloc(kferm2Halo,sizeof(Complex_f),AVX);
	Complex_f *x2_f=mkl_calloc(kferm2Halo, sizeof(Complex_f), AVX);
	Complex_f *x1_f=mkl_calloc(kferm2Halo, sizeof(Complex_f), AVX);
#else
	Complex *p  = aligned_alloc(AVX,kferm2*sizeof(Complex));
	Complex *x2=aligned_alloc(AVX,kferm2*sizeof(Complex));

	Complex_f *p_f=aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
	Complex_f *x1_f=aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
	Complex_f *x2_f=aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
#endif
	//Instead of copying elementwise in a loop, use memcpy.
	memcpy(p, X1, kferm2*sizeof(Complex));
#ifdef __NVCC__
	cudaMemPrefetchAsync(p,kferm2*sizeof(Complex),device,NULL);
#endif

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	double betan;
	for(*itercg=0; *itercg<niterc; (*itercg)++){
#pragma omp parallel for simd aligned(p_f,p:AVX)
		for(int i=0;i<kferm2;i++)
			p_f[i]=(Complex_f)p[i];
#ifdef	__NVCC__
		cudaMemPrefetchAsync(p_f,kferm2Halo*sizeof(Complex_f),device,NULL);
#endif
		//x2 =  (M^†M)p 
		Hdslash_f(x1_f,p_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
		Hdslashd_f(x2_f,x1_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
		//x2 =  (M^†M+J^2)p 
#ifdef	__NVCC__
		cublasCaxpy(cublas_handle,kferm2,(cuComplex *)&fac_f,(cuComplex *)p_f,1,(cuComplex *)x2_f,1);
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
		cblas_caxpy(kferm2, &fac_f, p_f, 1, x2_f, 1);
#else
#pragma omp parallel for simd aligned(p_f,x2_f:AVX)
		for(int i=0; i<kferm2; i++)
			x2_f[i]+=fac_f*p_f[i];
#endif
#pragma omp parallel for simd aligned(x2,x2_f:AVX)
		for(int i=0;i<kferm2;i++)
			x2[i]=(Complex)x2_f[i];
		//We can't evaluate α on the first *itercg because we need to get β_n.
		if(*itercg){
			//α_d= p* (M^†M+J^2)p
#ifdef __NVCC__
			cublasCdotc(cublas_handle,kferm2,(cuComplex *)p_f,1,(cuComplex *)x2_f,1,(cuComplex *)&alphad);
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
			cblas_cdotc_sub(kferm2, p_f, 1, x2_f, 1, &alphad);
#else
			alphad=0;
#pragma omp parallel for simd aligned(p_f,x2_f:AVX)
			for(int i=0; i<kferm2; i++)
				alphad+=conj(p_f[i])*x2_f[i];
#endif
			//For now I'll cast it into a double for the reduction. Each rank only sends and writes
			//to the real part so this is fine
			Par_fsum((float *)&alphad);
			//α=α_n/α_d = (r.r)/p(M^†M)p 
			alpha=alphan/creal(alphad);
			//x-αp, 
#ifdef __NVCC__
			cublasZaxpy(cublas_handle,kferm2,(cuDoubleComplex *)&alpha,(cuDoubleComplex *)p,1,(cuDoubleComplex *)X1,1);
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
			cblas_zaxpy(kferm2, &alpha, p, 1, X1, 1);
#else
			for(int i=0; i<kferm2; i++)
				X1[i]+=alpha*p[i];
#endif
		}			
		// r_n+1 = r_n-α(M^† M)p_n and β_n=r*.r
#ifdef	__NVCC__
		alpha*=-1;
		cublasZaxpy(cublas_handle, kferm2,(cuDoubleComplex *)&alpha,(cuDoubleComplex *)x2,1,(cuDoubleComplex *)r,1);
		alpha*=-1;
		cublasDznrm2(cublas_handle,kferm2,(cuDoubleComplex *)r,1,&betan);
		betan *= betan;
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
		alpha *= -1;
		cblas_zaxpy(kferm2, &alpha, x2, 1, r, 1);
		//Undo the negation for the BLAS routine
		alpha*=-1;
		betan = cblas_dznrm2(kferm2, r,1);
		//Gotta square it to "undo" the norm
		betan *= betan;
#else
		betan=0;
		for(int i=0; i<kferm2; i++){
			r[i]-=alpha*x2[i];
			betan += conj(r[i])*r[i];
		}
#endif
		//And... reduce.
		Par_dsum(&betan);
		if(betan<resid){ 
			(*itercg)++;
#ifdef _DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %e toler = %e\n", *itercg, betan, resid);
#endif
			ret_val=0;	break;
		}
		else if(*itercg==niterc-1){
			if(!rank) fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i β_n=%e\n", ITERLIM, funcname, *itercg, betan);
			ret_val=ITERLIM;	break;
		}
		//Here we evaluate β=(r_{k+1}.r_{k+1})/(r_k.r_k) and then shuffle our indices down the line.
		//On the first iteration we define beta to be zero.
		//Note that beta below is not the global beta and scoping is used to avoid conflict between them
		Complex beta = (*itercg) ?  betan/betad : 0;
		betad=betan; alphan=betan;
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multipyling y by
		//β instead of x.
#if (defined __INTEL_MKL__||defined USE_BLAS)
		Complex a = 1.0;
		//There is cblas_zaxpby in the MKL and AMD though, set a = 1 and b = β.
		//If we get a small enough β_n before hitting the iteration cap we break
		cblas_zaxpby(kferm2, &a, r, 1, &beta,  p, 1);
#else 
		for(int i=0; i<kferm2; i++)
			p[i]=r[i]+beta*p[i];
#endif
	}
#ifdef __NVCC__
	cudaFree(x2); cudaFree(p);
	cudaFree(x1_f);cudaFree(x2_f); cudaFree(p_f);
#elif defined __INTEL_MKL__
	mkl_free(x1_f); mkl_free(x2); mkl_free(p);
	mkl_free(p_f); mkl_free(x2_f);
#else
	free(x1_f); free(x2); free(p);
	free(p_f); free(x2_f);
#endif
	return ret_val;
}
int Congradp(int na,double res,Complex *Phi,Complex_f *xi_f,Complex_f *u11t_f,Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
		Complex_f gamval_f[5][4],int gamin[4][4],float *dk4m_f,float *dk4p_f,Complex_f jqq,float akappa,int *itercg){
	/*
	 * Matrix Inversion via Conjugate Gradient
	 * Solves (M^†)Mx=Phi
	 * No even/odd partitioning
	 *
	 * Calls:
	 * =====
	 * Dslash_f, Dslashd_f, Par_fsum, Par_dsum
	 *
	 * Parameters:
	 * ==========
	 * int			na:			Flavour index
	 * double		res:			Limit for conjugate gradient
	 * Complex		*Phi:			Phi initially, 
	 * Complex_f	*r:			Returned as (M†M)^{1} Phi
	 * Complex		*u11t_f:		First colour's trial field
	 * Complex		*u12t_f:		Second colour's trial field
	 * int			*iu:			Upper halo indices
	 * int			*id:			Lower halo indices
	 * Complex_f	*gamval_f:	Gamma matrices
	 * int			*gamin:		Dirac indices
	 * float			*dk4m_f:
	 * float			*dk4p_f:
	 * Complex_f	jqq:			Diquark source
	 * float			akappa:		Hopping Parameter
	 * int 			*itercg:		Counts the iterations of the conjugate gradient
	 *
	 * Returns:
	 * =======
	 * 0 on success, integer error code otherwise
	 */
	const char *funcname = "Congradp";
	//Return value
	int ret_val=0;
	const double resid = kferm*res*res;
	//These were evaluated only in the first loop of niterx so we'll just do it ouside of the loop.
	//These alpha and beta terms should be double, but that causes issues with BLAS. Instead we declare
	//them Complex and work with the real part (especially for α_d)
	//Give initial values Will be overwritten if niterx>0
#ifdef __NVCC__
	Complex *p, *r, *x2;
	Complex_f *p_f, *r_f;
	int device; cudaGetDevice(&device);
	cudaMallocManaged(&p, kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMemAdvise(p,kfermHalo*sizeof(Complex),cudaMemAdviseSetPreferredLocation,device);

	cudaMallocManaged(&r, kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMemAdvise(r,kfermHalo*sizeof(Complex),cudaMemAdviseSetPreferredLocation,device);

	cudaMallocManaged(&p_f, kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMemAdvise(p_f,kfermHalo*sizeof(Complex_f),cudaMemAdviseSetPreferredLocation,device);

	cudaMallocManaged(&x2, kferm*sizeof(Complex),cudaMemAttachGlobal);
	cudaMemAdvise(x2,kferm*sizeof(Complex),cudaMemAdviseSetPreferredLocation,device);
#elif defined __INTEL_MKL__
	Complex *p  = mkl_malloc(kfermHalo*sizeof(Complex),AVX);
	Complex *r  = mkl_malloc(kferm*sizeof(Complex),AVX);
	Complex *x2	= mkl_malloc(kferm*sizeof(Complex), AVX);

	Complex_f *p_f	= mkl_malloc(kfermHalo*sizeof(Complex_f),AVX);
	Complex_f *x1	=mkl_malloc(kfermHalo*sizeof(Complex_f), AVX);
	Complex_f *x2_f=mkl_malloc(kfermHalo*sizeof(Complex_f), AVX);
#else
	Complex *p  = aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex *r  = aligned_alloc(AVX,kferm*sizeof(Complex));
	Complex *x2  = aligned_alloc(AVX,kferm*sizeof(Complex));

	Complex_f *p_f  = aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *x1=aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *x2_f=aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
#endif
	double betad = 1.0; float alphad_f=0; Complex alpha = 1;
	float alphan_f=0.0;
	//Instead of copying elementwise in a loop, use memcpy.
	memcpy(p_f, xi_f, kferm*sizeof(Complex_f));
	memcpy(r, Phi+na*kferm, kferm*sizeof(Complex));

	// Declaring placeholder arrays 
	// This x1 is NOT related to the /common/vectorp/X1 in the FORTRAN code and should not
	// be confused with X1 the global variable
#ifdef __NVCC__
	Complex_f *x1, *x2_f;
	cudaMemPrefetchAsync(p,kfermHalo*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(r_f,kfermHalo*sizeof(Complex_f),device,NULL);
	cudaMalloc(&x1, kferm2Halo*sizeof(Complex_f));

	cudaMallocManaged(&x2_f, kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMemAdvise(x2_f,kfermHalo*sizeof(Complex_f),cudaMemAdviseSetPreferredLocation,device);
#endif

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	double betan;
	for((*itercg)=0; (*itercg)<=niterc; (*itercg)++){
		//Don't overwrite on first run. 
#ifdef	__NVCC__
		cudaMemPrefetchAsync(p_f,kfermHalo*sizeof(Complex_f),device,NULL);
		cudaMemPrefetchAsync(r_f,kfermHalo*sizeof(Complex_f),device,NULL);
#endif
		//x2=(M^†)x1=(M^†)Mp
		Dslash_f(x1,p_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
		Dslashd_f(x2_f,x1,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(*itercg){
			//x*.x
#ifdef __NVCC__
			cublasScnrm2(cublas_handle,kferm,(cuComplex*) x1, 1,&alphad_f);
			alphad_f *= alphad_f;
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
			alphad_f = cblas_scnrm2(kferm, x1, 1);
			alphad_f *= alphad_f;
#else
			alphad_f=0;
			for(int i = 0; i<kferm; i++)
				alphad_f+=conj(x1[i])*x1[i];
#endif
			Par_fsum((float *)&alphad_f);
			//α=(r.r)/p(M^†)Mp
			alpha=alphan_f/alphad_f;
			Complex_f alpha_f = (Complex_f)alpha;
			//x+αp
#ifdef __NVCC__
			cublasCaxpy(cublas_handle,kferm,(cuComplex*) &alpha_f,(cuComplex*) p_f,1,(cuComplex*) xi_f,1);
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
			cblas_caxpy(kferm, (Complex_f*)&alpha_f,(Complex_f*)p_f, 1, (Complex_f*)xi_f, 1);
#else
			for(int i = 0; i<kferm; i++)
				xi_f[i]+=alpha*p_f[i];
#endif
		}
#pragma omp parallel for simd aligned(p,p_f,x2,x2_f:AVX)
		for(int i=0;i<kferm;i++){
			p[i]=(Complex)p_f[i];
			x2[i]=(Complex)x2_f[i];
		}
		//r=α(M^†)Mp and β_n=r*.r
#ifdef __NVCC__
		alpha*=-1;
		cublasZaxpy(cublas_handle,kferm, (cuDoubleComplex *)&alpha,(cuDoubleComplex *) x2, 1,(cuDoubleComplex *) r, 1);
		alpha*=-1;
		//r*.r
		cublasDznrm2(cublas_handle,kferm,(cuDoubleComplex *) r,1,&betan);
		//Gotta square it to "undo" the norm
		betan*=betan;
#elif (defined __INTEL_MKL__ || defined USE_BLAS)
		alpha*=-1;
		cblas_zaxpy(kferm,(Complex*) &alpha,(Complex*) x2, 1,(Complex*) r, 1);
		alpha*=-1;
		//r*.r
		betan = cblas_dznrm2(kferm, (Complex*)r,1);
		//Gotta square it to "undo" the norm
		betan*=betan;
#else
		//Just like Congradq, this loop could be unrolled but will need a reduction to deal with the betan 
		//addition.
		betan = 0;
		//If we get a small enough β_n before hitting the iteration cap we break
#pragma omp parallel for simd aligned(x2_f,r_f:AVX)
		for(int i = 0; i<kferm;i++){
			r[i]-=alpha*x2[i];
			betan+=conj(r[i])*r[i];
		}
#endif
		//This is basically just congradq at the end. Check there for comments
		Par_dsum(&betan);
		if(betan<resid){
			//Started counting from zero so add one to make it accurate
			(*itercg)++;
#ifdef _DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %e toler = %e\n", *itercg, betan, resid);
#endif
			ret_val=0;	break;
		}
		else if(*itercg==niterc-1){
			if(!rank) fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i β_n=%e\n",
					ITERLIM, funcname, niterc, betan);
			ret_val=ITERLIM;	break;
		}
		//Note that beta below is not the global beta and scoping is used to avoid conflict between them
		Complex beta = (*itercg) ? betan/betad : 0;
		betad=betan; alphan_f=(float)betan;
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multipyling y by 
		//β instead of x.
		//There is cblas_zaxpby in the MKL though, set a = 1 and b = β.
#if (defined __INTEL_MKL__||defined USE_BLAS)
		Complex a = 1;
		cblas_zaxpby(kferm, &a, r, 1, &beta,  p, 1);
#else
		for(int i=0; i<kferm; i++)
			p[i]=r[i]+beta*p[i];
#endif
#pragma omp parallel for simd aligned(p_f,p:AVX)
		for(int i=0;i<kferm;i++)
			p_f[i]=(Complex_f)p[i];
	}
#ifdef	__NVCC__
	cudaFree(p); cudaFree(r);cudaFree(x1); 
	cudaFree(x2_f); cudaFree(p_f); cudaFree(x2);
#elif defined __INTEL_MKL__
	mkl_free(p); mkl_free(r); mkl_free(x1);
	mkl_free(p_f); mkl_free(x2); mkl_free(x2_f);
#else
	free(p); free(r); free(x1);
	free(p_f); free(x2); free(x2_f);
#endif
	return ret_val;
}
