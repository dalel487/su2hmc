/**
 * @file congrad.c
 *
 * @brief Conjugate gradients. Congradq for the solver and Congradp for the inversion
 */
#include	<matrices.h>
int Congradq(int na,double res,Complex *X1,Complex *r,Complex_f *u11t,Complex_f *u12t,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval_f,int *gamin,float *dk4m,float *dk4p,Complex_f jqq,float akappa,int *itercg){
	/*
	 * @brief Matrix Inversion via Mixed Precision Conjugate Gradient
	 * Solves @f$(M^\dagger)Mx=\Phi@f$
	 * Implements up/down partitioning
	 * The matrix multiplication step is done at single precision, while the update is done at double
	 *
	 * @param  na:				Flavour index
	 * @param  res:			Limit for conjugate gradient
	 * @param  X1:				@f(\Phi@f) initially, returned as @f((M^\dagger M)^{-1} \Phi@f)
	 * @param  r:				Partition of @f(\Phi@f) being used. Gets recycled as the residual vector
	 * @param  u11t:			First colour's trial field
	 * @param  u12t:			Second colour's trial field
	 * @param  iu:				Upper halo indices
	 * @param  id:				Lower halo indices
	 * @param  gamval_f:		Gamma matrices
	 * @param  gamin:			Dirac indices
	 * @param  dk4m:
	 * @param  dk4p:
	 * @param  jqq:			Diquark source
	 * @param  akappa:		Hopping Parameter
	 * @param  itercg:		Counts the iterations of the conjugate gradient
	 * 
	 * @see Hdslash_f(), Hdslashd_f(), Par_fsum(), Par_dsum()
	 *
	 * @return 0 on success, integer error code otherwise
	 */
	const char *funcname = "Congradq";
	int ret_val=0;
	const double resid = res*res;
	//The κ^2 factor is needed to normalise the fields correctly
	//jqq is the diquark condensate and is global scope.
	const Complex_f fac_f = conj(jqq)*jqq*akappa*akappa;
	//These were evaluated only in the first loop of niterx so we'll just do it outside of the loop.
	//n suffix is numerator, d is denominator
	double alphan=1;
	//The alpha and beta terms should be double, but that causes issues with BLAS pointers. Instead we declare
	//them complex and work with the real part (especially for α_d)
	//Give initial values Will be overwritten if niterx>0
	double betad = 1.0; Complex_f alphad=0; Complex alpha = 1;
	//Because we're dealing with flattened arrays here we can call cblas safely without the halo
#ifdef __NVCC__
	Complex_f *p_f, *x1_f, *x2_f, *r_f, *X1_f;
	int device=-1; cudaGetDevice(&device);

#ifdef _DEBUG
	cudaMallocManaged((void **)&p_f, kferm2Halo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&x1_f, kferm2Halo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&x2_f, kferm2*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&r_f, kferm2*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&X1_f, kferm2*sizeof(Complex_f),cudaMemAttachGlobal);
#else
	//First two have halo exchanges, so getting NCCL working is important
	cudaMallocAsync((void **)&p_f, kferm2Halo*sizeof(Complex_f),streams[0]);
	cudaMallocAsync((void **)&x1_f, kferm2Halo*sizeof(Complex_f),streams[1]);
	cudaMallocAsync((void **)&x2_f, kferm2*sizeof(Complex_f),streams[2]);
	cudaMallocAsync((void **)&r_f, kferm2*sizeof(Complex_f),streams[3]);
	cudaMallocAsync((void **)&X1_f, kferm2*sizeof(Complex_f),streams[4]);
#endif
#else
	Complex_f *p_f=aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
	Complex_f *x1_f=aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
	Complex_f *x2_f=aligned_alloc(AVX,kferm2*sizeof(Complex_f));
	Complex_f *X1_f=aligned_alloc(AVX,kferm2*sizeof(Complex_f));
	Complex_f *r_f=aligned_alloc(AVX,kferm2*sizeof(Complex_f));
#endif
	//Instead of copying element-wise in a loop, use memcpy.
#ifdef __NVCC__
	//Get X1 in single precision, then swap to AoS format
	cuComplex_convert(X1_f,X1,kferm2,true,dimBlock,dimGrid);
	Transpose_f(X1_f,ndirac*nc,kvol,dimGrid,dimBlock);

	//And repeat for r
	cuComplex_convert(r_f,r,kferm2,true,dimBlock,dimGrid);
	Transpose_f(r_f,ndirac*nc,kvol,dimGrid,dimBlock);

	//cudaMemcpy is blocking, so use async instead
	cudaMemcpyAsync(p_f, X1_f, kferm2*sizeof(Complex_f),cudaMemcpyDeviceToDevice,NULL);
	//Flip all the gauge fields around so memory is coalesced
	Transpose_f(u11t,ndim,kvol,dimGrid,dimBlock);
	Transpose_f(u12t,ndim,kvol,dimGrid,dimBlock);
#else
#pragma omp parallel for simd
	for(int i=0;i<kferm2;i++){
		r_f[i]=(Complex_f)r[i];
		X1_f[i]=(Complex_f)X1[i];
	}
	memcpy(p_f, X1_f, kferm2*sizeof(Complex_f));
#endif

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	double betan; bool pf=true;
	for(*itercg=0; *itercg<niterc; (*itercg)++){
		//x2 =  (M^†M)p 
		//No need to synchronise here. The memcpy in Hdslash is blocking
		Hdslash_f(x1_f,p_f,u11t,u12t,iu,id,gamval_f,gamin,dk4m,dk4p,akappa);
		Hdslashd_f(x2_f,x1_f,u11t,u12t,iu,id,gamval_f,gamin,dk4m,dk4p,akappa);
#ifdef	__NVCC__
		cudaDeviceSynchronise();
#endif
		//x2 =  (M^†M+J^2)p 
		//No point adding zero a couple of hundred times if the diquark source is zero
		if(fac_f!=0){
#ifdef	__NVCC__
			cublasCaxpy(cublas_handle,kferm2,(cuComplex *)&fac_f,(cuComplex *)p_f,1,(cuComplex *)x2_f,1);
#elif defined USE_BLAS
			cblas_caxpy(kferm2, &fac_f, p_f, 1, x2_f, 1);
#else
#pragma omp parallel for simd aligned(p_f,x2_f:AVX)
			for(int i=0; i<kferm2; i++)
				x2_f[i]+=fac_f*p_f[i];
#endif
		}
		//We can't evaluate α on the first *itercg because we need to get β_n.
		if(*itercg){
			//α_d= p* (M^†M+J^2)p
#ifdef __NVCC__
			cublasCdotc(cublas_handle,kferm2,(cuComplex *)p_f,1,(cuComplex *)x2_f,1,(cuComplex *)&alphad);
#elif defined USE_BLAS
			cblas_cdotc_sub(kferm2, p_f, 1, x2_f, 1, &alphad);
#else
			alphad=0;
#pragma omp parallel for simd aligned(p_f,x2_f:AVX)
			for(int i=0; i<kferm2; i++)
				alphad+=conj(p_f[i])*x2_f[i];
#endif
			//For now I'll cast it into a float for the reduction. Each rank only sends and writes
			//to the real part so this is fine
#if(nproc>1)
			Par_fsum((float *)&alphad);
#endif
			//α=α_n/α_d = (r.r)/p(M^†M)p 
			alpha=alphan/creal(alphad);
			//x-αp, 
#ifdef __NVCC__
			Complex_f alpha_f = (Complex_f)alpha;
			cublasCaxpy(cublas_handle,kferm2,(cuComplex *)&alpha_f,(cuComplex *)p_f,1,(cuComplex *)X1_f,1);
#elif defined USE_BLAS
			Complex_f alpha_f = (Complex_f)alpha;
			cblas_caxpy(kferm2, &alpha_f, p_f, 1, X1_f, 1);
#else
			for(int i=0; i<kferm2; i++)
				X1_f[i]+=alpha*p_f[i];
#endif
		}			
		// r_n+1 = r_n-α(M^† M)p_n and β_n=r*.r
#ifdef	__NVCC__
		Complex_f alpha_m=(Complex_f)(-alpha);
		cublasCaxpy(cublas_handle, kferm2,(cuComplex *)&alpha_m,(cuComplex *)x2_f,1,(cuComplex *)r_f,1);
		float betan_f;
		cublasScnrm2(cublas_handle,kferm2,(cuComplex *)r_f,1,&betan_f);
		betan = betan_f*betan_f;
#elif defined USE_BLAS
		Complex_f alpha_m = (Complex_f)(-alpha);
		cblas_caxpy(kferm2, &alpha_m, x2_f, 1, r_f, 1);
		//Undo the negation for the BLAS routine
		float betan_f = cblas_scnrm2(kferm2, r_f,1);
		//Gotta square it to "undo" the norm
		betan = betan_f*betan_f;
#else
		betan=0;
#pragma omp parallel for simd aligned(r_f,x2_f:AVX) reduction(+:betan) 
		for(int i=0; i<kferm2; i++){
			r_f[i]-=alpha*x2_f[i];
			betan += conj(r_f[i])*r_f[i];
		}
#endif
		//And... reduce.
#if(nproc>1)
		Par_dsum(&betan);
#endif
#ifdef _DEBUGCG
#warning "CG Debugging"
		char *endline = "\n";
#else
		char *endline = "\r";
#endif
#ifdef _DEBUG
		if(!rank) printf("Iter(CG)=%i\tβ_n=%e\tα=%e%s", *itercg, betan, alpha,endline);
#endif
		if(betan<resid){ 
			(*itercg)++;
#ifdef _DEBUG
			if(!rank) printf("\nIter(CG)=%i\tResidue: %e\tTolerance: %e\n", *itercg, betan, resid);
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
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multiplying y by
		//β instead of x.
#ifdef __NVCC__
		Complex_f beta_f=(Complex_f)beta;
		__managed__ Complex_f a = 1.0;
		cublasCscal(cublas_handle,kferm2,(cuComplex *)&beta_f,(cuComplex *)p_f,1);
		cublasCaxpy(cublas_handle,kferm2,(cuComplex *)&a,(cuComplex *)r_f,1,(cuComplex *)p_f,1);
#elif (defined __INTEL_MKL__)
		Complex_f a = 1.0;
		Complex_f beta_f=(Complex_f)beta;
		//There is cblas_?axpby in the MKL and AMD though, set a = 1 and b = β.
		//If we get a small enough β_n before hitting the iteration cap we break
		cblas_caxpby(kferm2, &a, r_f, 1, &beta_f,  p_f, 1);
#elif defined USE_BLAS
		Complex_f beta_f=(Complex_f)beta;
		cblas_cscal(kferm2,&beta_f,p_f,1);
		Complex_f a = 1.0;
		cblas_caxpy(kferm2,&a,r_f,1,p_f,1);
#else 
		for(int i=0; i<kferm2; i++)
			p_f[i]=r_f[i]+beta*p_f[i];
#endif
	}
#ifdef __NVCC__
//Restore arrays back to their previous salyout
	Transpose_f(X1_f,kvol,ndirac*nc,dimGrid,dimBlock);
	cuComplex_convert(X1_f,X1,kferm2,false,dimBlock,dimGrid);
	Transpose_f(r_f,kvol,ndirac*nc,dimGrid,dimBlock);
	cuComplex_convert(r_f,r,kferm2,false,dimBlock,dimGrid);
	Transpose_f(u11t,kvol,ndim,dimGrid,dimBlock);
	Transpose_f(u12t,kvol,ndim,dimGrid,dimBlock);
#else
	for(int i=0;i<kferm2;i++){
		X1[i]=(Complex)X1_f[i];
		r[i]=(Complex)r_f[i];
	}
#endif
#ifdef __NVCC__
#ifdef _DEBUG
	cudaDeviceSynchronise();
	cudaFree(x1_f);cudaFree(x2_f); cudaFree(p_f);
	cudaFree(r_f);cudaFree(X1_f);
#else
	//streams match the ones that allocated them.
	cudaFreeAsync(p_f,streams[0]);cudaFreeAsync(x1_f,streams[1]);cudaFreeAsync(x2_f,streams[2]);
	cudaDeviceSynchronise();
	cudaFreeAsync(r_f,streams[3]);cudaFreeAsync(X1_f,streams[4]);
#endif
#else
	free(x1_f);free(x2_f); free(p_f);  free(r_f); free(X1_f);
#endif
	return ret_val;
}
int Congradp(int na,double res,Complex *Phi,Complex *xi,Complex_f *u11t,Complex_f *u12t,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval,int *gamin, float *dk4m,float *dk4p,Complex_f jqq,float akappa,int *itercg){
	/*
	 * @brief Matrix Inversion via Conjugate Gradient
	 * Solves @f$(M^\dagger)Mx=\Phi@f$
	 * No even/odd partitioning.
	 * The matrix multiplication step is done at single precision, while the update is done at double
	 *
	 * @param 	na:			Flavour index
	 * @param 	res:			Limit for conjugate gradient
	 * @param 	Phi:			@f(\Phi@f) initially, 
	 * @param 	xi:			Returned as @f((M^\dagger M)^{-1} \Phi@f)
	 * @param 	u11t:			First colour's trial field
	 * @param 	u12t:			Second colour's trial field
	 * @param 	iu:			Upper halo indices
	 * @param 	id:			Lower halo indices
	 * @param 	gamval:		Gamma matrices
	 * @param 	gamin:		Dirac indices
	 * @param 	dk4m:
	 * @param 	dk4p:
	 * @param 	jqq:			Diquark source
	 * @param 	akappa:		Hopping Parameter
	 * @param 	itercg:		Counts the iterations of the conjugate gradient
	 *
	 * @return 0 on success, integer error code otherwise
	 */
	const char *funcname = "Congradp";
	//Return value
	int ret_val=0;
	const double resid = res*res;
	//These were evaluated only in the first loop of niterx so we'll just do it outside of the loop.
	//These alpha and beta terms should be double, but that causes issues with BLAS. Instead we declare
	//them Complex and work with the real part (especially for α_d)
	//Give initial values Will be overwritten if niterx>0
#ifdef __NVCC__
	Complex_f *p_f, *r_f, *xi_f, *x1_f, *x2_f;
	int device; cudaGetDevice(&device);
#ifdef _DEBUG
	cudaMallocManaged((void **)&p_f, kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&r_f, kferm*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&x1_f, kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&x2_f, kferm*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&xi_f, kferm*sizeof(Complex_f),cudaMemAttachGlobal);
#else
	cudaMalloc((void **)&p_f, kfermHalo*sizeof(Complex_f));
	cudaMalloc((void **)&r_f, kferm*sizeof(Complex_f));
	cudaMalloc((void **)&x1_f, kfermHalo*sizeof(Complex_f));
	cudaMalloc((void **)&x2_f, kferm*sizeof(Complex_f));
	cudaMalloc((void **)&xi_f, kferm*sizeof(Complex_f));
#endif
#else
	Complex_f *p_f  =	aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *r_f  =	aligned_alloc(AVX,kferm*sizeof(Complex_f));
	Complex_f *x1_f	=	aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *x2_f	=	aligned_alloc(AVX,kferm*sizeof(Complex_f));
	Complex_f *xi_f	=	aligned_alloc(AVX,kferm*sizeof(Complex_f));
#endif
	double betad = 1.0; Complex_f alphad=0; Complex alpha = 1;
	double alphan=0.0;
	//Instead of copying element-wise in a loop, use memcpy.
#ifdef __NVCC__
	//Get xi  in single precision, then swap to AoS format
	cuComplex_convert(p_f,xi,kferm,true,dimGrid,dimBlock);
	Transpose_f(p_f,ngorkov*nc,kvol,dimGrid,dimBlock);
	cudaMemcpy(xi_f,p_f,kferm*sizeof(Complex_f),cudaMemcpyDefault);

	//And repeat for r
	cuComplex_convert(r_f,Phi+na*kferm,kferm,true,dimGrid,dimBlock);
	Transpose_f(r_f,ngorkov*nc,kvol,dimGrid,dimBlock);

	//Flip all the gauge fields around so memory is coalesced
	Transpose_f(u11t,ndim,kvol,dimGrid,dimBlock);
	Transpose_f(u12t,ndim,kvol,dimGrid,dimBlock);
#else
#pragma omp parallel for simd aligned(p_f,xi_f,xi,r_f,Phi:AVX)
	for(int i =0;i<kferm;i++){
		p_f[i]=xi_f[i]=(Complex_f)xi[i];
		r_f[i]=(Complex_f)Phi[na*kferm+i];
	}
#endif

	// Declaring placeholder arrays 
	// This x1 is NOT related to the /common/vectorp/X1 in the FORTRAN code and should not
	// be confused with X1 the global variable

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	double betan;
#ifdef __NVCC__
	cudaDeviceSynchronise();
#endif
	for((*itercg)=0; (*itercg)<=niterc; (*itercg)++){
		//Don't overwrite on first run. 
		//x2=(M^†)x1=(M^†)Mp
		Dslash_f(x1_f,p_f,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
		Dslashd_f(x2_f,x1_f,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
#ifdef __NVCC__
		cudaDeviceSynchronise();
#endif
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(*itercg){
			//x*.x
#ifdef USE_BLAS
			float alphad_f;
#ifdef __NVCC__
			cublasScnrm2(cublas_handle,kferm,(cuComplex*) x1_f, 1,(float *)&alphad_f);
			alphad = alphad_f*alphad_f;
#else
			alphad_f = cblas_scnrm2(kferm, x1_f, 1);
#endif
			alphad = alphad_f*alphad_f;
#else
			alphad=0;
			for(int i = 0; i<kferm; i++)
				alphad+=conj(x1_f[i])*x1_f[i];
#endif
#if(nproc>1)
			Par_fsum((float *)&alphad);
#endif
			//α=(r.r)/p(M^†)Mp
			alpha=alphan/creal(alphad);
			//			Complex_f alpha_f = (Complex_f)alpha;
			//x+αp
#ifdef USE_BLAS
			Complex_f alpha_f=(float)alpha;
#ifdef __NVCC__
			cublasCaxpy(cublas_handle,kferm,(cuComplex*) &alpha_f,(cuComplex*) p_f,1,(cuComplex*) xi_f,1);
#else
			cblas_caxpy(kferm, (Complex_f*)&alpha_f,(Complex_f*)p_f, 1, (Complex_f*)xi_f, 1);
#endif
#else
#pragma omp parallel for simd aligned(xi_f,p_f:AVX)
			for(int i = 0; i<kferm; i++)
				xi_f[i]+=alpha*p_f[i];
#endif
		}

		//r=α(M^†)Mp and β_n=r*.r
#if defined USE_BLAS
		Complex_f alpha_m=(Complex_f)(-alpha);
		float betan_f=0;
#ifdef __NVCC__
		cublasCaxpy(cublas_handle,kferm, (cuComplex *)&alpha_m,(cuComplex *) x2_f, 1,(cuComplex *) r_f, 1);
		//cudaDeviceSynchronise();
		//r*.r
		cublasScnrm2(cublas_handle,kferm,(cuComplex *) r_f,1,(float *)&betan_f);
#else
		cblas_caxpy(kferm,(Complex_f*) &alpha_m,(Complex_f*) x2_f, 1,(Complex_f*) r_f, 1);
		//r*.r
		betan_f = cblas_scnrm2(kferm, (Complex_f*)r_f,1);
#endif
		//Gotta square it to "undo" the norm
		betan=betan_f*betan_f;
#else
		//Just like Congradq, this loop could be unrolled but will need a reduction to deal with the betan 
		//addition.
		betan = 0;
		//If we get a small enough β_n before hitting the iteration cap we break
#pragma omp parallel for simd aligned(x2_f,r_f:AVX) reduction(+:betan)
		for(int i = 0; i<kferm;i++){
			r_f[i]-=alpha*x2_f[i];
			betan+=conj(r_f[i])*r_f[i];
		}
#endif
		//This is basically just congradq at the end. Check there for comments
#if(nproc>1)
		Par_dsum(&betan);
#endif
#ifdef _DEBUG
#ifdef _DEBUGCG
		char *endline = "\n";
#else
		char *endline = "\r";
#endif
		if(!rank) printf("Iter (CG) = %i β_n= %e α= %e%s", *itercg, betan, alpha,endline);
#endif
		if(betan<resid){
			//Started counting from zero so add one to make it accurate
			(*itercg)++;
#ifdef _DEBUG
			if(!rank) printf("\nIter (CG) = %i resid = %e toler = %e\n", *itercg, betan, resid);
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
		betad=betan; alphan=betan;
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multiplying y by 
		//β instead of x.
		//There is cblas_zaxpby in the MKL though, set a = 1 and b = β.
#ifdef USE_BLAS
		Complex_f beta_f = (Complex_f)beta;
		Complex_f a = 1.0;
#ifdef __NVCC__
		cublasCscal(cublas_handle,kferm,(cuComplex *)&beta_f,(cuComplex *)p_f,1);
		cublasCaxpy(cublas_handle,kferm,(cuComplex *)&a,(cuComplex *)r_f,1,(cuComplex *)p_f,1);
		cudaDeviceSynchronise();
#elif (defined __INTEL_MKL__ || defined AMD_BLAS)
		cblas_caxpby(kferm, &a, r_f, 1, &beta_f,  p_f, 1);
#else
		cblas_cscal(kferm,&beta_f,p_f,1);
		cblas_caxpy(kferm,&a,r_f,1,p_f,1);
#endif
#else
#pragma omp parallel for simd aligned(r_f,p_f:AVX)
		for(int i=0; i<kferm; i++)
			p_f[i]=r_f[i]+beta*p_f[i];
#endif
	}
#ifdef __NVCC__
	Transpose_f(xi_f,kvol,ngorkov*nc,dimGrid,dimBlock);
	Transpose_f(r_f,kvol,ngorkov*nc,dimGrid,dimBlock);

	Transpose_f(u11t,kvol,ndim,dimGrid,dimBlock);
	Transpose_f(u12t,kvol,ndim,dimGrid,dimBlock);
	cudaDeviceSynchronise();
	cuComplex_convert(xi_f,xi,kferm,false,dimBlock,dimGrid);
#else
#pragma omp simd
	for(int i = 0; i <kferm;i++){
		xi[i]=(Complex)xi_f[i];
	}
#endif
#ifdef	__NVCC__
	cudaFree(p_f); cudaFree(r_f);cudaFree(x1_f); cudaFree(x2_f); cudaFree(xi_f); 
#else
	free(p_f); free(r_f); free(x1_f); free(x2_f); free(xi_f); 
#endif
	return ret_val;
}
/* Old clutter for debugging CG
 * Pre mult
#ifdef _DEBUGCG
memset(x1_f,0,kferm2Halo*sizeof(Complex_f));
#ifdef __NVCC__
cudaMemPrefetchAsync(x1_f,kferm2*sizeof(Complex_f),device,NULL);
cudaDeviceSynchronise();
#endif
printf("\nPre mult:\tp_f[kferm2-1]=%.5e+%.5ei\tx1_f[kferm2-1]=%.5e+%.5ei\tx2_f[kferm2-1]=%.5e+%.5ei\t",\
creal(p_f[kferm2-1]),cimag(p_f[kferm2-1]),creal(x1_f[kferm2-1]),cimag(x1_f[kferm2-1]),creal(x2_f[kferm2-1]),cimag(x2_f[kferm2-1]));
#endif

First mult
#ifdef _DEBUGCG
printf("\nHdslash_f:\tp_f[kferm2-1]=%.5e+%.5ei\tx1_f[kferm2-1]=%.5e+%.5ei\tx2_f[kferm2-1]=%.5e+%.5ei",\
creal(p_f[kferm2-1]),cimag(p_f[kferm2-1]),creal(x1_f[kferm2-1]),cimag(x1_f[kferm2-1]),creal(x2_f[kferm2-1]),cimag(x2_f[kferm2-1]));
#endif
Post mult
#ifdef _DEBUGCG
printf("\nHdslashd_f:\tp_f[kferm2-1]=%.5e+%.5ei\tx1_f[kferm2-1]=%.5e+%.5ei\tx2_f[kferm2-1]=%.5e+%.5ei\n",\
creal(p_f[kferm2-1]),cimag(p_f[kferm2-1]),creal(x1_f[kferm2-1]),cimag(x1_f[kferm2-1]),creal(x2_f[kferm2-1]),cimag(x2_f[kferm2-1]));
#endif

GAmmas
#ifdef _DEBUGCG
printf("Gammas:\n");
for(int i=0;i<5;i++){
for(int j=0;j<4;j++)
printf("%.5e+%.5ei\t",creal(gamval_f[i*4+j]),cimag(gamval_f[i*4+j]));
printf("\n");
}
printf("\nConstants (index %d):\nu11t[kvol-1]=%e+%.5ei\tu12t[kvol-1]=%e+%.5ei\tdk4m=%.5e\tdk4p=%.5e\tjqq=%.5e+I%.5f\tkappa=%.5f\n",\
kvol-1,creal(u11t[kvol-1]),cimag(u11t[kvol-1]),creal(u12t[kvol-1]),cimag(u12t[kvol-1]),dk4m[kvol-1],dk4p[kvol-1],creal(jqq),cimag(jqq),akappa);
#endif
*/
