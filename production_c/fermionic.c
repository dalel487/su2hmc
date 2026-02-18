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
	const char funcname[] = "Measure";
	//This x is just a storage container

#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	Complex	*x, *xi, *R1 ; Complex_f *xi_f, *R1_f, *clover[nc];
#ifdef _DEBUG
	cudaMallocManaged((void **)&R1,kfermHalo*sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged((void **)&R1_f,kferm*sizeof(Complex_f), cudaMemAttachGlobal);
	if(c_sw){
		cudaMallocManaged((void **)&clover[0], 6*kvol*sizeof(Complex),cudaMemAttachGlobal);
		cudaMallocManaged((void **)&clover[1], 6*kvol*sizeof(Complex),cudaMemAttachGlobal);
	}
#else
	cudaMallocAsync((void **)&R1,kfermHalo*sizeof(Complex),streams[0]);
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
	Complex *R1 = (Complex *)aligned_alloc(AVX,kfermHalo*sizeof(Complex));
#endif
	//Setting up noise. Again need that annoying stride 
	for(unsigned short j=0;j<nc*ngorkov;j++)
		Gauss_c(xi_f+j*kvolHalo, kvol, 0, (float)(1/sqrt(2)));
	ComplexConvert(xi_f,xi,kvol,false,nc*ngorkov);
#ifdef __NVCC__
#if (nproc>1) //strided
	for(unsigned short j=0;j<nc*ngorkov;j++){
		cudaMemcpyAsync(x+j*kvolHalo, xi+j*kvol, kvol*sizeof(Complex),cudaMemcpyDefault,0);
	}
#else
	cudaMemcpyAsync(x, xi, kferm*sizeof(Complex),cudaMemcpyDefault,0);
#endif
#else
#pragma omp parallel for
	for(unsigned short j=0;j<nc*ngorkov;j++){
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
	ComplexConvert(R1_f,R1,kvol,false,nc*ngorkov);
#ifdef __NVCC__
	cudaFree(xi_f);	
	cudaDeviceSynchronise();
#ifdef _DEBUG
	cudaFree(R1_f);
#else
	cudaFreeAsync(R1_f,streams[0]);
#endif
#if (nproc>1) //strided
	for(unsigned short j=0;j<ngorkov*nc;j++){
		//Phi has no halo
		cudaMemcpyAsync(Phi+j*kvol, R1+j*kvolHalo, kvol*sizeof(Complex),cudaMemcpyDefault,streams[j]);
	}
#else
	cudaMemcpyAsync(Phi, R1, kferm*sizeof(Complex),cudaMemcpyDefault,streams[0]);
#endif
	cudaDeviceSynchronise();
#else
	free(xi_f); free(R1_f);
#pragma omp parallel for 
	for(unsigned short j=0;j<ngorkov*nc;j++){
		memcpy(Phi+j*kvol, R1+j*kvolHalo, kvol*sizeof(Complex));
	}
#endif
	///Evaluate xi = (M^† M)^-1 R_1 
	if(Congradp(0, res, Phi,R1,ut,ut_f,clover,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,c_sw,itercg)==ITERLIM)
		return ITERLIM;
#ifdef __NVCC__
#if (nproc>1)
	for(unsigned short j=0;j<ngorkov*nc;j++)
		cudaMemcpyAsync(xi+j*kvol,R1+j*kvolHalo,kvol*sizeof(Complex),cudaMemcpyDefault,streams[j]);
#else
	cudaMemcpyAsync(xi,R1,kferm*sizeof(Complex),cudaMemcpyDefault,streams[0]);
#endif
#ifdef _DEBUG
	if(c_sw){
		cudaFree(clover[0]); cudaFree(clover[1]);
	}
	cudaFree(R1);
#else
	if(c_sw){
		cudaFreeAsync(clover[0],streams[1]); cudaFreeAsync(clover[1],streams[2]);
	}
	cudaDeviceSynchronise();
	cudaFreeAsync(R1,streams[0]);
#endif
	cudaDeviceSynchronise();
#else
	for(unsigned short j=0;j<ngorkov*nc;j++)
		memcpy(xi+j*kvol,R1+j*kvolHalo,kvol*sizeof(Complex));
	if(c_sw){
		free(clover[0]); free(clover[1]);
	}
	free(R1);
#endif
	*pbp = 0;
#ifdef USE_BLAS
	alignas(16) Complex buff;
#ifdef __NVCC__
#if(nproc>1)
	for(unsigned short j=0;j<ngorkov*nc;j++){
		buff=0;
		cublasZdotc(cublas_handle,kvol,(cuDoubleComplex *)x+j*kvolHalo,1,(cuDoubleComplex *)xi+j*kvol,1,(cuDoubleComplex *)&buff);
		*pbp+=creal(buff);
	}
#else
	cublasZdotc(cublas_handle,kferm,(cuDoubleComplex *)x,1,(cuDoubleComplex *)xi,1,(cuDoubleComplex *)&buff);
	*pbp+=creal(buff);
#endif
	cudaDeviceSynchronise();
#elif defined USE_BLAS
	for(unsigned short j=0;j<ngorkov*nc;j++){
		buff=0;
		cblas_zdotc_sub(kvol, x+j*kvolHalo, 1, xi+j*kvol,  1, &buff);
		*pbp+=creal(buff);
	}
#endif
#else
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
			alignas(16) Complex dot=0;
			//Because we have kvol on the outer index and are summing over it, we set the
			//step for BLAS to be ngorkov*nc=16. 
			//Does this make sense to do on the GPU?
#ifdef __NVCC__
			cublasZdotc(cublas_handle,kvol,(cuDoubleComplex *)x+kvolHalo*(idirac*nc+ic),1,(cuDoubleComplex *)xi+kvol*(igork*nc+ic), 1,(cuDoubleComplex *)&dot);
#else
			cblas_zdotc_sub(kvol, x+kvolHalo*(idirac*nc+ic), 1, xi+kvolHalo*(igork*nc+ic), 1, &dot);
#endif
			*qbqb+=gamval[4*ndirac+idirac]*dot;
#ifdef __NVCC__
			cublasZdotc(cublas_handle,kvol,(cuDoubleComplex *)x+kvolHalo*(igork*nc+ic),1,(cuDoubleComplex *)xi+kvol*(idirac*nc+ic), 1,(cuDoubleComplex *)&dot);
#else
			cblas_zdotc_sub(kvol, x+kvolHalo*(igork*nc+ic), 1, xi+kvol*(idirac*nc+ic), 1, &dot);
#endif
			*qq-=gamval[4*ndirac+idirac]*dot;
		}
	}
#else
	//What is the optimal order to evaluate these in?
#pragma omp parallel for simd collapse(2) aligned(x,xi:AVX) reduction(+:*qq,*qbqb)
	for(int idirac = 0; idirac<ndirac; idirac++)
		for(int i=0; i<kvol; i++){
			int igork=idirac+4;
			*qbqb+=gamval[4*ndirac+idirac]*conj(x[i+kvolHalo*(idirac*nc)])*xi[i+kvol*(igork*nc)];
			*qbqb+=gamval[4*ndirac+idirac]*conj(x[i+kvolHalo*(idirac*nc+1)])*xi[i+kvol*(igork*nc+1)];
			*qq-=gamval[4*ndirac+idirac]*conj(x[i*kvolHalo*(igork*nc)])*xi[i+kvol*(idirac*nc)];
			*qq-=gamval[4*ndirac+idirac]*conj(x[i*kvolHalo*(igork*nc+1)])*xi[i+kvol*(idirac*nc+1)];
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
	
	//TODO: Make the code below CUDA friendly.
	for(unsigned short igorkov=0; igorkov<4; igorkov++){
		const unsigned short igork1=gamin[3*ndirac+igorkov];
		//For the C Version I'll try and factorise where possible
#pragma omp parallel for simd aligned(dk,x,xi:AVX)  reduction(+:xu) 
		for(unsigned int i = 0; i<kvol; i++){
			unsigned int did=id[3*kvol+i];
			xu+=dk[1][did]*(conj(x[did+kvolHalo*(igorkov*nc)])*(\
						ut[0][did+kvol*3]*(xi[i+kvol*(igork1)*nc]-xi[i+kvol*(igorkov)*nc])+\
						ut[1][did+kvol*3]*(xi[i+kvol*(igork1)*nc+1]-xi[i+kvol*(igorkov)*nc+1]) )+\
					conj(x[did+kvolHalo*(igorkov*nc+1)])*(\
						conj(ut[0][did+kvol*3])*(xi[i+kvol*(igork1)*nc+1]-xi[i+kvol*(igorkov)*nc+1])+\
						conj(ut[1][did+kvol*3])*(xi[i+kvol*(igorkov)*nc]-xi[i+kvol*(igork1)*nc])));
		}
	}
	for(unsigned short igorkov=0; igorkov<4; igorkov++){
		const unsigned short igork1=gamin[3*ndirac+igorkov];
#pragma omp parallel for simd aligned(dk,x,xi:AVX)  reduction(+:xd) 
		for(unsigned int i = 0; i<kvol; i++){
			unsigned int uid=iu[3*kvol+i];
			xd+=dk[0][i]*(conj(x[uid+kvolHalo*(igorkov*nc)])*(\
						conj(ut[0][i+kvol*3])*(xi[i+kvol*(igork1*nc)]+xi[i+kvol*(igorkov*nc)])-\
						ut[1][i+kvol*3]*(xi[i+kvol*(igork1*nc+1)]+xi[i+kvol*(igorkov*nc+1)]) )+\
					conj(x[uid+kvolHalo*(igorkov*nc+1)])*(\
						ut[0][i+kvol*3]*(xi[i+kvol*(igork1*nc+1)]+xi[i+kvol*(igorkov*nc+1)])+\
						conj(ut[1][i+kvol*3])*(xi[i+kvol*(igorkov*nc)]+xi[i+kvol*(igork1*nc)]) ) );
		}
	}
	for(unsigned short igorkovPP=4; igorkovPP<8; igorkovPP++){
		const unsigned short igork1PP=4+gamin[3*ndirac+igorkovPP-4];
#pragma omp parallel for simd aligned(dk,x,xi:AVX)  reduction(+:xuu) 
		for(unsigned int i = 0; i<kvol; i++){
			unsigned int did=id[3*kvol+i];
			xuu-=dk[0][did]*(conj(x[did+kvolHalo*(igorkovPP*nc)])*(\
						ut[0][did+kvol*3]*(xi[i+kvol*(igork1PP*nc)]-xi[i+kvol*(igorkovPP*nc)])+\
						ut[1][did+kvol*3]*(xi[i+kvol*(igork1PP*nc+1)]-xi[i+kvol*(igorkovPP*nc+1)]) )+\
					conj(x[did+kvolHalo*(igorkovPP*nc+1)])*(\
						conj(ut[0][did+kvol*3])*(xi[i+kvol*(igork1PP)*nc+1]-xi[i+kvol*(igorkovPP)*nc+1])+\
						conj(ut[1][did+kvol*3])*(xi[i+kvol*(igorkovPP)*nc]-xi[i+kvol*(igork1PP)*nc]) ) );
		}
	}
	for(unsigned short igorkovPP=4; igorkovPP<8; igorkovPP++){
		const unsigned short igork1PP=4+gamin[3*ndirac+igorkovPP-4];
#pragma omp parallel for simd aligned(dk,x,xi:AVX)  reduction(+:xdd) 
		for(unsigned int i = 0; i<kvol; i++){
			unsigned int uid=iu[3*kvol+i];
			xdd-=dk[1][i]*(conj(x[uid+kvolHalo*(igorkovPP*nc)])*(\
						conj(ut[0][i+kvol*3])*(xi[i+kvol*(igork1PP*nc)]+xi[i+kvol*(igorkovPP*nc)])-\
						ut[1][i+kvol*3]*(xi[i+kvol*(igork1PP*nc+1)]+xi[i+kvol*(igorkovPP*nc+1)]) )+\
					conj(x[uid+kvolHalo*(igorkovPP*nc+1)])*(\
						ut[0][i+kvol*3]*(xi[i+kvol*(igork1PP*nc+1)]+xi[i+kvol*(igorkovPP*nc+1)])+\
						conj(ut[1][i+kvol*3])*(xi[i+kvol*(igorkovPP*nc)]+xi[i+kvol*(igork1PP*nc)]) ) );
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
