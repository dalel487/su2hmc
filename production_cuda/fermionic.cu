/*
 *Code for fermionic observables
 */
#include	<matrices.h>
#include	<random.h>
#include	<su2hmc.h>
int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg,\
		Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int *iu, unsigned int *id,\
		Complex gamval[5][4], Complex_f gamval_f[5][4],	int gamin[4][4], double *dk4m, double *dk4p,\
		float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa,	Complex *Phi, Complex *R1){
	/*
	 * Calculate fermion expectation values via a noisy estimator
	 * -matrix inversion via conjugate gradient algorithm
	 * solves Mx=x1
	 * (Numerical Recipes section 2.10 pp.70-73)   
	 * uses NEW lookup tables **
	 * Implimented in CongradX
	 *
	 * Calls:
	 * =====
	 * Gauss_z
	 * Par_dsum
	 * ZHalo_swap_dir
	 * DHalo_swap_dir
	 *
	 * Globals:
	 * =======
	 * Phi, X0, xi, R1, u11t, u12t 
	 *
	 * Parameters:
	 * ==========
	 * double *pbp:		Pointer to ψ-bar ψ
	 * double endenf:
	 * double denf:
	 * complex qq:
	 * complex qbqb:
	 * double res:
	 * int itercg:
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Measure";
	//This x is just a storage container

	int device=-1;
	cudaGetDevice(&device);
	Complex *x,*xi;
	Complex_f *xi_f, *R1_f;
	cudaMallocManaged(&x,kfermHalo*sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged(&xi,kferm*sizeof(Complex), cudaMemAttachGlobal);
	cudaMallocManaged(&xi_f,kfermHalo*sizeof(Complex_f), cudaMemAttachGlobal);
	cudaMallocManaged(&R1_f,kfermHalo*sizeof(Complex_f), cudaMemAttachGlobal);
	//Setting up noise. I don't see any reason to loop

	//The root two term comes from the fact we called gauss0 in the fortran code instead of gaussp
#if (defined(USE_RAN2)||!defined(USE_MKL))
	Gauss_c(xi_f, kferm, 0, 1/sqrt(2));
#else
	vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, (float*)xi_f, 0, 1/sqrt(2));
#endif
	cudaMemPrefetchAsync(xi_f, kferm*sizeof(Complex_f),device,NULL);
	for(int i=0;i<kferm;i++)
		xi[i]=(Complex)xi_f[i];
	memcpy(x, xi, kferm*sizeof(Complex));
	//R_1= M^† Ξ 
	//R1 is local in fortran but since its going to be reset anyway I'm going to recycle the
	//global
	Dslashd_f(R1_f, xi_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
	for(int i=0;i<kferm;i++)
		R1[i]=(Complex)R1_f[i];
	//Copying R1 to the first (zeroth) flavour index of Phi
	//This should be safe with memcpy since the pointer name
	//references the first block of memory for that pointer
	cudaMemPrefetchAsync(Phi, kfermHalo*sizeof(Complex),device,NULL);
	memcpy(Phi, R1, nc*ngorkov*kvol*sizeof(Complex));
	//Evaluate xi = (M^† M)^-1 R_1 
	cudaMemPrefetchAsync(R1, kfermHalo*sizeof(Complex),device,NULL);
	if(Congradp(0, res, Phi, R1_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa,itercg)==ITERLIM){
		itercg=0;
		fprintf(stderr, "Restarting conjugate gradient from %s\n", funcname);
		Congradp(0, res, Phi, R1_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa,itercg);
		itercg+=niterc;
	}
#pragma omp parallel for simd aligned(R1,R1_f:AVX)
	for(int i=0;i<kferm;i++)
		xi[i]=(Complex)R1_f[i];
	cudaFree(xi_f); cudaFree(R1_f);
	Complex buff;
	cublasZdotc(cublas_handle,kferm, (cuDoubleComplex *)x, 1, (cuDoubleComplex *)xi,  1, (cuDoubleComplex *)&buff);
	*pbp=buff.real();

	Par_dsum(pbp);
	*pbp/=4*gvol;

	*qbqb=0; *qq=0;

#pragma unroll
	for(int idirac = 0; idirac<ndirac; idirac++){
		int igork=idirac+4;
		//Unrolling the colour indices, Then its just (γ_5*x)*Ξ or (γ_5*Ξ)*x 
		for(int ic = 0; ic<nc; ic++){
			Complex dot;
			//Because we have kvol on the outer index and are summing over it, we set the
			//step for BLAS to be ngorkov*nc=16. 
			//Does this make sense to do on the GPU?
			cublasZdotc(cublas_handle,kferm, (cuDoubleComplex *)&x[idirac*nc+ic], ngorkov*nc,\
					(cuDoubleComplex *)&xi[igork*nc+ic],  ngorkov*nc, (cuDoubleComplex *)&dot);
			*qbqb+=gamval[4][idirac]*dot;

			cublasZdotc(cublas_handle,kferm, (cuDoubleComplex *)&x[igork*nc+ic], ngorkov*nc,\
					(cuDoubleComplex *)&xi[idirac*nc+ic],  ngorkov*nc, (cuDoubleComplex *)&dot);
			*qq-=gamval[4][idirac]*dot;
		}
	}

	//In the FORTRAN Code dsum was used instead despite qq and qbqb being complex
	Par_zsum(qq); Par_zsum(qbqb);
	*qq=(*qq+*qbqb)/(2.0*gvol);
	Complex xu, xd, xuu, xdd;
	xu=0;xd=0;xuu=0;xdd=0;

	//Halos
	ZHalo_swap_dir(x,16,3,DOWN);		ZHalo_swap_dir(x,16,3,UP);
	//Pesky halo exchange indices again
	//The halo exchange for the trial fields was done already at the end of the trajectory
	//No point doing it again

	//Instead of typing id[i*ndim+3] a lot, we'll just assign them to variables.
	//Idea. One loop instead of two loops but for xuu and xdd just use ngorkov-(igorkov+1) instead
#pragma omp parallel for //reduction(+:xd,xu,xdd,xuu) 
	for(int i = 0; i<kvol; i++){
		int did=id[3+ndim*i];
		int uid=iu[3+ndim*i];
#pragma unroll
#pragma omp simd aligned(u11t,u12t,xi,x,dk4m,dk4p:AVX) 
		for(int igorkov=0; igorkov<4; igorkov++){
			int igork1=gamin[3][igorkov];
			//For the C Version I'll try and factorise where possible

			xu+=dk4p[did]*(conj(x[(did*ngorkov+igorkov)*nc])*(\
						u11t[did*ndim+3]*(xi[(i*ngorkov+igork1)*nc]-xi[(i*ngorkov+igorkov)*nc])+\
						u12t[did*ndim+3]*(xi[(i*ngorkov+igork1)*nc+1]-xi[(i*ngorkov+igorkov)*nc+1]) )+\
					conj(x[(did*ngorkov+igorkov)*nc+1])*(\
						conj(u11t[did*ndim+3])*(xi[(i*ngorkov+igork1)*nc+1]-xi[(i*ngorkov+igorkov)*nc+1])+\
						conj(u12t[did*ndim+3])*(xi[(i*ngorkov+igorkov)*nc]-xi[(i*ngorkov+igork1)*nc])));

			xd+=dk4m[i]*(conj(x[(uid*ngorkov+igorkov)*nc])*(\
						conj(u11t[i*ndim+3])*(xi[(i*ngorkov+igork1)*nc]+xi[(i*ngorkov+igorkov)*nc])-\
						u12t[i*ndim+3]*(xi[(i*ngorkov+igork1)*nc+1]+xi[(i*ngorkov+igorkov)*nc+1]) )+\
					conj(x[(uid*ngorkov+igorkov)*nc+1])*(\
						u11t[i*ndim+3]*(xi[(i*ngorkov+igork1)*nc+1]+xi[(i*ngorkov+igorkov)*nc+1])+\
						conj(u12t[i*ndim+3])*(xi[(i*ngorkov+igorkov)*nc]+xi[(i*ngorkov+igork1)*nc]) ) );

			int igorkovPP=igorkov+4;
			int igork1PP=igork1+4;
			xuu-=dk4m[did]*(conj(x[(did*ngorkov+igorkovPP)*nc])*(\
						u11t[did*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc]-xi[(i*ngorkov+igorkovPP)*nc])+\
						u12t[did*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc+1]-xi[(i*ngorkov+igorkovPP)*nc+1]) )+\
					conj(x[(did*ngorkov+igorkovPP)*nc+1])*(\
						conj(u11t[did*ndim+3])*(xi[(i*ngorkov+igork1PP)*nc+1]-xi[(i*ngorkov+igorkovPP)*nc+1])+\
						conj(u12t[did*ndim+3])*(xi[(i*ngorkov+igorkovPP)*nc]-xi[(i*ngorkov+igork1PP)*nc]) ) );

			xdd-=dk4p[i]*(conj(x[(uid*ngorkov+igorkovPP)*nc])*(\
						conj(u11t[i*ndim+3])*(xi[(i*ngorkov+igork1PP)*nc]+xi[(i*ngorkov+igorkovPP)*nc])-\
						u12t[i*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc+1]+xi[(i*ngorkov+igorkovPP)*nc+1]) )+\
					conj(x[(uid*ngorkov+igorkovPP)*nc+1])*(\
						u11t[i*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc+1]+xi[(i*ngorkov+igorkovPP)*nc+1])+\
						conj(u12t[i*ndim+3])*(xi[(i*ngorkov+igorkovPP)*nc]+xi[(i*ngorkov+igork1PP)*nc]) ) );
		}
	}
	*endenf=(xu-xd-xuu+xdd).real();
	*denf=(xu+xd+xuu+xdd).real();

	Par_dsum(endenf); Par_dsum(denf);
	*endenf/=2*gvol; *denf/=2*gvol;
	//Future task. Chiral susceptibility measurements
	cudaFree(x);
	return 0;
}
