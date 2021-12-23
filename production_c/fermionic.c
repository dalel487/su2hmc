/*
 *Code for fermionic observables
 */

#include	<matrices.h>
#include	<random.h>
#include	<su2hmc.h>
int Measure(double *pbp, double *endenf, double *denf, Complex *qq, Complex *qbqb, double res, int *itercg){
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
	 * Gauss_z, Par_dsum, ZHalo_swap_dir, DHalo_swap_dir, Congradp, Dslashd
	 *
	 * Globals:
	 * =======
	 * Phi, X0, xi, R1, u11t, u12t, ganval, iu, id 
	 *
	 * Parameters:
	 * ==========
	 * double *pbp:		Pointer to ψ-bar ψ
	 * double endenf:		Energy density
	 * double denf:
	 * Complex qq:		Diquark
	 * Complex qbqb:		Antidiquark
	 * double res:		Conjugate Gradient Residue
	 * int itercg:		Iterations of Conjugate Gradient
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Measure";
	//This x is just a storage container

#ifdef __INTEL_MKL__
	Complex	*x = mkl_malloc(kfermHalo*sizeof(Complex), AVX);
	Complex *xi	= mkl_malloc(kfermHalo*sizeof(Complex),AVX);
	Complex_f	*xi_f = mkl_malloc(kfermHalo*sizeof(Complex_f), AVX);
	Complex_f	*R1_f = mkl_malloc(kferm*sizeof(Complex_f), AVX);
#else
	Complex *x = aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex_f *xi_f = aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *R1_f = aligned_alloc(AVX,kferm*sizeof(Complex_f));
#endif
	//Setting up noise.

	//The root two term comes from the fact we called gauss0 in the fortran code instead of gaussp
#if (defined(USE_RAN2)||!defined(__INTEL_MKL__))
	Gauss_z(xi, kferm, 0, 1/sqrt(2));
#else
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, xi, 0, 1/sqrt(2));
#endif
	memcpy(x, xi, kferm*sizeof(Complex));

	//R_1= M^† Ξ 
	//R1 is local in fortran but since its going to be reset anyway I'm going to recycle the
	//global
#pragma omp parallel for simd aligned(R1,xi,R1_f,xi_f:AVX)
	for(int i=0;i<kferm;i++){
		R1_f[i]=(Complex_f)R1[i];
		xi_f[i]=(Complex_f)xi[i];
	}
	Dslashd_f(R1_f, xi_f);
#pragma omp parallel for simd aligned(R1,R1_f:AVX)
	for(int i=0;i<kferm;i++)
		R1[i]=(Complex)R1_f[i];
#ifdef __NVCC__
	cudaFree(xi_f); cudaFree(R1_f);
#elif defined __INTEL_MKL__
	mkl_free(xi_f);	mkl_free(R1_f);
#else
	free(xi_f); free(R1_f);
#endif
	//Copying R1 to the first (zeroth) flavour index of Phi
	//This should be safe with memcpy since the pointer name
	//references the first block of memory for that pointer
	memcpy(Phi, R1, nc*ngorkov*kvol*sizeof(Complex));
	memcpy(xi, R1, nc*ngorkov*kvol*sizeof(Complex));

	//Evaluate xi = (M^† M)^-1 R_1 
	//This is still double precision for now
	Congradp(0, res, itercg);
#if (defined __INTEL_MKL__ || defined USE_BLAS)
	Complex buff;
	cblas_zdotc_sub(kferm, x, 1, xi,  1, &buff);
	*pbp=creal(buff);
#else
	*pbp = 0;
#pragma unroll
	for(int i=0;i<kferm;i++)
		*pbp+=creal(conj(x[i])*xi[i]);
#endif
	Par_dsum(pbp);
	*pbp/=4*gvol;

	*qbqb=0; *qq=0;
#if (defined __INTEL_MKL__ || defined USE_BLAS)
	for(int idirac = 0; idirac<ndirac; idirac++){
		int igork=idirac+4;
		//Unrolling the colour indices, Then its just (γ_5*x)*Ξ or (γ_5*Ξ)*x 
#pragma unroll
		for(int ic = 0; ic<nc; ic++){
			Complex dot;
			//Because we have kvol on the outer index and are summing over it, we set the
			//step for BLAS to be ngorkov*nc=16. 
			//Does this make sense to do on the GPU?
			cblas_zdotc_sub(kvol, &x[idirac*nc+ic], ngorkov*nc, &xi[igork*nc+ic], ngorkov*nc, &dot);
			*qbqb+=gamval[4][idirac]*dot;
			cblas_zdotc_sub(kvol, &x[igork*nc+ic], ngorkov*nc, &xi[idirac*nc+ic], ngorkov*nc, &dot);
			*qq-=gamval[4][idirac]*dot;
		}
	}
#else
#pragma unroll(2)
	for(int i=0; i<kvol; i++)
		//What is the optimal order to evaluate these in?
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork=idirac+4;
			*qbqb+=gamval[4][idirac]*conj(x[(i*ngorkov+idirac)*nc])*xi[(i*ngorkov+igork)*nc];
			*qq-=gamval[4][idirac]*conj(x[(i*ngorkov+igork)*nc])*xi[(i*ngorkov+idirac)*nc];
			*qbqb+=gamval[4][idirac]*conj(x[(i*ngorkov+idirac)*nc+1])*xi[(i*ngorkov+igork)*nc+1];
			*qq-=gamval[4][idirac]*conj(x[(i*ngorkov+igork)*nc+1])*xi[(i*ngorkov+idirac)*nc+1];
		}
#endif
	//In the FORTRAN Code dsum was used instead despite qq and qbqb being complex
	//Since we only care about the real part this shouldn't cause (m)any serious issues
	Par_dsum(qq); Par_dsum(qbqb);
	*qq=(*qq+*qbqb)/(2*gvol);
	Complex xu, xd, xuu, xdd;
	xu=0;xd=0;xuu=0;xdd=0;

	//Halos
	ZHalo_swap_dir(x,16,3,DOWN);		ZHalo_swap_dir(x,16,3,UP);
	//Pesky halo exchange indices again
	//The halo exchange for the trial fields was done already at the end of the trajectory
	//No point doing it again

	//Instead of typing id[i*ndim+3] a lot, we'll just assign them to variables.
	//Idea. One loop instead of two loops but for xuu and xdd just use ngorkov-(igorkov+1) instead
//#ifdef __clang__
//#pragma omp target teams distribute parallel for reduction(+:xd,xu,xdd,xuu)\
	map(tofrom:xu,xd,xuu,xdd)
//#else
#pragma omp parallel for reduction(+:xd,xu,xdd,xuu) 
//#endif
	for(int i = 0; i<kvol; i++){
		int did=id[3+ndim*i];
		int uid=iu[3+ndim*i];
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
	*endenf=creal(xu-xd-xuu+xdd);
	*denf=creal(xu+xd+xuu+xdd);

	Par_dsum(endenf); Par_dsum(denf);
	*endenf/=2*gvol; *denf/=2*gvol;
	//Future task. Chiral susceptibility measurements
#ifdef __NVCC__
	cudaFree(x); cudaFree(xi);
#elif defined __INTEL_MKL__
	mkl_free(x); mkl_free(xi);
#else
	free(x); free(xi);
#endif
	return 0;
}
