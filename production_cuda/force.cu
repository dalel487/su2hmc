/*
 * Code for force calculations.
 * Requires multiply.cu to work
 */
#include	<matrices.h>
#include	<par_mpi.h>
#include	<su2hmc.h>
int Gauge_force(double *dSdpi){
	/*
	 * Calculates dSdpi due to the Wilson Action at each intermediate time
	 *
	 * Globals:
	 * =======
	 * u11t, u12t, u11, u12, iu, id, beta
	 * Calls:
	 * =====
	 * Z_Halo_swap_all, Z_gather, Z_Halo_swap_dir
	 */
	const char *funcname = "Gauge_force";

	//We define zero halos for debugging
	//	#ifdef _DEBUG
	//		memset(u11t[kvol], 0, ndim*halo*sizeof(complex));	
	//		memset(u12t[kvol], 0, ndim*halo*sizeof(complex));	
	//	#endif
	//Was a trial field halo exchange here at one point.
#ifdef __CUDACC__
	int device=-1;
	cudaGetDevice(&device);
	Complex *Sigma11, *Sigma12, *u11sh, *u12sh;
	cudaMallocManaged(&Sigma11,kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Sigma12,kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11sh,(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12sh,(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
#elif defined USE_MKL
	complex *Sigma11 = mkl_malloc(kvol*sizeof(complex),AVX); 
	complex *Sigma12= mkl_malloc(kvol*sizeof(complex),AVX); 
	complex *u11sh = mkl_malloc((kvol+halo)*sizeof(complex),AVX); 
	complex *u12sh = mkl_malloc((kvol+halo)*sizeof(complex),AVX); 
#else
	complex *Sigma11 = malloc(kvol*sizeof(complex)); 
	complex *Sigma12= malloc(kvol*sizeof(complex)); 
	complex *u11sh = malloc((kvol+halo)*sizeof(complex)); 
	complex *u12sh = malloc((kvol+halo)*sizeof(complex)); 
#endif
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
		memset(Sigma11,0, kvol*sizeof(Complex));
		cudaMemPrefetchAsync(Sigma11,kvol*sizeof(Complex),device,NULL);
		memset(Sigma12,0, kvol*sizeof(Complex));
		cudaMemPrefetchAsync(Sigma12,kvol*sizeof(Complex),device,NULL);
		for(int nu=0; nu<ndim; nu++){
			if(nu!=mu){
				//The +ν Staple
				Plus_staple<<<dimGrid,dimBlock>>>(mu, nu, Sigma11, Sigma12);
				Z_gather(u11sh, u11t, kvol, id, nu);
				Z_gather(u12sh, u12t, kvol, id, nu);
				ZHalo_swap_dir(u11sh, 1, mu, DOWN);
				cudaMemPrefetchAsync(u11sh, (kvol+halo)*sizeof(Complex),device,NULL);
				ZHalo_swap_dir(u12sh, 1, mu, DOWN);
				cudaMemPrefetchAsync(u12sh, (kvol+halo)*sizeof(Complex),device,NULL);
				//Next up, the -ν staple
				Minus_staple<<<dimGrid,dimBlock>>>(mu, nu, Sigma11, Sigma12, u11sh, u12sh);
			}
		}
		cuGaugeForce<<<dimGrid,dimBlock>>>(mu,Sigma11,Sigma12,dSdpi);
	}
#ifdef __NVCC__
	cudaFree(Sigma11); cudaFree(Sigma12); cudaFree(u11sh); cudaFree(u12sh);
#elif defined USE_MKL
	mkl_free(u11sh); mkl_free(u12sh); mkl_free(Sigma11); mkl_free(Sigma12);
#else
	free(u11sh); free(u12sh); free(Sigma11); free(Sigma12);
#endif
	return 0;
}
int Force(double *dSdpi, int iflag, double res1){
	/*
	 *	Calculates dSds at each intermediate time
	 *	
	 *	Calls:
	 *	=====
	 *
	 *	Globals:
	 *	=======
	 *	u11t, u12t, X1, Phi
	 *
	 *	This X1 is the one being referred to in the common/vector/ statement in the original FORTRAN
	 *	code. There may subroutines with a different X1 (or use a different common block definition
	 *	for this X1) so keep your wits about you
	 *
	 *	Parameters:
	 *	===========
	 *	double dSdpi[][3][kvol+halo]
	 *	int	iflag
	 *	double	res1;
	 *
	 *	Returns:
	 *	=======
	 *	Zero on success, integer error code otherwise
	 */
	const char *funcname = "Force";
	int device=-1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(u11t,(kvol+halo)*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(u12t,(kvol+halo)*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(X0,nf*kfermHalo*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(Phi,nf*kfermHalo*sizeof(Complex),device,NULL);
#ifndef NO_GAUGE
	Gauge_force(dSdpi);
#endif
	//X1=(M†M)^{1} Phi
	int itercg;
#ifdef __CUDACC__
	Complex *X2, *smallPhi;
	cudaMallocManaged(&X2,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&smallPhi,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
#elif defined USE_MKL
	Complex *X2= mkl_malloc(kferm2Halo*sizeof(Complex), AVX);
	Complex *smallPhi =mkl_malloc(kferm2Halo*sizeof(Complex), AVX); 
#else
	Complex *X2= aligned_alloc(AVX,kferm2Halo*sizeof(Complex));
	Complex *smallPhi = aligned_alloc(AVX,kferm2Halo*sizeof(Complex)); 
#endif
	for(int na = 0; na<nf; na++){
		memcpy(X1, X0+na*kferm2Halo, nc*ndirac*kvol*sizeof(Complex));
		//FORTRAN's logic is backwards due to the implied goto method
		//If iflag is zero we do some initalisation stuff 
		if(!iflag){
			Congradq(na, res1,smallPhi, &itercg );
			ancg+=itercg;
			//This is not a general BLAS Routine, just an MKL/AMD one
#if (defined USE_MKL||defined USE_BLAS)
			Complex blasa=2.0; Complex blasb=-1.0;
			cblas_zaxpby(kvol*ndirac*nc, &blasa, X1, 1, &blasb, X0+na*kferm2Halo, 1); 
#else
			for(int i=0;i<kvol;i++){
#pragma unroll
				for(int idirac=0;idirac<ndirac;idirac++){
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc]=
						2.0*X1[(i*ndirac+idirac)*nc]-X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc];
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1]=
						2.0*X1[(i*ndirac+idirac)*nc+1]-X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1];
				}
			}
#endif
		}
		Hdslash(X2,X1);
#ifdef __NVCC__
		double blasd=2.0;
		cublasZdscal(cublas_handle,kferm2, &blasd, (cuDoubleComplex *)X2, 1);
#elif (defined USE_MKL || defined USE_BLAS)
		double blasd=2.0;
		cblas_zdscal(kferm2, blasd, X2, 1);
#else
#pragma unroll
		for(int i=0;i<kferm2;i++)
			X2[i]*=2;
#endif
#pragma unroll
		for(int mu=0;mu<4;mu++){
			ZHalo_swap_dir(X1,8,mu,DOWN);
			ZHalo_swap_dir(X2,8,mu,DOWN);
		}

		//	The original FORTRAN Comment:
		//    dSdpi=dSdpi-Re(X1*(d(Mdagger)dp)*X2) -- Yikes!
		//   we're gonna need drugs for this one......
		//
		//  Makes references to X1(.,.,iu(i,mu)) AND X2(.,.,iu(i,mu))
		//  as a result, need to swap the DOWN halos in all dirs for
		//  both these arrays, each of which has 8 cpts
		//
		cuForce<<<dimGrid,dimBlock>>>(dSdpi,X2);
	}
#ifdef __CUDACC__
	cudaFree(X2); cudaFree(smallPhi);
#elif defined USE_MKL
	mkl_free(X2); mkl_free(smallPhi);
#else
	free(X2); free(smallPhi);
#endif
	return 0;
}

//CUDA Kernels
__global__ void cuForce(double *dSdpi, Complex *X2){
	char *funcname = "cuForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize)
		for(int idirac=0;idirac<ndirac;idirac++){
			int mu, uid, igork1;
#ifndef NO_SPACE
			for(mu=0; mu<3; mu++){
				//Long term ambition. I used the diff command on the different
				//spacial components of dSdpi and saw a lot of the values required
				//for them are duplicates (u11(i,mu)*X2(1,idirac,i) is used again with
				//a minus in front for example. Why not evaluate them first /and then plug 
				//them into the equation? Reduce the number of evaluations needed and look
				//a bit neater (although harder to follow as a consequence).

				//Up indices
				uid = iu[mu+ndim*i];
				igork1 = gamin_d[mu*ndirac+idirac];	
				dSdpi[(i*nadj)*ndim+mu]+=(*akappa_d)*(I*
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
						  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))).real();
				dSdpi[(i*nadj)*ndim+mu]+=(I*gamval_d[mu*ndirac+idirac]*
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
						  +conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))).real();

				dSdpi[(i*nadj+1)*ndim+mu]+=(*akappa_d)*(
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
						  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))).real();
				dSdpi[(i*nadj+1)*ndim+mu]+=(gamval_d[mu*ndirac+idirac]*
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
						  +conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))).real();

				dSdpi[(i*nadj+2)*ndim+mu]+=(*akappa_d)*(I*
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
						  +u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1]))).real();
				dSdpi[(i*nadj+2)*ndim+mu]+=(I*gamval_d[mu*ndirac+idirac]*
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
						  -u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1]))).real();

			}
#endif
			//We're not done tripping yet!! Time like term is different. dk4? shows up
			//For consistency we'll leave mu in instead of hard coding.
			mu=3;
			uid = iu[mu+ndim*i];
			//We are mutiplying terms by dk4?[i] Also there is no (*akappa_d) or gamval_d factor in the time direction	
			//for the "gamval_d" terms the sign of d4kp flips
#ifndef NO_TIME
			dSdpi[(i*nadj)*ndim+mu]+=(I*
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
							     -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])))).real()
				+(I*
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
									-conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])))).real();

			dSdpi[(i*nadj+1)*ndim+mu]+=(
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
							     -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))).real()
				+(
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
									 -conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])))).real();

			dSdpi[(i*nadj+2)*ndim+mu]+=(I*
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
						     +u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])))).real()
				+(I*
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
								+u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1])))).real();

#endif
		}
}
__global__ void Plus_staple(int mu, int nu, Complex *Sigma11, Complex *Sigma12){
	char *funcname = "Plus_staple";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		int uidm = iu[mu+ndim*i];
		int uidn = iu[nu+ndim*i];
		Complex	a11=u11t[uidm*ndim+nu]*conj(u11t[uidn*ndim+mu])+\
				    u12t[uidm*ndim+nu]*conj(u12t[uidn*ndim+mu]);
		Complex	a12=-u11t[uidm*ndim+nu]*u12t[uidn*ndim+mu]+\
				    u12t[uidm*ndim+nu]*u11t[uidn*ndim+mu];
		Sigma11[i]+=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
		Sigma12[i]+=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+nu];
	}
}
__global__ void Minus_staple(int mu, int nu, Complex *Sigma11, Complex *Sigma12, Complex *u11sh, Complex *u12sh){
	char *funcname = "Minus_staple";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
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
}
__global__ void cuGaugeForce(int mu, Complex *Sigma11, Complex *Sigma12,double*dSdpi){
	char *funcname = "cuGaugeForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		Complex a11 = u11t[i*ndim+mu]*Sigma12[i]+u12t[i*ndim+mu]*conj(Sigma11[i]);
		Complex a12 = u11t[i*ndim+mu]*Sigma11[i]+conj(u12t[i*ndim+mu])*Sigma12[i];

		dSdpi[(i*nadj)*ndim+mu]=(*beta_d)*a11.imag();
		dSdpi[(i*nadj+1)*ndim+mu]=(*beta_d)*a11.real();
		dSdpi[(i*nadj+2)*ndim+mu]=(*beta_d)*a12.imag();
	}
}
