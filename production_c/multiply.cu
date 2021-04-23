#include <cuComplex.h>
#include <multiply.h>
#include <string.h>
//TO DO: Check and see are there any terms we are evaluating twice in the same loop
//and use a variable to hold them instead to reduce the number of evaluations.
//Host/Series Code
int Dslash(cuDoubleComplex *phi, cuDoubleComplex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * cuDoubleComplex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * cuDoubleComplex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslash";
	//Get the halos in order
	ZHalo_swap_all(r, 16);

	//Mass term
	memcpy(phi, r, kferm*sizeof(cuDoubleComplex));
	cudaDeviceSynchronize();
	cuDslash<<<dimGrid,dimBlock>>>(phi,r);
	cudaDeviceSynchronize();
	return 0;
}
int Dslashd(cuDoubleComplex *phi, cuDoubleComplex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 *
	 * cuDoubleComplex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * cuDoubleComplex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslashd";
	//Get the halos in order
	ZHalo_swap_all(r, 16);

	//Mass term
	memcpy(phi, r, kferm*sizeof(cuDoubleComplex));
	cudaDeviceSynchronize();
	cuDslashd<<<dimGrid,dimBlock>>>(phi,r);
	cudaDeviceSynchronize();
	return 0;
}
int Hdslash(cuDoubleComplex *phi, cuDoubleComplex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * cuDoubleComplex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * cuDoubleComplex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslash";
	//Get the halos in order
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Mass term
	memcpy(phi, r, kferm2*sizeof(cuDoubleComplex));
	cudaDeviceSynchronize();
	cuHdslash<<<dimGrid,dimBlock>>>(phi,r);
	cudaDeviceSynchronize();
	return 0;
}
int Hdslashd(cuDoubleComplex *phi, cuDoubleComplex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * cuDoubleComplex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * cuDoubleComplex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslashd";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
	ZHalo_swap_all(r, 8);

	//Mass term
	memcpy(phi, r, kferm2*sizeof(cuDoubleComplex));
	memcpy(phi, r, kferm2*sizeof(cuDoubleComplex));
	cudaDeviceSynchronize();
	cuHdslashd<<<dimGrid,dimBlock>>>(phi,r);
	cudaDeviceSynchronize();
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

	Gauge_force(dSdpi);
	//X1=(M†M)^{1} Phi
	int itercg;
#ifdef USE_CUDA
	cuDoubleComplex *X2, *smallPhi;
	cudaMallocManaged(&X2, kferm2Halo*sizeof(cuDoubleComplex));
#elif defined USE_MKL
	cuDoubleComplex *X2= mkl_malloc(kferm2Halo*sizeof(cuDoubleComplex), AVX);
	cuDoubleComplex *smallPhi =mkl_malloc(kferm2Halo*sizeof(cuDoubleComplex), AVX); 
#else
	cuDoubleComplex *X2= malloc(kferm2Halo*sizeof(cuDoubleComplex));
	cuDoubleComplex *smallPhi = malloc(kferm2Halo*sizeof(cuDoubleComplex)); 
#endif
	for(int na = 0; na<nf; na++){
		memcpy(X1, X0+na*kferm2Halo, nc*ndirac*kvol*sizeof(cuDoubleComplex));
		//FORTRAN's logic is backwards due to the implied goto method
		//If iflag is zero we do some initalisation stuff? 
		if(!iflag){
			Congradq(na, res1,smallPhi, &itercg );
			ancg+=itercg;
			//BLASable? If we cheat and flatten the array it is!
			//This is not a general BLAS Routine, just an MKL one
#ifdef USE_MKL
			cuDoubleComplex blasa=2.0; cuDoubleComplex blasb=-1.0;
			cblas_zaxpby(kvol*ndirac*nc, &blasa, X1, 1, &blasb, X0+na*kferm2Halo, 1); 
#else
			for(int i=0;i<kvol;i++){
#pragma unroll
				for(int idirac=0;idirac<ndirac;idirac++){
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc]=
						2*X1[(i*ndirac+idirac)*nc]-X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc];
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1]=
						2*X1[(i*ndirac+idirac)*nc+1]-X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1];
				}
			}
#endif
		}
		Hdslash(X2,X1);
#if (defined USE_MKL || defined USE_BLAS)
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
		cudaDeviceSynchronize();
		cuForce<<<dimGrid,dimBlock>>>(dSdpi,X2);
		cudaDeviceSynchronize();
	}
#ifdef USE_CUDA
	cudaFree(X2); cudaFree(smallPhi);
#elif defined USE_MKL
	mkl_free(X2); mkl_free(smallPhi);
#else
	free(X2); free(smallPhi);
#endif
	return 0;
}

//Device code: Mainly the loops
dim3 dimGrid(ksizez,ksizet,1);
dim3 dimBlock(ksizex,ksizey,1);
__global__ void cuDslash(cuDoubleComplex *phi, cuDoubleComplex *r){
	int blockID = (gridDim.x*blockIdx.y)+blockIdx.x;
	int threadID = (blockID*(blockDim.x*blockDim.y))+(threadIdx.y*blockDim.x)+threadIdx.y;
	int thread_count = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	for(int i=threadId;i<kvol;i+=thread_count){

		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			cuDoubleComplex a_1, a_2;
			a_1=conj(jqq)*gamval[4][idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval[4][idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc+0];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+0]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!

		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];

			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu][idirac] : gamin[mu][idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]-\
						u12t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
								     //Dirac term
								     gamval[mu][idirac]*(u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]+\
										     u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]-\
										     conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]+\
										     u12t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]+\
						u11t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
									 //Dirac term
									 gamval[mu][idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]+\
											 conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]-\
											 conj(u12t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]-\
											 u11t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];

		for(int igorkov=0; igorkov<4; igorkov++){
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			//the FORTRAN code did it.
			int igork1 = gamin[3][igorkov];	int igork1PP = igork1+4;

			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4p[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));

			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])+\
					u12t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))-\
								 dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])-\
										 u12t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=-dk4m[i]*(conj(-u12t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])+\
					conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))-\
								   dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])+\
										   u11t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));
		}
	}
}
__global__ void cuDslashd(cuDoubleComplex *phi, cuDoubleComplex *r){
	int blockID = (gridDim.x*blockIdx.y)+blockIdx.x;
	int threadID = (blockID*(blockDim.x*blockDim.y))+(threadIdx.y*blockDim.x)+threadIdx.y;
	int thread_count = gridDim.x*gridDim.y*blockDim.x*blockDim.y;

	for(int i=threadId;i<kvol;i+=thread_count){
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			cuDoubleComplex a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval[4][idirac];
			a_2=jqq*gamval[4][idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc+0];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+0]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!

		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];

			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu][idirac] : gamin[mu][idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=
					-akappa*(      u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]
							+u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							-u12t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval[mu][idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]
						     +u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]
						     -conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
						     +u12t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							+u11t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval[mu][idirac]*
					(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]
					 +conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]
					 -conj(u12t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
					 -u11t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];

		for(int igorkov=0; igorkov<4; igorkov++){
			//the FORTRAN code did it.
			int igork1 = gamin[3][igorkov];	
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4m[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));


			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])+\
					u12t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))-\
								 dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])-\
										 u12t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=dk4p[i]*(conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])-\
					conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))-\
								   dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])+
										   u11t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

		}
	}
}
__global__ void cuHdslash(cuDoubleComplex *phi, cuDoubleComplex *r){
	int blockID = (gridDim.x*blockIdx.y)+blockIdx.x;
	int threadID = (blockID*(blockDim.x*blockDim.y))+(threadIdx.y*blockDim.x)+threadIdx.y;
	int thread_count = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	for(int i=threadId;i<kvol;i+=thread_count){
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]-\
						u12t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
								   //Dirac term
								   gamval[mu][idirac]*(u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]+\
										   u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]-\
										   conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]+\
										   u12t[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]+\
						u11t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
								     //Dirac term
								     gamval[mu][idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]+\
										     conj(u11t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]-\
										     conj(u12t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]-\
										     u11t[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);
			}
		}
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3][idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ndirac+idirac)*nc]+=
				-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));
			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4p[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
		}
	}
}
__global__ void cuHdslashd(cuDoubleComplex *phi, cuDoubleComplex *r){
	int blockID = (gridDim.x*blockIdx.y)+blockIdx.x;
	int threadID = (blockID*(blockDim.x*blockDim.y))+(threadIdx.y*blockDim.x)+threadIdx.y;
	int thread_count = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	for(int i=threadId;i<kvol;i+=thread_count){
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				phi[(i*ndirac+idirac)*nc]+=
					-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]
							+u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							-u12t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu][idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]
						     +u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]
						     -conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
						     +u12t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							+u11t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu][idirac]*
					(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]
					 +conj(u11t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]
					 -conj(u12t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
					 -u11t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);
			}
		}
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3][idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
			phi[(i*ndirac+idirac)*nc]+=
				-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4m[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));
		}
	}
}
__global__ void cuForce(double *dSdpi, cuDoubleComplex *X2){
	int blockID = (gridDim.x*blockIdx.y)+blockIdx.x;
	int threadID = (blockID*(blockDim.x*blockDim.y))+(threadIdx.y*blockDim.x)+threadIdx.y;
	int thread_count = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
	//Number of iterations per thread
	for(int i=threadId;i<kvol;i+=thread_count){
		for(int idirac=0;idirac<ndirac;idirac++){
			int mu, uid, igork1;
			//Unrolling the loop
			//Tells the compiler that no vector dependencies exist

			for(mu=0; mu<3; mu++){
				//Long term ambition. I used the diff command on the different
				//spacial components of dSdpi and saw a lot of the values required
				//for them are duplicates (u11(i,mu)*X2(1,idirac,i) is used again with
				//a minus in front for example. Why not evaluate them first /and then plug 
				//them into the equation? Reduce the number of evaluations needed and look
				//a bit neater (although harder to follow as a consequence).

				//Up indices
				uid = iu[mu+ndim*i];
				igork1 = gamin[mu][idirac];	
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
						  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])))
					+creal(I*gamval[idirac][mu]*
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
						  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])))
					+creal(gamval[idirac][mu]*
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
						  +u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])))
					+creal(I*gamval[idirac][mu]*
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
			//We're not done tripping yet!! Time like term is different. dk4? shows up
			//For consistency we'll leave mu in instead of hard coding.
			mu=3;
			uid = iu[mu+ndim*i];
			//We are mutiplying terms by dk4?[i] Also there is no akappa or gamval factor in the time direction	
			//for the "gamval" terms the sign of d4kp flips
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
		}
	}
}
