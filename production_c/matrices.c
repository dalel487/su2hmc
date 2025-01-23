/**
 * @file matrices.c
 *
 * @brief Matrix multiplication and related routines
 *
 * There are two four matrix mutiplication routines, and each had a double and single (_f) version
 * The Hdslash? routines are called when acting on half of the fermions (up/down flavour partitioning)
 * The Dslash routines act on everything
 *
 * Any routine ending in a d is the daggered multiplication
 */
#include <assert.h>
#include <complex.h>
#include <matrices.h>
#include <string.h>
#include <stdalign.h>
//TO DO: Check and see are there any terms we are evaluating twice in the same loop
//and use a variable to hold them instead to reduce the number of evaluations.
int Dslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t, unsigned int *iu,unsigned int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M r@f) in double precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk4m:		
	 *	@param	dk4p:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @see ZHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslash";
	//Get the halos in order
#if(nproc>1)
	ZHalo_swap_all(r, 16);
#endif

	//Mass term
	//Diquark Term (antihermitian)
#ifdef __NVCC__
	cuDslash(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma omp simd aligned(phi,r,gamval:AVX)
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			a_1=conj(jqq)*gamval[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc+0];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+0]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t,u12t,gamval:AVX)
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu*ndirac+idirac] : gamin[mu*ndirac+idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]-\
						u12t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
													  //Dirac term. Reminder! gamval was rescaled by kappa when we defined it
													  gamval[mu*ndirac+idirac]*(u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]+\
															  u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]-\
															  conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]+\
															  u12t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]+\
						u11t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
														 //Dirac term
														 gamval[mu*ndirac+idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]+\
																 conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]-\
																 conj(u12t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]-\
																 u11t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t,u12t,dk4m,dk4p:AVX)
		for(int igorkov=0; igorkov<4; igorkov++){
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1 = gamin[3*ndirac+igorkov];	int igork1PP = igork1+4;

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
#endif
	}
#endif
	return 0;
}
int Dslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in double precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk4m:		
	 *	@param	dk4p:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @see ZHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslashd";
	//Get the halos in order
#if(nproc>1)
	ZHalo_swap_all(r, 16);
#endif

	//Mass term
#ifdef __NVCC__
	cuDslashd(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma omp simd aligned(phi,r,gamval:AVX)
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval[4*ndirac+idirac];
			a_2=jqq*gamval[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t,u12t,gamval:AVX)
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu*ndirac+idirac] : gamin[mu*ndirac+idirac]+4;
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//Reminder! gamval was rescaled by kappa when we defined it
				phi[(i*ngorkov+igorkov)*nc]+=
					-akappa*(      u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]
							+u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							-u12t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval[mu*ndirac+idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]
								  +u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]
								  -conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
								  +u12t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							+u11t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval[mu*ndirac+idirac]*
					(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]
					 +conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]
					 -conj(u12t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
					 -u11t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t,u12t,dk4m,dk4p:AVX)
		for(int igorkov=0; igorkov<4; igorkov++){
			//the FORTRAN code did it.
			int igork1 = gamin[3*ndirac+igorkov];	
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
#endif
	}
#endif
	return 0;
}
int Hdslash(Complex *phi, Complex *r, Complex *ut[2],unsigned  int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk[2], float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M r@f) in double precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	ut[0]:	First colour trial field
	 * @param	ut[1]:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk[0]:	
	 *	@param	dk[1]:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @see ZHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslash";
	//Get the halos in order
#if(nproc>1)
	ZHalo_swap_all(r, 8);
#endif

	//Mass term
	//Spacelike term
#ifdef __NVCC__
	cuHdslash(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,gamval:AVX)
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(ut[0][i*ndim+mu]*r[(uid*ndirac+idirac)*nc]+\
						ut[1][i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conj(ut[0][did*ndim+mu])*r[(did*ndirac+idirac)*nc]-\
						ut[1][did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
													//Dirac term
													gamval[mu*ndirac+idirac]*(ut[0][i*ndim+mu]*r[(uid*ndirac+igork1)*nc]+\
															ut[1][i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]-\
															conj(ut[0][did*ndim+mu])*r[(did*ndirac+igork1)*nc]+\
															ut[1][did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conj(ut[1][i*ndim+mu])*r[(uid*ndirac+idirac)*nc]+\
						conj(ut[0][i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conj(ut[1][did*ndim+mu])*r[(did*ndirac+idirac)*nc]+\
						ut[0][did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
													  //Dirac term
													  gamval[mu*ndirac+idirac]*(-conj(ut[1][i*ndim+mu])*r[(uid*ndirac+igork1)*nc]+\
															  conj(ut[0][i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]-\
															  conj(ut[1][did*ndim+mu])*r[(did*ndirac+igork1)*nc]-\
															  ut[0][did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r:AVX)
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//Reminder! gamval was rescaled by kappa when we defined it
			phi[(i*ndirac+idirac)*nc]+=
				-dk[1][i]*(ut[0][i*ndim+3]*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+ut[1][i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk[0][did]*(conj(ut[0][did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						-ut[1][did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
			phi[(i*ndirac+idirac)*nc+1]+=
				-dk[1][i]*(-conj(ut[1][i*ndim+3])*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+conj(ut[0][i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk[0][did]*(conj(ut[1][did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						+ut[0][did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
int Hdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned  int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in double precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk4m:	
	 *	@param	dk4p:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @see ZHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslashd";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
#if(nproc>1)
	ZHalo_swap_all(r, 8);
#endif

	//Looks like flipping the array ordering for C has meant a lot
	//of for loops. Sense we're jumping around quite a bit the cache is probably getting refreshed
	//anyways so memory access patterns mightn't be as big of an limiting factor here anyway

	//Mass term
#ifdef __NVCC__
	cuHdslashd(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex));
	//Spacelike term
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t,u12t,gamval:AVX)
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				//Reminder! gamval was rescaled by kappa when we defined it
				phi[(i*ndirac+idirac)*nc]+=
					-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]
							+u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							-u12t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu*ndirac+idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]
								  +u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]
								  -conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
								  +u12t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							+u11t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu*ndirac+idirac]*
					(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]
					 +conj(u11t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]
					 -conj(u12t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
					 -u11t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t,u12t,dk4m,dk4p:AVX)
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
			phi[(i*ndirac+idirac)*nc]+=
				-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4m[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
//Float Versions
//int Dslash_f(Complex_f *phi, Complex_f *r){
int Dslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_f,	int *gamin,	float *dk_f[2], Complex_f jqq, float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M r@f) in single precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t_f:		First colour trial field
	 * @param	u12t_f:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval_f:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk_f[0]:		
	 *	@param	dk_f[1]:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @see CHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslash_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
	//Diquark Term (antihermitian)
#ifdef __NVCC__
	cuDslash_f(phi,r,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex_f));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma omp simd aligned(phi,r,gamval_f:AVX)
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			a_1=conj(jqq)*gamval_f[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval_f[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc+0];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+0]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,gamval_f,gamin:AVX)
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu*ndirac+idirac] : gamin[mu*ndirac+idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=-akappa*(u11t_f[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]+\
						u12t_f[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u11t_f[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]-\
						u12t_f[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
													  //Dirac term. Reminder! gamval was rescaled by kappa when we defined it
													  gamval_f[mu*ndirac+idirac]*(u11t_f[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]+\
															  u12t_f[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]-\
															  conj(u11t_f[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]+\
															  u12t_f[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=-akappa*(-conj(u12t_f[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]+\
						conj(u11t_f[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u12t_f[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]+\
						u11t_f[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
														 //Dirac term
														 gamval_f[mu*ndirac+idirac]*(-conj(u12t_f[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]+\
																 conj(u11t_f[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]-\
																 conj(u12t_f[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]-\
																 u11t_f[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,gamin:AVX)
		for(int igorkov=0; igorkov<4; igorkov++){
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1 = gamin[3*ndirac+igorkov];	int igork1PP = igork1+4;

			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk_f[1][i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk_f[0][did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk_f[1][i]*(-conj(u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk_f[0][did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));

			//And the +4 terms. Note that dk_f[1] and dk_f[0] swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk_f[0][i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])
					+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk_f[1][did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])
						-u12t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=-dk_f[0][i]*(conj(-u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])
					+conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk_f[1][did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])
						+u11t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
int Dslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval_f, int *gamin, float *dk_f[2], Complex_f jqq, float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in single precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t_f:		First colour trial field
	 * @param	u12t_f:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval_f:	Gamma matrices
	 *	@param	gamin:		Indices for dirac terms
	 *	@param	dk_f[0]:		
	 *	@param	dk_f[1]:		
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @see CHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslashd_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
#ifdef __NVCC__
	cuDslashd_f(phi,r,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex_f));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma omp simd aligned(phi,r,gamval_f:AVX)
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval_f[4*ndirac+idirac];
			a_2=jqq*gamval_f[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,gamval_f:AVX)
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu*ndirac+idirac] : gamin[mu*ndirac+idirac]+4;
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//Reminder! gamval was rescaled by kappa when we defined it
				phi[(i*ngorkov+igorkov)*nc]+=
					-akappa*(      u11t_f[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]
							+u12t_f[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u11t_f[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							-u12t_f[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval_f[mu*ndirac+idirac]*
					(          u11t_f[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]
								  +u12t_f[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]
								  -conj(u11t_f[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
								  +u12t_f[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=
					-akappa*(-conj(u12t_f[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]
							+conj(u11t_f[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u12t_f[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							+u11t_f[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval_f[mu*ndirac+idirac]*
					(-conj(u12t_f[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]
					 +conj(u11t_f[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]
					 -conj(u12t_f[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
					 -u11t_f[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk_f[1] and dk_f[0] get swapped and the dirac component flips sign.
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t_f,u12t_f:AVX)
		for(int igorkov=0; igorkov<4; igorkov++){
			//the FORTRAN code did it.
			int igork1 = gamin[3*ndirac+igorkov];	
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk_f[0][i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk_f[1][did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk_f[0][i]*(-conj(u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk_f[1][did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));


			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk_f[1] and dk_f[0] swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk_f[1][i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])
					+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk_f[0][did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])
						-u12t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=dk_f[1][i]*(conj(u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])
					-conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk_f[0][did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])
						+u11t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

		}
#endif
	}
#endif
	return 0;
}
int Hdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned  int *iu,unsigned  int *id,\
		Complex_f *gamval, int *gamin, float *dk[2], float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M r@f) in single precision.
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	ut[0]_f:	First colour trial field
	 * @param	ut[1]_f:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval_f:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk4m_f:	
	 *	@param	dk4p_f:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @see CHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslash_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 8);
#endif
#ifdef __NVCC__
	cuHdslash_f(phi,r,ut,iu,id,gamval,gamin,dk,akappa,dimGrid,dimBlock);
#else
	//Mass term
	memcpy(phi, r, kferm2*sizeof(Complex_f));
#pragma omp parallel for
	for(int i=0;i<kvol;i+=AVX){
		alignas(AVX) Complex_f u11s[AVX];	 alignas(AVX) Complex_f u12s[AVX];
		alignas(AVX) Complex_f u11sd[AVX];	 alignas(AVX) Complex_f u12sd[AVX];
		alignas(AVX) Complex_f ru[2][AVX];   alignas(AVX) Complex_f rd[2][AVX];
		alignas(AVX) Complex_f rgu[2][AVX];  alignas(AVX) Complex_f rgd[2][AVX];
		alignas(AVX) Complex_f phi_s[ndirac*nc][AVX];
		//Do we need to sync threads if each thread only accesses the value it put in shared memory?
#pragma unroll(2)
		for(int idirac=0; idirac<ndirac; idirac++)
			for(int c=0; c<nc; c++)
#pragma omp simd aligned(phi_s,phi:AVX)
				for(int j=0;j<AVX;j++)
					phi_s[idirac*nc+c][j]=phi[((i+j)*ndirac+idirac)*nc+c];
		alignas(AVX) int did[AVX], uid[AVX];
#pragma unroll
		for(int mu = 0; mu <3; mu++){
#pragma omp simd aligned(u11s,u12s,did,uid,id,iu,u11sd,u12sd:AVX)
			for(int j =0;j<AVX;j++){
				did[j]=id[(i+j)*ndim+mu]; uid[j] = iu[(i+j)*ndim+mu];
				u11s[j]=ut[0][(i+j)*ndim+mu];	u12s[j]=ut[1][(i+j)*ndim+mu];
				u11sd[j]=ut[0][did[j]*ndim+mu];	u12sd[j]=ut[1][did[j]*ndim+mu];
			}
#pragma unroll
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin[mu*ndirac+idirac];
#pragma unroll
				for(int c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
					for(int j =0;j<AVX;j++){
						ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
						rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
						rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
						rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
					}
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd:AVX)
				for(int j =0;j<AVX;j++){
					phi_s[idirac*nc][j]+=-akappa*(u11s[j]*ru[0][j]+\
							u12s[j]*ru[1][j]+\
							conj(u11sd[j])*rd[0][j]-\
							u12sd[j]*rd[1][j]);
					//Dirac term
					phi_s[idirac*nc][j]+=gamval[mu*ndirac+idirac]*(u11s[j]*rgu[0][j]+\
							u12s[j]*rgu[1][j]-\
							conj(u11sd[j])*rgd[0][j]+\
							u12sd[j]*rgd[1][j]);

					phi_s[idirac*nc+1][j]+=-akappa*(-conj(u12s[j])*ru[0][j]+\
							conj(u11s[j])*ru[1][j]+\
							conj(u12sd[j])*rd[0][j]+\
							u11sd[j]*rd[1][j]);
					//Dirac term
					phi_s[idirac*nc+1][j]+=gamval[mu*ndirac+idirac]*(-conj(u12s[j])*rgu[0][j]+\
							conj(u11s[j])*rgu[1][j]-\
							conj(u12sd[j])*rgd[0][j]-\
							u11sd[j]*rgd[1][j]);
				}
			}
		}
#ifndef NO_TIME
		//Timelike terms
		alignas(AVX) float dk4ms[AVX],dk4ps[AVX];
#pragma omp simd
		for(int j=0;j<AVX;j++){
			u11s[j]=ut[0][(i+j)*ndim+3];	u12s[j]=ut[1][(i+j)*ndim+3];
			did[j]=id[(i+j)*ndim+3];uid[j]= iu[(i+j)*ndim+3];
			u11sd[j]=ut[0][did[j]*ndim+3];	u12sd[j]=ut[1][did[j]*ndim+3];
			dk4ms[j]=dk[0][did[j]];   dk4ps[j]=dk[1][i+j];
		}

#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
#pragma unroll
			for(int c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
				for(int j =0;j<AVX;j++){
					ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
					rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
					rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
					rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
				}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd,dk4ms,dk4ps,phi:AVX)
			for(int j =0;j<AVX;j++){
				phi_s[idirac*nc+0][j]-=
					dk4ps[j]*(u11s[j]*(ru[0][j]-rgu[0][j])
							+u12s[j]*(ru[1][j]-rgu[1][j]));
				phi_s[idirac*nc+0][j]-=
					dk4ms[j]*(conj(u11sd[j])*(rd[0][j]+rgd[0][j])
							-u12sd[j]*(rd[1][j]+rgd[1][j]));
				phi[((i+j)*ndirac+idirac)*nc]=phi_s[idirac*nc][j];

				phi_s[idirac*nc+1][j]-=
					dk4ps[j]*(-conj(u12s[j])*(ru[0][j]-rgu[0][j])
							+conj(u11s[j])*(ru[1][j]-rgu[1][j]));
				phi_s[idirac*nc+1][j]-=
					dk4ms[j]*(conj(u12sd[j])*(rd[0][j]+rgd[0][j])
							+u11sd[j]*(rd[1][j]+rgd[1][j]));
				phi[((i+j)*ndirac+idirac)*nc+1]=phi_s[idirac*nc+1][j];
			}
		}
#endif
	}
#endif
	return 0;
}
int Hdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu,unsigned int *id,\
		Complex_f *gamval, int *gamin, float *dk[2], float akappa){
	/*
	 * @brief Evaluates @f(\Phi=M^\dagger r@f) in single precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	ut[0]_f:	First colour trial field
	 * @param	ut[1]_f:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval_f:	Gamma matrices
	 *	@param	gamin:	Indices for dirac terms
	 *	@param	dk4m_f:	
	 *	@param	dk4p_f:	
	 *	@param	akappa:	Hopping parameter
	 *
	 * @see CHalo_swap_all (MPI only)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslashd_f";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
#if(nproc>1)
	CHalo_swap_all(r, 8);
#endif

	//Looks like flipping the array ordering for C has meant a lot
	//of for loops. Sense we're jumping around quite a bit the cache is probably getting refreshed
	//anyways so memory access patterns mightn't be as big of an limiting factor here anyway

	//Mass term
#ifdef __NVCC__
	cuHdslashd_f(phi,r,ut,iu,id,gamval,gamin,dk,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex_f));

	//Spacelike term
	//Enough room on L1 data cache for Zen 2 to hold 160 elements at a time
	//Vectorise with 128 maybe?
#pragma omp parallel for
	for(int i=0;i<kvol;i+=AVX){
		//Right. Time to prefetch
		alignas(AVX) Complex_f u11s[AVX];		alignas(AVX) Complex_f u12s[AVX];
		alignas(AVX) Complex_f u11sd[AVX];		alignas(AVX) Complex_f u12sd[AVX];
		alignas(AVX) Complex_f ru[2][AVX]; 		alignas(AVX) Complex_f rd[2][AVX];
		alignas(AVX) Complex_f rgu[2][AVX];		alignas(AVX) Complex_f rgd[2][AVX];
		alignas(AVX) Complex_f phi_s[ndirac*nc][AVX];
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++)
#pragma unroll
			for(int c=0; c<nc; c++)
#pragma omp simd aligned(phi_s,phi:AVX)
				for(int j=0;j<AVX;j++)
					phi_s[idirac*nc+c][j]=phi[((i+j)*ndirac+idirac)*nc+c];
		alignas(AVX) int did[AVX], uid[AVX];
#ifndef NO_SPACE
#pragma unroll
		for(int mu = 0; mu <ndim-1; mu++){
			//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
#pragma omp simd aligned(u11s,u12s,did,uid,id,iu,u11sd,u12sd:AVX)
			for(int j =0;j<AVX;j++){
				did[j]=id[(i+j)*ndim+mu]; uid[j] = iu[(i+j)*ndim+mu];
				u11s[j]=ut[0][(i+j)*ndim+mu];	u12s[j]=ut[1][(i+j)*ndim+mu];
				u11sd[j]=ut[0][did[j]*ndim+mu];	u12sd[j]=ut[1][did[j]*ndim+mu];
			}
#pragma unroll
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin[mu*ndirac+idirac];
#pragma unroll
				for(int c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
					for(int j =0;j<AVX;j++){
						ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
						rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
						rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
						rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
					}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd:AVX)
				for(int j =0;j<AVX;j++){
					phi_s[idirac*nc][j]-=akappa*(u11s[j]*ru[0][j]
							+u12s[j]*ru[1][j]
							+conj(u11sd[j])*rd[0][j]
							-u12sd[j] *rd[1][j]);
					//Dirac term
					phi_s[idirac*nc][j]-=gamval[mu*ndirac+idirac]*
						(u11s[j]*rgu[0][j]
						 +u12s[j]*rgu[1][j]
						 -conj(u11sd[j])*rgd[0][j]
						 +u12sd[j] *rgd[1][j]);

					phi_s[idirac*nc+1][j]-=akappa*(-conj(u12s[j])*ru[0][j]
							+conj(u11s[j])*ru[1][j]
							+conj(u12sd[j])*rd[0][j]
							+u11sd[j] *rd[1][j]);
					//Dirac term
					phi_s[idirac*nc+1][j]-=gamval[mu*ndirac+idirac]*(-conj(u12s[j])*rgu[0][j]
							+conj(u11s[j])*rgu[1][j]
							-conj(u12sd[j])*rgd[0][j]
							-u11sd[j] *rgd[1][j]);
				}
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		alignas(AVX) float dk4ms[AVX],dk4ps[AVX];
#pragma omp simd aligned(u11s,u12s,did,uid,id,iu,u11sd,u12sd,dk4ms,dk4ps:AVX)
		for(int j=0;j<AVX;j++){
			u11s[j]=ut[0][(i+j)*ndim+3];	u12s[j]=ut[1][(i+j)*ndim+3];
			did[j]=id[(i+j)*ndim+3];		uid[j]= iu[(i+j)*ndim+3];
			u11sd[j]=ut[0][did[j]*ndim+3];	u12sd[j]=ut[1][did[j]*ndim+3];
			dk4ms[j]=dk[0][i+j];   			dk4ps[j]=dk[1][did[j]];
		}
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
#pragma unroll
			for(int c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
				for(int j =0;j<AVX;j++){
					ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
					rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
					rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
					rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
				}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd,dk4ms,dk4ps,phi:AVX)
			for(int j =0;j<AVX;j++){
				phi_s[idirac*nc][j]+=
					-dk4ms[j]*(u11s[j]*(ru[0][j]+rgu[0][j])
							+u12s[j]*(ru[1][j]+rgu[1][j]));
				phi_s[idirac*nc][j]+=
					-dk4ps[j]*(conj(u11sd[j])*(rd[0][j]-rgd[0][j])
							-u12sd[j] *(rd[1][j]-rgd[1][j]));
				phi[((i+j)*ndirac+idirac)*nc]=phi_s[idirac*nc][j];

				phi_s[idirac*nc+1][j]-=
					dk4ms[j]*(-conj(u12s[j])*(ru[0][j]+rgu[0][j])
							+conj(u11s[j])*(ru[1][j]+rgu[1][j]));
				phi_s[idirac*nc+1][j]-=
					+dk4ps[j]*(conj(u12sd[j])*(rd[0][j]-rgd[0][j])
							+u11sd[j] *(rd[1][j]-rgd[1][j]));
				phi[((i+j)*ndirac+idirac)*nc+1]=phi_s[idirac*nc+1][j];
			}
		}
#endif
	}
#endif
	return 0;
}
inline int Reunitarise(Complex *ut[2]){
	/*
	 * @brief Reunitarises ut[0] and ut[1] as in conj(ut[0][i])*ut[0][i]+conj(ut[1][i])*ut[1][i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with ut[0] and ut[1].
	 *
	 * @see cuReunitarise (CUDA Wrapper)
	 *
	 * @param ut[0], ut[1] Trial fields to be reunitarised
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Reunitarise";
#ifdef __NVCC__
	cuReunitarise(ut[0],ut[1],dimGrid,dimBlock);
#else
#pragma omp parallel for simd
	for(int i=0; i<kvol*ndim; i++){
		//Declaring anorm inside the loop will hopefully let the compiler know it
		//is safe to vectorise aggressively
		double anorm=sqrt(conj(ut[0][i])*ut[0][i]+conj(ut[1][i])*ut[1][i]);
		//		Exception handling code. May be faster to leave out as the exit prevents vectorisation.
		//		if(anorm==0){
		//			fprintf(stderr, "Error %i in %s on rank %i: anorm = 0 for μ=%i and i=%i.\nExiting...\n\n",
		//					DIVZERO, funcname, rank, mu, i);
		//			MPI_Finalise();
		//			exit(DIVZERO);
		//		}
		ut[0][i]/=anorm;
		ut[1][i]/=anorm;
	}
#endif
	return 0;
}


inline void Transpose_c(Complex_f *out, const int fast_in, const int fast_out){
	const volatile char *funcname="Transpose_c";

#ifdef __NVCC__
	cuTranspose_c(out,fast_in,fast_out,dimGrid,dimBlock);
#else
	Complex_f *in = (Complex_f *)aligned_alloc(AVX,fast_in*fast_out*sizeof(Complex_f));
	memcpy(in,out,fast_in*fast_out*sizeof(Complex_f));
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=0;x<fast_out;x++)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=0;y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	free(in);
#endif
}
inline void Transpose_z(Complex *out, const int fast_in, const int fast_out){
	const volatile char *funcname="Transpose_c";

#ifdef __NVCC__
	cuTranspose_z(out,fast_in,fast_out,dimGrid,dimBlock);
#else
	Complex *in = (Complex *)aligned_alloc(AVX,fast_in*fast_out*sizeof(Complex));
	memcpy(in,out,fast_in*fast_out*sizeof(Complex));
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=0;x<fast_out;x++)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=0;y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	free(in);
#endif
}
inline void Transpose_f(float *out, const int fast_in, const int fast_out){
	const char *funcname="Transpose_f";

#ifdef __NVCC__
	cuTranspose_f(out,fast_in,fast_out,dimGrid,dimBlock);
#else
	float *in = (float *)aligned_alloc(AVX,fast_in*fast_out*sizeof(float));
	memcpy(in,out,fast_in*fast_out*sizeof(float));
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=0;x<fast_out;x++)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=0;y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	free(in);
#endif
}
inline void Transpose_d(double *out, const int fast_in, const int fast_out){
	const char *funcname="Transpose_f";

#ifdef __NVCC__
	cuTranspose_d(out,fast_in,fast_out,dimGrid,dimBlock);
#else
	double *in = (double *)aligned_alloc(AVX,fast_in*fast_out*sizeof(double));
	memcpy(in,out,fast_in*fast_out*sizeof(double));
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=0;x<fast_out;x++)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=0;y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	free(in);
#endif
}
inline void Transpose_I(int *out, const int fast_in, const int fast_out){
	const char *funcname="Transpose_I";

#ifdef __NVCC__
	cuTranspose_I(out,fast_in,fast_out,dimGrid,dimBlock);
#else
	int *in = (int *)aligned_alloc(AVX,fast_in*fast_out*sizeof(int));
	memcpy(in,out,fast_in*fast_out*sizeof(int));
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=0;x<fast_out;x++)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=0;y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	free(in);
#endif
}
inline void Transpose_U(unsigned int *out, const int fast_in, const int fast_out){
	const char *funcname="Transpose_I";

#ifdef __NVCC__
	cuTranspose_U(out,fast_in,fast_out,dimGrid,dimBlock);
#else
	unsigned int *in = (unsigned int *)aligned_alloc(AVX,fast_in*fast_out*sizeof(unsigned int));
	memcpy(in,out,fast_in*fast_out*sizeof(unsigned int));
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(unsigned int x=0;x<fast_out;x++)
			for(unsigned int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(unsigned int x=0; x<fast_out;x++)
			for(unsigned int y=0;y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	free(in);
#endif
}
