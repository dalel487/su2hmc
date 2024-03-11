#include <assert.h>
#include <complex.h>
#include <matrices.h>
#include <string.h>
//TO DO: Check and see are there any terms we are evaluating twice in the same loop
//and use a variable to hold them instead to reduce the number of evaluations.
int Dslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t, unsigned int *iu,unsigned int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa){
	/*
	 * Evaluates phi= M*r
	 *
	 * Calls:
	 * ======
	 * ZHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex	*phi:		The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 						for consistency with the FORTRAN code I'll keep the name here
	 * Complex	*r:		The array being acted on by M
	 * Complex	*u11t:	First colour trial field
	 * Complex	*u12t:	Second colour trial field
	 *	int		*iu:		Upper halo indices
	 *	int		*id:		Lower halo indices
	 *	Complex	*gamval:	Gamma matrices
	 *	int		*gamin:	Indices for dirac terms
	 *	double	*dk4m:	
	 *	double	*dk4p:	
	 *	Complex	jqq:		Diquark source
	 *	double	akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslash";
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
			a_1=conj(jqq)*gamval[idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval[idirac];
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
													  //Dirac term
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
	 * Evaluates phi= M^†*r
	 *
	 * Calls:
	 * ======
	 * ZHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex	*phi:		The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 						for consistency with the FORTRAN code I'll keep the name here
	 * Complex	*r:		The array being acted on by M
	 * Complex	*u11t:	First colour trial field
	 * Complex	*u12t:	Second colour trial field
	 *	int		*iu:		Upper halo indices
	 *	int		*id:		Lower halo indices
	 *	Complex	*gamval:	Gamma matrices
	 *	int		*gamin:	Indices for dirac terms
	 *	double	*dk4m:	
	 *	double	*dk4p:	
	 *	Complex	jqq:		Diquark source
	 *	double	akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslashd";
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
			a_1=-conj(jqq)*gamval[idirac];
			a_2=jqq*gamval[idirac];
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
int Hdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned  int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa){
	//int Hdslash(Complex *phi, Complex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Calls:
	 * ======
	 * ZHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex	*phi:		The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 						for consistency with the FORTRAN code I'll keep the name here
	 * Complex	*r:		The array being acted on by M
	 * Complex	*u11t:	First colour trial field
	 * Complex	*u12t:	Second colour trial field
	 *	int		*iu:		Upper halo indices
	 *	int		*id:		Lower halo indices
	 *	Complex	*gamval:	Gamma matrices
	 *	int		*gamin:	Indices for dirac terms
	 *	double	*dk4m:	
	 *	double	*dk4p:	
	 *	Complex	jqq:		Diquark source
	 *	double	akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslash";
	//Get the halos in order
#if(nproc>1)
	ZHalo_swap_all(r, 8);
#endif

	//Mass term
	//Spacelike term
#ifdef __NVCC__
	cuHdslash(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t,u12t,gamval:AVX)
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]-\
						u12t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
													//Dirac term
													gamval[mu*ndirac+idirac]*(u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]+\
															u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]-\
															conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]+\
															u12t[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]+\
						u11t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
													  //Dirac term
													  gamval[mu*ndirac+idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]+\
															  conj(u11t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]-\
															  conj(u12t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]-\
															  u11t[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);
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
			phi[(i*ndirac+idirac)*nc]+=
				-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4p[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
int Hdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned  int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa){
	/*
	 * Evaluates phi= M^†*r
	 *
	 * Calls:
	 * ======
	 * ZHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex	*phi:		The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 						for consistency with the FORTRAN code I'll keep the name here
	 * Complex	*r:		The array being acted on by M
	 * Complex	*u11t:	First colour trial field
	 * Complex	*u12t:	Second colour trial field
	 *	int		*iu:		Upper halo indices
	 *	int		*id:		Lower halo indices
	 *	Complex	*gamval:	Gamma matrices
	 *	int		*gamin:	Indices for dirac terms
	 *	double	*dk4m:	
	 *	double	*dk4p:	
	 *	Complex	jqq:		Diquark source
	 *	double	akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslashd";
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
		Complex_f *gamval_f,	int *gamin,	float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa){
	/*
	 * Evaluates phi= M*r
	 *
	 * Calls:
	 * ======
	 * CHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex_f	*phi:			The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 								for consistency with the FORTRAN code I'll keep the name here
	 * Complex_f	*r:			The array being acted on by M
	 * Complex_f	*u11t_f:		First colour trial field
	 * Complex_f	*u12t_f:		Second colour trial field
	 *	int			*iu:			Upper halo indices
	 *	int			*id:			Lower halo indices
	 *	Complex_f	*gamval_f:	Gamma matrices
	 *	int			*gamin:		Indices for dirac terms
	 *	float			*dk4m_f:	
	 *	float			*dk4p_f:	
	 *	Complex_f	jqq:		Diquark source
	 *	float			akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslash_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
	//Diquark Term (antihermitian)
#ifdef __NVCC__
	cudaMemcpy(phi, r, kferm*sizeof(Complex_f),cudaMemcpyDefault);
	cuDslash_f(phi,r,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex_f));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma omp simd aligned(phi,r,gamval_f:AVX)
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			a_1=conj(jqq)*gamval_f[idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval_f[idirac];
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
													  //Dirac term
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
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,dk4m_f,dk4p_f,gamin:AVX)
		for(int igorkov=0; igorkov<4; igorkov++){
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1 = gamin[3*ndirac+igorkov];	int igork1PP = igork1+4;

			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4p_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m_f[did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4p_f[i]*(-conj(u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m_f[did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));

			//And the +4 terms. Note that dk4p_f and dk4m_f swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4m_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])
					+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk4p_f[did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])
						-u12t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=-dk4m_f[i]*(conj(-u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])
					+conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk4p_f[did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])
						+u11t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
int Dslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval_f, int *gamin, float *dk4m_f, float *dk4p_f, Complex_f jqq, float akappa){
	/*
	 * Evaluates phi= M*r
	 *
	 * Calls:
	 * ======
	 * CHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex_f	*phi:			The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 								for consistency with the FORTRAN code I'll keep the name here
	 * Complex_f	*r:			The array being acted on by M
	 * Complex_f	*u11t_f:		First colour trial field
	 * Complex_f	*u12t_f:		Second colour trial field
	 *	int			*iu:			Upper halo indices
	 *	int			*id:			Lower halo indices
	 *	Complex_f	*gamval_f:	Gamma matrices
	 *	int			*gamin:		Indices for dirac terms
	 *	float			*dk4m_f:	
	 *	float			*dk4p_f:	
	 *	Complex_f	jqq:		Diquark source
	 *	float			akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslashd_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
#ifdef __NVCC__
#ifdef _DEBUG
	int errc=
#endif
		cudaMemcpy(phi, r, kferm*sizeof(Complex_f),cudaMemcpyDefault);
#ifdef _DEBUG
	printf("cudaMemcpy returned %d\n");
#endif
	cuDslashd_f(phi,r,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa,dimGrid,dimBlock);
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
			a_1=-conj(jqq)*gamval_f[idirac];
			a_2=jqq*gamval_f[idirac];
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
		//Under dagger, dk4p_f and dk4m_f get swapped and the dirac component flips sign.
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,dk4m_f,dk4p_f:AVX)
		for(int igorkov=0; igorkov<4; igorkov++){
			//the FORTRAN code did it.
			int igork1 = gamin[3*ndirac+igorkov];	
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4m_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p_f[did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4m_f[i]*(-conj(u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p_f[did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));


			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk4p_f and dk4m_f swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4p_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])
					+u12t_f[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk4m_f[did]*(conj(u11t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])
						-u12t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=dk4p_f[i]*(conj(u12t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])
					-conj(u11t_f[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk4m_f[did]*(conj(u12t_f[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])
						+u11t_f[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

		}
#endif
	}
#endif
	return 0;
}
int Hdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned  int *iu,unsigned  int *id,\
		Complex_f *gamval_f, int *gamin, float *dk4m_f, float *dk4p_f, float akappa){
	/*
	 * Evaluates phi= M*r
	 *
	 * Calls:
	 * ======
	 * CHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex_f	*phi:			The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 								for consistency with the FORTRAN code I'll keep the name here
	 * Complex_f	*r:			The array being acted on by M
	 * Complex_f	*u11t_f:		First colour trial field
	 * Complex_f	*u12t_f:		Second colour trial field
	 *	int			*iu:			Upper halo indices
	 *	int			*id:			Lower halo indices
	 *	Complex_f	*gamval_f:	Gamma matrices
	 *	int			*gamin:		Indices for dirac terms
	 *	float			*dk4m_f:	
	 *	float			*dk4p_f:	
	 *	Complex_f	jqq:		Diquark source
	 *	float			akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslash_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 8);
#endif
	//TODO: Get u11t_f and u12t_f sorted
	//Mass term
	//Spacelike term
#ifdef __NVCC__
	cuHdslash_f(phi,r,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex_f));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,gamval_f:AVX)
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(u11t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]+\
						u12t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conjf(u11t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]-\
						u12t_f[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1]);
				//Dirac term
				phi[(i*ndirac+idirac)*nc]+=gamval_f[mu*ndirac+idirac]*(u11t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]+\
						u12t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]-\
						conjf(u11t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]+\
						u12t_f[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conjf(u12t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]+\
						conjf(u11t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conjf(u12t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]+\
						u11t_f[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1]);
				//Dirac term
				phi[(i*ndirac+idirac)*nc+1]+=gamval_f[mu*ndirac+idirac]*(-conjf(u12t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]+\
						conjf(u11t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]-\
						conjf(u12t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]-\
						u11t_f[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,dk4m_f,dk4p_f:AVX)
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
			//Factorising for performance, we get dk4?*(float)u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ndirac+idirac)*nc]+=
				-dk4p_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m_f[did]*(conjf(u11t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4p_f[i]*(-conjf(u12t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+conjf(u11t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m_f[did]*(conjf(u12t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
int Hdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval_f, int *gamin, float *dk4m_f, float *dk4p_f, float akappa){
	/*
	 * Evaluates phi= M*r
	 *
	 * Calls:
	 * ======
	 * CHalo_swap_all (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 * Complex_f	*phi:			The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 								for consistency with the FORTRAN code I'll keep the name here
	 * Complex_f	*r:			The array being acted on by M
	 * Complex_f	*u11t_f:		First colour trial field
	 * Complex_f	*u12t_f:		Second colour trial field
	 *	int			*iu:			Upper halo indices
	 *	int			*id:			Lower halo indices
	 *	Complex_f	*gamval_f:	Gamma matrices
	 *	int			*gamin:		Indices for dirac terms
	 *	float			*dk4m_f:	
	 *	float			*dk4p_f:	
	 *	Complex_f	jqq:		Diquark source
	 *	float			akappa:	Hopping parameter
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslashd_f";
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
	cuHdslashd_f(phi,r,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex_f));
	//Spacelike term
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,gamval_f:AVX)
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				phi[(i*ndirac+idirac)*nc]+=
					-akappa*(u11t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]
							+u12t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]
							+conjf(u11t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							-u12t_f[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval_f[mu*ndirac+idirac]*
					(          u11t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]
								  +u12t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]
								  -conjf(u11t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
								  +u12t_f[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=
					-akappa*(-conjf(u12t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]
							+conjf(u11t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]
							+conjf(u12t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							+u11t_f[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval_f[mu*ndirac+idirac]*
					(-conjf(u12t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]
					 +conjf(u11t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]
					 -conjf(u12t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
					 -u11t_f[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r,u11t_f,u12t_f,gamval_f:AVX)
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
			//Factorising for performance, we get (float)dk4?*(float)u1?*(+/-r_wilson -/+ r_dirac)
			//(float)dk4m and dk4p_f swap under dagger
			phi[(i*ndirac+idirac)*nc]+=
				-dk4m_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p_f[did]*(conjf(u11t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4m_f[i]*(-conjf(u12t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+conjf(u11t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p_f[did]*(conjf(u12t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
int New_trial(double dt, double *pp, Complex *u11t, Complex *u12t){
	/*
	 * Generates new trial fields
	 *
	 * Calls:
	 * =====
	 *
	 * Parameters:
	 * =========
	 * double	dt:		Half lattice spacing
	 * double *pp:		Momentum field
	 * Complex	*u11t:	First colour field
	 * Complex	*u12t:	Second colour field
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "New_trial"; //
											//#ifdef __clang__
											//Double precision bad for offloading
#ifdef __NVCC__
	cuNew_trial(dt,pp,u11t,u12t,dimGrid,dimBlock);
#else
#pragma omp parallel for simd collapse(2) aligned(pp,u11t,u12t:AVX) 
	for(int i=0;i<kvol;i++)
		for(int mu = 0; mu<ndim; mu++){
			//Sticking to what was in the FORTRAN for variable names.
			//CCC for cosine SSS for sine AAA for...
			//Re-exponentiating the force field. Can be done analytically in SU(2)
			//using sine and cosine which is nice

			double AAA = dt*sqrt(pp[i*nadj*ndim+mu]*pp[i*nadj*ndim+mu]\
					+pp[(i*nadj+1)*ndim+mu]*pp[(i*nadj+1)*ndim+mu]\
					+pp[(i*nadj+2)*ndim+mu]*pp[(i*nadj+2)*ndim+mu]);
			double CCC = cos(AAA);
			double SSS = dt*sin(AAA)/AAA;
			Complex a11 = CCC+I*SSS*pp[(i*nadj+2)*ndim+mu];
			Complex a12 = pp[(i*nadj+1)*ndim+mu]*SSS + I*SSS*pp[i*nadj*ndim+mu];
			//b11 and b12 are u11t and u12t terms, so we'll use u12t directly
			//but use b11 for u11t to prevent RAW dependency
			complex b11 = u11t[i*ndim+mu];
			u11t[i*ndim+mu] = a11*b11-a12*conj(u12t[i*ndim+mu]);
			u12t[i*ndim+mu] = a11*u12t[i*ndim+mu]+a12*conj(b11);
		}
#endif
	return 0;
}
inline int Reunitarise(Complex *u11t, Complex *u12t){
	/*
	 * Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * Globals:
	 * =======
	 * u11t, u12t
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Reunitarise";
#ifdef __NVCC__
	cuReunitarise(u11t,u12t,dimGrid,dimBlock);
#else
#pragma omp parallel for simd aligned(u11t,u12t:AVX)
	for(int i=0; i<kvol*ndim; i++){
		//Declaring anorm inside the loop will hopefully let the compiler know it
		//is safe to vectorise aggressively
		double anorm=sqrt(conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]);
		//		Exception handling code. May be faster to leave out as the exit prevents vectorisation.
		//		if(anorm==0){
		//			fprintf(stderr, "Error %i in %s on rank %i: anorm = 0 for μ=%i and i=%i.\nExiting...\n\n",
		//					DIVZERO, funcname, rank, mu, i);
		//			MPI_Finalise();
		//			exit(DIVZERO);
		//		}
		u11t[i]/=anorm;
		u12t[i]/=anorm;
	}
#endif
	return 0;
}
#ifdef DIAGNOSTIC
int Diagnostics(int istart, Complex *u11, Complex *u12,Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
		unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk4m, double *dk4p,\
		float *dk4m_f, float *dk4p_f, int *gamin, Complex *gamval, Complex_f *gamval_f,\
		Complex_f jqq,float akappa,float beta, double ancg){
	/*
	 * Routine to check if the multiplication routines are working or not
	 * How I hope this will work is that
	 * 1)	Initialise the system
	 * 2) Just after the initialisation of the system but before anything
	 * 	else call this routine using the C Preprocessor.
	 * 3) Give dummy values for the fields and then do work with them
	 * Caveats? Well this could get messy if we call something we didn't
	 * realise was being called and hadn't initialised it properly (Congradq
	 * springs to mind straight away)
	 */
	char *funcname = "Diagnostics";

	//Initialise the arrays being used. Just going to assume MKL is being
	//used here will also assert the number of flavours for now to avoid issues
	//later
	assert(nf==1);
#include<float.h>
	printf("FLT_EVAL_METHOD is %i. Check online for what this means\n", FLT_EVAL_METHOD);

#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	Complex *xi,*R1,*Phi,*X0,*X1;
	Complex_f *X0_f, *X1_f, *xi_f, *R1_f, *Phi_f;
	double *dSdpi,*pp;
	cudaMallocManaged(&R1,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&xi,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&R1_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&xi_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&X0,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X1,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X0_f,kferm2Halo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&X1_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&pp,kmomHalo*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dSdpi,kmom*sizeof(double),cudaMemAttachGlobal);
#else
	Complex *R1= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex *xi= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex_f *R1_f= aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *xi_f= aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex *Phi= aligned_alloc(AVX,nf*kfermHalo*sizeof(Complex)); 
	Complex_f *Phi_f= aligned_alloc(AVX,nf*kfermHalo*sizeof(Complex_f)); 
	Complex *X0= aligned_alloc(AVX,nf*kferm2Halo*sizeof(Complex)); 
	Complex *X1= aligned_alloc(AVX,kferm2Halo*sizeof(Complex)); 
	double *pp = aligned_alloc(AVX,kmomHalo*sizeof(double));
	Complex_f *X0_f= aligned_alloc(AVX,nf*kferm2Halo*sizeof(Complex_f)); 
	Complex_f *X1_f= aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f)); 
	double *dSdpi = aligned_alloc(AVX,kmom*sizeof(double));
#endif
	//pp is the momentum field

#pragma omp parallel sections
	{
#pragma omp section
		{
			FILE *dk4m_File = fopen("dk4m","w");
			for(int i=0;i<kvol;i+=4)
				fprintf(dk4m_File,"%f\t%f\t%f\t%f\n",dk4m[i],dk4m[i+1],dk4m[i+2],dk4m[i+3]);
		}
#pragma omp section
		{
			FILE *dk4p_File = fopen("dk4p","w");
			for(int i=0;i<kvol;i+=4)
				fprintf(dk4p_File,"%f\t%f\t%f\t%f\n",dk4p[i],dk4p[i+1],dk4p[i+2],dk4p[i+3]);
		}
	}
	for(int test = 0; test<=7; test++){
		//Trial fields shouldn't get modified so were previously set up outside
		switch(istart){
			case(1):
#if(nproc>1)
				Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
#endif
#pragma omp parallel sections num_threads(4)
				{
#pragma omp section
					{
						FILE *trial_out = fopen("u11t", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(u11t[i]),cimag(u11t[i]),creal(u11t[i+1]),cimag(u11t[i+1]),
									creal(u11t[2+i]),cimag(u11t[2+i]),creal(u11t[i+3]),cimag(u11t[i+3]));
						fclose(trial_out);
					}
#pragma omp section
					{
						FILE *trial_out = fopen("u12t", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(u12t[i]),cimag(u12t[i]),creal(u12t[i+1]),cimag(u12t[i+1]),
									creal(u12t[2+i]),cimag(u12t[2+i]),creal(u12t[i+3]),cimag(u12t[i+3]));
						fclose(trial_out);
					}
				}
				break;
			case(2):
#pragma omp parallel for simd aligned(u11t,u12t:AVX)
				for(int i =0; i<ndim*kvol; i+=8){
					u11t[i+0]=1+I; u12t[i]=1+I;
					u11t[i+1]=1-I; u12t[i+1]=1+I;
					u11t[i+2]=1+I; u12t[i+2]=1-I;
					u11t[i+3]=1-I; u12t[i+3]=1-I;
					u11t[i+4]=-1+I; u12t[i+4]=1+I;
					u11t[i+5]=1+I; u12t[i+5]=-1+I;
					u11t[i+6]=-1+I; u12t[i+6]=-1+I;
					u11t[i+7]=-1-I; u12t[i+7]=-1-I;
				}
				break;
			default:
				//Cold start as a default
				memcpy(u11t,u11,kvol*ndim*sizeof(Complex));
				memcpy(u12t,u12,kvol*ndim*sizeof(Complex));
				break;
		}
		Reunitarise(u11t,u12t);
		Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
		Gauss_d(pp,kmomHalo,0,1);
		Gauss_z(R1, kferm, 0, 1/sqrt(2));
		Gauss_z(Phi, kferm, 0, 1/sqrt(2));
		Gauss_z(xi, kferm, 0, 1/sqrt(2));
		Gauss_c(R1_f, kferm, 0, 1/sqrt(2));
		Gauss_c(Phi_f, kferm, 0, 1/sqrt(2));
		Gauss_c(xi_f, kferm, 0, 1/sqrt(2));
#else
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R1_f, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, Phi_f, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, xi_f, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R1, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, Phi, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, xi, 0, 1/sqrt(2));
#endif
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
		Gauss_z(X0, kferm2, 0, 1/sqrt(2));
		Gauss_z(X1, kferm2, 0, 1/sqrt(2));
		Gauss_c(X0_f, kferm2, 0, 1/sqrt(2));
		Gauss_c(X1_f, kferm2, 0, 1/sqrt(2));
#else
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X0, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X1, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X0_f, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X1_f, 0, 1/sqrt(2));
#endif

//Random nomalised momentum field
		Gauss_d(dSdpi,kmom,0,1/sqrt(2));
#pragma omp for simd aligned(dSdpi:AVX) nowait
		for(int i=0; i<kmom; i+=4){
			double norm = sqrt(dSdpi[i]*dSdpi[i]+dSdpi[i+1]*dSdpi[i+1]+dSdpi[i+2]*dSdpi[i+2]+dSdpi[i+3]*dSdpi[i+3]);
			dSdpi[i]/=norm; dSdpi[i+1]/=norm; dSdpi[i+2]/=norm;dSdpi[i+3]/=norm;
		}
		FILE *output_old, *output;
		FILE *output_f_old, *output_f;
		switch(test){
			case(0):
#pragma omp parallel for simd
				for(int i = 0; i< kferm; i++){
					R1_f[i]=(Complex_f)R1[i];
					xi_f[i]=(Complex_f)xi[i];
				}
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				output_old = fopen("dslash_old", "w");
				output_f_old = fopen("dslash_f_old", "w");
#ifdef __NVCC__
				cudaMemPrefetchAsync(R1,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(xi,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(R1_f,kferm*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(xi_f,kferm*sizeof(Complex_f),device,NULL);
#endif
				for(int i = 0; i< kferm; i+=8){
					fprintf(output_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]),cimag(R1[i]),creal(R1[i+1]),cimag(R1[i+1]),
							creal(R1[i+2]),cimag(R1[i+2]),creal(R1[i+3]),cimag(R1[i+3]),
							creal(R1[i+4]),cimag(R1[i+4]),creal(R1[i+5]),cimag(R1[i+5]),
							creal(R1[i+6]),cimag(R1[i+6]),creal(R1[i+7]),cimag(R1[i+7])	);
					fprintf(output_f_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1_f[i]),cimag(R1_f[i]),creal(R1_f[i+1]),cimag(R1_f[i+1]),
							creal(R1_f[i+2]),cimag(R1_f[i+2]),creal(R1_f[i+3]),cimag(R1_f[i+3]),
							creal(R1_f[i+4]),cimag(R1_f[i+4]),creal(R1_f[i+5]),cimag(R1_f[i+5]),
							creal(R1_f[i+6]),cimag(R1_f[i+6]),creal(R1_f[i+7]),cimag(R1_f[i+7]));
					printf("Difference in dslash double and float R1[%d] to R1[%d+7]:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]-R1_f[i]),cimag(R1[i]-R1_f[i]),creal(R1[i+1]-R1_f[i+1]),cimag(R1[i+1]-R1_f[i+1]),
							creal(R1[i+2]-R1_f[i+2]),cimag(R1[i+2]-R1_f[i+2]),creal(R1[i+3]-R1_f[i+3]),cimag(R1[i+3]-R1_f[i+3]),
							creal(R1[i+4]-R1_f[i+4]),cimag(R1[i+4]-R1_f[i+4]),creal(R1[i+5]-R1_f[i+5]),cimag(R1[i+5]-R1_f[i+5]),
							creal(R1[i+6]-R1_f[i+6]),cimag(R1[i+6]-R1_f[i+6]),creal(R1[i+7]-R1_f[i+7]),cimag(R1[i+7]-R1_f[i+7]));
				}
				fclose(output_old); fclose(output_f_old);
				Dslash(xi,R1,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
				Dslash_f(xi_f,R1_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
				output = fopen("dslash", "w");
				output_f = fopen("dslash_f", "w");
				for(int i = 0; i< kferm; i+=8){
					fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
					fprintf(output_f, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi_f[i]),cimag(xi_f[i]),creal(xi_f[i+1]),cimag(xi_f[i+1]),
							creal(xi_f[i+2]),cimag(xi_f[i+2]),creal(xi_f[i+3]),cimag(xi_f[i+3]),
							creal(xi_f[i+4]),cimag(xi_f[i+4]),creal(xi_f[i+5]),cimag(xi_f[i+5]),
							creal(xi_f[i+6]),cimag(xi_f[i+6]),creal(xi_f[i+7]),cimag(xi_f[i+7]));
					printf("Difference in dslash double and float xi[%d] to xi[%d+7] after mult:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]-xi_f[i]),cimag(xi[i]-xi_f[i]),creal(xi[i+1]-xi_f[i+1]),cimag(xi[i+1]-xi_f[i+1]),
							creal(xi[i+2]-xi_f[i+2]),cimag(xi[i+2]-xi_f[i+2]),creal(xi[i+3]-xi_f[i+3]),cimag(xi[i+3]-xi_f[i+3]),
							creal(xi[i+4]-xi_f[i+4]),cimag(xi[i+4]-xi_f[i+4]),creal(xi[i+5]-xi_f[i+5]),cimag(xi[i+5]-xi_f[i+5]),
							creal(xi[i+6]-xi_f[i+6]),cimag(xi[i+6]-xi_f[i+6]),creal(xi[i+7]-xi_f[i+7]),cimag(xi[i+7]-xi_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(1):
#pragma omp parallel for simd
				for(int i = 0; i< kferm; i++){
					R1_f[i]=(Complex_f)R1[i];
					xi_f[i]=(Complex_f)xi[i];
				}
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				output_old = fopen("dslashd_old", "w");
				output_f_old = fopen("dslashd_f_old", "w");
#ifdef __NVCC__
				cudaMemPrefetchAsync(R1,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(xi,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(R1_f,kferm*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(xi_f,kferm*sizeof(Complex_f),device,NULL);
#endif
				for(int i = 0; i< kferm; i+=8){
					fprintf(output_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]),cimag(R1[i]),creal(R1[i+1]),cimag(R1[i+1]),
							creal(R1[i+2]),cimag(R1[i+2]),creal(R1[i+3]),cimag(R1[i+3]),
							creal(R1[i+4]),cimag(R1[i+4]),creal(R1[i+5]),cimag(R1[i+5]),
							creal(R1[i+6]),cimag(R1[i+6]),creal(R1[i+7]),cimag(R1[i+7])	);
					fprintf(output_f_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1_f[i]),cimag(R1_f[i]),creal(R1_f[i+1]),cimag(R1_f[i+1]),
							creal(R1_f[i+2]),cimag(R1_f[i+2]),creal(R1_f[i+3]),cimag(R1_f[i+3]),
							creal(R1_f[i+4]),cimag(R1_f[i+4]),creal(R1_f[i+5]),cimag(R1_f[i+5]),
							creal(R1_f[i+6]),cimag(R1_f[i+6]),creal(R1_f[i+7]),cimag(R1_f[i+7]));
					printf("Difference in dslashd double and float R1[%d] to R1[%d+7]:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]-R1_f[i]),cimag(R1[i]-R1_f[i]),creal(R1[i+1]-R1_f[i+1]),cimag(R1[i+1]-R1_f[i+1]),
							creal(R1[i+2]-R1_f[i+2]),cimag(R1[i+2]-R1_f[i+2]),creal(R1[i+3]-R1_f[i+3]),cimag(R1[i+3]-R1_f[i+3]),
							creal(R1[i+4]-R1_f[i+4]),cimag(R1[i+4]-R1_f[i+4]),creal(R1[i+5]-R1_f[i+5]),cimag(R1[i+5]-R1_f[i+5]),
							creal(R1[i+6]-R1_f[i+6]),cimag(R1[i+6]-R1_f[i+6]),creal(R1[i+7]-R1_f[i+7]),cimag(R1[i+7]-R1_f[i+7]));
				}
				fclose(output_old); fclose(output_f_old);
				Dslashd(xi,R1,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
				Dslashd_f(xi_f,R1_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
				output = fopen("dslashd", "w");
				output_f = fopen("dslashd_f", "w");
				for(int i = 0; i< kferm; i+=8){
					fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
					fprintf(output_f, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi_f[i]),cimag(xi_f[i]),creal(xi_f[i+1]),cimag(xi_f[i+1]),
							creal(xi_f[i+2]),cimag(xi_f[i+2]),creal(xi_f[i+3]),cimag(xi_f[i+3]),
							creal(xi_f[i+4]),cimag(xi_f[i+4]),creal(xi_f[i+5]),cimag(xi_f[i+5]),
							creal(xi_f[i+6]),cimag(xi_f[i+6]),creal(xi_f[i+7]),cimag(xi_f[i+7]));
					printf("Difference in dslashd double and float xi[%d] to xi[%d+7] after mult:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]-xi_f[i]),cimag(xi[i]-xi_f[i]),creal(xi[i+1]-xi_f[i+1]),cimag(xi[i+1]-xi_f[i+1]),
							creal(xi[i+2]-xi_f[i+2]),cimag(xi[i+2]-xi_f[i+2]),creal(xi[i+3]-xi_f[i+3]),cimag(xi[i+3]-xi_f[i+3]),
							creal(xi[i+4]-xi_f[i+4]),cimag(xi[i+4]-xi_f[i+4]),creal(xi[i+5]-xi_f[i+5]),cimag(xi[i+5]-xi_f[i+5]),
							creal(xi[i+6]-xi_f[i+6]),cimag(xi[i+6]-xi_f[i+6]),creal(xi[i+7]-xi_f[i+7]),cimag(xi[i+7]-xi_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(2):	
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
#pragma omp parallel for simd
				for(int i = 0; i< kferm2; i++){
					X0_f[i]=(Complex_f)X0[i];
					X1_f[i]=(Complex_f)X1[i];
				}
#ifdef __NVCC__
				cudaMemPrefetchAsync(X0,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X1,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X0_f,kferm2*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(X1_f,kferm2*sizeof(Complex_f),device,NULL);
#endif
				output_old = fopen("hdslash_old", "w");output_f_old = fopen("hdslash_f_old", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output_old, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X0[i]),cimag(X0[i]),creal(X0[i+1]),cimag(X0[i+1]),
							creal(X0[i+2]),cimag(X0[i+2]),creal(X0[i+3]),cimag(X0[i+3]),
							creal(X0[i+4]),cimag(X0[i+4]),creal(X0[i+5]),cimag(X0[i+5]),
							creal(X0[i+6]),cimag(X0[i+6]),creal(X0[i+7]),cimag(X0[i+7]));
					fprintf(output_f_old, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X0_f[i]),cimag(X0_f[i]),creal(X0_f[i+1]),cimag(X0_f[i+1]),
							creal(X0_f[i+2]),cimag(X0_f[i+2]),creal(X0_f[i+3]),cimag(X0_f[i+3]),
							creal(X0_f[i+4]),cimag(X0_f[i+4]),creal(X0_f[i+5]),cimag(X0_f[i+5]),
							creal(X0_f[i+6]),cimag(X0_f[i+6]),creal(X0_f[i+7]),cimag(X0_f[i+7]));
					printf("Difference in hdslash double and float X0[%d] to X0[%d+7]:\n",i,i);
					printf("%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X0[i]-X0_f[i]),cimag(X0[i]-X0_f[i]),creal(X0[i+1]-X0_f[i+1]),cimag(X0[i+1]-X0_f[i+1]),
							creal(X0[i+2]-X0_f[i+2]),cimag(X0[i+2]-X0_f[i+2]),creal(X0[i+3]-X0_f[i+3]),cimag(X0[i+3]-X0_f[i+3]),
							creal(X0[i+4]-X0_f[i+4]),cimag(X0[i+4]-X0_f[i+4]),creal(X0[i+5]-X0_f[i+5]),cimag(X0[i+5]-X0_f[i+5]),
							creal(X0[i+6]-X0_f[i+6]),cimag(X0[i+6]-X0_f[i+6]),creal(X0[i+7]-X0_f[i+7]),cimag(X0[i+7]-X0_f[i+7]));
				}
				fclose(output_old);fclose(output_f_old);
				Hdslash(X1,X0,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
				Hdslash_f(X1_f,X0_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslash", "w");	output_f = fopen("hdslash_f", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
					fprintf(output_f, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1_f[i]),cimag(X1_f[i]),creal(X1_f[i+1]),cimag(X1_f[i+1]),
							creal(X1_f[i+2]),cimag(X1_f[i+2]),creal(X1_f[i+3]),cimag(X1_f[i+3]),
							creal(X1_f[i+4]),cimag(X1_f[i+4]),creal(X1_f[i+5]),cimag(X1_f[i+5]),
							creal(X1_f[i+6]),cimag(X1_f[i+6]),creal(X1_f[i+7]),cimag(X1_f[i+7]));
					printf("Difference in hdslash double and float X1[%d] to X1[%d+7] after mult.:\n",i,i);
					printf("%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1[i]-X1_f[i]),cimag(X1[i]-X1_f[i]),creal(X1[i+1]-X1_f[i+1]),cimag(X1[i+1]-X1_f[i+1]),
							creal(X1[i+2]-X1_f[i+2]),cimag(X1[i+2]-X1_f[i+2]),creal(X1[i+3]-X1_f[i+3]),cimag(X1[i+3]-X1_f[i+3]),
							creal(X1[i+4]-X1_f[i+4]),cimag(X1[i+4]-X1_f[i+4]),creal(X1[i+5]-X1_f[i+5]),cimag(X1[i+5]-X1_f[i+5]),
							creal(X1[i+6]-X1_f[i+6]),cimag(X1[i+6]-X1_f[i+6]),creal(X1[i+7]-X1_f[i+7]),cimag(X1[i+7]-X1_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(3):	
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				for(int i = 0; i< kferm2; i++){
					X0_f[i]=(Complex_f)X0[i];
					X1_f[i]=(Complex_f)X1[i];
				}
#ifdef __NVCC__
				cudaMemPrefetchAsync(X0,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X1,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X0_f,kferm2*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(X1_f,kferm2*sizeof(Complex_f),device,NULL);
#endif
				output_old = fopen("hdslashd_old", "w");output_f_old = fopen("hdslashd_f_old", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output_old, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X0[i]),cimag(X0[i]),creal(X0[i+1]),cimag(X0[i+1]),
							creal(X0[i+2]),cimag(X0[i+2]),creal(X0[i+3]),cimag(X0[i+3]),
							creal(X0[i+4]),cimag(X0[i+4]),creal(X0[i+5]),cimag(X0[i+5]),
							creal(X0[i+6]),cimag(X0[i+6]),creal(X0[i+7]),cimag(X0[i+7]));
					fprintf(output_f_old, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X0_f[i]),cimag(X0_f[i]),creal(X0_f[i+1]),cimag(X0_f[i+1]),
							creal(X0_f[i+2]),cimag(X0_f[i+2]),creal(X0_f[i+3]),cimag(X0_f[i+3]),
							creal(X0_f[i+4]),cimag(X0_f[i+4]),creal(X0_f[i+5]),cimag(X0_f[i+5]),
							creal(X0_f[i+6]),cimag(X0_f[i+6]),creal(X0_f[i+7]),cimag(X0_f[i+7]));
					printf("Difference in hdslashd double and float X0[%d] to X0[%d+7]:\n",i,i);
					printf("%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X0[i]-X0_f[i]),cimag(X0[i]-X0_f[i]),creal(X0[i+1]-X0_f[i+1]),cimag(X0[i+1]-X0_f[i+1]),
							creal(X0[i+2]-X0_f[i+2]),cimag(X0[i+2]-X0_f[i+2]),creal(X0[i+3]-X0_f[i+3]),cimag(X0[i+3]-X0_f[i+3]),
							creal(X0[i+4]-X0_f[i+4]),cimag(X0[i+4]-X0_f[i+4]),creal(X0[i+5]-X0_f[i+5]),cimag(X0[i+5]-X0_f[i+5]),
							creal(X0[i+6]-X0_f[i+6]),cimag(X0[i+6]-X0_f[i+6]),creal(X0[i+7]-X0_f[i+7]),cimag(X0[i+7]-X0_f[i+7]));
				}
				Hdslashd(X1,X0,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
				Hdslashd_f(X1_f,X0_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslashd", "w");	output_f = fopen("hdslashd_f", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
					fprintf(output_f, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1_f[i]),cimag(X1_f[i]),creal(X1_f[i+1]),cimag(X1_f[i+1]),
							creal(X1_f[i+2]),cimag(X1_f[i+2]),creal(X1_f[i+3]),cimag(X1_f[i+3]),
							creal(X1_f[i+4]),cimag(X1_f[i+4]),creal(X1_f[i+5]),cimag(X1_f[i+5]),
							creal(X1_f[i+6]),cimag(X1_f[i+6]),creal(X1_f[i+7]),cimag(X1_f[i+7]));
					printf("Difference in hdslashd double and float X1[%d] to X1[%d+7] after mult.:\n",i,i);
					printf("%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1[i]-X1_f[i]),cimag(X1[i]-X1_f[i]),creal(X1[i+1]-X1_f[i+1]),cimag(X1[i+1]-X1_f[i+1]),
							creal(X1[i+2]-X1_f[i+2]),cimag(X1[i+2]-X1_f[i+2]),creal(X1[i+3]-X1_f[i+3]),cimag(X1[i+3]-X1_f[i+3]),
							creal(X1[i+4]-X1_f[i+4]),cimag(X1[i+4]-X1_f[i+4]),creal(X1[i+5]-X1_f[i+5]),cimag(X1[i+5]-X1_f[i+5]),
							creal(X1[i+6]-X1_f[i+6]),cimag(X1[i+6]-X1_f[i+6]),creal(X1[i+7]-X1_f[i+7]),cimag(X1[i+7]-X1_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(4):	
				output_old = fopen("hamiltonian_old", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output_old, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				}
				fclose(output_old);
				output = fopen("hamiltonian", "w");
				double h,s,ancgh;  h=s=ancgh=0;
				Hamilton(&h,&s,rescgg,pp,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,\
						akappa,beta,&ancgh);
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output, "%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n%.9f+%.9fI\t%.9f+%.9fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				}
				fclose(output);
				break;
				//Two force cases because of the flag
			case(6):	
				output_old = fopen("force_0_old", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output_old);
				Force(dSdpi, 1, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
						dk4m_f,dk4p_f,jqq,akappa,beta,&ancg);
				output = fopen("force_0", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
			case(7):	
				output_old = fopen("force_1_old", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output_old);
				output = fopen("force_1", "w");
				Force(dSdpi, 0, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
						dk4m_f,dk4p_f,jqq,akappa,beta,&ancg);
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
				/*			case(6):
							output = fopen("Measure", "w");
							int itercg=0;
							double pbp, endenf, denf; complex qq, qbqb;
							Measure(&pbp, &endenf, &denf, &qq, &qbqb, respbp, &itercg);
							fprintf(output,"pbp=%.5f\tendenf=%.5f\tdenf=%.5f\nqq=%.5f+(%.5f)i\tqbqb=%.5f+(%.5f)i\titercg=%i\n\n",
							pbp,endenf,denf,creal(qq),cimag(qq),creal(qbqb),cimag(qbqb),itercg);
				//				Congradp(0,respbp,&itercg);
				for(int i = 0; i< kferm; i+=8)
				fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
				creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
				creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
				creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
				creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
				fclose(output);
				break;
				*/
			case(5):
				output_old = fopen("Gauge_Force_old","w");
				for(int i = 0; i< kmom; i+=12){
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i+4], dSdpi[i+5], dSdpi[i+6], dSdpi[i+7]);
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n\n", dSdpi[i+8], dSdpi[i+9], dSdpi[i+10], dSdpi[i+11]);
				}
				fclose(output_old);	
#ifdef __NVCC__
				cudaMemPrefetchAsync(dSdpi,kmom*sizeof(double),device,NULL);
#endif
				Gauge_force(dSdpi,u11t_f,u12t_f,iu,id,beta);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("Gauge_Force","w");
				for(int i = 0; i< kmom; i+=12){
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i+4], dSdpi[i+5], dSdpi[i+6], dSdpi[i+7]);
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n\n", dSdpi[i+8], dSdpi[i+9], dSdpi[i+10], dSdpi[i+11]);
				}
				fclose(output);	
				break;

		}
	}
	//George Michael's favourite bit of the code
#ifdef __NVCC__
	//Make a routine that does this for us
	cudaFree(dk4m); cudaFree(dk4p); cudaFree(R1); cudaFree(dSdpi); cudaFree(pp);
	cudaFree(Phi); cudaFree(u11t); cudaFree(u12t);
	cudaFree(X0); cudaFree(X1); cudaFree(u11); cudaFree(u12);
	cudaFree(X0_f); cudaFree(X1_f); cudaFree(u11t_f); cudaFree(u12t_f);
	cudaFree(id); cudaFree(iu); cudaFree(hd); cudaFree(hu);
#else
	free(dk4m); free(dk4p); free(R1); free(dSdpi); free(pp);
	free(Phi); free(u11t); free(u12t); free(xi);
	free(X0); free(X1); free(u11); free(u12);
	free(id); free(iu); free(hd); free(hu);
	free(pcoord);
#endif

#if(nproc>1)
	MPI_Finalise();
#endif
	exit(0);
}
#endif
