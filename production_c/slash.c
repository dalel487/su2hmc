#include <slash.h>
#include <string.h>
//TO DO: Check and see are there any terms we are evaluating twice in the same loop
//and use a variable to hold them instead to reduce the number of evaluations.
int Dslash(complex *phi, complex *r){
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
	 * complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslash";
	//Get the halos in order
	ZHalo_swap_all(r, 16);
#ifdef USE_MKL
	complex *z = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
#else
	complex *z = malloc((kvol+halo)*sizeof(complex));
#endif
	for(int mu=0;mu<ndim;mu++){
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, u11t+mu, ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, u11t+mu, ndim);
#else
		for(int i=0; i<kvol;i++)
			u11t[i*ndim+mu]=z[i];
#endif
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, u12t+mu, 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, u12t+mu, 4);
#else
		for(int i=0; i<kvol;i++)
			u12t[i*ndim+mu]=z[i];
#endif
	}
#ifdef USE_MKL
	mkl_free(z);
#else
	free(z);
#endif
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Mass term
	memcpy(phi, r, kferm*sizeof(complex));
	//Diquark Term (antihermitian)
#pragma omp parallel for 
	for(int i=0;i<kvol;i++){
#pragma ivdep
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			complex a_1, a_2;
			a_1=conj(jqq)*gamval[4][idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval[4][idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc+0];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+0]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#pragma ivdep 
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma ivdep 
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
#pragma ivdep
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
	return 0;
}
int Dslashd(complex *phi, complex *r){
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
	 * complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Dslashd";
	//Get the halos in order
	ZHalo_swap_all(r, 16);
#ifdef USE_MKL
	complex *z = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
#else
	complex *z = malloc((kvol+halo)*sizeof(complex));
#endif
	for(int mu=0;mu<ndim;mu++){
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, u11t+mu, ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, u11t+mu, ndim);
#else
		for(int i=0; i<kvol;i++)
			u11t[i*ndim+mu]=z[i];
#endif
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, u12t+mu, 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, u12t+mu, 4);
#else
		for(int i=0; i<kvol;i++)
			u12t[i*ndim+mu]=z[i];
#endif
	}
#ifdef USE_MKL
	mkl_free(z);
#else
	free(z);
#endif
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Mass term
	memcpy(phi, r, kferm*sizeof(complex));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma ivdep
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			complex a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval[4][idirac];
			a_2=jqq*gamval[4][idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc+0];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+0]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#pragma ivdep
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma ivdep 
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
#pragma ivdep
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
	return 0;
}
int Hdslash(complex *phi, complex *r){
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
	 * complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslash";
	//Get the halos in order
	ZHalo_swap_all(r, 8);
#ifdef USE_MKL
	complex *z = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
#else
	complex *z = malloc((kvol+halo)*sizeof(complex));
#endif
	for(int mu=0;mu<ndim;mu++){
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, &u11t[mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, &u11t[mu], ndim);
#else
		for(int i=0; i<kvol+halo;i++)
			u11t[i*ndim+mu]=z[i];
#endif
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, &u12t[mu], 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, &u12t[mu], 4);
#else
		for(int i=0; i<kvol+halo;i++)
			u12t[i*ndim+mu]=z[i];
#endif
	}
#ifdef USE_MKL
	mkl_free(z);
#else
	free(z);
#endif
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Mass term
	memcpy(phi, r, kferm2*sizeof(complex));
	//Spacelike term
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma ivdep 
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma ivdep 
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
#pragma ivdep
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
	return 0;
}
int Hdslashd(complex *phi, complex *r){
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
	 * complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Hdslashd";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
	ZHalo_swap_all(r, 8);
#ifdef USE_MKL
	complex *z = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
#else
	complex *z = malloc((kvol+halo)*sizeof(complex));
#endif
	for(int mu=0;mu<ndim;mu++){
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, &u11t[mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, &u11t[mu], ndim);
#else
		for(int i=0; i<kvol+halo;i++)
			u11t[i*ndim+mu]=z[i];
#endif
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, &u12t[mu], 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i*ndim+mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, &u12t[mu], 4);
#else
		for(int i=0; i<kvol+halo;i++)
			u12t[i*ndim+mu]=z[i];
#endif
	}
#ifdef USE_MKL
	mkl_free(z);
#else
	free(z);
#endif
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Looks like flipping the array ordering for C has meant a lot
	//of for loops. Sense we're jumping around quite a bit the cache is probably getting refreshed
	//anyways so memory access patterns mightn't be as big of an limiting factor here anyway

	//Mass term
	memcpy(phi, r, kferm2*sizeof(complex));
	//Spacelike term
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma ivdep
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
#pragma ivdep 
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
#pragma ivdep 
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
	return 0;
}
