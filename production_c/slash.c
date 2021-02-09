#include <slash.h>
#include <string.h>
//TO DO: Check and see are there any terms we are evaluating twice in the same loop
//and use a variable to hold them instead to reduce the number of evaluations.
int Dslash(complex phi[][ngorkov][nc], complex r[][ngorkov][nc]){
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
	for(int mu=0;mu<ndim;mu++){
		complex z[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
		cblas_zcopy(kvol, &u11t[0][mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u11t[0][mu], ndim);
#else
		for(int i=0; i<kvol;i++)
			u11t[i][mu]=z[i];
#endif
#ifdef USE_MKL
		cblas_zcopy(kvol, &u12t[0][mu], 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u12t[0][mu], 4);
#else
		for(int i=0; i<kvol;i++)
			u12t[i][mu]=z[i];
#endif
	}
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Mass term
	memcpy(phi, r, kferm*sizeof(complex));
	//Diquark Term (antihermitian)
#pragma omp parallel for 
	for(int i=0;i<kvol;i++){
#pragma unroll
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			complex a_1, a_2;
			a_1=conj(jqq)*gamval[4][idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval[4][idirac];
			phi[i][idirac][0]+=a_1*r[i][igork][0];
			phi[i][idirac][1]+=a_1*r[i][igork][1];
			phi[i][igork][0]+=a_2*r[i][idirac][0];
			phi[i][igork][1]+=a_2*r[i][idirac][1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
		for(int mu = 0; mu <3; mu++){
			int did=id[mu][i]; int uid = iu[mu][i];
#pragma unroll
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu][idirac] : gamin[mu][idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[i][igorkov][0]+=-akappa*(u11t[i][mu]*r[uid][igorkov][0]+\
						u12t[i][mu]*r[uid][igorkov][1]+\
						conj(u11t[did][mu])*r[did][igorkov][0]-\
						u12t[did][mu]*r[did][igorkov][1])+\
							  //Dirac term
							  gamval[mu][idirac]*(u11t[i][mu]*r[uid][igork1][0]+\
									  u12t[i][mu]*r[uid][igork1][1]-\
									  conj(u11t[did][mu])*r[did][igork1][0]+\
									  u12t[did][mu]*r[did][igork1][1]);

				phi[i][igorkov][1]+=-akappa*(-conj(u12t[i][mu])*r[uid][igorkov][0]+\
						conj(u11t[i][mu])*r[uid][igorkov][1]+\
						conj(u12t[did][mu])*r[did][igorkov][0]+\
						u11t[did][mu]*r[did][igorkov][1])+\
							  //Dirac term
							  gamval[mu][idirac]*(-conj(u12t[i][mu])*r[uid][igork1][0]+\
									  conj(u11t[i][mu])*r[uid][igork1][1]-\
									  conj(u12t[did][mu])*r[did][igork1][0]-\
									  u11t[did][mu]*r[did][igork1][1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		int did=id[3][i]; int uid = iu[3][i];
#pragma unroll
		for(int igorkov=0; igorkov<4; igorkov++){
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			//the FORTRAN code did it.
			int igork1 = gamin[3][igorkov];	int igork1PP = igork1+4;

			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[i][igorkov][0]+=
				-dk4p[i]*(u11t[i][3]*(r[uid][igorkov][0]-r[uid][igork1][0])
						+u12t[i][3]*(r[uid][igorkov][1]-r[uid][igork1][1]))
				-dk4m[did]*(conj(u11t[did][3])*(r[did][igorkov][0]+r[did][igork1][0])
						-u12t[did][3] *(r[did][igorkov][1]+r[did][igork1][1]));
			phi[i][igorkov][1]+=
				-dk4p[i]*(-conj(u12t[i][3])*(r[uid][igorkov][0]+r[uid][igork1][0])
						+conj(u11t[i][3])*(r[uid][igorkov][1]+r[uid][igork1][1]))
				-dk4m[did]*(conj(u12t[did][3])*(r[did][igorkov][0]-r[did][igork1][0])
						+u11t[did][3] *(r[did][igorkov][1]-r[did][igork1][1]));

			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[i][igorkovPP][0]+=-dk4m[i]*(u11t[i][3]*(r[uid][igorkovPP][0]-r[uid][igork1PP][0])+\
					u12t[i][3]*(r[uid][igorkovPP][1]-r[uid][igork1PP][1]))-\
						    dk4p[did]*(conj(u11t[did][3])*(r[did][igorkovPP][0]+r[did][igork1PP][0])-\
								    u12t[did][3]*(r[did][igorkovPP][1]+r[did][igork1PP][1]));

			phi[i][igorkovPP][1]+=-dk4m[i]*(conj(-u12t[i][3])*(r[uid][igorkovPP][0]-r[uid][igork1PP][0])+\
					conj(u11t[i][3])*(r[uid][igorkovPP][1]-r[uid][igork1PP][1]))-\
						    dk4p[did]*(conj(u12t[did][3])*(r[did][igorkovPP][0]+r[did][igork1PP][0])+\
								    u11t[did][3]*(r[did][igorkovPP][1]+r[did][igork1PP][1]));
		}
	}
	return 0;
}
int Dslashd(complex phi[][ngorkov][nc], complex r[][ngorkov][nc]){
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
	for(int mu=0;mu<ndim;mu++){
		complex z[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
		cblas_zcopy(kvol, &u11t[0][mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u11t[0][mu], ndim);
#else
		for(int i=0; i<kvol;i++)
			u11t[i][mu]=z[i];
#endif
#ifdef USE_MKL
		cblas_zcopy(kvol, &u12t[0][mu], 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u12t[0][mu], 4);
#else
		for(int i=0; i<kvol;i++)
			u12t[i][mu]=z[i];
#endif
	}
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Mass term
	memcpy(phi, r, kferm*sizeof(complex));
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#pragma unroll
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			complex a_1, a_2;
			a_1=-conj(jqq)*gamval[4][idirac];
			//We subtract a_2, hence the minus
			a_2=jqq*gamval[4][idirac];
			phi[i][idirac][0]+=a_1*r[i][igork][0];
			phi[i][idirac][1]+=a_1*r[i][igork][1];
			phi[i][igork][0]+=a_2*r[i][idirac][0];
			phi[i][igork][1]+=a_2*r[i][idirac][1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
		for(int mu = 0; mu <3; mu++){
			int did=id[mu][i]; int uid = iu[mu][i];
#pragma unroll
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu][idirac] : gamin[mu][idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[i][igorkov][0]+=-akappa*(u11t[i][mu]*r[uid][igorkov][0]+\
						u12t[i][mu]*r[uid][igorkov][1]+\
						conj(u11t[did][mu])*r[did][igorkov][0]-\
						u12t[did][mu]*r[did][igorkov][1])-\
							  //Dirac term. Sign flips under dagger
							  gamval[mu][idirac]*(u11t[i][mu]*r[uid][igork1][0]+\
									  u12t[i][mu]*r[uid][igork1][1]-\
									  conj(u11t[did][mu])*r[did][igork1][0]+\
									  u12t[did][mu]*r[did][igork1][1]);

				phi[i][igorkov][1]+=-akappa*(-conj(u12t[i][mu])*r[uid][igorkov][0]+\
						conj(u11t[i][mu])*r[uid][igorkov][1]+\
						conj(u12t[did][mu])*r[did][igorkov][0]+\
						u11t[did][mu]*r[did][igorkov][1])-\
							  //Dirac term. Sign flips under dagger
							  gamval[mu][idirac]*(-conj(u12t[i][mu])*r[uid][igork1][0]+\
									  conj(u11t[i][mu])*r[uid][igork1][1]-\
									  conj(u12t[did][mu])*r[did][igork1][0]-\
									  u11t[did][mu]*r[did][igork1][1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
		int did=id[3][i]; int uid = iu[3][i];
#pragma unroll
		for(int igorkov=0; igorkov<4; igorkov++){
			//the FORTRAN code did it.
			int igork1 = gamin[3][igorkov];	
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[i][igorkov][0]+=
				-dk4m[i]*(u11t[i][3]*(r[uid][igorkov][0]+r[uid][igork1][0])
						+u12t[i][3]*(r[uid][igorkov][1]+r[uid][igork1][1]))
				-dk4p[did]*(conj(u11t[did][3])*(r[did][igorkov][0]-r[did][igork1][0])
						-u12t[did][3] *(r[did][igorkov][1]-r[did][igork1][1]));
			phi[i][igorkov][1]+=
				-dk4m[i]*(-conj(u12t[i][3])*(r[uid][igorkov][0]+r[uid][igork1][0])
						+conj(u11t[i][3])*(r[uid][igorkov][1]+r[uid][igork1][1]))
				-dk4p[did]*(conj(u12t[did][3])*(r[did][igorkov][0]-r[did][igork1][0])
						+u11t[did][3] *(r[did][igorkov][1]-r[did][igork1][1]));


			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[i][igorkovPP][0]+=-dk4p[i]*(u11t[i][3]*(r[uid][igorkovPP][0]+r[uid][igork1PP][0])+\
					u12t[i][3]*(r[uid][igorkovPP][1]+r[uid][igork1PP][1]))-\
						    dk4m[did]*(conj(u11t[did][3])*(r[did][igorkovPP][0]-r[did][igork1PP][0])-\
								    u12t[did][3]*(r[did][igorkovPP][1]-r[did][igork1PP][1]));

			phi[i][igorkovPP][1]+=dk4p[i]*(conj(u12t[i][3])*(r[uid][igorkovPP][0]+r[uid][igork1PP][0])-\
					conj(u11t[i][3])*(r[uid][igorkovPP][1]+r[uid][igork1PP][1]))-\
						    dk4m[did]*(conj(u12t[did][3])*(r[did][igorkovPP][0]-r[did][igork1PP][0])+								    u11t[did][3]*(r[did][igorkovPP][1]-r[did][igork1PP][1]));
		}
	}
	return 0;
}
int Hdslash(complex phi[][ndirac][nc], complex r[][ndirac][nc]){
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
	for(int mu=0;mu<ndim;mu++){
		complex z[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
		cblas_zcopy(kvol, &u11t[0][mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u11t[0][mu], ndim);
#else
		for(int i=0; i<kvol+halo;i++)
			u11t[i][mu]=z[i];
#endif
#ifdef USE_MKL
		cblas_zcopy(kvol, &u12t[0][mu], 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u12t[0][mu], 4);
#else
		for(int i=0; i<kvol+halo;i++)
			u12t[i][mu]=z[i];
#endif
	}
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Mass term
	memcpy(phi, r, kferm2*sizeof(complex));
	//Spacelike term
#pragma omp parallel for private(akappa)
	for(int i=0;i<kvol;i++){
		for(int mu = 0; mu <3; mu++){
			int did=id[mu][i]; int uid = iu[mu][i];
#pragma unroll
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[i][idirac][0]+=-akappa*(u11t[i][mu]*r[uid][idirac][0]+\
						u12t[i][mu]*r[uid][idirac][1]+\
						conj(u11t[did][mu])*r[did][idirac][0]-\
						u12t[did][mu]*r[did][idirac][1])+\
							 //Dirac term
							 gamval[mu][idirac]*(u11t[i][mu]*r[uid][igork1][0]+\
									 u12t[i][mu]*r[uid][igork1][1]-\
									 conj(u11t[did][mu])*r[did][igork1][0]+\
									 u12t[did][mu]*r[did][igork1][1]);

				phi[i][idirac][1]+=-akappa*(-conj(u12t[i][mu])*r[uid][idirac][0]+\
						conj(u11t[i][mu])*r[uid][idirac][1]+\
						conj(u12t[did][mu])*r[did][idirac][0]+\
						u11t[did][mu]*r[did][idirac][1])+\
							 //Dirac term
							 gamval[mu][idirac]*(-conj(u12t[i][mu])*r[uid][igork1][0]+\
									 conj(u11t[i][mu])*r[uid][igork1][1]-\
									 conj(u12t[did][mu])*r[did][igork1][0]-\
									 u11t[did][mu]*r[did][igork1][1]);
			}
		}
		//Timelike terms
		int did=id[3][i]; int uid = iu[3][i];
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3][idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[i][idirac][0]+=
				-dk4p[i]*(u11t[i][3]*(r[uid][idirac][0]-r[uid][igork1][0])
						+u12t[i][3]*(r[uid][idirac][1]-r[uid][igork1][1]))
				-dk4m[did]*(conj(u11t[did][3])*(r[did][idirac][0]+r[did][igork1][0])
						-u12t[did][3] *(r[did][idirac][1]-r[did][igork1][1]));
			phi[i][idirac][1]+=
				-dk4p[i]*(-conj(u12t[i][3])*(r[uid][idirac][0]-r[uid][igork1][0])
						+conj(u11t[i][3])*(r[uid][idirac][1]-r[uid][igork1][1]))
				-dk4m[did]*(conj(u12t[did][3])*(r[did][idirac][0]+r[did][igork1][0])
						+u11t[did][3] *(r[did][idirac][1]+r[did][igork1][1]));
		}
	}
	return 0;
}
int Hdslashd(complex phi[][ndirac][nc], complex r[][ndirac][nc]){
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
	for(int mu=0;mu<ndim;mu++){
		complex z[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
		cblas_zcopy(kvol, &u11t[0][mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
		//And the swap back
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u11t[0][mu], ndim);
#else
		for(int i=0; i<kvol+halo;i++)
			u11t[i][mu]=z[i];
#endif
#ifdef USE_MKL
		cblas_zcopy(kvol, &u12t[0][mu], 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i][mu];
#endif
		ZHalo_swap_dir(z, 1, mu, UP);
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u12t[0][mu], 4);
#else
		for(int i=0; i<kvol+halo;i++)
			u12t[i][mu]=z[i];
#endif
	}
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);

	//Looks like flipping the array ordering for C has meant a lot
	//of for loops. Sense we're jumping around quite a bit the cache is probably getting refreshed
	//anyways so memory access patterns mightn't be as big of an limiting factor here anyway

	//Mass term
	memcpy(phi, r, kferm2*sizeof(complex));
	//Spacelike term
#pragma omp parallel for private(akappa)
	for(int i=0;i<kvol;i++){
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu][i]; int uid = iu[mu][i];
#pragma unroll
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[i][idirac][0]+=-akappa*(u11t[i][mu]*r[uid][idirac][0]+\
						u12t[i][mu]*r[uid][idirac][1]+\
						conj(u11t[did][mu])*r[did][idirac][0]-\
						u12t[did][mu]*r[did][idirac][1])-\
							 //Dirac term. Subtract under dagger
							 gamval[mu][idirac]*(u11t[i][mu]*r[uid][igork1][0]+\
									 u12t[i][mu]*r[uid][igork1][1]-\
									 conj(u11t[did][mu])*r[did][igork1][0]+\
									 u12t[did][mu]*r[did][igork1][1]);

				phi[i][idirac][1]+=-akappa*(-conj(u12t[i][mu])*r[uid][idirac][0]+\
						conj(u11t[i][mu])*r[uid][idirac][1]+\
						conj(u12t[did][mu])*r[did][idirac][0]+\
						u11t[did][mu]*r[did][idirac][1])-\
							 //Dirac term. Subtract under dagger
							 gamval[mu][idirac]*(-conj(u12t[i][mu])*r[uid][igork1][0]+\
									 conj(u11t[i][mu])*r[uid][igork1][1]-\
									 conj(u12t[did][mu])*r[did][igork1][0]-\
									 u11t[did][mu]*r[did][igork1][1]);
			}
		}
		//Timelike terms
		int did=id[3][i]; int uid = iu[3][i];
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3][idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
			phi[i][idirac][0]+=
				-dk4m[i]*(u11t[i][3]*(r[uid][idirac][0]+r[uid][igork1][0])
						+u12t[i][3]*(r[uid][idirac][1]+r[uid][igork1][1]))
				-dk4p[did]*(conj(u11t[did][3])*(r[did][idirac][0]-r[did][igork1][0])
						-u12t[did][3] *(r[did][idirac][1]+r[did][igork1][1]));

			phi[i][idirac][1]+=
				-dk4m[i]*(-conj(u12t[i][3])*(r[uid][idirac][0]+r[uid][igork1][0])
						+conj(u11t[i][3])*(r[uid][idirac][1]+r[uid][igork1][1]))
				-dk4p[did]*(conj(u12t[did][3])*(r[did][idirac][0]-r[did][igork1][0])
						+u11t[did][3] *(r[did][idirac][1]-r[did][igork1][1]));
		}
	}
	return 0;
}
