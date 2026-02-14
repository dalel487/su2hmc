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
#include <matrices.h>
//TODO: Check and see are there any terms we are evaluating twice in the same loop
//and use a variable to hold them instead to reduce the number of evaluations.
int Dslash(Complex *phi, Complex *r, Complex *ut[2], unsigned int *iu,unsigned int *id,\
		Complex gamval[20], const unsigned short gamin[16], double *dk[2], Complex_f jqq, float akappa){
	const char *funcname = "Dslash";
	//Get the halos in order
#if(nproc>1)
	ZHalo_swap_all(r, 16);
#endif

	//Mass term
	//Diquark Term (antihermitian)
#ifdef __NVCC__
	cuDslash(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex));
#pragma omp parallel for
	for(unsigned int i=0;i<kvol;i++){
		Complex ru[nc]; Complex rd[nc];
		Complex rgu[nc]; Complex rgd[nc];
		Complex phi_s[ngorkov*nc];
		for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
			unsigned short igork = ((idirac>>1)+4)<<1;
			unsigned int ind_d =4*ndirac+(idirac>>1);
			Complex a_1=conj(jqq)*gamval[ind_d];
			//We subtract a_2, hence the minus
			Complex a_2=-jqq*gamval[ind_d];
			ind_d=i+kvolHalo*(idirac); unsigned int ind_g=i+kvolHalo*(igork);
			phi_s[idirac]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork]=phi[ind_g]+a_2*r[ind_d];
			ind_d+=kvol; ind_g+=kvol;
			phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
		}
		Complex u11s;	Complex u12s;
		Complex u11sd; Complex u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvolHalo*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			u11s=ut[0][ind]; u12s=ut[1][ind];
			ind = did+kvolHalo*mu;
			u11sd=ut[0][ind]; u12sd=ut[1][ind];
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				unsigned short idirac=igorkov&3;		
				unsigned short gind=mu*ndirac+idirac;
				const Complex gam=gamval[gind];
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				unsigned short igork1 = (igorkov<4) ? gamin[gind] : gamin[gind]+4;
				for(unsigned short c=0;c<nc;c++){
					ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
					rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
				}
				//Wilson + Dirac term in that order. Definitely easier
				phi_s[igorkov*nc]+=-akappa*(u11s*ru[0]+ u12s*ru[1]+\
						conj(u11sd)*rd[0]- u12sd*rd[1]);
				//Dirac term
				phi_s[igorkov*nc]+=gam*(u11s*rgu[0]+ u12s*rgu[1]-\
						conj(u11sd)*rgd[0]+ u12sd*rgd[1]);

				phi_s[igorkov*nc+1]+=-akappa*(-conj(u12s)*ru[0]+ conj(u11s)*ru[1]+\
						conj(u12sd)*rd[0]+ u11sd*rd[1]);
				//Dirac term
				phi_s[igorkov*nc+1]+=gam*(-conj(u12s)*rgu[0]+ conj(u11s)*rgu[1]-\
						conj(u12sd)*rgd[0]- u11sd*rgd[1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
#ifndef NO_TIME
		ind=i+kvolHalo*3;
		u11s=ut[0][ind]; u12s=ut[1][ind];
		const double dk4ms=dk[0][i];	const double dk4ps=dk[1][i];
		const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
		ind=did+kvolHalo*3;
		u11sd=ut[0][ind]; u12sd=ut[1][ind];
		const double dk4msd=dk[0][did];	const double dk4psd=dk[1][did];
		for(unsigned short igorkov=0;igorkov<ndirac;igorkov++){
			unsigned short igork1 = gamin[3*ndirac+igorkov];
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi_s[igorkov*nc]+=
				-dk4ps*(u11s*(ru[0]-rgu[0]) +u12s*(ru[1]-rgu[1]))
				-dk4msd*(conj(u11sd)*(rd[0]+rgd[0]) -u12sd *(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkov*nc)]=phi_s[igorkov*nc];

			phi_s[igorkov*nc+1]+=
				-dk4ps*(-conj(u12s)*(ru[0]-rgu[0]) +conj(u11s)*(ru[1]-rgu[1]))
				-dk4msd*(conj(u12sd)*(rd[0]+rgd[0]) +u11sd *(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkov*nc+1)]=phi_s[igorkov*nc+1];
			const unsigned short igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
																		//the FORTRAN code did it.
			igork1 += 4;
			//And the gorkov terms. Note that dk4p and dk4m swap positions compared to the above				
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkovPP*nc+c)]; rd[c]=r[did+kvolHalo*(igorkovPP*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//And the Gor'kov terms. Note that dk4p and dk4m swap positions compared to the above				
			phi_s[igorkovPP*nc]+=-dk4ms*(u11s*(ru[0]-rgu[0])+ u12s*(ru[1]-rgu[1]))-
				dk4psd*(conj(u11sd)*(rd[0]+rgd[0])- u12sd*(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc)]=phi_s[igorkovPP*nc];

			phi_s[igorkovPP*nc+1]+=-dk4ms*(conj(-u12s)*(ru[0]-rgu[0]) +conj(u11s)*(ru[1]-rgu[1]))
				-dk4psd*(conj(u12sd)*(rd[0]+rgd[0]) +u11sd*(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc+1)]=phi_s[igorkovPP*nc+1];
		}
#endif
	}
#endif
	return 0;
}
int Dslashd(Complex *phi, Complex *r, Complex *ut[2],unsigned int *iu,unsigned int *id,\
		Complex gamval[20], const unsigned short gamin[16], double *dk[2],Complex_f jqq, float akappa){
	const char *funcname = "Dslashd";
	//Get the halos in order
#if(nproc>1)
	ZHalo_swap_all(r, 16);
#endif

	//Mass term
#ifdef __NVCC__
	cuDslashd(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex));
#pragma omp parallel for
	for(unsigned int i=0;i<kvol;i++){
		Complex ru[nc];  Complex rd[nc];
		Complex rgu[nc];  Complex rgd[nc];
		Complex phi_s[ngorkov*nc];
		#pragma omp simd
		for(unsigned short idirac = 0; idirac<ndirac; idirac++){
			unsigned short igork = idirac+4;
			//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
			//We subtract a_1, hence the minus
			Complex a_1=-conj(jqq)*gamval[4*ndirac+idirac];
			Complex a_2=jqq*gamval[4*ndirac+idirac];
			phi_s[idirac*nc]=phi[i+kvolHalo*(idirac*nc)]+a_1*r[i+kvolHalo*(igork*nc)];
			phi_s[igork*nc]=phi[i+kvolHalo*(igork*nc)]+a_2*r[i+kvolHalo*(idirac*nc)];
			phi_s[idirac*nc+1]=phi[i+kvolHalo*(idirac*nc+1)]+a_1*r[i+kvolHalo*(igork*nc+1)];
			phi_s[igork*nc+1]=phi[i+kvolHalo*(igork*nc+1)]+a_2*r[i+kvolHalo*(idirac*nc+1)];
		}
		Complex u11s;	 Complex u12s;
		Complex u11sd;	 Complex u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvolHalo*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			u11s=ut[0][ind]; u12s=ut[1][ind];
			ind = did+kvolHalo*mu;
			u11sd=ut[0][ind]; u12sd=ut[1][ind];
			#pragma omp simd
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				unsigned short idirac=igorkov&3;		
				const Complex gam=gamval[mu*ndirac+idirac];
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				unsigned short igork1 = (igorkov<4) ? gamin[mu*ndirac+idirac] : gamin[mu*ndirac+idirac]+4;
				for(unsigned short c=0;c<nc;c++){
					ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
					rgd[c]=r[did+kvolHalo*(igork1*nc+c)]; rgu[c]=r[uid+kvolHalo*(igork1*nc+c)];
				}
				//Wilson + Dirac term in that order. Definitely easier
				phi_s[igorkov*nc]-= akappa*(u11s*ru[0] +u12s*ru[1]
						+conj(u11sd)*rd[0] -u12sd *rd[1]);

				//Dirac term
				phi_s[igorkov*nc]-=gam* (u11s*rgu[0] +u12s*rgu[1]
						-conj(u11sd)*rgd[0] +u12sd *rgd[1]);

				phi_s[igorkov*nc+1]-= akappa*(-conj(u12s)*ru[0] +conj(u11s)*ru[1]
						+conj(u12sd)*rd[0] +u11sd *rd[1]);
				//Dirac term
				phi_s[igorkov*nc+1]-=gam* (-conj(u12s)*rgu[0] +conj(u11s)*rgu[1]
						-conj(u12sd)*rgd[0] -u11sd *rgd[1]);

			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
#ifndef NO_TIME
		ind=i+kvolHalo*3;
		u11s=ut[0][ind]; u12s=ut[1][ind];
		const double dk4ms=dk[0][i];	const double dk4ps=dk[1][i];
		const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
		ind=did+kvolHalo*3;
		u11sd=ut[0][ind]; u12sd=ut[1][ind];
		const double dk4msd=dk[0][did];	const double dk4psd=dk[1][did];
		#pragma omp simd
		for(unsigned short igorkov=0; igorkov<ndirac; igorkov++){
			unsigned short igork1 = gamin[3*ndirac+igorkov];	
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi_s[igorkov*nc]+=
				-dk4ms*(u11s*(ru[0]+rgu[0]) +u12s*(ru[1]+rgu[1]))
				-dk4psd*(conj(u11sd)*(rd[0]-rgd[0]) -u12sd *(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkov*nc)]=phi_s[igorkov*nc];

			phi_s[igorkov*nc+1]+=
				-dk4ms*(-conj(u12s)*(ru[0]+rgu[0]) +conj(u11s)*(ru[1]+rgu[1]))
				-dk4psd*(conj(u12sd)*(rd[0]-rgd[0]) +u11sd *(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkov*nc+1)]=phi_s[igorkov*nc+1];
			const unsigned short igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
																		//the FORTRAN code did it.
			igork1 += 4;
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkovPP*nc+c)]; rd[c]=r[did+kvolHalo*(igorkovPP*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//And the Gor'kov terms. Note that dk4p and dk4m swap positions compared to the above				
			phi_s[igorkovPP*nc]+=-dk4ps*(u11s*(ru[0]+rgu[0]) +u12s*(ru[1]+rgu[1]))
				-dk4msd*(conj(u11sd)*(rd[0]-rgd[0]) -u12sd*(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc)]=phi_s[igorkovPP*nc];

			phi_s[igorkovPP*nc+1]+=dk4ps*(conj(u12s)*(ru[0]+rgu[0]) -conj(u11s)*(ru[1]+rgu[1]))
				-dk4msd*(conj(u12sd)*(rd[0]-rgd[0]) +u11sd*(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc+1)]=phi_s[igorkovPP*nc+1];
		}
#endif
	}
#endif
	return 0;
}
int Hdslash(Complex *phi, Complex *r, Complex *ut[2],unsigned  int *iu,unsigned  int *id,\
		Complex gamval[20], const unsigned short gamin[16], double *dk[2], float akappa){
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
			int did=id[mu*kvol+i]; int uid = iu[mu*kvol+i];
#pragma omp simd aligned(phi,r,gamval:AVX)
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(ut[0][i+kvol*mu]*r[(uid*ndirac+idirac)*nc]+\
						ut[1][i+kvol*mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conj(ut[0][did+kvol*mu])*r[(did*ndirac+idirac)*nc]-\
						ut[1][did+kvol*mu]*r[(did*ndirac+idirac)*nc+1])+\
													//Dirac term
													gamval[mu*ndirac+idirac]*(ut[0][i+kvol*mu]*r[(uid*ndirac+igork1)*nc]+\
															ut[1][i+kvol*mu]*r[(uid*ndirac+igork1)*nc+1]-\
															conj(ut[0][did+kvol*mu])*r[(did*ndirac+igork1)*nc]+\
															ut[1][did+kvol*mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conj(ut[1][i+kvol*mu])*r[(uid*ndirac+idirac)*nc]+\
						conj(ut[0][i+kvol*mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conj(ut[1][did+kvol*mu])*r[(did*ndirac+idirac)*nc]+\
						ut[0][did+kvol*mu]*r[(did*ndirac+idirac)*nc+1])+\
													  //Dirac term
													  gamval[mu*ndirac+idirac]*(-conj(ut[1][i+kvol*mu])*r[(uid*ndirac+igork1)*nc]+\
															  conj(ut[0][i+kvol*mu])*r[(uid*ndirac+igork1)*nc+1]-\
															  conj(ut[1][did+kvol*mu])*r[(did*ndirac+igork1)*nc]-\
															  ut[0][did+kvol*mu]*r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3*kvol+i]; int uid = iu[3*kvol+i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r:AVX)
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//Reminder! gamval was rescaled by kappa when we defined it
			phi[(i*ndirac+idirac)*nc]+=
				-dk[1][i]*(ut[0][i+kvol*3]*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+ut[1][i+kvol*3]*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk[0][did]*(conj(ut[0][did+kvol*3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						-ut[1][did+kvol*3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
			phi[(i*ndirac+idirac)*nc+1]+=
				-dk[1][i]*(-conj(ut[1][i+kvol*3])*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+conj(ut[0][i+kvol*3])*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk[0][did]*(conj(ut[1][did+kvol*3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						+ut[0][did+kvol*3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
int Hdslashd(Complex *phi, Complex *r, Complex *ut[2],unsigned  int *iu,unsigned  int *id,\
		Complex gamval[20], const unsigned short gamin[16], double *dk[2], float akappa){
	const char *funcname = "Hdslashd";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
#if(nproc>1)
	ZHalo_swap_all(r, 8);
#endif

	//Mass term
#ifdef __NVCC__
	cuHdslashd(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex));
	//Spacelike term
#pragma omp parallel for
	for(int i=0;i<kvol;i++){
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu*kvol+i]; int uid = iu[mu*kvol+i];
#pragma omp simd aligned(phi,r,gamval:AVX)
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				//Reminder! gamval was rescaled by kappa when we defined it
				phi[(i*ndirac+idirac)*nc]+=
					-akappa*(ut[0][i+kvol*mu]*r[(uid*ndirac+idirac)*nc]
							+ut[1][i+kvol*mu]*r[(uid*ndirac+idirac)*nc+1]
							+conj(ut[0][did+kvol*mu])*r[(did*ndirac+idirac)*nc]
							-ut[1][did+kvol*mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu*ndirac+idirac]*
					(          ut[0][i+kvol*mu]*r[(uid*ndirac+igork1)*nc]
								  +ut[1][i+kvol*mu]*r[(uid*ndirac+igork1)*nc+1]
								  -conj(ut[0][did+kvol*mu])*r[(did*ndirac+igork1)*nc]
								  +ut[1][did+kvol*mu] *r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=
					-akappa*(-conj(ut[1][i+kvol*mu])*r[(uid*ndirac+idirac)*nc]
							+conj(ut[0][i+kvol*mu])*r[(uid*ndirac+idirac)*nc+1]
							+conj(ut[1][did+kvol*mu])*r[(did*ndirac+idirac)*nc]
							+ut[0][did+kvol*mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu*ndirac+idirac]*
					(-conj(ut[1][i+kvol*mu])*r[(uid*ndirac+igork1)*nc]
					 +conj(ut[0][i+kvol*mu])*r[(uid*ndirac+igork1)*nc+1]
					 -conj(ut[1][did+kvol*mu])*r[(did*ndirac+igork1)*nc]
					 -ut[0][did+kvol*mu] *r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3*kvol+i]; int uid = iu[3*kvol+i];
#ifndef NO_TIME
#pragma omp simd aligned(phi,r:AVX)
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3*ndirac+idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk[0] and dk[1] swap under dagger
			phi[(i*ndirac+idirac)*nc]+=
				-dk[0][i]*(ut[0][i+kvol*3]*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+ut[1][i+kvol*3]*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk[1][did]*(conj(ut[0][did+kvol*3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						-ut[1][did+kvol*3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]+=
				-dk[0][i]*(-conj(ut[1][i+kvol*3])*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+conj(ut[0][i+kvol*3])*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk[1][did]*(conj(ut[1][did+kvol*3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						+ut[0][did+kvol*3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
#endif
	return 0;
}
//Float Versions
//int Dslash_f(Complex_f *phi, Complex_f *r){
int Dslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu, unsigned int *id,\
		Complex_f gamval[20],	const unsigned short gamin[16],	float *dk[2], Complex_f jqq, float akappa){
	const char *funcname = "Dslash_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
	//Diquark Term (antihermitian)
#ifdef __NVCC__
	cuDslash_f(phi,r,ut[0],ut[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex_f));
#pragma omp parallel for
	for(unsigned int i=0;i<kvol;i++){
		Complex_f ru[nc]; Complex_f rd[nc];
		Complex_f rgu[nc]; Complex_f rgd[nc];
		Complex_f phi_s[ngorkov*nc];
		for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
			unsigned short igork = ((idirac>>1)+4)<<1;
			unsigned int ind_d =4*ndirac+(idirac>>1);
			Complex_f a_1=conj(jqq)*gamval[ind_d];
			//We subtract a_2, hence the minus
			Complex_f a_2=-jqq*gamval[ind_d];
			ind_d=i+kvolHalo*(idirac); unsigned int ind_g=i+kvolHalo*(igork);
			phi_s[idirac]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork]=phi[ind_g]+a_2*r[ind_d];
			ind_d+=kvol; ind_g+=kvol;
			phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
		}
		Complex_f u11s;	Complex_f u12s;
		Complex_f u11sd; Complex_f u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvolHalo*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			u11s=ut[0][ind]; u12s=ut[1][ind];
			ind = did+kvolHalo*mu;
			u11sd=ut[0][ind]; u12sd=ut[1][ind];
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				unsigned short idirac=igorkov&3;		
				unsigned short gind=mu*ndirac+idirac;
				const Complex_f gam=gamval[gind];
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				unsigned short igork1 = (igorkov<4) ? gamin[gind] : gamin[gind]+4;
				for(unsigned short c=0;c<nc;c++){
					ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
					rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
				}
				//Wilson + Dirac term in that order. Definitely easier
				phi_s[igorkov*nc]+=-akappa*(u11s*ru[0]+ u12s*ru[1]+\
						conj(u11sd)*rd[0]- u12sd*rd[1]);
				//Dirac term
				phi_s[igorkov*nc]+=gam*(u11s*rgu[0]+ u12s*rgu[1]-\
						conj(u11sd)*rgd[0]+ u12sd*rgd[1]);

				phi_s[igorkov*nc+1]+=-akappa*(-conj(u12s)*ru[0]+ conj(u11s)*ru[1]+\
						conj(u12sd)*rd[0]+ u11sd*rd[1]);
				//Dirac term
				phi_s[igorkov*nc+1]+=gam*(-conj(u12s)*rgu[0]+ conj(u11s)*rgu[1]-\
						conj(u12sd)*rgd[0]- u11sd*rgd[1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
#ifndef NO_TIME
		ind=i+kvolHalo*3;
		u11s=ut[0][ind]; u12s=ut[1][ind];
		const float dk4ms=dk[0][i];	const float dk4ps=dk[1][i];
		const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
		ind=did+kvolHalo*3;
		u11sd=ut[0][ind]; u12sd=ut[1][ind];
		const float dk4msd=dk[0][did];	const float dk4psd=dk[1][did];
		for(unsigned short igorkov=0;igorkov<ndirac;igorkov++){
			unsigned short igork1 = gamin[3*ndirac+igorkov];
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi_s[igorkov*nc]+=
				-dk4ps*(u11s*(ru[0]-rgu[0]) +u12s*(ru[1]-rgu[1]))
				-dk4msd*(conj(u11sd)*(rd[0]+rgd[0]) -u12sd *(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkov*nc)]=phi_s[igorkov*nc];

			phi_s[igorkov*nc+1]+=
				-dk4ps*(-conj(u12s)*(ru[0]-rgu[0]) +conj(u11s)*(ru[1]-rgu[1]))
				-dk4msd*(conj(u12sd)*(rd[0]+rgd[0]) +u11sd *(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkov*nc+1)]=phi_s[igorkov*nc+1];
			const unsigned short igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
																		//the FORTRAN code did it.
			igork1 += 4;
			//And the gorkov terms. Note that dk4p and dk4m swap positions compared to the above				
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkovPP*nc+c)]; rd[c]=r[did+kvolHalo*(igorkovPP*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//And the Gor'kov terms. Note that dk4p and dk4m swap positions compared to the above				
			phi_s[igorkovPP*nc]+=-dk4ms*(u11s*(ru[0]-rgu[0])+ u12s*(ru[1]-rgu[1]))-
				dk4psd*(conj(u11sd)*(rd[0]+rgd[0])- u12sd*(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc)]=phi_s[igorkovPP*nc];

			phi_s[igorkovPP*nc+1]+=-dk4ms*(conj(-u12s)*(ru[0]-rgu[0]) +conj(u11s)*(ru[1]-rgu[1]))
				-dk4psd*(conj(u12sd)*(rd[0]+rgd[0]) +u11sd*(rd[1]+rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc+1)]=phi_s[igorkovPP*nc+1];
		}
#endif
	}
#endif
	return 0;
}
int Dslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu,unsigned int *id,\
		Complex_f gamval[20], const unsigned short gamin[16], float *dk[2], Complex_f jqq, float akappa){
	const char *funcname = "Dslashd_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
#ifdef __NVCC__
	cuDslashd_f(phi,r,ut[0],ut[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex_f));
#pragma omp parallel for
	for(unsigned int i=0;i<kvol;i++){
		Complex_f ru[nc];  Complex_f rd[nc];
		Complex_f rgu[nc];  Complex_f rgd[nc];
		Complex_f phi_s[ngorkov*nc];
#pragma omp simd
		for(unsigned short idirac = 0; idirac<ndirac; idirac++){
			unsigned short igork = idirac+4;
			//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
			//We subtract a_1, hence the minus
			Complex_f a_1=-conj(jqq)*gamval[4*ndirac+idirac];
			Complex_f a_2=jqq*gamval[4*ndirac+idirac];
			phi_s[idirac*nc]=phi[i+kvolHalo*(idirac*nc)]+a_1*r[i+kvolHalo*(igork*nc)];
			phi_s[igork*nc]=phi[i+kvolHalo*(igork*nc)]+a_2*r[i+kvolHalo*(idirac*nc)];
			phi_s[idirac*nc+1]=phi[i+kvolHalo*(idirac*nc+1)]+a_1*r[i+kvolHalo*(igork*nc+1)];
			phi_s[igork*nc+1]=phi[i+kvolHalo*(igork*nc+1)]+a_2*r[i+kvolHalo*(idirac*nc+1)];
		}
		Complex_f u11s;	 Complex_f u12s;
		Complex_f u11sd;	 Complex_f u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvolHalo*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			u11s=ut[0][ind]; u12s=ut[1][ind];
			ind = did+kvolHalo*mu;
			u11sd=ut[0][ind]; u12sd=ut[1][ind];
#pragma omp simd
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				unsigned short idirac=igorkov&3;		
				const Complex_f gam=gamval[mu*ndirac+idirac];
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				unsigned short igork1 = (igorkov<4) ? gamin[mu*ndirac+idirac] : gamin[mu*ndirac+idirac]+4;
				for(unsigned short c=0;c<nc;c++){
					ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
					rgd[c]=r[did+kvolHalo*(igork1*nc+c)]; rgu[c]=r[uid+kvolHalo*(igork1*nc+c)];
				}
				//Wilson + Dirac term in that order. Definitely easier
				phi_s[igorkov*nc]-= akappa*(u11s*ru[0] +u12s*ru[1]
						+conj(u11sd)*rd[0] -u12sd *rd[1]);

				//Dirac term
				phi_s[igorkov*nc]-=gam* (u11s*rgu[0] +u12s*rgu[1]
						-conj(u11sd)*rgd[0] +u12sd *rgd[1]);

				phi_s[igorkov*nc+1]-= akappa*(-conj(u12s)*ru[0] +conj(u11s)*ru[1]
						+conj(u12sd)*rd[0] +u11sd *rd[1]);
				//Dirac term
				phi_s[igorkov*nc+1]-=gam* (-conj(u12s)*rgu[0] +conj(u11s)*rgu[1]
						-conj(u12sd)*rgd[0] -u11sd *rgd[1]);

			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
#ifndef NO_TIME
		ind=i+kvolHalo*3;
		u11s=ut[0][ind]; u12s=ut[1][ind];
		const float dk4ms=dk[0][i];	const float dk4ps=dk[1][i];
		const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
		ind=did+kvolHalo*3;
		u11sd=ut[0][ind]; u12sd=ut[1][ind];
		const float dk4msd=dk[0][did];	const float dk4psd=dk[1][did];
#pragma omp simd
		for(unsigned short igorkov=0; igorkov<ndirac; igorkov++){
			unsigned short igork1 = gamin[3*ndirac+igorkov];	
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkov*nc+c)]; rd[c]=r[did+kvolHalo*(igorkov*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi_s[igorkov*nc]+=
				-dk4ms*(u11s*(ru[0]+rgu[0]) +u12s*(ru[1]+rgu[1]))
				-dk4psd*(conj(u11sd)*(rd[0]-rgd[0]) -u12sd *(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkov*nc)]=phi_s[igorkov*nc];

			phi_s[igorkov*nc+1]+=
				-dk4ms*(-conj(u12s)*(ru[0]+rgu[0]) +conj(u11s)*(ru[1]+rgu[1]))
				-dk4psd*(conj(u12sd)*(rd[0]-rgd[0]) +u11sd *(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkov*nc+1)]=phi_s[igorkov*nc+1];
			const unsigned short igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
																		//the FORTRAN code did it.
			igork1 += 4;
			for(unsigned short c=0;c<nc;c++){
				ru[c]=r[uid+kvolHalo*(igorkovPP*nc+c)]; rd[c]=r[did+kvolHalo*(igorkovPP*nc+c)];
				rgu[c]=r[uid+kvolHalo*(igork1*nc+c)]; rgd[c]=r[did+kvolHalo*(igork1*nc+c)];
			}
			//And the Gor'kov terms. Note that dk4p and dk4m swap positions compared to the above				
			phi_s[igorkovPP*nc]+=-dk4ps*(u11s*(ru[0]+rgu[0]) +u12s*(ru[1]+rgu[1]))
				-dk4msd*(conj(u11sd)*(rd[0]-rgd[0]) -u12sd*(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc)]=phi_s[igorkovPP*nc];

			phi_s[igorkovPP*nc+1]+=dk4ps*(conj(u12s)*(ru[0]+rgu[0]) -conj(u11s)*(ru[1]+rgu[1]))
				-dk4msd*(conj(u12sd)*(rd[0]-rgd[0]) +u11sd*(rd[1]-rgd[1]));
			phi[i+kvolHalo*(igorkovPP*nc+1)]=phi_s[igorkovPP*nc+1];
		}
#endif
	}
#endif
	return 0;
}
int Hdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned  int *iu,unsigned  int *id,\
		Complex_f gamval[20], const unsigned short gamin[16], float *dk[2], float akappa){
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
	for(unsigned int i=0;i<kvol;i+=AVX){
		alignas(AVX) Complex_f u11s[AVX];	 alignas(AVX) Complex_f u12s[AVX];
		alignas(AVX) Complex_f u11sd[AVX];	 alignas(AVX) Complex_f u12sd[AVX];
		alignas(AVX) Complex_f ru[2][AVX];   alignas(AVX) Complex_f rd[2][AVX];
		alignas(AVX) Complex_f rgu[2][AVX];  alignas(AVX) Complex_f rgd[2][AVX];
		alignas(AVX) Complex_f phi_s[ndirac*nc][AVX];
		//Do we need to sync threads if each thread only accesses the value it put in shared memory?
#pragma unroll(2)
		for(unsigned short idirac=0; idirac<ndirac; idirac++)
			for(unsigned short c=0; c<nc; c++)
#pragma omp simd aligned(phi_s,phi:AVX)
				for(unsigned short j=0;j<AVX;j++)
					phi_s[idirac*nc+c][j]=phi[((i+j)*ndirac+idirac)*nc+c];
		alignas(AVX) unsigned int did[AVX], uid[AVX];
#pragma unroll
		for(unsigned short mu = 0; mu <3; mu++){
#pragma omp simd aligned(u11s,u12s,did,uid,id,iu,u11sd,u12sd:AVX)
			for(unsigned short j =0;j<AVX;j++){
				did[j]=id[(i+j)+kvol*mu]; uid[j] = iu[(i+j)+kvol*mu];
				u11s[j]=ut[0][(i+j)+kvol*mu];	u12s[j]=ut[1][(i+j)+kvol*mu];
				u11sd[j]=ut[0][did[j]+kvol*mu];	u12sd[j]=ut[1][did[j]+kvol*mu];
			}
#pragma unroll
			for(unsigned short idirac=0; idirac<ndirac; idirac++){
				unsigned short igork1 = gamin[mu*ndirac+idirac];
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
					for(unsigned short j =0;j<AVX;j++){
						ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
						rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
						rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
						rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
					}
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd:AVX)
				for(unsigned short j =0;j<AVX;j++){
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
		for(unsigned short j=0;j<AVX;j++){
			u11s[j]=ut[0][(i+j)+kvol*3];	u12s[j]=ut[1][(i+j)+kvol*3];
			did[j]=id[(i+j)+kvol*3];uid[j]= iu[(i+j)+kvol*3];
			u11sd[j]=ut[0][did[j]+kvol*3];	u12sd[j]=ut[1][did[j]+kvol*3];
			dk4ms[j]=dk[0][did[j]];   dk4ps[j]=dk[1][i+j];
		}

#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac; idirac++){
			unsigned short igork1 = gamin[3*ndirac+idirac];
#pragma unroll
			for(unsigned short c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
				for(unsigned short j =0;j<AVX;j++){
					ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
					rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
					rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
					rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
				}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd,dk4ms,dk4ps,phi:AVX)
			for(unsigned short j =0;j<AVX;j++){
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
		Complex_f gamval[20], const unsigned short gamin[16], float *dk[2], float akappa){
	const char *funcname = "Hdslashd_f";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
#if(nproc>1)
	CHalo_swap_all(r, 8);
#endif

	//Mass term
#ifdef __NVCC__
	cuHdslashd_f(phi,r,ut,iu,id,gamval,gamin,dk,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm2*sizeof(Complex_f));

	//Spacelike term
	//Enough room on L1 data cache for Zen 2 to hold 160 elements at a time
	//Vectorise with 128 maybe?
#pragma omp parallel for
	for(unsigned int i=0;i<kvol;i+=AVX){
		//Right. Time to prefetch
		alignas(AVX) Complex_f u11s[AVX];		alignas(AVX) Complex_f u12s[AVX];
		alignas(AVX) Complex_f u11sd[AVX];		alignas(AVX) Complex_f u12sd[AVX];
		alignas(AVX) Complex_f ru[2][AVX]; 		alignas(AVX) Complex_f rd[2][AVX];
		alignas(AVX) Complex_f rgu[2][AVX];		alignas(AVX) Complex_f rgd[2][AVX];
		alignas(AVX) Complex_f phi_s[ndirac*nc][AVX];
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac; idirac++)
#pragma unroll
			for(unsigned short c=0; c<nc; c++)
#pragma omp simd aligned(phi_s,phi:AVX)
				for(unsigned short j=0;j<AVX;j++)
					phi_s[idirac*nc+c][j]=phi[((i+j)*ndirac+idirac)*nc+c];
		alignas(AVX) unsigned int did[AVX], uid[AVX];
#ifndef NO_SPACE
#pragma unroll
		for(unsigned short mu = 0; mu <ndim-1; mu++){
			//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
#pragma omp simd aligned(u11s,u12s,did,uid,id,iu,u11sd,u12sd:AVX)
			for(unsigned short j =0;j<AVX;j++){
				did[j]=id[(i+j)+kvol*mu]; uid[j] = iu[(i+j)+kvol*mu];
				u11s[j]=ut[0][(i+j)+kvol*mu];	u12s[j]=ut[1][(i+j)+kvol*mu];
				u11sd[j]=ut[0][did[j]+kvol*mu];	u12sd[j]=ut[1][did[j]+kvol*mu];
			}
#pragma unroll
			for(unsigned short idirac=0; idirac<ndirac; idirac++){
				unsigned short igork1 = gamin[mu*ndirac+idirac];
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
					for(unsigned short j =0;j<AVX;j++){
						ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
						rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
						rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
						rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
					}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd:AVX)
				for(unsigned short j =0;j<AVX;j++){
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
		for(unsigned short j=0;j<AVX;j++){
			u11s[j]=ut[0][(i+j)+kvol*3];	u12s[j]=ut[1][(i+j)+kvol*3];
			did[j]=id[(i+j)+kvol*3];		uid[j]= iu[(i+j)+kvol*3];
			u11sd[j]=ut[0][did[j]+kvol*3];	u12sd[j]=ut[1][did[j]+kvol*3];
			dk4ms[j]=dk[0][i+j];   			dk4ps[j]=dk[1][did[j]];
		}
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac; idirac++){
			unsigned short igork1 = gamin[3*ndirac+idirac];
#pragma unroll
			for(unsigned short c=0; c<nc; c++)
#pragma omp simd aligned(ru,rd,rgu,rgd,r,uid,did:AVX)
				for(unsigned short j =0;j<AVX;j++){
					ru[c][j]=r[(uid[j]*ndirac+idirac)*nc+c];
					rd[c][j]=r[(did[j]*ndirac+idirac)*nc+c];
					rgu[c][j]=r[(uid[j]*ndirac+igork1)*nc+c];
					rgd[c][j]=r[(did[j]*ndirac+igork1)*nc+c];
				}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
#pragma omp simd aligned(phi_s,u11s,u12s,u11sd,u12sd,ru,rd,rgu,rgd,dk4ms,dk4ps,phi:AVX)
			for(unsigned short j =0;j<AVX;j++){
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
