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
	const char funcname[] = "Dslash";
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
			ind_d+=kvolHalo; ind_g+=kvolHalo;
			phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
		}
		Complex u11s;	Complex u12s;
		Complex u11sd; Complex u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvol*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			ind = i+kvolHalo*mu;
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
		ind=i+kvol*3;
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
	const char funcname[] = "Dslashd";
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
		for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
			unsigned short igork = ((idirac>>1)+4)<<1;
			unsigned int ind_d =4*ndirac+(idirac>>1);
			Complex a_1=-conj(jqq)*gamval[ind_d];
			//We subtract a_2, hence the minus
			Complex a_2=jqq*gamval[ind_d];
			ind_d=i+kvolHalo*(idirac); unsigned int ind_g=i+kvolHalo*(igork);
			phi_s[idirac]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork]=phi[ind_g]+a_2*r[ind_d];
			ind_d+=kvolHalo; ind_g+=kvolHalo;
			phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
		}
		Complex u11s;	 Complex u12s;
		Complex u11sd;	 Complex u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvol*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			ind = i+kvolHalo*mu;
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
		ind = i+kvol*3;
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
	const char funcname[] = "Hdslash";
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
	for(unsigned int i=0;i<kvol;i++){
	Complex ru[2];  Complex rd[2];
	Complex rgu[2];  Complex rgd[2];
	Complex phi_s[ndirac*nc];
#pragma omp simd
		for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc)
#pragma unroll
			for(unsigned short c=0; c<nc; c++)
				//NOTE: idirac is increasing by nc each time. So should be read as idirac*nc 
				phi_s[idirac+c]=phi[i+kvolHalo*(c+idirac)];

		//#pragma unroll
		for(unsigned short mu = 0; mu <ndim; mu++){
			unsigned int ind=i+kvolHalo*mu;
			const Complex u11s=ut[0][ind];	const Complex u12s=ut[1][ind];
			ind = i+kvol*mu;
			const int did=id[ind];	const int uid = iu[ind];
			ind=did+kvolHalo*mu;
			const Complex u11sd=ut[0][ind];	const Complex u12sd=ut[1][ind];
#pragma omp simd
			for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
				const unsigned short igork1 = gamin[mu*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0;c<nc;c++){
					ind =kvolHalo*(idirac+c);
					ru[c]=r[uid+ind]; rd[c]=r[did+ind];
					ind =kvolHalo*(igork1+c);
					rgu[c]=r[uid+ind]; rgd[c]=r[did+ind];
				}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//Spacelike terms
				if(mu<3){
					const Complex gam=gamval[mu*ndirac+(idirac>>1)];
					phi_s[idirac]+=-akappa*(u11s*ru[0]+u12s*ru[1]+\
							conj(u11sd)*rd[0]-u12sd*rd[1]);
					//Dirac term
					phi_s[idirac]+=gam*(u11s*rgu[0]+u12s*rgu[1]-\
							conj(u11sd)*rgd[0]+ u12sd*rgd[1]);

					phi_s[idirac+1]+=-akappa*(-conj(u12s)*ru[0]+ conj(u11s)*ru[1]+\
							conj(u12sd)*rd[0]+ u11sd*rd[1]);
					//Dirac term
					phi_s[idirac+1]+=gam*(-conj(u12s)*rgu[0]+ conj(u11s)*rgu[1]-\
							conj(u12sd)*rgd[0]- u11sd*rgd[1]);
				}
				//Timelike terms
				else{
					const double dk4ms=dk[0][did];   const double dk4ps=dk[1][i];
					//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

					phi_s[idirac+0]-= dk4ps*(u11s*(ru[0]-rgu[0])
							+u12s*(ru[1]-rgu[1]));
					phi_s[idirac+0]-= dk4ms*(conj(u11sd)*(rd[0]+rgd[0])
							-u12sd *(rd[1]+rgd[1]));
					phi[i+kvolHalo*(0+idirac)]=phi_s[idirac+0];

					phi_s[idirac+1]-= dk4ps*(-conj(u12s)*(ru[0]-rgu[0])
							+conj(u11s)*(ru[1]-rgu[1]));
					phi_s[idirac+1]-= dk4ms*(conj(u12sd)*(rd[0]+rgd[0])
							+u11sd *(rd[1]+rgd[1]));
					phi[i+kvolHalo*(1+idirac)]=phi_s[idirac+1];
				}
			}
		}
	}
#endif
	return 0;
}
int Hdslashd(Complex *phi, Complex *r, Complex *ut[2],unsigned  int *iu,unsigned  int *id,\
		Complex gamval[20], const unsigned short gamin[16], double *dk[2], float akappa){
	const char funcname[] = "Hdslashd";
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
	for(unsigned int i=0;i<kvol;i++){
	//Right. Time to prefetch
	Complex ru[2];  Complex rd[2];
	Complex rgu[2];  Complex rgd[2];
	Complex phi_s[ndirac*nc];
#pragma omp simd
		for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc)
#pragma unroll
			for(unsigned short c=0; c<nc; c++)
				//NOTE: idirac is increasing by nc each time. So should be read as idirac*nc 
				phi_s[idirac+c]=phi[i+kvolHalo*(c+idirac)];

		//#pragma unroll
		for(unsigned short mu = 0; mu <ndim; mu++){
			unsigned int ind=i+kvolHalo*mu;
			const Complex u11s=ut[0][ind];	const Complex u12s=ut[1][ind];
			ind = i+kvol*mu;
			const int did=id[ind];	const int uid = iu[ind];
			ind=did+kvolHalo*mu;
			const Complex u11sd=ut[0][ind];	const Complex u12sd=ut[1][ind];
#pragma omp simd
			for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc){
				unsigned short igork1 = gamin[mu*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0;c<nc;c++){
					ind =kvolHalo*(idirac+c);
					ru[c]=r[uid+ind]; rd[c]=r[did+ind];
					ind =kvolHalo*(igork1+c);
					rgu[c]=r[uid+ind]; rgd[c]=r[did+ind];
				}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//Spacelike terms
				if(mu<3){
					const Complex gam=gamval[mu*ndirac+(idirac>>1)];
					phi_s[idirac]-=akappa*(u11s*ru[0] +u12s*ru[1]
							+conj(u11sd)*rd[0] -u12sd *rd[1]);
					//Dirac term
					phi_s[idirac]-=gam* (u11s*rgu[0] +u12s*rgu[1]
							-conj(u11sd)*rgd[0] +u12sd *rgd[1]);

					phi_s[idirac+1]-=akappa*(-conj(u12s)*ru[0] +conj(u11s)*ru[1]
							+conj(u12sd)*rd[0] +u11sd *rd[1]);
					//Dirac term
					phi_s[idirac+1]-=gam*(-conj(u12s)*rgu[0] +conj(u11s)*rgu[1]
							-conj(u12sd)*rgd[0] -u11sd *rgd[1]);
				}
				//Timelike terms
				else{
					const double  dk4ms=dk[0][i];  const double dk4ps=dk[1][did];
					//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

					phi_s[idirac]+= -dk4ms*(u11s*(ru[0]+rgu[0])
							+u12s*(ru[1]+rgu[1]));
					phi_s[idirac]+= -dk4ps*(conj(u11sd)*(rd[0]-rgd[0])
							-u12sd *(rd[1]-rgd[1]));
					phi[i+kvolHalo*(0+idirac)]=phi_s[idirac+0];

					phi_s[idirac+1]-= dk4ms*(-conj(u12s)*(ru[0]+rgu[0])
							+conj(u11s)*(ru[1]+rgu[1]));
					phi_s[idirac+1]-= +dk4ps*(conj(u12sd)*(rd[0]-rgd[0])
							+u11sd *(rd[1]-rgd[1]));
					phi[i+kvolHalo*(1+idirac)]=phi_s[idirac+1];
				}
			}
		}
	}
#endif
	return 0;
}
//Float Versions
//int Dslash_f(Complex_f *phi, Complex_f *r){
int Dslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu, unsigned int *id,\
		Complex_f gamval[20],	const unsigned short gamin[16],	float *dk[2], Complex_f jqq, float akappa){
	const char funcname[] = "Dslash_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
	//Diquark Term (antihermitian)
#ifdef __NVCC__
	cuDslash_f(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa,dimGrid,dimBlock);
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
			ind_d+=kvolHalo; ind_g+=kvolHalo;
			phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
		}
		Complex_f u11s;	Complex_f u12s;
		Complex_f u11sd; Complex_f u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvol*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			ind = i+kvolHalo*mu;
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
		ind = i+kvol*3;
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
	const char funcname[] = "Dslashd_f";
	//Get the halos in order
#if(nproc>1)
	CHalo_swap_all(r, 16);
#endif

	//Mass term
#ifdef __NVCC__
	cuDslashd_f(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa,dimGrid,dimBlock);
#else
	memcpy(phi, r, kferm*sizeof(Complex_f));
#pragma omp parallel for
	for(unsigned int i=0;i<kvol;i++){
		Complex_f ru[nc];  Complex_f rd[nc];
		Complex_f rgu[nc];  Complex_f rgd[nc];
		Complex_f phi_s[ngorkov*nc];
#pragma omp simd
		for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
			unsigned short igork = ((idirac>>1)+4)<<1;
			unsigned int ind_d =4*ndirac+(idirac>>1);
			Complex_f a_1=-conj(jqq)*gamval[ind_d];
			//We subtract a_2, hence the minus
			Complex_f a_2=jqq*gamval[ind_d];
			ind_d=i+kvolHalo*(idirac); unsigned int ind_g=i+kvolHalo*(igork);
			phi_s[idirac]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork]=phi[ind_g]+a_2*r[ind_d];
			ind_d+=kvolHalo; ind_g+=kvolHalo;
			phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
			phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
		}
		Complex_f u11s;	 Complex_f u12s;
		Complex_f u11sd;	 Complex_f u12sd;
		unsigned int ind;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(unsigned short mu = 0; mu <3; mu++){
			ind = i+kvol*mu;
			const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
			ind = i+kvolHalo*mu;
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
		ind = i+kvol*3;
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
	const char funcname[] = "Hdslash_f";
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
	for(unsigned int i=0;i<kvol;i++){
	Complex_f ru[2];  Complex_f rd[2];
	Complex_f rgu[2];  Complex_f rgd[2];
	Complex_f phi_s[ndirac*nc];
#pragma omp simd
		for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc)
#pragma unroll
			for(unsigned short c=0; c<nc; c++)
				//NOTE: idirac is increasing by nc each time. So should be read as idirac*nc 
				phi_s[idirac+c]=phi[i+kvolHalo*(c+idirac)];

		//#pragma unroll
		for(unsigned short mu = 0; mu <ndim; mu++){
			unsigned int ind=i+kvolHalo*mu;
			const Complex_f u11s=ut[0][ind];	const Complex_f u12s=ut[1][ind];
			ind = i+kvol*mu;
			const int did=id[ind];	const int uid = iu[ind];
			ind=did+kvolHalo*mu;
			const Complex_f u11sd=ut[0][ind];	const Complex_f u12sd=ut[1][ind];
#pragma omp simd
			for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
				const unsigned short igork1 = gamin[mu*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0;c<nc;c++){
					ind =kvolHalo*(idirac+c);
					ru[c]=r[uid+ind]; rd[c]=r[did+ind];
					ind =kvolHalo*(igork1+c);
					rgu[c]=r[uid+ind]; rgd[c]=r[did+ind];
				}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//Spacelike terms
				if(mu<3){
					const Complex_f gam=gamval[mu*ndirac+(idirac>>1)];
					phi_s[idirac]+=-akappa*(u11s*ru[0]+u12s*ru[1]+\
							conj(u11sd)*rd[0]-u12sd*rd[1]);
					//Dirac term
					phi_s[idirac]+=gam*(u11s*rgu[0]+u12s*rgu[1]-\
							conj(u11sd)*rgd[0]+ u12sd*rgd[1]);

					phi_s[idirac+1]+=-akappa*(-conj(u12s)*ru[0]+ conj(u11s)*ru[1]+\
							conj(u12sd)*rd[0]+ u11sd*rd[1]);
					//Dirac term
					phi_s[idirac+1]+=gam*(-conj(u12s)*rgu[0]+ conj(u11s)*rgu[1]-\
							conj(u12sd)*rgd[0]- u11sd*rgd[1]);
				}
				//Timelike terms
				else{
					const float dk4ms=dk[0][did];   const float dk4ps=dk[1][i];
					//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

					phi_s[idirac+0]-= dk4ps*(u11s*(ru[0]-rgu[0])
							+u12s*(ru[1]-rgu[1]));
					phi_s[idirac+0]-= dk4ms*(conj(u11sd)*(rd[0]+rgd[0])
							-u12sd *(rd[1]+rgd[1]));
					phi[i+kvolHalo*(0+idirac)]=phi_s[idirac+0];

					phi_s[idirac+1]-= dk4ps*(-conj(u12s)*(ru[0]-rgu[0])
							+conj(u11s)*(ru[1]-rgu[1]));
					phi_s[idirac+1]-= dk4ms*(conj(u12sd)*(rd[0]+rgd[0])
							+u11sd *(rd[1]+rgd[1]));
					phi[i+kvolHalo*(1+idirac)]=phi_s[idirac+1];
				}
			}
		}
	}
#endif
	return 0;
}
int Hdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu,unsigned int *id,\
		Complex_f gamval[20], const unsigned short gamin[16], float *dk[2], float akappa){
	const char funcname[] = "Hdslashd_f";
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
	for(unsigned int i=0;i<kvol;i++){
	//Right. Time to prefetch
	Complex_f ru[2];  Complex_f rd[2];
	Complex_f rgu[2];  Complex_f rgd[2];
	Complex_f phi_s[ndirac*nc];
#pragma omp simd
		for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc)
#pragma unroll
			for(unsigned short c=0; c<nc; c++)
				//NOTE: idirac is increasing by nc each time. So should be read as idirac*nc 
				phi_s[idirac+c]=phi[i+kvolHalo*(c+idirac)];

		//#pragma unroll
		for(unsigned short mu = 0; mu <ndim; mu++){
			unsigned int ind=i+kvolHalo*mu;
			const Complex_f u11s=ut[0][ind];	const Complex_f u12s=ut[1][ind];
			ind = i+kvol*mu;
			const int did=id[ind];	const int uid = iu[ind];
			ind=did+kvolHalo*mu;
			const Complex_f u11sd=ut[0][ind];	const Complex_f u12sd=ut[1][ind];
#pragma omp simd
			for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc){
				unsigned short igork1 = gamin[mu*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0;c<nc;c++){
					ind =kvolHalo*(idirac+c);
					ru[c]=r[uid+ind]; rd[c]=r[did+ind];
					ind =kvolHalo*(igork1+c);
					rgu[c]=r[uid+ind]; rgd[c]=r[did+ind];
				}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//Spacelike terms
				if(mu<3){
					const Complex_f gam=gamval[mu*ndirac+(idirac>>1)];
					phi_s[idirac]-=akappa*(u11s*ru[0] +u12s*ru[1]
							+conj(u11sd)*rd[0] -u12sd *rd[1]);
					//Dirac term
					phi_s[idirac]-=gam* (u11s*rgu[0] +u12s*rgu[1]
							-conj(u11sd)*rgd[0] +u12sd *rgd[1]);

					phi_s[idirac+1]-=akappa*(-conj(u12s)*ru[0] +conj(u11s)*ru[1]
							+conj(u12sd)*rd[0] +u11sd *rd[1]);
					//Dirac term
					phi_s[idirac+1]-=gam*(-conj(u12s)*rgu[0] +conj(u11s)*rgu[1]
							-conj(u12sd)*rgd[0] -u11sd *rgd[1]);
				}
				//Timelike terms
				else{
					const float  dk4ms=dk[0][i];  const float dk4ps=dk[1][did];
					//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

					phi_s[idirac]+= -dk4ms*(u11s*(ru[0]+rgu[0])
							+u12s*(ru[1]+rgu[1]));
					phi_s[idirac]+= -dk4ps*(conj(u11sd)*(rd[0]-rgd[0])
							-u12sd *(rd[1]-rgd[1]));
					phi[i+kvolHalo*(0+idirac)]=phi_s[idirac+0];

					phi_s[idirac+1]-= dk4ms*(-conj(u12s)*(ru[0]+rgu[0])
							+conj(u11s)*(ru[1]+rgu[1]));
					phi_s[idirac+1]-= +dk4ps*(conj(u12sd)*(rd[0]-rgd[0])
							+u11sd *(rd[1]-rgd[1]));
					phi[i+kvolHalo*(1+idirac)]=phi_s[idirac+1];
				}
			}
		}
	}
#endif
	return 0;
}


inline void Transpose_c(Complex_f *out, const int fast_in, const int fast_out){
	const volatile char funcname[]="Transpose_c";

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
	const volatile char funcname[]="Transpose_c";

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
	const char funcname[]="Transpose_f";

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
	const char funcname[]="Transpose_f";

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
	const char funcname[]="Transpose_I";

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
	const char funcname[]="Transpose_I";

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
