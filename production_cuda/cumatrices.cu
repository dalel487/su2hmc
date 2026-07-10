/**
 *	@file
 *	@brief	CUDA version of matrix manipulation routines
 *	@author	D. Lawlor
 */
#include <assert.h>
#include <su2hmc.h>
#include <matrices.h>
#include	<thrust_complex.h>
namespace Device{
	/**
	 * @brief Performs a warp reduction for sum
	 * @ingroup Helper
	 *
	 * @param[out,in] sdata:	The shared data array
	 * @param[in] tid:		The thread ID
	 *
	 * @post	Sum written to zeroth index of @p sdata
	 */
	template <typename T,unsigned int bsize>
		__device__ void warpReduce_sum(volatile T* sdata, const unsigned int tid){
			if(bsize >= 64) sdata[tid] += sdata[tid + 32];
			if(bsize >= 32) sdata[tid] += sdata[tid + 16];
			if(bsize >= 16) sdata[tid] += sdata[tid + 8];
			if(bsize >= 8) sdata[tid] += sdata[tid + 4];
			if(bsize >= 4) sdata[tid] += sdata[tid + 2];
			if(bsize >= 2) sdata[tid] += sdata[tid + 1];
		}
}
namespace Kernels{
	/**
	 * @brief Evaluates @f$\Phi=M r@f$
	 * @ingroup Dslashes
	 *
	 * @param[in,out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	u11t,u12t	Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk4m,dk4p:	@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post	Result added to @p phi
	 */
	template <typename T>
		__global__ void cuDslash(complex<T> *phi, complex<T> *r, complex<T> *u11t, complex<T> *u12t,const unsigned int *iu, const unsigned int *id,\
				complex<T> gamval[20],	const unsigned short gamin[16], const T *dk4m, const T *dk4p, const Complex_f jqq, const float akappa){
			const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
			const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
			const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
			const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
			const unsigned int gthreadId= blockId * bsize+bthreadId;

			for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
				complex<T> ru[nc]; complex<T> rd[nc];
				complex<T> rgu[nc]; complex<T> rgd[nc];
				complex<T> phi_s[ngorkov*nc];
				for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
					unsigned short igork = ((idirac>>1)+4)<<1;
					unsigned int ind_d =4*ndirac+(idirac>>1);
					complex<T> a_1=conj(jqq)*gamval[ind_d];
					//We subtract a_2, hence the minus
					complex<T> a_2=-jqq*gamval[ind_d];
					ind_d=i+kvolHalo*(idirac); unsigned int ind_g=i+kvolHalo*(igork);
					phi_s[idirac]=phi[ind_d]+a_1*r[ind_g];
					phi_s[igork]=phi[ind_g]+a_2*r[ind_d];
					ind_d+=kvolHalo; ind_g+=kvolHalo;
					phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
					phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
				}
				complex<T> u11s;	complex<T> u12s;
				complex<T> u11sd; complex<T> u12sd;
				unsigned int ind;
				//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
				for(unsigned short mu = 0; mu <3; mu++){
					ind = i+kvol*mu;
					const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
					ind = i+kvolHalo*mu;
					u11s=u11t[ind]; u12s=u12t[ind];
					ind = did+kvolHalo*mu;
					u11sd=u11t[ind]; u12sd=u12t[ind];
					for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
						unsigned short idirac=igorkov&3;		
						unsigned short gind=mu*ndirac+idirac;
						const complex<T> gam=gamval[gind];
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
				u11s=u11t[ind]; u12s=u12t[ind];
				const T dk4ms=dk4m[i];	const T dk4ps=dk4p[i];
				ind=i+kvol*3;
				const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
				ind=did+kvolHalo*3;
				u11sd=u11t[ind]; u12sd=u12t[ind];
				const T dk4msd=dk4m[did];	const T dk4psd=dk4p[did];
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
		}
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$
	 * @ingroup Dslashes
	 *
	 * @param[in,out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	u11t,u12t	Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk4m,dk4p:	@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post	Result added to @p phi
	 */
	template <typename T>
		__global__ void cuDslashd(complex<T> *phi, const complex<T> *r, const complex<T> *u11t, const complex<T> *u12t,const unsigned int *iu, const unsigned int *id,\
				complex<T> gamval[20], const unsigned short gamin[16], const T *dk4m, const T *dk4p, const Complex_f jqq, const float akappa){
			const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
			const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
			const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
			const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
			const unsigned int gthreadId= blockId * bsize+bthreadId;

			for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
				complex<T> ru[nc];  complex<T> rd[nc];
				complex<T> rgu[nc];  complex<T> rgd[nc];
				complex<T> phi_s[ngorkov*nc];
				for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
					unsigned short igork = ((idirac>>1)+4)<<1;
					unsigned int ind_d =4*ndirac+(idirac>>1);
					complex<T> a_1=-conj(jqq)*gamval[ind_d];
					complex<T> a_2=jqq*gamval[ind_d];
					ind_d=i+kvol*(idirac); unsigned int ind_g=i+kvol*(igork);
					phi_s[idirac]=phi[ind_d]+a_1*r[ind_g];
					phi_s[igork]=phi[ind_g]+a_2*r[ind_d];
					ind_d+=kvol; ind_g+=kvol;
					phi_s[idirac+1]=phi[ind_d]+a_1*r[ind_g];
					phi_s[igork+1]=phi[ind_g]+a_2*r[ind_d];
				}
				complex<T> u11s;	 complex<T> u12s;
				complex<T> u11sd;	 complex<T> u12sd;
				unsigned int ind;
				//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
				for(unsigned short mu = 0; mu <3; mu++){
					ind = i+kvol*mu;
					const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
					ind = i+kvolHalo*mu;
					u11s=u11t[ind]; u12s=u12t[ind];
					ind = did+kvolHalo*mu;
					u11sd=u11t[ind]; u12sd=u12t[ind];
					for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
						unsigned short idirac=igorkov&3;		
						const complex<T> gam=gamval[mu*ndirac+idirac];
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
				u11s=u11t[ind]; u12s=u12t[ind];
				const T dk4ms=dk4m[i];	const T dk4ps=dk4p[i];
				ind = i+kvol*3;
				const unsigned int did=id[ind]; const unsigned int uid = iu[ind];
				ind=did+kvolHalo*3;
				u11sd=u11t[ind]; u12sd=u12t[ind];
				const T dk4msd=dk4m[did];	const T dk4psd=dk4p[did];
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
					phi[i+kvol*(igorkov*nc)]=phi_s[igorkov*nc];

					phi_s[igorkov*nc+1]+=
						-dk4ms*(-conj(u12s)*(ru[0]+rgu[0]) +conj(u11s)*(ru[1]+rgu[1]))
						-dk4psd*(conj(u12sd)*(rd[0]-rgd[0]) +u11sd *(rd[1]-rgd[1]));
					phi[i+kvol*(igorkov*nc+1)]=phi_s[igorkov*nc+1];
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
					phi[i+kvol*(igorkovPP*nc)]=phi_s[igorkovPP*nc];

					phi_s[igorkovPP*nc+1]+=dk4ps*(conj(u12s)*(ru[0]+rgu[0]) -conj(u11s)*(ru[1]+rgu[1]))
						-dk4msd*(conj(u12sd)*(rd[0]-rgd[0]) +u11sd*(rd[1]-rgd[1]));
					phi[i+kvol*(igorkovPP*nc+1)]=phi_s[igorkovPP*nc+1];
				}
#endif
			}
		}

	/**
	 * @brief Evaluates @f$\Phi=Mr@f$ using up/down partitioning
	 * @ingroup Dslashes
	 *
	 * @param[out,in]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	u11t,u12t	Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk4m,dk4p:	@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post	Result added to @p phi
	 */
	template <typename T>
		__global__ void cuHdslash(complex<T> *phi, const complex<T> *r, const complex<T> *u11t, const complex<T> *u12t,unsigned int *iu, unsigned int *id,\
				__constant__ complex<T> gamval[20],	const unsigned short gamin[16],	const T *dk4m, const T *dk4p, const __grid_constant__ float akappa){
			/*
			 * Half Dslash T precision
			 */
			const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
			const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
			const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
			const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
			const unsigned int gthreadId= blockId * bsize+bthreadId;

			//Right. Time to prefetch
			complex<T> ru[2];  complex<T> rd[2];
			complex<T> rgu[2];  complex<T> rgd[2];
			complex<T> phi_s[ndirac*nc];
			for(unsigned int i=gthreadId;i<kvol;i+=bsize*gsize){
#pragma unroll
				for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc)
#pragma unroll
					for(unsigned short c=0; c<nc; c++)
						//NOTE: idirac is increasing by nc each time. So should be read as idirac*nc 
						phi_s[idirac+c]=phi[i+kvolHalo*(c+idirac)];

				//#pragma unroll
				for(unsigned short mu = 0; mu <ndim; mu++){
					unsigned int ind=i+kvolHalo*mu;
					const complex<T> u11s=u11t[ind];	const complex<T> u12s=u12t[ind];
					ind = i+kvol*mu;
					const int did=id[ind];	const int uid = iu[ind];
					ind=did+kvolHalo*mu;
					const complex<T> u11sd=u11t[ind];	const complex<T> u12sd=u12t[ind];
#pragma unroll
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
							const complex<T> gam=gamval[mu*ndirac+(idirac>>1)];
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
							const T dk4ms=dk4m[did];   const T dk4ps=dk4p[i];
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
		}
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ using up/down partitioning
	 * @ingroup Dslashes
	 *
	 * @param[in,out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	u11t,u12t	Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk4m,dk4p:	@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post	Result added to @p phi
	 */
	template <typename T>
		__global__ void cuHdslashd(complex<T> *phi, const complex<T>* r, const complex<T>* u11t, const complex<T>* u12t,unsigned int* iu, unsigned int* id,\
				__constant__ complex<T> gamval[20],	const unsigned short gamin[16],	const T* dk4m, const T* dk4p, const __grid_constant__ float akappa){
			/*
			 * Half Dslash Dagger T precision 
			 */
			const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
			const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
			const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
			const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
			const unsigned int gthreadId= blockId * bsize+bthreadId;

			//Right. Time to prefetch
			for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
				complex<T> phi_s[ndirac*nc];
#pragma unroll
				for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc)
#pragma unroll
					for(unsigned short c=0; c<nc; c++)
						//NOTE: idirac is increasing by nc each time. So should be read as idirac*nc 
						phi_s[idirac+c]=phi[i+kvol*(c+idirac)];

				//#pragma unroll
				for(unsigned short mu = 0; mu <ndim; mu++){
					unsigned int ind=i+kvolHalo*mu;
					const complex<T> u11s=u11t[ind];	const complex<T> u12s=u12t[ind];
					ind = i+kvol*mu;
					const int did=id[ind];	const int uid = iu[ind];
					ind=did+kvolHalo*mu;
					const complex<T> u11sd=u11t[ind];	const complex<T> u12sd=u12t[ind];
#pragma unroll
					for(unsigned short idirac=0; idirac<nc*ndirac; idirac+=nc){
						const unsigned short igork1 = gamin[mu*ndirac+(idirac>>1)] << (nc-1);
						complex<T> ru[2];  complex<T> rd[2];
						complex<T> rgu[2];  complex<T> rgd[2];
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
							const complex<T> gam=gamval[mu*ndirac+(idirac>>1)];
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
							const T  dk4ms=dk4m[i];  const T dk4ps=dk4p[did];
							//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

							phi_s[idirac]+= -dk4ms*(u11s*(ru[0]+rgu[0])
									+u12s*(ru[1]+rgu[1]));
							phi_s[idirac]+= -dk4ps*(conj(u11sd)*(rd[0]-rgd[0])
									-u12sd *(rd[1]-rgd[1]));
							phi[i+kvol*(0+idirac)]=phi_s[idirac+0];

							phi_s[idirac+1]-= dk4ms*(-conj(u12s)*(ru[0]+rgu[0])
									+conj(u11s)*(ru[1]+rgu[1]));
							phi_s[idirac+1]-= +dk4ps*(conj(u12sd)*(rd[0]-rgd[0])
									+u11sd *(rd[1]-rgd[1]));
							phi[i+kvol*(1+idirac)]=phi_s[idirac+1];
						}
					}
				}
			}
		}

	/**
	 * @brief Swaps the order of the gauge field so that it is now SoA instead of AoS and it is nice and coalesced in memory
	 * @ingroup Helper
	 * 
	 * @param[out] out:			The flipped array
	 * @param[in] in:			The original array
	 * @param[in] fast_out:	The size of the slowest moving dimension. This is the lattice site when read in from disk
	 * @param[in] fast_in:	The size of the fastest moving dimension. This is the direction index when read in from disk.
	 * 
	 */
	template <typename T>
		__global__ void Transpose(T *out, const T *in, const int fast_in, const int fast_out){
			const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
			const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
			const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
			const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
			const unsigned int gthreadId= blockId * bsize+bthreadId;

			//The if/else here is only to ensure we maximise GPU bandwidth
			//Typically this is used to write back to the AoS/Coalseced format
			if(fast_out>fast_in){
				for(unsigned int x=gthreadId;x<fast_out;x+=gsize*bsize)
					for(unsigned int y=0; y<fast_in;y++)
						out[y*fast_out+x]=in[x*fast_in+y];
			}
			//Typically this is used to write back to the SoA/saved config format
			else{
				for(unsigned int x=0; x<fast_out;x++)
					for(unsigned int y=gthreadId;y<fast_in;y+=gsize*bsize)
						out[y*fast_out+x]=in[x*fast_in+y];
			}
		}

	/**
	 * @brief Sums a float array into a double array
	 * @ingroup Helper
	 *
	 * @param[out] d:		The double array
	 * @param[in] f:		The float array
	 * @param[in] n:		The size of the arrays
	 */
	__global__ void Mixed_Sumto(double *d, float *f, const unsigned int n){
		const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
		const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
		const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
		const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
		const unsigned int gthreadId= blockId * bsize+bthreadId;

		for(unsigned int i=gthreadId; i<n;i+=bsize*gsize)
			d[i]+=(double)f[i];
		return;
	}

	/**
	 * @brief Performs a block reduction for sum
	 * @ingroup Helper
	 *
	 * @param[in] g_in_data:		The input global data array
	 * @param[out] g_out_data:	The output global data array
	 * @param[in] n:				The size of the input array
	 *
	 * @post sum saved in zeroth entry of @p g_out_data
	 */
	template <typename T,unsigned int bsize>
		__global__ void reduce_sum(T *g_in_data, T *g_out_data, const unsigned int n){
			extern __shared__ T sdata[];  // stored in the shared memory

			// Each thread loading one element from global onto shared memory
			const unsigned short tid = threadIdx.x;
			unsigned int i = blockIdx.x*(bsize*2) + tid;
			const unsigned int gridSize = blockDim.x * 2 * gridDim.x;
			sdata[tid] = 0;

			while (i < n) {
				sdata[tid] += g_in_data[i];
				if (i + bsize < n) {
					sdata[tid] += g_in_data[i + bsize];
				}
				i += gridSize;
			}
			__syncthreads();

			// Perform reductions in steps, reducing thread synchronization
			if (bsize >= 512) {
				if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
			}
			if (bsize >= 256) {
				if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
			}
			if (bsize >= 128) {
				if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
			}

			if (tid < 32)
				//Device::warpReduce_sum<T,bsize>(sdata, tid);
			{
				T val = sdata[tid];
				for (int offset = 16; offset > 0; offset >>= 1)
					val += __shfl_down_sync(0xffffffff, val, offset);
				if (tid == 0) sdata[0] = val;
			}

			if (tid == 0){
				g_out_data[blockIdx.x] = sdata[0];
			}
		}
}
//Calling Functions
//================
double cureduce_sum_d(double *input, const unsigned int n,const unsigned short stream){
	const unsigned int bsize=256;
	unsigned int gsize=(n + (2 * bsize) - 1) / (2 * bsize); 
	double *cachein, *cacheout;
	cudaMallocAsync(&cacheout,gsize*sizeof(double),streams[stream]);
	Kernels::reduce_sum<double,bsize><<<gsize,bsize,bsize*sizeof(double),streams[stream]>>>(input,cacheout,n);
	while(gsize>1){
		cudaMallocAsync(&cachein,gsize*sizeof(double),streams[stream]);
		cudaMemcpyAsync(cachein,cacheout,gsize*sizeof(double),cudaMemcpyDefault,streams[stream]);
		cudaFreeAsync(cacheout,streams[stream]);
		gsize>>=1;
		cudaMallocAsync(&cacheout,gsize*sizeof(double),streams[stream]);
		Kernels::reduce_sum<double,bsize><<<gsize,bsize,bsize*sizeof(double),streams[stream]>>>(cachein,cacheout,gsize);
		cudaFreeAsync(cachein,streams[stream]);
	}
	double output=0;
	cudaStreamSynchronize(streams[stream]);
	cudaMemcpyAsync(&output,cacheout,sizeof(double),cudaMemcpyDefault,streams[stream]);
	cudaStreamSynchronize(streams[stream]);
	cudaFreeAsync(cacheout,streams[stream]);
	return output;
}
void cuDslash(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
		Complex gamval[20], const unsigned short gamin[16], double *dk[nc], Complex_f jqq, float akappa,
		dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Dslash";
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ngorkov;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvolHalo, r+j*kvolHalo, kvol*sizeof(Complex),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	Kernels::cuDslash<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa);
	return;
}
void cuDslashd(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
		Complex gamval[20], const unsigned short gamin[16], double *dk[nc], Complex_f jqq, float akappa,
		dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Dslashd";
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ngorkov;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvol, r+j*kvolHalo, kvol*sizeof(Complex),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	Kernels::cuDslashd<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa);
	return;
}
void cuHdslash(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
		Complex gamval[20], const unsigned short gamin[16], double *dk[nc], float akappa, 
		dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Hdslash";
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ndirac;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvolHalo, r+j*kvolHalo, kvol*sizeof(Complex),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	Kernels::cuHdslash<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
	return;
}
void cuHdslashd(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
		Complex gamval[20], const unsigned short gamin[16],double *dk[nc], float akappa, 
		dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Hdslashd";
	//Spacelike term
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ndirac;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvol, r+j*kvolHalo, kvol*sizeof(Complex),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	Kernels::cuHdslashd<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
	return;
}

//Float editions
void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,
		Complex_f gamval[20],const unsigned short gamin[16],	float *dk[nc], Complex_f jqq, float akappa,
		dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Dslash_f";
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ngorkov;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvolHalo, r+j*kvolHalo, kvol*sizeof(Complex_f),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	Kernels::cuDslash<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa);
	return;
}
void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,
		Complex_f gamval[20],const unsigned short gamin[16],	float *dk[nc], Complex_f jqq, float akappa,
		dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Dslashd_f";
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ngorkov;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvol, r+j*kvolHalo, kvol*sizeof(Complex_f),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	Kernels::cuDslashd<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa);
	return;
}
void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id, Complex_f gamval[20],
		const unsigned short gamin[16],	float *dk[nc], float akappa, dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Hdslash_f";
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ndirac;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvolHalo, r+j*kvolHalo, kvol*sizeof(Complex_f),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	const int bsize=dimGrid.x*dimGrid.y*dimGrid.z;
	const int shareSize= ndim*bsize*nc*sizeof(Complex_f);
	Kernels::cuHdslash<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
	return;
}
void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,
		Complex_f gamval[20],const unsigned short gamin[16],float *dk[nc], float akappa,dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Hdslashd_f";
	int cuCpyStat=0;
	for(unsigned short j=0;j<nc*ndirac;j++)
		if((cuCpyStat=cudaMemcpy(phi+j*kvol, r+j*kvolHalo, kvol*sizeof(Complex_f),cudaMemcpyDefault))){
			fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
					CPYERROR,funcname,cuCpyStat);
			exit(cuCpyStat);
		}
	Kernels::cuHdslashd<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
	return;
}

void cuTranspose_z(Complex *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	Complex *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(Complex));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(Complex),cudaMemcpyDefault);
	Kernels::Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	cudaFree(holder);
}
void cuTranspose_c(Complex_f *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	Complex_f *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(Complex_f));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(Complex_f),cudaMemcpyDefault);
	Kernels::Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	cudaDeviceSynchronise();
	cudaFree(holder);
}
void cuTranspose_d(double *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	double *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(double));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(double),cudaMemcpyDefault);
	Kernels::Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	cudaFree(holder);
}
void cuTranspose_f(float *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	float *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(float));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(float),cudaMemcpyDefault);
	Kernels::Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	cudaFree(holder);
}
void cuTranspose_I(int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	int *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(int));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(int),cudaMemcpyDefault);
	Kernels::Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	cudaFree(holder);
}
void cuTranspose_U(unsigned int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	unsigned int *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(unsigned int));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(unsigned int),cudaMemcpyDefault);
	Kernels::Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	cudaFree(holder);
}

void cuMixed_Sumto(double *d, float *f,const unsigned int n,const dim3 dimGrid,const dim3 dimBlock){
	Kernels::Mixed_Sumto<<<dimGrid,dimBlock>>>(d,f,n);
	return;
}
