#include <clover.h>
#include <math.h>
#include <stdalign.h>
#include <clover.h>

//Calculating the clover and the leaves
//=====================================
#pragma omp declare simd
inline int Clover_SU2plaq(Complex_f *ut[nc], Complex_f *Leaves[nc], unsigned int *iu,  int i, int mu, int nu){
	const char *funcname = "SU2plaq";
	int uidm = iu[mu+ndim*i]; 
	/***
	 *	Let's take a quick moment to compare this to the analysis code.
	 *	The analysis code stores the gauge field as a 4 component real valued vector, whereas the produciton code
	 *	used two complex numbers.
	 *
	 *	Analysis code: u=(Re(u11),Im(u12),Re(u12),Im(u11))
	 *	Production code: u11=u[0]+I*u[3]	u12=u[2]+I*u[1]
	 *
	 *	This applies to the Leavess and a's below too
	 */
	Leaves[0][i]=ut[0][i*ndim+mu]*ut[0][uidm*ndim+nu]-ut[1][i*ndim+mu]*conj(ut[1][uidm*ndim+nu]);
	Leaves[1][i]=ut[0][i*ndim+mu]*ut[1][uidm*ndim+nu]+ut[1][i*ndim+mu]*conj(ut[0][uidm*ndim+nu]);

	int uidn = iu[nu+ndim*i]; 
	Complex_f a11=Leaves[0][i]*conj(ut[0][uidn*ndim+mu])+Leaves[1][i]*conj(ut[1][uidn*ndim+mu]);
	Complex_f a12=-Leaves[0][i]*ut[1][uidn*ndim+mu]+Leaves[1][i]*ut[0][uidn*ndim+mu];

	Leaves[0][i]=a11*conj(ut[0][i*ndim+nu])+a12*conj(ut[1][i*ndim+nu]);
	Leaves[1][i]=-a11*ut[1][i*ndim+nu]+a12*ut[0][i*ndim+nu];
	return 0;
}
#pragma omp declare simd
int Leaf(Complex_f *ut[nc], Complex_f *Leaves[nc], unsigned int *iu, unsigned int *id, int i, int mu, int nu, short leaf){
	char *funcname="Leaf";
	Complex_f a[2];
	unsigned int didm,didn,uidn,uidm;
	///NOTE: The multiplication order is the opposite of the textbook version. This is to maintain compatability with the
	///rest of the code which blindly copied the order from the earlier FORTRAN code
	switch(leaf){
		case(0):
			//Both positive is just a standard plaquette
			Clover_SU2plaq(ut,Leaves,iu,i,mu,nu);
			break;
		case(1):
			//\mu<0 and \nu>=0
			didm = id[mu+ndim*i];
			/// @f$U_\mu^\dagger\(x-\hat{\mu})U_\nu(x-\hat{\mu}\)@f$
			Leaves[0][i+kvol*leaf]=conj(ut[0][didm*ndim+mu])*ut[0][didm*ndim+nu]+ut[1][didm*ndim+mu]*conj(ut[1][didm*ndim+nu]);
			Leaves[1][i+kvol*leaf]=conj(ut[0][didm*ndim+mu])*ut[1][didm*ndim+nu]-ut[1][didm*ndim+mu]*conj(ut[0][didm*ndim+nu]);

			int uin_didm=id[nu+ndim*didm];
			/// @f$U_\mu^\dagger\(x-\hat{\mu})U_\nu(x+-hat{\mu}\)U_\mu(x-\hat{mu}+\hat{nu})@f$
			//a[0]=Leaves[0][i+kvol*leaf]*conj(ut[0][didm*ndim+nu])+conj(Leaves[1][i+kvol*leaf])*ut[1][didm*ndim+nu];
			a[0]=Leaves[0][i+kvol*leaf]*ut[0][uin_didm*ndim+mu]-Leaves[1][i+kvol*leaf]*conj(ut[1][uin_didm*ndim+mu]);
			//a[1]=Leaves[1][i+kvol*leaf]*conj(ut[0][didm*ndim+nu])-conj(Leaves[0][i+kvol*leaf])*ut[1][didm*ndim+nu];
			a[1]=Leaves[0][i+kvol*leaf]*ut[1][uin_didm*ndim+mu]+Leaves[1][i+kvol*leaf]*conj(ut[0][uin_didm*ndim+mu]);

			/// @f$U_\mu^\dagger\(x)U_\nu^\dagger(x+\hat{\mu}-\hat{\nu}\)U_\mu(x+\hat{mu}-\hat{nu})U_\nu^\dagger(x-\hat{\nu})@f$
			//Leaves[0][i+kvol*leaf]=a[0]*ut[0][didm*ndim+mu]-conj(a[1])*ut[1][didm*ndim+mu];
			Leaves[0][i+kvol*leaf]=a[0]*conj(ut[0][i*ndim+nu])+a[1]*conj(ut[1][i*ndim+nu]);
			Leaves[1][i+kvol*leaf]=-a[0]*ut[1][i*ndim+nu]+a[1]*ut[0][i*ndim+nu];
			break;
		case(2):
			//\mu>=0 and \nu<0
			//TODO: Figure out down site index
			//Another awkward index
			uidm = iu[mu+ndim*i]; int din_uidm=id[nu+ndim*uidm];
			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{mu}-\hat{\nu})@f$
			Leaves[0][i+kvol*leaf]=ut[0][i*ndim+mu]*conj(ut[0][din_uidm*ndim+nu])+ut[1][i*ndim+mu]*conj(ut[1][din_uidm*ndim+nu]);
			Leaves[1][i+kvol*leaf]=-ut[0][i*ndim+mu]*ut[1][din_uidm*ndim+nu]+ut[1][i*ndim+mu]*ut[0][din_uidm*ndim+nu];

			didn = id[nu+ndim*i]; 
			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{mu}-\hat{\nu})U_\mu^\dagger(x-\hat{nu}\)@f$
			a[0]=Leaves[0][i+kvol*leaf]*conj(ut[0][didn*ndim+mu])+Leaves[1][i+kvol*leaf]*conj(ut[1][didn*ndim+mu]);
			a[1]=-Leaves[0][i+kvol*leaf]*ut[1][didn*ndim+mu]+Leaves[1][i+kvol*leaf]*ut[0][didn*ndim+mu];

			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{mu}-\hat{\nu})U_\mu^\dagger(x-\hat{nu}\)U_\nu(x-\hat{\nu})@f$
			Leaves[0][i+kvol*leaf]=a[0]*ut[0][didn*ndim+nu]-a[1]*conj(ut[1][didn*ndim+nu]);
			Leaves[1][i+kvol*leaf]=a[0]*ut[1][didn*ndim+nu]+a[1]*conj(ut[0][didn*ndim+nu]);

			break;
		case(3):
			//\mu<0 and \nu<0
			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu})@f$
			didm = id[mu+ndim*i];int dim_didn=id[nu+ndim*didm];
			Leaves[0][i+kvol*leaf]=conj(ut[0][didm*ndim+mu])*conj(ut[0][dim_didn*ndim+nu])-ut[1][didm*ndim+mu]*conj(ut[1][dim_didn*ndim+nu]);
			Leaves[1][i+kvol*leaf]=-conj(ut[0][didm*ndim+mu])*ut[1][dim_didn*ndim+nu]-ut[1][didm*ndim+mu]*ut[0][dim_didn*ndim+nu];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(x-\hat{\mu}-\hat{\nu})@f$
			a[0]=Leaves[0][i+kvol*leaf]*ut[0][dim_didn*ndim+mu]-Leaves[1][i+kvol*leaf]*conj(ut[1][dim_didn*ndim+mu]);
			a[1]=Leaves[0][i+kvol*leaf]*ut[1][dim_didn*ndim+mu]+Leaves[1][i+kvol*leaf]*conj(ut[0][dim_didn*ndim+mu]);

			didn = id[nu+ndim*i]; 
			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(x-\hat{\mu}-\hat{\nu})U_\nu(x-\hat{\nu})@f$
			Leaves[0][i+kvol*leaf]=a[0]*ut[0][didn*ndim+nu]-a[1]*conj(ut[1][didn*ndim+nu]);
			Leaves[1][i+kvol*leaf]=a[0]*ut[1][didn*ndim+nu]+a[1]*conj(ut[0][didn*ndim+nu]);
			break;
	}
#ifdef _DEBUG
	if(isnan(creal(Leaves[0][i+kvol*leaf]))||isnan(cimag(Leaves[0][i+kvol*leaf]))|| \
			isnan(creal(Leaves[1][i+kvol*leaf]))||isnan(cimag(Leaves[1][i+kvol*leaf]))){
		printf("Leaves: Index %d, mu %d, nu %d, leaf %d is NaN\n"\
				"Leaf 0=%e+i%e\tLeaf 1=%e+i%e\n",i,mu,nu,leaf,\
				creal(Leaves[0][i+kvol*leaf]),cimag(Leaves[0][i+kvol*leaf]),\
				creal(Leaves[1][i+kvol*leaf]),cimag(Leaves[1][i+kvol*leaf]));
		abort();
	}
	//	Leaves[0][i+kvol*leaf]=(1.0+I)/sqrt(4.0);Leaves[1][i+kvol*leaf]=Leaves[0][i+kvol*leaf];
	//Leaves[0][i+kvol*leaf]=I;Leaves[1][i+kvol*leaf]=0;
	float norm=sqrt(creal(conj(Leaves[0][i+kvol*leaf])*Leaves[0][i+kvol*leaf]+Leaves[1][i+kvol*leaf]*conj(Leaves[1][i+kvol*leaf])));
	if(fabs(norm-1.0f)>=1e-6){
		printf("Leaves: Index %d, mu %d, nu %d, leaf %d is not unitary\n"\
				"Leaf 0=%e+i%e\tLeaf 1=%e+i%e\tnorm=%e\n",i,mu,nu,leaf,\
				creal(Leaves[0][i+kvol*leaf]),cimag(Leaves[0][i+kvol*leaf]),\
				creal(Leaves[1][i+kvol*leaf]),cimag(Leaves[1][i+kvol*leaf]),sqrt(norm));
		abort();
	}
#endif
	return 0;
}
inline int Half_Clover(Complex_f *clover[nc],Complex_f *Leaves[nc], Complex_f *ut[nc], unsigned int *iu, unsigned int *id, int i, int mu, int nu, short clov){
	const char funcname[] ="Half_Clover";
#pragma omp simd
	for(short leaf=0;leaf<ndim;leaf++)
	{
		Leaf(ut,Leaves,iu,id,i,mu,nu,leaf);
		clover[0][clov*kvol+i]+=Leaves[0][i+kvol*leaf]; clover[1][clov*kvol+i]+=Leaves[1][i+kvol*leaf];
	}
	return 0;
}
int Clover(Complex_f *clover[nc],Complex_f *Leaves[6][2],Complex_f *ut[2], unsigned int *iu, unsigned int *id){
	const char funcname[]="Clover";
#ifdef __NVCC__
	cuClover(clover,Leaves,ut,iu,id);
#else
	//Six clovers
	clover[0]=(Complex_f *)aligned_alloc(AVX,6*kvol*sizeof(Complex_f));
	clover[1]=(Complex_f *)aligned_alloc(AVX,6*kvol*sizeof(Complex_f));
	for(unsigned int mu=0;mu<ndim-1;mu++)
		for(unsigned int nu=mu+1;nu<ndim;nu++)
			if(mu!=nu){
				//Clover index
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
				//Allocate clover memory
				//Note that the clover is completely local, so doesn't need a halo for MPI
				Leaves[clov][0]=(Complex_f *)aligned_alloc(AVX,kvol*ndim*sizeof(Complex_f));
				Leaves[clov][1]=(Complex_f *)aligned_alloc(AVX,kvol*ndim*sizeof(Complex_f));
#pragma omp parallel for
				for(unsigned int i=0;i<kvol;i++)
				{
					clover[0][clov*kvol+i]=0;clover[1][clov*kvol+i]=0;
					Half_Clover(clover,Leaves[clov],ut,iu,id,i,mu,nu,clov);	
					//creal(clover[0]) drops so we are traceless. And everything else just gets doubled
					clover[0][clov*kvol+i]-=conj(clover[0][clov*kvol+i]);	clover[1][clov*kvol+i]+=clover[1][clov*kvol+i];
#ifdef _DEBUG
					if(isnan(creal(clover[0][clov*kvol+i]))||isnan(cimag(clover[0][clov*kvol+i]))||isnan(creal(clover[1][clov*kvol+i]))|| \
							isnan(cimag(clover[1][clov*kvol+i]))){
						printf("Clover: Index %d, mu %d, nu %d, clover %d is NaN\n"\
								"Clover 0=%e+i%e\tClover 1=%e+i%e\n",i,mu,nu,clov,\
								creal(clover[0][clov*kvol+i]),cimag(clover[0][clov*kvol+i]),\
								creal(clover[1][clov*kvol+i]),cimag(clover[1][clov*kvol+i]));
						abort();
					}
#endif
					//Don't forget the factor out front!
					//Uh Oh. G&L says -i/8 here. But hep-lat/9605038 and other sources say +1/8
					//It gets worse in the C_sw definition. We have a 1/2. They have +i/4
					clover[0][clov*kvol+i]*=(-I/8.0);	clover[1][clov*kvol+i]*=(-I/8.0);
#ifdef _DEBUG
					if(fabs(cimag(clover[0][clov*kvol+i]))>1e-6){
						printf("Clover: Index %d, mu %d, nu %d, clover %d is not hermitian\n"\
								"Clover 0=%e+i%e\tClover 1=%e+i%e\n",i,mu,nu,clov,\
								creal(clover[0][clov*kvol+i]),cimag(clover[0][clov*kvol+i]),\
								creal(clover[1][clov*kvol+i]),cimag(clover[1][clov*kvol+i]));
						abort();
					}
#endif
				}
			}
#endif
	return 0;
}

//Multiplication for Congradq
//=========================
// Congradq only acts on flavour 1
int ByClover(Complex *phi, Complex *r, Complex *clover[nc], Complex *sigval, unsigned short *sigin){
	const char funcname[] = "ByClover";
#ifdef __NVCC__
	cuByClover(phi, r, clover, sigval, sigin);
#else
#pragma omp parallel for
	for(int i=0;i<kvol;i+=AVX){
		//Prefetched r and Phi array
#pragma omp simd
		for(unsigned short j =0;j<AVX;j++)
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				Complex phi_s[nc];
				Complex r_s[nc];
				Complex clov_s[nc];
				unsigned short idirac = igorkov%4;
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
					phi_s[c]=0;

				for(unsigned short clov=0;clov<6;clov++){
					const unsigned short igork1 = (igorkov<4) ? sigin[clov*ndirac+idirac] : sigin[clov*ndirac+idirac]+4;
#pragma unroll
					for(unsigned short c=0; c<nc; c++){
						r_s[c]=r[((i+j)*ngorkov+igork1)*nc+c];
						clov_s[c]=clover[c][clov*kvol+i+j];

#ifdef _DEBUG
						if(isnan(creal(r_s[c]))||isnan(cimag(r_s[c]))){
							printf("r_s: Index %d, colour %d c and gorkov index %d (idirac %d and igork1 %d) is NaN\n"\
									"Sigma matrices = %e+%e\n"\
									"Clover: %e+i%e\\n"\
									"r_s=%e+i%e\n",i+j,c,igorkov,idirac,igork1,
									creal(sigval[clov*ndirac+idirac]),cimag(sigval[clov*ndirac+idirac]),
									creal(clov_s[c]),cimag(clov_s[c]), creal(r_s[c]),cimag(r_s[c]));
							abort();
						}
#endif
					}

					///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
					phi_s[0]+=sigval[clov*ndirac+idirac]*(clov_s[0]*r_s[0]+clov_s[1]*r_s[1]);
					phi_s[1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]+conj(clov_s[0])*r_s[1]);
#ifdef _DEBUG
					if(isnan(creal(phi_s[0]))||isnan(cimag(phi_s[0]))||isnan(creal(phi_s[1]))||isnan(cimag(phi_s[1]))){
						fprintf(stderr, "phi_s: Index %d and gorkov index %d (idirac %d and igork1 %d) is NaN\n"\
								"Sigma matrices = %e+%e\n"\
								"Clover 0: %e+i%e\tClover 1:%e+i%e\n"\
								"r_s 0=%e+i%e\tr_s 1=%e+i%e\n"\
								"phi_s 0=%e+i%e\tphi_s 1=%e+i%e\n",i+j,igorkov,idirac,igork1,
								creal(sigval[clov*ndirac+idirac]),cimag(sigval[clov*ndirac+idirac]),
								creal(clov_s[0]),cimag(clov_s[0]),creal(clov_s[1]),cimag(clov_s[1]),
								creal(r_s[0]),cimag(r_s[0]),creal(r_s[1]),cimag(r_s[1]),
								creal(phi_s[0]),cimag(phi_s[0]),creal(phi_s[1]),cimag(phi_s[1]));
						abort();
					}
#endif
				}
#pragma unroll
				for(int c=0; c<nc; c++){
					phi[((i+j)*ngorkov+igorkov)*nc+c]+=2*phi_s[c];
				}
			}
	}
#endif
	return 0;
}
int HbyClover(Complex *phi, Complex *r, Complex *clover[nc], Complex *sigval, const float akappa, unsigned short *sigin){
	const char funcname[] = "HbyClover";
#ifdef __NVCC__
	cuHbyClover(phi, r, clover, sigval, sigin);
#else
#pragma omp parallel for
	for(int i=0;i<kvol;i+=AVX){
		//Prefetched r and Phi array
#pragma omp simd
		for(int j =0;j<AVX;j++)
			for(int idirac=0; idirac<ndirac; idirac++){
				alignas(AVX) Complex phi_s[nc][AVX];
				alignas(AVX) Complex r_s[nc][AVX];
				alignas(AVX) Complex clov_s[nc][AVX];
#pragma unroll
				for(int c=0; c<nc; c++){
					phi_s[c][j]=0;
				}
				for(unsigned int clov=0;clov<6;clov++){
					const unsigned short igork1 = sigin[clov*ndirac+idirac];	
#pragma unroll
					for(int c=0; c<nc; c++){
						r_s[c][j]=r[((i+j)*ndirac+igork1)*nc+c];
						clov_s[c][j]=clover[c][clov*kvol+i+j];
					}
					///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
					///Also note that clov_s[0] is real as the clover is hermitian
					phi_s[0][j]+=akappa*sigval[clov*ndirac+idirac]*(creal(clov_s[0][j])*r_s[0][j]+clov_s[1][j]*r_s[1][j]);
					phi_s[1][j]+=akappa*sigval[clov*ndirac+idirac]*(conj(clov_s[1][j])*r_s[0][j]+creal(clov_s[0][j])*r_s[1][j]);
				}
#pragma unroll
				for(int c=0; c<nc; c++)
					///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
					phi[((i+j)*ndirac+idirac)*nc+c]+=2*phi_s[c][j];
			}
	}
#endif
	return 0;
}
//Float versions
int ByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc], Complex_f *sigval, unsigned short *sigin){
	const char funcname[] = "ByClover";
#ifdef __NVCC__
	cuByClover(phi, r, clover, sigval, sigin);
#else
#pragma omp parallel for
	for(int i=0;i<kvol;i+=AVX){
		//Prefetched r and Phi array
#pragma omp simd
		for(unsigned short j =0;j<AVX;j++)
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				Complex_f phi_s[nc];
				Complex_f r_s[nc];
				Complex_f clov_s[nc];
				unsigned short idirac = igorkov%4;
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
					phi_s[c]=0;

				for(unsigned short clov=0;clov<6;clov++){
					const unsigned short igork1 = (igorkov<4) ? sigin[clov*ndirac+idirac] : sigin[clov*ndirac+idirac]+4;
#pragma unroll
					for(unsigned short c=0; c<nc; c++){
						r_s[c]=r[((i+j)*ngorkov+igork1)*nc+c];
						clov_s[c]=clover[c][clov*kvol+i+j];

#ifdef _DEBUG
						if(isnan(creal(r_s[c]))||isnan(cimag(r_s[c]))){
							printf("r_s: Index %d, colour %d c and gorkov index %d (idirac %d and igork1 %d) is NaN\n"\
									"Sigma matrices = %e+%e\n"\
									"Clover: %e+i%e\\n"\
									"r_s=%e+i%e\n",i+j,c,igorkov,idirac,igork1,
									creal(sigval[clov*ndirac+idirac]),cimag(sigval[clov*ndirac+idirac]),
									creal(clov_s[c]),cimag(clov_s[c]), creal(r_s[c]),cimag(r_s[c]));
							abort();
						}
#endif
					}

					///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
					phi_s[0]+=sigval[clov*ndirac+idirac]*(creal(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
					phi_s[1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]+creal(clov_s[0])*r_s[1]);
#ifdef _DEBUG
					if(isnan(creal(phi_s[0]))||isnan(cimag(phi_s[0]))||isnan(creal(phi_s[1]))||isnan(cimag(phi_s[1]))){
						fprintf(stderr, "phi_s: Index %d and gorkov index %d (idirac %d and igork1 %d) is NaN\n"\
								"Sigma matrices = %e+%e\n"\
								"Clover 0: %e+i%e\tClover 1:%e+i%e\n"\
								"r_s 0=%e+i%e\tr_s 1=%e+i%e\n"\
								"phi_s 0=%e+i%e\tphi_s 1=%e+i%e\n",i+j,igorkov,idirac,igork1,
								creal(sigval[clov*ndirac+idirac]),cimag(sigval[clov*ndirac+idirac]),
								creal(clov_s[0]),cimag(clov_s[0]),creal(clov_s[1]),cimag(clov_s[1]),
								creal(r_s[0]),cimag(r_s[0]),creal(r_s[1]),cimag(r_s[1]),
								creal(phi_s[0]),cimag(phi_s[0]),creal(phi_s[1]),cimag(phi_s[1]));
						abort();
					}
#endif
				}
#pragma unroll
				for(int c=0; c<nc; c++){
					phi[((i+j)*ngorkov+igorkov)*nc+c]+=2*phi_s[c];
				}
			}
	}
#endif
	return 0;
}
int HbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc], Complex_f *sigval, const float akappa, unsigned short *sigin){
	const char funcname[] = "HbyClover";
#ifdef __NVCC__
	cuHbyClover(phi, r, clover, sigval, sigin);
#else
#pragma omp parallel for
	for(int i=0;i<kvol;i+=AVX){
		//Prefetched r and Phi array
#pragma omp simd
		for(int j =0;j<AVX;j++)
			for(int idirac=0; idirac<ndirac; idirac++){
				alignas(AVX) Complex_f phi_s[nc][AVX];
				alignas(AVX) Complex_f r_s[nc][AVX];
				alignas(AVX) Complex_f clov_s[nc][AVX];
#pragma unroll
				for(int c=0; c<nc; c++){
					phi_s[c][j]=0;
				}
				for(unsigned int clov=0;clov<6;clov++){
					const unsigned short igork1 = sigin[clov*ndirac+idirac];	
#pragma unroll
					for(int c=0; c<nc; c++){
						r_s[c][j]=r[((i+j)*ndirac+igork1)*nc+c];
						clov_s[c][j]=clover[c][clov*kvol+i+j];
					}
					///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
					phi_s[0][j]+=akappa*sigval[clov*ndirac+idirac]*(creal(clov_s[0][j])*r_s[0][j]+clov_s[1][j]*r_s[1][j]);
					phi_s[1][j]+=akappa*sigval[clov*ndirac+idirac]*(conj(clov_s[1][j])*r_s[0][j]+creal(clov_s[0][j])*r_s[1][j]);
				}
#pragma unroll
				for(int c=0; c<nc; c++)
					///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
					phi[((i+j)*ndirac+idirac)*nc+c]+=2*phi_s[c][j];
			}
	}
#endif
	return 0;
}

//Clover force terms 144 in total...
//=================================
//Calling function
//TODO: X1 and X2 contributions
int Clover_Force(double *dSdpi, Complex_f *Leaves[6][nc], Complex_f *X1, Complex_f *X2, Complex_f *sigval,unsigned short *sigin){
	const char funcname[]="Clover_Force";
#ifdef __NVCC__
	cuClover_Force(dSdpi,Leaves,X1,X2,sigval,sigin);
#else
	//TODO: Make this more CUDA friendly? Or just have a CUDA call above
	for(unsigned short adj=0;adj>nadj;adj++)
		for(unsigned short mu=0;mu<ndim;mu++)
			for(unsigned short nu=mu+1;nu<ndim;nu++){
				///Only three clovers @f$\mu\ne\nu@f$ contribute to the force term in each @f$\mu@f$
				///Out of these clovers only three leaves contribute, with the leaf in the @f$\mu@f$ direction contributing
				///twice (daggered and not daggered)
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
#pragma omp parallel for
				for(unsigned int i=0;i<kvol;i++){
					Complex_f Fleaf[2]={0,0};
					///Only three clovers @f$\mu\ne\nu@f$ contribute to the force term
					///Out of these clovers only two leaves contribute (containing @f$U_\mu@f$). Daggered and undaggered
					///Additionally, @f$\sigma_{\nu\mu}F_{\nu\mu}=\sigma_{\mu\nu}F_{\mu\nu}@f$ so we can double the final answer

					///Contribution from @f$(f_{\mu\nu})@f$
					///Contribution 1 is the normal plaquette so leaf 0
					GenLeaf(Fleaf,Leaves[clov],i,0,adj,true);
					///Contribution 3 is the normal plaquette daggered so leaf 0
					GenLeafd(Fleaf,Leaves[clov],i,0,adj,false);

					///Contribution 2 is the leaf containing @f$ U_\mu@f$ link and @f$ \nu<0@f$)
					GenLeaf(Fleaf,Leaves[clov],i,2,adj,true);
					///Contribution 4 is the leaf containing @f$ U^dagger_\mu@f$ link and @f$ \nu<0@f$)
					GenLeafd(Fleaf,Leaves[clov],i,2,adj,false);

					///NOTE: The clover is scaled by -i/8.0, but the leaves were not. We do that scaling here.
					///		The 4.0 instead of 8.0 is due to the second contribution from @f$\nu\mu@f$
					Fleaf[0]*=-I/4.0f; Fleaf[1]*=-I/4.0f;

					//Actual force stuff
					for(unsigned short idirac=0;idirac<ndirac;idirac++){
						const unsigned short igork1 = sigin[clov*ndirac+idirac];	
						dSdpi[(i*nadj+adj)*ndim+mu]-=cimag(sigval[clov*ndirac+idirac]*(
									conj(X1[(i*ndirac)*nc])*(Fleaf[0]*X2[(i*ndirac+igork1)*nc]+Fleaf[1]*X2[(i*ndirac+igork1)*nc+1])+
									conj(X1[(i*ndirac)*nc+1])*(-conj(Fleaf[1])*X2[(i*ndirac+igork1*nc)]+conj(Fleaf[0])*X2[(i*ndirac+igork1)*nc+1])));
					}
				}
			}
#endif
	return 0;
}
inline int Clover_free(Complex_f *clover[nc],Complex_f *Leaves[6][nc]){
	for(unsigned short c=0;c<nc;c++){
#ifdef __NVCC__
		cudaFreeAsync(clover[c],streams[c]);
#else
		free(clover[c]);
#endif
		for(unsigned short clov=0;clov<6;clov++){
#ifdef __NVCC__
			cudaFreeAsync(Leaves[clov][c],streams[clov+nc]);
#else
			free(Leaves[clov][c]);
#endif
		}
	}
	return 0;	
}

//Generator by Leaf
inline int GenLeaf(Complex_f Fleaf[nc], Complex_f *Leaves[nc],const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm){
	const char funcname[] = "GenLeaf";
	//Adding or subtracting this term
	const short sign = (pm) ? 1 : -1;
	//Which generator are we multiplying by? Zero indexed so subtract one from your usual index in textbooks
	switch(adj){
		case(0):
			Fleaf[0]+=sign*Leaves[1][i+kvol*leaf];		Fleaf[1]+=sign*Leaves[0][i+kvol*leaf];
		case(1):
			Fleaf[0]+=-sign*I*Leaves[1][i+kvol*leaf];	Fleaf[1]+=sign*I*Leaves[0][i+kvol*leaf];
		case(2):
			Fleaf[0]+=sign*Leaves[0][i+kvol*leaf];		Fleaf[1]+=-sign*Leaves[1][i+kvol*leaf];
	}
	return 0;
}
inline int GenLeafd(Complex_f Fleaf[nc], Complex_f *Leaves[nc],const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm){
	const char funcname[] = "GenLeafd";
	//Adding or subtracting this term
	const short sign = (pm) ? 1 : -1;
	//Which generator are we multiplying by? Zero indexed so subtract one from your usual index in textbooks
	switch(adj){
		case(0):
			Fleaf[0]+=-sign*Leaves[1][i+kvol*leaf]; 		Fleaf[1]+=sign*conj(Leaves[0][i+kvol*leaf]);
		case(1):
			Fleaf[0]+=sign*I*Leaves[1][i+kvol*leaf];		Fleaf[1]+=sign*I*conj(Leaves[0][i+kvol*leaf]);
		case(2):
			Fleaf[0]+=sign*conj(Leaves[0][i+kvol*leaf]); 	Fleaf[1]+=sign*conj(Leaves[1][i+kvol*leaf]);
	}
	return 0;
}
//initialisation
//
int Init_clover(Complex *sigval, Complex_f *sigval_f,unsigned short *sigin, float c_sw){
	const char funcname[] = "Init_clover";
	unsigned short __attribute__((aligned(AVX))) sigin_t[6][4] =	{{0,1,2,3},{1,0,3,2},{1,0,3,2},{1,0,3,2},{1,0,3,2},{0,1,2,3}};
	//The sigma matrices are the commutators of the gamma matrices. These are antisymmetric when you swap the indices
	//0 is sigma_0,1
	//1 is sigma_0,2
	//2 is sigma_0,3
	//3 is sigma_1,2
	//4 is sigma_1,3
	//5 is sigma_2,3
	Complex	__attribute__((aligned(AVX)))	sigval_t[6][4] =	{{-1,1,-1,1},{-I,I,-I,I},{1,1,-1,-1},{-1,-1,-1,-1},{-I,I,I,-I},{1,-1,-1,1}};
	//Complex	__attribute__((aligned(AVX)))	sigval_t[6][4] =	{{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1}};
	//We mutiply by 1/2 and c_sw here since sigval is never used without them.
#if defined USE_BLAS
	cblas_zdscal(6*4, 0.5*c_sw, sigval_t, 1);
#else
#pragma omp parallel for simd collapse(2) aligned(sigval,sigval_f:AVX)
	for(int i=1;i<6;i++)
		for(int j=0;j<4;j++)
			sigval_t[i][j]*=c_sw*0.5;
#endif

#ifdef __NVCC__
	cudaMemcpy(sigval,sigval_t,6*4*sizeof(Complex),cudaMemcpyDefault);
	cudaMemcpy(sigin,sigin_t,6*4*sizeof(short),cudaMemcpyDefault);
	cuComplex_convert(sigval_f,sigval,24,true,dimBlockOne,dimGridOne);	
#else
	memcpy(sigval,sigval_t,6*4*sizeof(Complex));
	memcpy(sigin,sigin_t,6*4*sizeof(short));
	for(int i=0;i<6*4;i++)
		sigval_f[i]=(Complex_f)sigval[i];
#endif
}
