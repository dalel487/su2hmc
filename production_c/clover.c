/**
 * @file 	clover.c
 *
 * @brief	Routines related to clover improved wilson fermions
 *
 * @author	D. Lawlor
 */
#include <clover.h>
#include <stdalign.h>

//Calculating the clover and the leaves
//=====================================
inline int Clover_SU2plaq(Complex_f *ut[2], Complex_f *Leaves[2], unsigned int *iu,  int i, int mu, int nu){
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

	//TODO: Figure out how we want to label the leaves. 12 clovers in total. Each with 4 leaves. The below should work for
	//the plaquette as the first leaf
	Leaves[0][i*ndim]=ut[0][i*ndim+mu]*ut[0][uidm*ndim+nu]-ut[1][i*ndim+mu]*conj(ut[1][uidm*ndim+nu]);
	//Leaves[0][i*ndim]=ut[0][i*ndim+mu]*ut[0][uidm*ndim+nu]-conj(ut[1][i*ndim+mu])*ut[1][uidm*ndim+nu];
	Leaves[1][i*ndim]=ut[0][i*ndim+mu]*ut[1][uidm*ndim+nu]+ut[1][i*ndim+mu]*conj(ut[0][uidm*ndim+nu]);
	//Leaves[1][i*ndim]=ut[1][i*ndim+mu]*ut[0][uidm*ndim+nu]+conj(ut[0][i*ndim+mu])*ut[1][uidm*ndim+nu];

	int uidn = iu[nu+ndim*i]; 
	Complex_f a11=Leaves[0][i*ndim]*conj(ut[0][uidn*ndim+mu])+Leaves[1][i*ndim]*conj(ut[1][uidn*ndim+mu]);
	//Complex_f a11=Leaves[0][i*ndim]*conj(ut[0][uidn*ndim+mu])+conj(Leaves[1][i*ndim])*ut[1][uidn*ndim+mu];
	Complex_f a12=-Leaves[0][i*ndim]*ut[1][uidn*ndim+mu]+Leaves[1][i*ndim]*ut[0][uidn*ndim+mu];
	//Complex_f a12=Leaves[1][i*ndim]*conj(ut[0][uidn*ndim+mu])-conj(Leaves[0][i*ndim])*ut[1][uidn*ndim+mu];

	Leaves[0][i*ndim]=a11*conj(ut[0][i*ndim+nu])+a12*conj(ut[1][i*ndim+nu]);
	//Leaves[0][i*ndim]=a11*conj(ut[0][i*ndim+nu])+conj(a12)*ut[1][i*ndim+nu];
	Leaves[1][i*ndim]=-a11*ut[1][i*ndim+nu]+a12*ut[0][i*ndim+mu];
	//Leaves[1][i*ndim]=a12*conj(ut[0][i*ndim+nu])-conj(a11)*ut[1][i*ndim+mu];
	return 0;
}
int Leaf(Complex_f *ut[2], Complex_f *Leaves[2], unsigned int *iu, unsigned int *id, int i, int mu, int nu, short leaf){
	char *funcname="Leaf";
	Complex_f a[2];
	unsigned int didm,didn,uidn,uidm;
	///NOTE: The multiplication order is the opposite of the textbook version. This is to maintain compatability with the
	///rest of the code which blindly copied the order from the earlier FORTRAN code
	switch(leaf){
		case(0):
			//Both positive is just a standard plaquette
			return Clover_SU2plaq(ut,Leaves,iu,i,mu,nu);
		case(1):
			//\mu<0 and \nu>=0
			didm = id[mu+ndim*i];
			/// @f$U_\nu\left(x-\hat{\mu}\right)U^\dagger_\mu(x-\mu)@f$ 
			//Leaves[0][i*ndim+leaf]=ut[0][i*ndim+nu]*conj(ut[0][uin_didm*ndim+mu])+conj(ut[1][i*ndim+nu])*ut[1][uin_didm*ndim+mu];
			Leaves[0][i*ndim+leaf]=ut[0][didm*ndim+nu]*conj(ut[0][didm*ndim+mu])+conj(ut[1][didm*ndim+nu])*ut[1][didm*ndim+mu];
			//Leaves[1][i*ndim+leaf]=ut[1][i*ndim+nu]*conj(ut[0][uin_didm*ndim+mu])-conj(ut[0][i*ndim+nu])*ut[1][uin_didm*ndim+mu];
			Leaves[1][i*ndim+leaf]=ut[1][didm*ndim+nu]*conj(ut[0][didm*ndim+mu])-conj(ut[0][didm*ndim+nu])*ut[1][didm*ndim+mu];

			int uin_didm=id[nu+ndim*didm];
			/// @f$\left(U_\mu\left(x-\hat{\mu}+\hat{\nu}\right)\right)*U_\nu\left(x-\hat{\mu}\right)U^\dagger_\mu(x-\mu)@f$ 
			//a[0]=Leaves[0][i*ndim+leaf]*conj(ut[0][didm*ndim+nu])+conj(Leaves[1][i*ndim+leaf])*ut[1][didm*ndim+nu];
			a[0]=ut[0][uin_didm*ndim+mu]*Leaves[0][i*ndim+leaf]-conj(ut[1][uin_didm*ndim+mu])*Leaves[1][i*ndim+leaf];
			//a[1]=Leaves[1][i*ndim+leaf]*conj(ut[0][didm*ndim+nu])-conj(Leaves[0][i*ndim+leaf])*ut[1][didm*ndim+nu];
			a[1]=ut[1][uin_didm*ndim+mu]*Leaves[1][i*ndim+leaf]+conj(ut[0][uin_didm*ndim+mu])*Leaves[1][i*ndim+leaf];

			/// @f$\left(U^\dagger_\nu(x)\right)*U_\mu\left(x-\hat{\mu}+\hat{\nu}\right)U_\nu\left(x-\hat{\mu}\right)U^\dagger_\mu(x-\mu)@f$ 
			//Leaves[0][i*ndim+leaf]=a[0]*ut[0][didm*ndim+mu]-conj(a[1])*ut[1][didm*ndim+mu];
			Leaves[0][i*ndim+leaf]=conj(ut[0][i*ndim+nu])*a[0]+conj(ut[1][i*ndim+nu])*a[1];
			Leaves[1][i*ndim+leaf]=ut[0][i*ndim+nu]*a[1]-ut[1][i*ndim+nu]*conj(a[0]);
			return 0;
		case(2):
			//\mu>=0 and \nu<0
			//TODO: Figure out down site index
			//Another awkward index
			uidm = iu[mu+ndim*i]; int din_uidm=id[nu+ndim*uidm];
			/// @f$U^\dagger_\nu\left(x+\hat{\mu}-\hat{nu}\right)U_\mu\left(x\right)@f$
			//Leaves[0][i*ndim+leaf]=conj(ut[0][didn*ndim+nu])*ut[0][didn*ndim+mu]+conj(ut[1][didn*ndim+nu])*ut[1][didn*ndim+mu];
			Leaves[0][i*ndim+leaf]=ut[0][din_uidm*ndim+nu]*conj(ut[0][i*ndim+mu])+conj(ut[1][din_uidm*ndim+nu])*ut[1][i*ndim+mu];
			//Leaves[1][i*ndim+leaf]=-ut[1][didn*ndim+mu]*ut[0][didn*ndim+nu]+ut[0][didn*ndim+nu]*ut[1][didn*ndim+mu];
			Leaves[0][i*ndim+leaf]=ut[1][din_uidm*ndim+nu]*conj(ut[0][i*ndim+mu])-conj(ut[0][din_uidm*ndim+nu])*ut[1][i*ndim+mu];

			didn = id[nu+ndim*i]; 
			/// @f$U_\mu^\dagger\left(x-\hat{\nu}\right)U^\dagger_\nu\left(x+\hat{\mu}-\hat{nu}\right)U_\mu\left(x\right)@f$
			a[0]=ut[0][didn*ndim+mu]*Leaves[0][i*ndim+leaf]-conj(ut[1][didn*ndim+mu])*Leaves[1][i*ndim+leaf];
			a[1]=ut[1][didn*ndim+mu]*Leaves[0][i*ndim+leaf]+conj(ut[0][didn*ndim+mu])*Leaves[1][i*ndim+leaf];

			/// @f$((U_-\nu(x)*U_\mu(x-\nu))*U_ν(x+\mu-\nu))*U_-\mu(x+\mu)=((U^\dagger_\nu(x-\nu)*U_\mu(x-ν))*U_\nu(x+\mu-\nu))*U^\dagger_\mu(x)@f$
			Leaves[0][i*ndim+leaf]=ut[0][didn*ndim+nu]*a[0]-conj(ut[1][didn*ndim+nu])*a[1];
			Leaves[1][i*ndim+leaf]=ut[1][didn*ndim+nu]*a[0]+conj(ut[0][didn*ndim+nu])*a[1];
			return 0;
		case(3):
			//\mu<0 and \nu<0
			/// @f$U_\nu^\dagger\left(x-\hat{\mu}-\hat{\nu}\right)U_\mu^\dagger\left(x-\hat{\mu}\right)@f$
			didm = id[mu+ndim*i];int dim_didn=id[nu+ndim*didm];
			Leaves[0][i*ndim+leaf]=conj(ut[0][dim_didn*ndim+nu])*conj(ut[0][didm*ndim+mu])-conj(ut[1][dim_didn*ndim+nu])*ut[1][didm*ndim+mu];
			Leaves[0][i*ndim+leaf]=-ut[1][dim_didn*ndim+nu]*conj(ut[0][didm*ndim+mu])-ut[0][dim_didn*ndim+nu]*ut[1][didm*ndim+mu];

			/// @f$U_\mu\left(x-\hat{\mu}-\hat{\nu}\right)U_\nu^\dagger\left(x-\hat{\mu}-\hat{\nu}\right)U_\mu^\dagger\left(x-\hat{\mu}\right)@f$
			a[0]=ut[0][dim_didn*ndim+mu]*Leaves[0][i*ndim+leaf]-conj(ut[1][dim_didn*ndim+mu])*Leaves[1][i*ndim+leaf];
			a[1]=ut[1][dim_didn*ndim+mu]*Leaves[0][i*ndim+leaf]+conj(ut[0][dim_didn*ndim+mu])*Leaves[1][i*ndim+leaf];

			didn = id[nu+ndim*i]; 
			/// @f$U_\nu\left(x-\hat{\nu}\right)U_\mu\left(x-\hat{\mu}-\hat{\nu}\right)U_\nu^\dagger\left(x-\hat{\mu}-\hat{\nu}\right)U_\mu^\dagger\left(x-\hat{\mu}\right)@f$
			Leaves[0][i*ndim+leaf]=ut[0][didn*ndim+nu]*a[0]-conj(ut[1][didn*ndim+nu])*a[1];
			Leaves[1][i*ndim+leaf]=ut[1][didn*ndim+nu]*a[0]+conj(ut[0][didn*ndim+nu])*a[1];
			return 0;
	}
}
inline int Half_Clover(Complex_f *clover[2],	Complex_f *Leaves[2], Complex_f *ut[2], unsigned int *iu, unsigned int *id, int i, int mu, int nu){
	const char funcname[] ="Half_Clover";
	for(short leaf=0;leaf<ndim;leaf++)
	{
		Leaf(ut,Leaves,iu,id,i,mu,nu,leaf);
		clover[0][i]+=Leaves[0][i*ndim+leaf]; clover[1][i]+=Leaves[1][i*ndim+leaf];
	}
	return 0;
}
int Clover(Complex_f *clover[6][2],Complex_f *Leaves[6][2],Complex_f *ut[2], unsigned int *iu, unsigned int *id){
	const char funcname[]="Clover";
	for(unsigned int mu=0;mu<ndim-1;mu++)
		for(unsigned int nu=mu+1;nu<ndim;nu++)
			if(mu!=nu){
				//Clover index
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
				//Allocate clover memory
				//Note that the clover is completely local, so doesn't need a halo for MPI
				clover[clov][0]=(Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f));
				clover[clov][1]=(Complex_f *)aligned_alloc(AVX,kvol*sizeof(Complex_f));
				Leaves[clov][0]=(Complex_f *)aligned_alloc(AVX,kvol*ndim*sizeof(Complex_f));
				Leaves[clov][1]=(Complex_f *)aligned_alloc(AVX,kvol*ndim*sizeof(Complex_f));
#pragma omp parallel for
				for(unsigned int i=0;i<kvol;i++)
				{
					Half_Clover(clover[clov],Leaves[clov],ut,iu,id,i,mu,nu);	
					//creal(clover[0]) drops so we are traceless. And everything else just gets doubled
					clover[clov][0][i]-=conj(clover[clov][0][i]);	clover[clov][1][i]+=clover[clov][1][i];
					//Don't forget the factor out front!
					//Uh Oh. G&L says -i/8 here. But hep-lat/9605038 and other sources say +1/8
					//It gets worse in the C_sw definition. We have a 1/2. They have +i/4
					clover[clov][0][i]*=(-I/16.0);	clover[clov][1][i]*=(-I/16.0);
				}
			}
	return 0;
}

//Multiplication for Congrad
//=========================
// Congradq only acts on flavour 1
int HbyClover(Complex_f *phi, Complex_f *r, Complex_f *clover[6][2], Complex_f *sigval, unsigned short *sigin){
	const char funcname[] = "HbyClover";
#pragma omp parallel for
	for(int i=0;i<kvol;i+=AVX)
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
						clov_s[c][j]=clover[clov][c][i+j];
					}
					///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
					phi_s[0][j]+=sigval[clov*ndirac+idirac]*(clov_s[0][j]*r_s[0][j]-conj(clov_s[1][j])*r_s[1][j]);
					phi_s[1][j]+=sigval[clov*ndirac+idirac]*(clov_s[1][j]*r_s[0][j]+conj(clov_s[0][j])*r_s[1][j]);
				}
#pragma unroll
				for(int c=0; c<nc; c++)
					phi[((i+j)*ndirac+idirac)*nc+c]+=phi_s[c][j];
			}
	return 0;
}

//Clover force terms 144 in total...
//=================================
//Calling function
//TODO: X1 and X2 contributions
int Clover_Force(double *dSdpi, Complex_f *Leaves[6][2], Complex_f *X1, Complex_f *X2, Complex_f *sigval,unsigned short *sigin){
	const char funcname[]="Clover_Force";
	//TODO: Make this more CUDA friendly? Or just have a CUDA call above
#pragma omp parallel for
	for(unsigned int i=0;i<kvol;i++)
		for(unsigned short mu=0;mu<ndim;mu++)
			for(unsigned short nu=0;nu<ndim;nu++)
				if(mu!=nu)
				{
					///Only three clovers @f$\mu\ne\nu@f$ contribute to the force term
					///Out of these clovers only three leaves contribute, with the leaf in the @f$\mu@f$ direction contributing
					///twice (daggered and not daggered)
					Complex_f Fleaf[2]={0,0};
					unsigned short clov;
					for(unsigned short adj=0;adj>nadj;adj++){
						///There are cases here where @f$\mu>\nu@f$. For those we use @f$ F_{\nu\mu}=F^\dagger_{\mu\nu}@f$
						if(mu<nu){
							//Clover index
							clov= (mu==0) ? nu-1 :mu+nu;
							///Contributions from @f$(f_{\mu\nu})@f$ first
							///Fleaf1 is the normal plaquette so leaf 0
							GenLeaf(Fleaf,Leaves[clov],i,0,adj,true);
							///Fleaf2 is the leaf containing @f$ U^dagger_\mu@f$ link and @f$ \nu<0@f$)
							GenLeaf(Fleaf,Leaves[clov],i,2,adj,true);
							///Contributions from @f$(f^\dagger_{\mu\nu})@f$ next
							///Fleaf3 is the normal plaquette daggered so leaf 0
							GenLeafd(Fleaf,Leaves[clov],i,0,adj,false);
							///Fleaf4 is the leaf containing @f$ U_\mu@f$ link and @f$ \nu<0@f$)
							GenLeafd(Fleaf,Leaves[clov],i,2,adj,false);
						}
						else{
							/// Clover index. Swapped from the @f$\mu<\nu@f$ case
							clov= (nu==0) ? mu-1 :nu+mu;

							///Contributions from @f$(f_{\mu\nu})@f$ first
							///Fleaf1 is the normal plaquette so leaf 0 daggered
							GenLeafd(Fleaf,Leaves[clov],i,0,adj,true);
							///Fleaf2 is the leaf containing @f$ U_\mu@f$ link and @f$ \nu<0@f$)
							GenLeaf(Fleaf,Leaves[clov],i,1,adj,true);
							///Contributions from @f$(f^\dagger_{\mu\nu})@f$ next
							///Fleaf3 is the normal plaquette daggered so leaf 0
							GenLeaf(Fleaf,Leaves[clov],i,0,adj,false);
							///Fleaf4 is the leaf containing @f$ U_\mu@f$ link and @f$ \nu<0@f$)
							GenLeafd(Fleaf,Leaves[clov],i,1,adj,false);
						}
						/// @f$\sigma_{\nu\mu}=-\sigma_{\mu\nu}@f$
						short pm= (mu<nu) ? 1 : -1;
						//Actual force stuff
						for(unsigned short idirac=0;idirac<ndirac;idirac++){
							const unsigned short igork1 = sigin[clov*ndirac+idirac];	
							dSdpi[(i*nadj+adj)*ndim+mu]+=pm*creal(I*sigval[clov*ndirac+idirac]*(
										conj(X1[(i*ndirac)*nc])*(Fleaf[0]*X2[(i*ndirac+igork1)*nc]-conj(Fleaf[1])*X2[(i*ndirac+igork1)*nc+1])+
										conj(X1[(i*ndirac)*nc+1])*(Fleaf[1]*X2[(i*ndirac+igork1*nc)]+conj(Fleaf[0])*X2[(i*ndirac+igork1)*nc+1])));
						}
					}
				}
	return 0;
}
inline int Clover_free(Complex_f *clover[6][2],Complex_f *Leaves[6][2]){
	for(unsigned short clov=0;clov<5;clov++)
		for(unsigned short c=0;c<nc;c++){
			free(clover[clov][c]); free(Leaves[clov][c]);
		}
	return 0;	
}

//Generator by Leaf
inline int GenLeaf(Complex_f Fleaf[2], Complex_f *Leaves[2],const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm){
	const char funcname[] = "GenLeaf";
	//Adding or subtracting this term
	const short sign = (pm) ? 1 : -1;
	//Which generator are we multiplying by? Zero indexed so subtract one from your usual index in textbooks
	switch(adj){
		case(0):
			Fleaf[0]+=sign*Leaves[1][i*ndim+leaf];		Fleaf[1]+=sign*Leaves[0][i*ndim+leaf];
		case(1):
			Fleaf[0]+=-sign*I*Leaves[1][i*ndim+leaf];	Fleaf[1]+=sign*I*Leaves[0][i*ndim+leaf];
		case(2):
			Fleaf[0]+=sign*Leaves[0][i*ndim+leaf];		Fleaf[1]+=-sign*Leaves[1][i*ndim+leaf];
	}
	return 0;
}
inline int GenLeafd(Complex_f Fleaf[2], Complex_f *Leaves[2],const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm){
	const char funcname[] = "GenLeafd";
	//Adding or subtracting this term
	const short sign = (pm) ? 1 : -1;
	//Which generator are we multiplying by? Zero indexed so subtract one from your usual index in textbooks
	switch(adj){
		case(0):
			Fleaf[0]+=-sign*Leaves[1][i*ndim+leaf]; 		Fleaf[1]+=sign*conj(Leaves[0][i*ndim+leaf]);
		case(1):
			Fleaf[0]+=sign*I*Leaves[1][i*ndim+leaf];		Fleaf[1]+=sign*I*conj(Leaves[0][i*ndim+leaf]);
		case(2):
			Fleaf[0]+=sign*conj(Leaves[0][i*ndim+leaf]); 	Fleaf[1]+=sign*conj(Leaves[1][i*ndim+leaf]);
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
