/**
 * @file clover.c
 * @brief Clover fermion routines
 * @author D. Lawlor
 */
#include <clover.h>
//Multiplying by generators
#pragma omp declare simd
void ByGenLeft(Complex_f a[nc],const unsigned short gen){
	Complex_f tmp = a[0];
	switch(gen){
		///@f$i\sigma_x@f$
		case(0):
			a[0] = -cimagf(a[1])-crealf(a[1])*I;
			a[1] =  cimagf(tmp)+crealf(tmp)*I;
			break;
			///@f$i\sigma_y@f$
		case(1):
			a[0] = -conjf(a[1]);
			a[1] = conjf(tmp);
			break;
			///@f$i\sigma_z@f$
		case(2):
			a[0] = -cimagf(a[0])+crealf(a[0])*I;
			a[1] = -cimagf(a[1])+crealf(a[1])*I;
			break;
	}
	return;
}
#pragma omp declare simd
void ByGenRight(Complex_f a[nc],const unsigned short gen){
	Complex_f tmp = a[0];
	switch(gen){
		///@f$i\sigma_x@f$
		case(0):
			a[0] = -cimagf(a[1])+crealf(a[1])*I;
			a[1] = -cimagf(tmp)+ crealf(tmp)*I;
			break;
			///@f$i\sigma_y@f$
		case(1):
			a[0]=-a[1]; a[1]=tmp;
			break;
			///@f$i\sigma_z@f$
		case(2):
			a[0] = -cimagf(a[0])+crealf(a[0])*I;
			a[1] =  cimagf(a[1])-crealf(a[1])*I;
			break;
	}
	return;
}

//Calculating the clover and the leaves
//=====================================
/**
 *	@brief	Calculates the first half of the leaf for a clover term. We split it so that the force term can reuse the
 *				first half of the leaf
 *
 *	@param	Leaves:	Leaf
 *	@param	ut:		Gauge fields
 *	@param	a:			Buffer array
 *	@param	iu,id:	Upper and lower site indices
 *	@param	i:			Lattice index of the clover in question
 *	@param	mu,nu:	Direction in which we're evaluating the leaf
 *	@param	leaf:		Which leaf of the clover is being calculated
 *	
 */
#pragma omp declare simd
int Half_Leaf(Complex_f Leaves[nc], Complex_f *ut[nc], Complex_f a[nc], unsigned int *iu,\
		unsigned int *id, const unsigned int i, const unsigned short mu, const unsigned short nu, const unsigned short leaf){
	unsigned int uidm;
	switch(leaf){
		case(0):
			///Both positive is just a standard plaquette
			a[0]=ut[0][i+kvolHalo*mu]; a[1]=ut[1][i+kvolHalo*mu];
			uidm = iu[mu*kvol+i]; 

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})@f$
			Leaves[0]=a[0]*ut[0][uidm+kvolHalo*nu]-a[1]*conjf(ut[1][uidm+kvolHalo*nu]);
			Leaves[1]=a[0]*ut[1][uidm+kvolHalo*nu]+a[1]*conjf(ut[0][uidm+kvolHalo*nu]);
			break;
		case(1):
			///Leaf in the forward nu and backwards mu direction
			//Should really read didm, but I've already declared this 
			uidm = id[mu*kvol+i];
			a[0]=ut[0][i+kvolHalo*nu]; a[1]=ut[1][i+kvolHalo*nu];
			//Awkward index...
			const unsigned int uin_didm=iu[nu*kvol+uidm];
			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})@f$
			Leaves[0]=a[0]*conjf(ut[0][uin_didm+kvolHalo*mu])+a[1]*conjf(ut[1][uin_didm+kvolHalo*mu]);
			Leaves[1]=-a[0]*ut[1][uin_didm+kvolHalo*mu]+a[1]*ut[0][uin_didm+kvolHalo*mu];
			break;
		case(2):
			///Leaf in the backwards nu and forwards mu direction
			//Should really read didn, but I've already declared this 
			uidm = id[nu*kvol+i];
			//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
			a[0]=conjf(ut[0][uidm+kvolHalo*nu]); a[1]=-ut[1][uidm+kvolHalo*nu];

			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})@f$
			Leaves[0]=a[0]*ut[0][uidm+kvolHalo*mu]-a[1]*conjf(ut[1][uidm+kvolHalo*mu]);
			//Don't forget negatiion of second term was handled earlier!
			Leaves[1]=a[0]*ut[1][uidm+kvolHalo*mu]+a[1]*conjf(ut[0][uidm+kvolHalo*mu]);
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			//Should really read didm, but I've already declared this 
			uidm  =  id[i+kvol*mu];
			//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
			a[0]=conjf(ut[0][uidm+kvolHalo*mu]); a[1]=-ut[1][uidm+kvolHalo*mu];
			//Another awkward index
			const unsigned int din_didm=id[nu*kvol+uidm];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})@f$
			/// TODO: Copy to CUDA if working
			Leaves[0]=a[0]*conjf(ut[0][din_didm+kvolHalo*nu])+a[1]*conjf(ut[1][din_didm+kvolHalo*nu]);
			Leaves[1]=-a[0]*ut[1][din_didm+kvolHalo*nu]+a[1]*ut[0][din_didm+kvolHalo*nu];
			break;
	}
	return 0;
}
void Half_Leaves(Complex_f *hLeaves[2],Complex_f *ut[2], unsigned int *iu,unsigned int *id,\
		const unsigned short mu,const unsigned short nu){

#pragma omp parallel for simd collapse(2)
	for(unsigned short leaf=0;leaf<ndim;leaf++)
		for(unsigned int i=0;i<kvol;i++){
			Complex_f Leaves[nc], a[nc];
			Half_Leaf(Leaves,ut,a,iu,id,i,mu,nu,leaf);
			hLeaves[0][i+kvol*leaf]=Leaves[0]; hLeaves[1][i+kvol*leaf]=Leaves[1];
		}
	return;
}
#pragma omp declare simd
int Leaf(Complex_f Leaves[nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id, unsigned int i,\
		const unsigned short mu, const unsigned short nu,const unsigned short leaf){
	Complex_f a[nc];
	Half_Leaf(Leaves,ut,a,iu,id,i,mu,nu,leaf);
	unsigned int didm,didn,uidm;
	switch(leaf){
		case(0):
			unsigned int uidn = iu[nu*kvol+i]; 
			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})@f$
			a[0]=Leaves[0]*conjf(ut[0][uidn+kvolHalo*mu])+Leaves[1]*conjf(ut[1][uidn+kvolHalo*mu]);
			a[1]=-Leaves[0]*ut[1][uidn+kvolHalo*mu]+Leaves[1]*ut[0][uidn+kvolHalo*mu];

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})U^\dagger_\nu(x)@f$
			Leaves[0]=a[0]*conjf(ut[0][i+kvolHalo*nu])+a[1]*conjf(ut[1][i+kvolHalo*nu]);
			Leaves[1]=-a[0]*ut[1][i+kvolHalo*nu]+a[1]*ut[0][i+kvolHalo*nu];

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(1):
			didm = id[mu*kvol+i];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})@f$
			a[0]=Leaves[0]*conjf(ut[0][didm+kvolHalo*nu])+Leaves[1]*conjf(ut[1][didm+kvolHalo*nu]);
			a[1]=-Leaves[0]*ut[1][didm+kvolHalo*nu]+Leaves[1]*ut[0][didm+kvolHalo*nu];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*ut[0][didm+kvolHalo*mu]-a[1]*conjf(ut[1][didm+kvolHalo*mu]);
			Leaves[1]=a[0]*ut[1][didm+kvolHalo*mu]+a[1]*conjf(ut[0][didm+kvolHalo*mu]);
			//DEBUG
			//			Leaves[0]=0; Leaves[1]=0;
			break;
		case(2):
			///Leaf in the forwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			unsigned int uim_didn=iu[mu*kvol+didn];
			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})@f$
			a[0]=Leaves[0]*ut[0][uim_didn+kvolHalo*nu]-Leaves[1]*conjf(ut[1][uim_didn+kvolHalo*nu]);
			a[1]=Leaves[0]*ut[1][uim_didn+kvolHalo*nu]+Leaves[1]*conjf(ut[0][uim_didn+kvolHalo*nu]);

			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})U^\dagger_\mu(x)@f$
			/// TODO: If works, copy to CUDA
			Leaves[0]=a[0]*conjf(ut[0][i+kvolHalo*mu])+a[1]*conjf(ut[1][i+kvolHalo*mu]);
			Leaves[1]=-a[0]*ut[1][i+kvolHalo*mu]+a[1]*ut[0][i+kvolHalo*mu];

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			unsigned int din_didm=id[mu*kvol+didn];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})@f$
			a[0]=Leaves[0]*ut[0][din_didm+kvolHalo*mu]-Leaves[1]*conjf(ut[1][din_didm+kvolHalo*mu]);
			a[1]=Leaves[0]*ut[1][din_didm+kvolHalo*mu]+Leaves[1]*conjf(ut[0][din_didm+kvolHalo*mu]);

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})U_\nu(n-\hat{\nu})@f$
			Leaves[0]=a[0]*ut[0][didn+kvolHalo*nu]-a[1]*conjf(ut[1][didn+kvolHalo*nu]);
			Leaves[1]=a[0]*ut[1][didn+kvolHalo*nu]+a[1]*conjf(ut[0][didn+kvolHalo*nu]);

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
	}
	return 0;
}
void Clover(Complex_f *clover[2], Complex_f *ut[2], unsigned int *iu, unsigned int *id){
	const char funcname[]="Full_Clover";
#ifdef __NVCC__
	cuClover(clover,ut,iu,id);
#else
	clover[0]=aligned_alloc(AVX,6*kvol*sizeof(Complex_f));
	clover[1]=aligned_alloc(AVX,6*kvol*sizeof(Complex_f));
	for(unsigned short mu=0;mu<ndim-1;mu++)
		for(unsigned short nu=mu+1;nu<ndim;nu++)
			if(mu!=nu){
				//Clover index
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
#pragma omp parallel for 
				for(unsigned int i=0;i<kvol;i++){
					clover[0][i+clov*kvol]=0;
					clover[1][i+clov*kvol]=0;
					Complex_f Leaves[nc];
					for(unsigned short leaf=0;leaf<ndim;leaf++)
					{
						//Pointer arithemetic on the leaves.
						Leaf(Leaves,ut,iu,id,i,mu,nu,leaf);
						clover[0][i+clov*kvol]+=Leaves[0]; clover[1][i+clov*kvol]+=Leaves[1];
					}
					///The clover is given by @f$F_{\mu\nu}=\frac{-i}{8}\left(Q_{\mu\nu}-Q_{\nu\mu}\right)@f$. We do that
					///manually below.

					///The @f$\alpha@f$ component. Only the imaginary part survives. And since it is multiplied by @f$-i@f$ it is real.
					///Need to be extra cautious here though cimag() returns a real value. So we multiply by I manually (by
					///using (cimagf) and the minuses cancel.
					///The 8.0f becomes a 4.0f to account for the factor of two
					clover[0][i+clov*kvol]=cimagf(clover[0][i+clov*kvol]);		clover[0][i+clov*kvol]*=(1.0f/4.0f);

					///The @f$\beta@f$ component. Both real and imaginary components survive. It ends up getting doubled.
					clover[1][i+clov*kvol]+=clover[1][i+clov*kvol];	clover[1][i+clov*kvol]*=(-I/8.0f);
				}
			}
#endif
	return;
}

//Multiplication for Congradq
//=========================
void ByClover(Complex *phi, Complex *r, Complex *clover[2], Complex *sigval, const float akappa, unsigned short *sigin, bool dag){
#ifdef __NVCC__
	cuByClover(phi,r,clover,sigval,akappa,sigin,dag);
#else
#pragma omp parallel for simd
	for(unsigned int i=0;i<kvol;i++){
		//Prefetched r and Phi array
		Complex phi_s[ngorkov][nc];
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++)
			for(unsigned short c=0; c<nc; c++){
				phi_s[igorkov][c]=0;
			}
		Complex r_s[nc];
		Complex clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover[0][clov*kvol+i]; clov_s[1]=clover[1][clov*kvol+i];
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				//Mod 4 done bitwise. In general n mod 2^m = n & (2^m-1)
				const unsigned short idirac = igorkov&3;
				const unsigned short sind = (igorkov<4) ? sigin[clov*ndirac+idirac] : sigin[clov*ndirac+idirac]+4;
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
					r_s[c]= r[i+kvolHalo*(sind*nc+c)];
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				phi_s[igorkov][0]+=sigval[clov*ndirac+idirac]*(creal(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				//Clover is in the Lie Algebra, not Lie group. So signs are correct here.
				phi_s[igorkov][1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]-creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++)
			for(unsigned short c=0; c<nc; c++){
				///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
				//dag is just to do with the output layout and if it has a halo
				if(dag)
					phi[i+kvol*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
				else
					phi[i+kvolHalo*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
			}
	}
#endif
	return;
}
void HbyClover(Complex *phi, Complex *r, Complex *clover[2],Complex *sigval, const float akappa, unsigned short *sigin,bool dag){
	const char funcname[] = "HbyClover";
#ifdef __NVCC__
	cuHbyClover(phi,r,clover,sigval,akappa,sigin,dag);
#else
#pragma omp parallel for simd
	for(unsigned int i=0;i<kvol;i++){
		//Prefetched r and Phi array
		Complex phi_s[ndirac*nc];
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc)
			for(unsigned short c=0; c<nc; c++){
				phi_s[idirac+c]=0;
			}
		Complex r_s[nc]; Complex clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover[0][clov*kvol+i]; clov_s[1]=clover[1][clov*kvol+i];
			for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
				const unsigned short sind = sigin[clov*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0; c<nc; c++){
					r_s[c]= r[i+kvolHalo*(sind+c)];
				}
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				const Complex sig=sigval[clov*ndirac+(idirac>>1)];
				phi_s[idirac+0]+=sig*(creal(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				//Clover is in the Lie Algebra, not Lie group. So signs are correct here.
				phi_s[idirac+1]+=sig*(conj(clov_s[1])*r_s[0]-creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc)
			for(unsigned short c=0; c<nc; c++)
				///@f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
				//dag is just to do with the output layout and if it has a halo
				if(dag)
					phi[i+kvol*(c+idirac)]-=akappa*phi_s[idirac+c];
				else
					phi[i+kvolHalo*(c+idirac)]-=akappa*phi_s[idirac+c];
	}
#endif
	return;
}
//Float versions
void ByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[2], Complex_f *sigval, const float akappa, unsigned short *sigin, bool dag){
#ifdef __NVCC__
	cuByClover_f(phi,r,clover,sigval,akappa,sigin,dag);
#else
#pragma omp parallel for simd
	for(unsigned int i=0;i<kvol;i++){
		//Prefetched r and Phi array
		Complex_f phi_s[ngorkov][nc];
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++)
			for(unsigned short c=0; c<nc; c++){
				phi_s[igorkov][c]=0;
			}
		Complex_f r_s[nc];
		Complex_f clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover[0][clov*kvol+i]; clov_s[1]=clover[1][clov*kvol+i];
			for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++){
				//Mod 4 done bitwise. In general n mod 2^m = n & (2^m-1)
				const unsigned short idirac = igorkov&3;
				const unsigned short sind = (igorkov<4) ? sigin[clov*ndirac+idirac] : sigin[clov*ndirac+idirac]+4;
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
					r_s[c]= r[i+kvolHalo*(sind*nc+c)];
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				phi_s[igorkov][0]+=sigval[clov*ndirac+idirac]*(crealf(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				//Clover is in the Lie Algebra, not Lie group. So signs are correct here.
				phi_s[igorkov][1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]-crealf(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++)
			for(unsigned short c=0; c<nc; c++){
				///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
				//dag is just to do with the output layout and if it has a halo
				if(dag)
					phi[i+kvol*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
				else
					phi[i+kvolHalo*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
			}
	}
#endif
	return;
}
void HbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[2],Complex_f *sigval, const float akappa, unsigned short *sigin,bool dag){
	const char funcname[] = "HbyClover_f";
#ifdef __NVCC__
	cuHbyClover_f(phi,r,clover,sigval,akappa,sigin,dag);
#else
#pragma omp parallel for simd
	for(unsigned int i=0;i<kvol;i++){
		//Prefetched r and Phi array
		Complex_f phi_s[ndirac*nc];
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc)
			for(unsigned short c=0; c<nc; c++){
				phi_s[idirac+c]=0;
			}
		Complex_f r_s[nc]; Complex_f clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover[0][clov*kvol+i]; clov_s[1]=clover[1][clov*kvol+i];
			for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
				const unsigned short sind = sigin[clov*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0; c<nc; c++){
					r_s[c]= r[i+kvolHalo*(sind+c)];
				}
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				const Complex_f sig=sigval[clov*ndirac+(idirac>>1)];
				phi_s[idirac+0]+=sig*(crealf(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				//Clover is in the Lie Algebra, not Lie group. So signs are correct here.
				phi_s[idirac+1]+=sig*(conj(clov_s[1])*r_s[0]-crealf(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc)
			for(unsigned short c=0; c<nc; c++)
				///@f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
				//dag is just to do with the output layout and if it has a halo
				if(dag)
					phi[i+kvol*(c+idirac)]-=akappa*phi_s[idirac+c];
				else
					phi[i+kvolHalo*(c+idirac)]-=akappa*phi_s[idirac+c];
	}
#endif
	return;
}

//Clover Force
//===========
void CalcXmunu(Complex_f *Xmunu, Complex_f *X1, Complex_f *X2, const Complex_f *sigval, const unsigned short *sigin,\
					const unsigned short mu, const unsigned short nu){
	const char funcname[] = "Xmunu";
#ifdef __NVCC__
	cuCalcXmunu(Xmunu,X1,X2,sigval,sigin,mu,nu);
#else
	unsigned short clov;
	//Get sign and index of @f$\sigma_{\mu\nu}@f correct
		clov = (mu==0) ? nu-1 : mu+nu;
#pragma omp parallel for simd aligned(X1,X2,Xmunu:AVX)
	for(unsigned int i=0;i<kvol;i++){
		//Buffer. Eight registers...
		Complex_f Xmn[4]={0,0,0,0};
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
			const unsigned short sind = sigin[clov*ndirac+(idirac>>1)]<<1;
			const Complex_f sig = sigval[clov*ndirac+(idirac>>1)];
#pragma unroll
			for(unsigned short c1=0;c1<nc;c1++){
				//Spinors (rows) So we only load from memory once.
				const Complex_f X1s = X1[i+kvolHalo*(sind+c1)];
				const Complex_f X2s = X2[i+kvolHalo*(sind+c1)];
#pragma unroll
				for(unsigned short c2=0;c2<nc;c2++){
					//Conjugated spinor (columns).
					const Complex_f X1c = conjf(X1[i+kvolHalo*(idirac+c2)]);
					const Complex_f X2c = conjf(X2[i+kvolHalo*(idirac+c2)]);
					Xmn[(c1*nc+c2)]+=sig*(X2s*X1c+X1s*X2c);
				}
			}
		}
		//And write back to global memory.
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Xmunu[i+kvol*c]=Xmn[c];
	}
#endif
	return;
}

/**
 *	@brief	Multiplies @f$ X_{\mu\nu}@f$ by a gauge field from the left
 *
 *	@param	out:	Result
 *	@param	X:		@f$X_{\mu\nu}(x)@f$
 *	@param	G:		Gauge field
 */
static inline void GLeft(Complex_f out[4],const Complex_f G[2], const Complex_f X[4]){
	out[0]=G[0]*X[0]+G[1]*X[2];
	out[1]=G[0]*X[1]+G[1]*X[3];
	out[2]=-conj(G[1])*X[0]+conj(G[0])*X[2];
	out[3]=-conj(G[1])*X[1]+conj(G[0])*X[3];
	return;
}
/**
 *	@brief	Multiplies @f$ X_{\mu\nu}@f$ by a gauge field from the right
 *
 *	@param	out:	Result
 *	@param	X:		@f$X_{\mu\nu}(x)@f$
 *	@param	G:		Gauge field
 */
static inline void GRight(Complex_f out[4],const Complex_f G[2], const Complex_f X[4]){
	out[0]=G[0]*X[0]-conj(G[1])*X[1];
	out[1]=G[1]*X[0]+conj(G[0])*X[1];
	out[2]=G[0]*X[2]-conj(G[1])*X[3];
	out[3]=G[1]*X[2]+conj(G[0])*X[3];
	return;
}
/**
 *	@brief	Multiplies @f$ X_{\mu\nu}@f$ by a gauge field from the left and the right
 *
 *	@param	out:		Result
 *	@param	tmp:		Buffer for intermediate result. Passing as an argument to reduce register pressure.
 *	@param	X:			@f$X_{\mu\nu}(x)@f$
 *	@param	Gl,Gr:	Left/Right Gauge fields
 */
static inline void GSandwich(Complex_f out[4],Complex_f tmp[4], const Complex_f Gl[2], const Complex_f X[4],const Complex_f Gr[2]){
	GRight(tmp,Gr,X);
	GLeft(out,Gl,tmp);
	return;
}

void Clov_Force(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
		const short *sigin, unsigned int *iu, unsigned int *id, const float akappa){
	const char funcname[] = "Clov_Force";
#ifdef __NVCC__
	cuClov_Force(dSdpi,ut,X1,X2,sigval,sigin,iu,id,akappa);
#else
	//Allocate the @f$X_{\mu\nu}@f$ array
	short nclov=6;
	Complex_f *Xmn=(Complex_f *)aligned_alloc(AVX,kvol*nc*nc*sizeof(Complex_f));
	//And get the @f$X_{\mu\nu}@f$ values
	//Loop over @f$\mu@f$ and @f$\nu@f$. Symmetry means we actually only need half the terms
	for(unsigned short mu=0;mu<ndim-1;mu++)
		for(unsigned short nu=mu;nu<ndim;nu++)
			if(mu!=nu){
				CalcXmunu(Xmn,X1,X2,sigval,sigin,mu,nu);
#pragma omp parallel for
				for(unsigned int i=0;i<kvol;i++){
					//Buffer for intermediate force calculation. One for each generator.
					float dSdpis[3] = {0,0,0};
					//This is where it gets messy. Using HiRep/OpenQCD labelling for different intermediate values
					//But recycling to reduce register pressure on GPU
					//First up, W0, W1 and W6 match their Documentation values
					Complex_f W0[2], W1[2], W6[2];	
					//Get the correct site. Originally uid and did stood for up and down. Then I realised only one was needed
					//at a time and am too lazy to change it everywhere.
					unsigned int uid = id[i+kvol*nu];
					//Gauge field @f$U_\nu\left(i-\hat{\nu}\right)
					W1[0]=ut[0][uid+kvolHalo*nu]; W1[1]=ut[1][uid+kvolHalo*nu];

					//@f$Z_2=X_{\mu\nu}\left(i-\hat{\nu}\right)@f$
					Complex_f Z[nc*nc];
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Z[c]=Xmn[uid+kvol*c];

					//W0 is @f$U^\dagger_\mu@f(x-\hat{nu}\right)@f$
					W0[0]=conjf(ut[0][uid+kvolHalo*mu]); W0[1]=-ut[1][uid+kvolHalo*mu];

					//Need a temporary Z buffers for the intermediate result
					Complex_f Zbuff1[nc*nc], Zbuff2[nc*nc];
					GSandwich(Zbuff1,Zbuff2,W0,Z,W1);

					//@f$W_6=W_0 W_1@f$
					W6[0]=W0[0]*W1[0]-W0[1]*conjf(W1[1]); W6[1]=W0[0]*W1[1]+W0[1]*conjf(W1[0]);

					//Z3 is the @f$X_{\mu\nu}\left(x+\hat{\mu}-\hat{\nu}\right)@f$. Store in Z
					uid=iu[uid+kvol*mu];
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Z[c]=Xmn[uid+kvol*c];

					//Need a second Zbuffer for another intermediate result.
					GRight(Zbuff2,W6,Z);
					//Sum the two results into Zbuff1. Then scale by -W5
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Zbuff1[c]+=Zbuff2[c];
					//W5 is @f$U^\dagger_\nu\left(x+\hat{\mu}-\hat{\nu}\right)@f$
					Complex_f W5[2];
					W5[0]=conjf(ut[0][uid+kvolHalo*nu]); W5[1]=-ut[1][uid+kvolHalo*nu];
					//Now multiply by @f$W_5@f$ from the left into Zbuff2
					GLeft(Zbuff2,W5,Zbuff1);

					//Intermediate results from the four parts of the sum.
					Complex_f F_int[4];
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						//Negative as it is @f$-W_5@f$
						F_int[c]=-Zbuff2[c];

					//Now we repeat for the last term in the sum. Recycling along the way.
					//First store @f$W_2=U_\nu\left(x+\hat{\mu}\right)@f$ into W0.
					uid=iu[i+kvol*mu];
					W0[0]=ut[0][uid+kvolHalo*nu]; W0[1]=ut[1][uid+kvolHalo*nu];
					//@f$W_3=U^\dagger_\mu\left(x+\hat{\nu}\right). Storing it in W1
					uid=iu[i+kvol*nu];
					W1[0]=conjf(ut[0][uid+kvolHalo*mu]); W1[1]=-ut[1][uid+kvolHalo*mu];
					//@f$Z_4=X_{\mu\nu}\left(x+\hat{\mu}+\hat{\nu}\right)@f$. Storing in Z
					uid=iu[uid+kvol*mu];
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Z[c]=Xmn[uid+kvol*c];
					//Calculate and write into Zbuff1
					GSandwich(Zbuff1,Zbuff2,W0,Z,W1);

					//@f$W_7=W_0 W_1@f$
					Complex_f W7[2];
					W7[0]=W0[0]*W1[0]-W0[1]*conjf(W1[1]); W7[1]=W0[0]*W1[1]+W0[1]*conjf(W1[0]);
					//@f$Z_5=X_{\mu\nu}\left(x+\hat{\nu}\right)@f$
					uid=iu[i+kvol*nu]; 
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Z[c]=Xmn[uid+kvol*c];
					//And calculate the second term
					GLeft(Zbuff2,W7,Z);
					//Sum the two results into Zbuff1.
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Zbuff1[c]+=Zbuff2[c];
					//W4 is @f$U^\dagger_\nu\left(x\right)@f$
					Complex_f W4[2];
					W4[0]=conjf(ut[0][i+kvolHalo*nu]); W4[1]=-ut[1][i+kvolHalo*nu];
					//Now multiply by @f$W_4@f$ from the right into Zbuff2
					GRight(Zbuff2,W4,Zbuff1);

					//Intermediate results from the four parts of the sum.
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						F_int[c]+=Zbuff2[c];
					//The last thing we need is @f$W_8=W_7W_4-W_5W_6@f$. Do it in parts and store intermediates in W0 and W1
					W0[0]=W7[0]*W4[0]-W7[1]*conjf(W4[1]); W0[1]=W7[0]*W4[1]+W7[1]*conjf(W4[0]);
					W1[0]=W5[0]*W6[0]-W5[1]*conjf(W6[1]); W1[1]=W5[0]*W6[1]+W5[1]*conjf(W6[0]);
					//Store W8 in W0
					W0[0]-=W1[0]; W0[1]-=W1[1];

					//Now load @f$@Z_0=X_{\mu\nu}(x)@f$
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Z[c]=Xmn[i+kvol*c];
					GLeft(Zbuff1,W0,Z);
					//And sum intermediate
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						F_int[c]+=Zbuff1[c];

					//Now load @f$@Z_1=X_{\mu\nu}(x)@f$
					uid=iu[i+kvol*mu];
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						Z[c]=Xmn[uid+kvol*c];
					GRight(Zbuff1,W0,Z);
					//And sum intermediate
#pragma unroll
					for(unsigned short c=0;c<nc*nc;c++)
						F_int[c]+=Zbuff1[c];

					//Excellent. Now we just need to multiply by the derivative term
					W0[0]=ut[0][i+kvolHalo*mu]; W0[1]=ut[1][i+kvolHalo*mu];
					for(unsigned short gen=0;gen<nadj;gen++){
						W1[0]=W0[0]; W1[1]=W0[1];
						ByGenLeft(W1,gen);
						GLeft(Zbuff1,W1,F_int);
						//Sum of the real part of the trace.
						dSdpis[gen]=crealf(Zbuff1[0])+crealf(Zbuff1[3]);
						dSdpi[i+kvol*(gen*ndim+mu)]-=akappa*dSdpis[gen]/8.0f;
						//There's a minus sign from @f$\sigma_{\nu\mu}@f$ and from @f$F_{\mu\nu}@f$ which cancel
						dSdpi[i+kvol*(gen*ndim+nu)]-=akappa*dSdpis[gen]/8.0f;
					}
				}
			}
	free(Xmn);
#endif
	return;
}

//Initialisation and freeing
int Init_clover(Complex **sigval, Complex_f **sigval_f,unsigned short **sigin, float c_sw){
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
	for(int i=0;i<6;i++)
		for(int j=0;j<4;j++)
			sigval_t[i][j]*=c_sw*0.5;
#endif

#ifdef __NVCC__
	int device = -1; 
	cudaGetDevice(&device);

	cudaMalloc((void **)sigin,6*4*sizeof(short));
	cudaMalloc((void **)sigval,6*4*sizeof(Complex));
	cudaMalloc((void **)sigval_f,6*4*sizeof(Complex_f));

	cudaMemcpy(*sigin,sigin_t,6*4*sizeof(short),cudaMemcpyDefault);
	cudaMemcpy(*sigval,sigval_t,6*4*sizeof(Complex),cudaMemcpyDefault);

	cuComplex_convert(*sigval_f,*sigval,24,true,dimBlockOne,dimGridOne);	
#else
	*sigin = (unsigned short *)malloc(6*4*sizeof(short));
	*sigval=(Complex *)malloc(6*4*sizeof(Complex));
	*sigval_f=(Complex_f *)malloc(6*4*sizeof(Complex_f));;
	memcpy(*sigval,sigval_t,6*4*sizeof(Complex));
	memcpy(*sigin,sigin_t,6*4*sizeof(short));
	for(int i=0;i<6*4;i++)
		*(*sigval_f+i)=(Complex_f)*(*sigval+i);
#endif
}
inline int Clover_free(Complex_f *clover[nc]){
	for(unsigned short c=0;c<nc;c++){
#ifdef __NVCC__
#ifdef _DEBUG
		cudaFree(clover[c]);
#else
		cudaFreeAsync(clover[c],streams[c]);
#endif
#else
		free(clover[c]);
#endif
	}
	return 0;	
}
