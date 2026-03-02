/**
 * @file clover.c
 * @brief Clover fermion routines
 * @author D. Lawlor
 */
#include <clover.h>
//Multiplying by generators
/**
 * @brief Multiply leaf (or part of one) by generator from left
 *
 *	The leaves contributing to each force term need to be scaled by the generator, but the generator appears at
 *	different points in each leaf.  This routine multiples by the generator from the left side.
 *
 *	@param	a:		The leaf or partial leaf
 *	@param	gen:	What generator are we multiplying by?
 */
void ByGenLeft(Complex a[nc],const unsigned short gen){
	Complex tmp = a[0];
	switch(gen){
		///@f$i\sigma_x@f$
		case(0):
			a[0] = -cimagf(a[1])-crealf(a[1])*I;
			a[1] =  cimagf(tmp)+crealf(tmp)*I;
			break;
			///@f$i\sigma_y@f$
		case(1):
			a[0] = a[1];
			a[1] = -tmp;
			break;
			///@f$i\sigma_z@f$
		case(2):
			a[0] = -cimagf(a[0])+crealf(a[0])*I;
			a[1] = -cimagf(a[1])+crealf(a[1])*I;
			break;
	}
	return;
}
/**
 * @brief Multiply leaf (or part of one) by generator from right
 *
 *	The leaves contributing to each force term need to be scaled by the generator, but the generator appears at
 *	different points in each leaf.  This routine multiples by the generator from the right side.
 *
 *	@param	a:		The leaf or partial leaf
 *	@param	gen:	What generator are we multiplying by?
 */
void ByGenRight(Complex a[nc],const unsigned short gen){
	Complex tmp = a[0];
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
#pragma omp simd
int Half_Leaf(Complex_f Leaves[nc], Complex_f *ut[nc], Complex_f a[nc], unsigned int *iu,\
		unsigned int *id, const unsigned int i, const unsigned short mu, const unsigned short nu, const unsigned short leaf){
	unsigned int uidm;
	switch(leaf){
		case(0):
			///Both positive is just a standard plaquette
			a[0]=ut[0][i+kvolHalo*mu]; a[1]=ut[1][i+kvolHalo*mu];
			uidm = iu[mu*kvol+i]; 

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})@f$
			Leaves[0]=a[0]*ut[0][uidm+kvolHalo*nu]-a[1]*conj(ut[1][uidm+kvolHalo*nu]);
			Leaves[1]=a[0]*ut[1][uidm+kvolHalo*nu]+a[1]*conj(ut[0][uidm+kvolHalo*nu]);
			break;
		case(1):
			///Leaf in the forward nu and backwards mu direction
			//Should really read didm, but I've already declared this 
			uidm = id[mu*kvol+i];
			a[0]=ut[0][i+kvolHalo*nu]; a[1]=ut[1][i+kvolHalo*nu];
			//Awkward index...
			const unsigned int uin_didm=iu[nu*kvol+uidm];
			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})@f$
			Leaves[0]=a[0]*conj(ut[0][uin_didm+kvolHalo*mu])+a[1]*conj(ut[1][uin_didm+kvolHalo*mu]);
			Leaves[1]=-a[0]*ut[1][uin_didm+kvolHalo*mu]+a[1]*ut[0][uin_didm+kvolHalo*mu];
			break;
		case(2):
			///Leaf in the backwards nu and forwards mu direction
			//Should really read didn, but I've already declared this 
			uidm = id[nu*kvol+i];
			//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
			a[0]=conj(ut[0][uidm+kvolHalo*nu]); a[1]=-ut[1][uidm+kvolHalo*nu];

			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})@f$
			Leaves[0]=a[0]*ut[0][uidm+kvolHalo*mu]-a[1]*conj(ut[1][uidm+kvolHalo*mu]);
			//Don't forget negatiion of second term was handled earlier!
			Leaves[1]=a[0]*ut[1][uidm+kvolHalo*mu]+a[1]*conj(ut[0][uidm+kvolHalo*mu]);
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			//Should really read didm, but I've already declared this 
			uidm  =  id[i+kvol*mu];
			//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
			a[0]=conj(ut[0][uidm+kvolHalo*mu]); a[1]=-ut[1][uidm+kvolHalo*mu];
			//Another awkward index
			const unsigned int din_didm=id[nu*kvol+uidm];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})@f$
			Leaves[0]=a[0]*conj(ut[0][din_didm+kvolHalo*nu])+a[1]*ut[1][din_didm+kvolHalo*nu];
			Leaves[1]=-a[0]*conj(ut[1][din_didm+kvolHalo*nu])+a[1]*ut[0][din_didm+kvolHalo*nu];
			break;
	}
	return 0;
}
/**
 *	@brief	Calculates a leaf for a clover term.
 *
 *	@param	Leaves:	Array of leaves
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower site indices
 *	@param	i:			Lattice index of the clover in question
 *	@param	mu,nu:	Direction in which we're evaluating the leaf
 *	@param	leaf:		Which leaf of the clover is being calculated
 *	
 */
#pragma omp simd
int Leaf(Complex_f Leaves[nc],Complex_f *ut[nc], nsigned int *iu, unsigned int *id, unsigned int i,\
		const unsigned short mu, const unsigned short nu,const unsigned short leaf){
	Complex_f a[nc];
	Half_Leaf(Leaves,ut[0],ut[1],a,iu,id,i,mu,nu,leaf);
	unsigned int didm,didn,uidm;
	switch(leaf){
		case(0):
			unsigned int uidn = iu[nu*kvol+i]; 
			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})@f$
			a[0]=Leaves[0]*conj(ut[0][uidn+kvolHalo*mu])+Leaves[1]*conj(ut[1][uidn+kvolHalo*mu]);
			a[1]=-Leaves[0]*ut[1][uidn+kvolHalo*mu]+Leaves[1]*ut[0][uidn+kvolHalo*mu];

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})U^\dagger_\nu(x)@f$
			Leaves[0]=a[0]*conj(ut[0][i+kvolHalo*nu])+a[1]*conj(ut[1][i+kvolHalo*nu]);
			Leaves[1]=-a[0]*ut[1][i+kvolHalo*nu]+a[1]*ut[0][i+kvolHalo*nu];

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(1):
			didm = id[mu*kvol+i];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})@f$
			a[0]=Leaves[0]*conj(ut[0][didm+kvolHalo*nu])+Leaves[1]*conj(ut[1][didm+kvolHalo*nu]);
			a[1]=-Leaves[0]*ut[1][didm+kvolHalo*nu]+Leaves[1]*ut[0][didm+kvolHalo*nu];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*ut[0][didm+kvolHalo*mu]-a[1]*conj(ut[1][didm+kvolHalo*mu]);
			Leaves[1]=a[0]*ut[1][didm+kvolHalo*mu]+a[1]*conj(ut[0][didm+kvolHalo*mu]);
			//DEBUG
			//			Leaves[0]=0; Leaves[1]=0;
			break;
		case(2):
			///Leaf in the forwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			unsigned int uim_didn=iu[mu*kvol+didn];
			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})@f$
			a[0]=Leaves[0]*ut[0][uim_didn+kvolHalo*nu]-Leaves[1]*conj(ut[1][uim_didn+kvolHalo*nu]);
			a[1]=Leaves[0]*ut[1][uim_didn+kvolHalo*nu]+Leaves[1]*conj(ut[0][uim_didn+kvolHalo*nu]);

			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})U^\dagger_\mu(x)@f$
			Leaves[0]=a[0]*conj(ut[0][i+kvolHalo*mu])+a[1]*ut[1][i+kvolHalo*mu];
			Leaves[1]=-a[0]*conj(ut[1][i+kvolHalo*mu])+a[1]*ut[0][i+kvolHalo*mu];

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			unsigned int din_didm=id[mu*kvol+didn];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})@f$
			a[0]=Leaves[0]*ut[0][din_didm+kvolHalo*mu]-Leaves[1]*conj(ut[1][din_didm+kvolHalo*mu]);
			a[1]=Leaves[0]*ut[1][din_didm+kvolHalo*mu]+Leaves[1]*conj(ut[0][din_didm+kvolHalo*mu]);

			didm = id[mu*kvol+i];
			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})U_\nu(n-\hat{\nu})@f$
			Leaves[0]=a[0]*ut[0][didm+kvolHalo*nu]-a[1]*conj(ut[1][didm+kvolHalo*nu]);
			Leaves[1]=a[0]*ut[1][didm+kvolHalo*nu]+a[1]*conj(ut[0][didm+kvolHalo*nu]);

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
	}
	return 0;
}
/**
 *	@brief Calculates the products of the first two links in a plaquette
 *
 *	@param	hleaves:		Product of first two links in
 *	@param	ut:			Gauge fields
 *	@param	iu,id:		Upper and lower indices
 *	@param	mu,nu:		Clover direction
 */
void Half_Leaves(Complex_f *hLeaves[2],Complex_f *ut[2], unsigned int *iu,unsigned int *id,\
		const unsigned short mu,const unsigned short nu){

#pragma omp parallel for simd collapse(2)
	for(unsigned short leaf=0;leaf<ndim;leaf++)
		for(unsigned int i=0;i<kvol;i++){
			Complex_f Leaves[nc], a[nc];
			Half_Leaf(Leaves,ut[0],ut[1],a,iu,id,i,mu,nu,leaf);
			hLeaves[0][i+kvol*leaf]=Leaves[0]; hLeaves[1][i+kvol*leaf]=Leaves[1];
		}
	return;
}
/**
 *	@brief Calculates the clovers in all directions at all sites
 *	@f$ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f$
 *
 *	@param	clover:	Array of clovers
 *	@param	ut:		Gauge fields
 *	@param	iu,id:	Upper and lower indices
 *	@param	mu,nu:	Clover direction
 */
void Full_Clover(Complex_f *clover[2], Complex_f *ut[2], unsigned int *iu, unsigned int *id){
	const char funcname[]="Full_Clover";
#ifdef __NVCC__
	cuClover(clover,ut,iu,id);
#else
	for(unsigned short mu=0;mu<ndim-1;mu++)
		for(unsigned short nu=mu+1;nu<ndim;nu++)
			if(mu!=nu){
				//Clover index
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
#pragma omp parallel for simd
				for(unsigned int i=0;i<kvol;i++){
					clover[0][i]=0;clover[1][i]=0;
					Complex_f Leaves[nc];
					for(unsigned short leaf=0;leaf<ndim;leaf++)
					{
						//Pointer arithemetic on the leaves.
						Leaf(ut[0],ut[1],Leaves,iu,id,i,mu,nu,leaf);
						clover[0][i+clov*kvol]+=Leaves[0]; clover[1][i+clov*kvol]+=Leaves[1];
					}
					///The clover is given by @f$F_{\mu\nu}=\frac{-i}{8}\left(Q_{\mu\nu}-Q_{\nu\mu}\right)@f$. We do that
					///manually below.

					///The @f$\alpha@f$ component. Only the imaginary part survives. And since it is multiplied by @f$-i@f$ it is real.
					///Need to be extra cautious here though cimag() returns a real value. So we multiply by I_f manually 
					///The 8.0f becomes a 4.0f to account for the factor of two
					clover[0][i+clov*kvol]=cimagf(clover[0][i+clov*kvol]);		clover[0][i+clov*kvol]*=(1.0f/4.0f);

					///The @f$\beta@f$ component. Both real and imaginary components survive. It ends up getting doubled.
					clover[1][i+clov*kvol]+=clover[1][i+clov*kvol];	clover[1][i+clov*kvol]*=(-I_f/8.0f);
				}
			}
#endif
	return;
}

//Multiplication for Congradq
//=========================
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 */
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
				phi_s[igorkov][1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]+creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++)
			for(unsigned short c=0; c<nc; c++){
				///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
				if(dag)
					phi[i+kvol*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
				else
					phi[i+kvolHalo*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
			}
	}
#endif
	return;
}
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 */
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
				phi_s[idirac+1]+=sig*(conj(clov_s[1])*r_s[0]+creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc)
			for(unsigned short c=0; c<nc; c++)
				///@f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
#if(dag)
				phi[i+kvol*(c+idirac)]-=akappa*phi_s[idirac+c];
#else
		phi[i+kvolHalo*(c+idirac)]-=akappa*phi_s[idirac+c];
#endif
	}
#endif
	return;
}
//Float versions
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 */
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
				phi_s[igorkov][0]+=sigval[clov*ndirac+idirac]*(creal(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				//Clover is in the Lie Algebra, not Lie group. So signs are correct here.
				phi_s[igorkov][1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]+creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++)
			for(unsigned short c=0; c<nc; c++){
				///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
				if(dag)
					phi[i+kvol*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
				else
					phi[i+kvolHalo*(nc*igorkov+c)]-=akappa*phi_s[igorkov][c];
			}
	}
#endif
	return;
}
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover:				Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 */
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
				phi_s[idirac+0]+=sig*(creal(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				//Clover is in the Lie Algebra, not Lie group. So signs are correct here.
				phi_s[idirac+1]+=sig*(conj(clov_s[1])*r_s[0]+creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc)
			for(unsigned short c=0; c<nc; c++)
				///@f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
#if(dag)
				phi[i+kvol*(c+idirac)]-=akappa*phi_s[idirac+c];
#else
		phi[i+kvolHalo*(c+idirac)]-=akappa*phi_s[idirac+c];
#endif
	}
#endif
	return;
}

//Clover Force
//===========
/**
 *	@brief	Calculates a leaf for a clover term.
 *
 *	@param	ut:			Gauge fields
 *	@param	Leaves:		Array of leaves
 *	@param	iu,id:		Upper and lower site indices
 *	@param	i:				Lattice index of the clover in question
 *	@param	mu,nu:		Direction in which we're evaluating the leaf
 *	@param	leaf:			Which leaf of the clover is being calculated
 *	@param	gen:			Which generator do we multiply the leaves by. Used for the force terms
 *	@param	gen_pos:		Where does the generator appear in the multiplication. Used for the force terms.
 *	
 */
int Force_Leaf(complex<T> *ut[nc], complex<T> Leaves[nc],\
		unsigned int *iu, unsigned int *id, unsigned int i,const unsigned short mu,const unsigned short nu,\
		const unsigned short leaf,short gen,short gen_pos){
	complex<T> a[nc];
	unsigned int didm,didn,uidm;
	switch(leaf){
		case(0):
			//If the generator is between the first two links, then we can't use the precomputed half-leaves
			if(gen_pos==1){
				///Both positive is just a standard plaquette
				a[0]=ut[0][i+kvolHalo*mu]; a[1]=ut[1][i+kvolHalo*mu];
				//Multiply first link by generator from the right
				ByGenRight(a,gen);
				uidm = iu[mu*kvol+i]; 
				/// @f$U_\mu(x)U^\nu(x+\hat{\mu})@f$
				Leaves[0]=a[0]*ut[0][uidm+kvolHalo*nu]-a[1]*conj(ut[1][uidm+kvolHalo*nu]);
				Leaves[1]=a[0]*ut[1][uidm+kvolHalo*nu]+a[1]*conj(ut[0][uidm+kvolHalo*nu]);
			}
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);

			unsigned int uidn = iu[nu*kvol+i]; 
			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})@f$
			a[0]=Leaves[0]*conj(ut[0][uidn+kvolHalo*mu])+Leaves[1]*conj(ut[1][uidn+kvolHalo*mu]);
			a[1]=-Leaves[0]*ut[1][uidn+kvolHalo*mu]+Leaves[1]*ut[0][uidn+kvolHalo*mu];
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})U^\dagger_\nu(x)@f$
			Leaves[0]=a[0]*conj(ut[0][i+kvolHalo*nu])+a[1]*conj(ut[1][i+kvolHalo*nu]);
			Leaves[1]=-a[0]*ut[1][i+kvolHalo*nu]+a[1]*ut[0][i+kvolHalo*nu];

			//DEBUG
			//					Leaves[0]=0; Leaves[1]=0;
			break;
		case(1):
			//If the generator is between the first two links, then we can't use the precomputed half-leaves
			if(gen_pos==1){
				//Should really read didm, but I've already declared this 
				uidm = id[mu*kvol+i];
				a[0]=ut[0][i+kvolHalo*nu]; a[1]=ut[1][i+kvolHalo*nu];
				//Multiply first link by generator from the right
				ByGenRight(a,gen);
				//Awkward index...
				const unsigned int uin_didm=iu[nu*kvol+uidm];
				/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})@f$
				Leaves[0]=a[0]*conj(ut[0][uin_didm+kvolHalo*mu])+a[1]*conj(ut[1][uin_didm+kvolHalo*mu]);
				Leaves[1]=-a[0]*ut[1][uin_didm+kvolHalo*mu]+a[1]*ut[0][uin_didm+kvolHalo*mu];
			}
			didm = id[mu*kvol+i];
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})@f$
			a[0]=Leaves[0]*conj(ut[0][didm+kvolHalo*nu])+Leaves[1]*conj(ut[1][didm+kvolHalo*nu]);
			a[1]=-Leaves[0]*ut[1][didm+kvolHalo*nu]+Leaves[1]*ut[0][didm+kvolHalo*nu];
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*ut[0][didm+kvolHalo*mu]-a[1]*conj(ut[1][didm+kvolHalo*mu]);
			Leaves[1]=a[0]*ut[1][didm+kvolHalo*mu]+a[1]*conj(ut[0][didm+kvolHalo*mu]);
			//DEBUG
			//			Leaves[0]=0; Leaves[1]=0;
			break;
		case(2):
			//If the generator is between the first two links, then we can't use the precomputed half-leaves
			if(gen_pos==1){
				//Should really read didn, but I've already declared this 
				uidm = id[nu*kvol+i];
				//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
				a[0]=conj(ut[0][uidm+kvolHalo*nu]); a[1]=-ut[1][uidm+kvolHalo*nu];
				//Multiply first link by generator from the right
				ByGenRight(a,gen);

				/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})@f$
				Leaves[0]=a[0]*ut[0][uidm+kvolHalo*mu]-a[1]*conj(ut[1][uidm+kvolHalo*mu]);
				//Don't forget negatiion of second term was handled earlier!
				Leaves[1]=a[0]*ut[1][uidm+kvolHalo*mu]+a[1]*conj(ut[0][uidm+kvolHalo*mu]);
			}
			///Leaf in the forwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);
			unsigned int uim_didn=iu[mu*kvol+didn];
			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})@f$
			a[0]=Leaves[0]*ut[0][uim_didn+kvolHalo*nu]-Leaves[1]*conj(ut[1][uim_didn+kvolHalo*nu]);
			a[1]=Leaves[0]*ut[1][uim_didn+kvolHalo*nu]+Leaves[1]*conj(ut[0][uim_didn+kvolHalo*nu]);
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})U^\dagger_\mu(x)@f$
			Leaves[0]=a[0]*conj(ut[0][i+kvolHalo*mu])+a[1]*ut[1][i+kvolHalo*mu];
			Leaves[1]=-a[0]*conj(ut[1][i+kvolHalo*mu])+a[1]*ut[0][i+kvolHalo*mu];

			//DEBUG
			//					Leaves[0]=0; Leaves[1]=0;
			break;
		case(3):
			//If the generator is between the first two links, then we can't use the precomputed half-leaves
			if(gen_pos==1){
				//Should really read didm, but I've already declared this 
				uidm  =  id[i+kvol*mu];
				//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
				a[0]=conj(ut[0][uidm+kvolHalo*mu]); a[1]=-ut[1][uidm+kvolHalo*mu];
				ByGenRight(a,gen);
				//Another awkward index
				const unsigned int din_didm=id[nu*kvol+uidm];

				/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})@f$
				Leaves[0]=a[0]*conj(ut[0][din_didm+kvolHalo*nu])+a[1]*ut[1][din_didm+kvolHalo*nu];
				Leaves[1]=-a[0]*conj(ut[1][din_didm+kvolHalo*nu])+a[1]*ut[0][din_didm+kvolHalo*nu];

			}
			didn = id[nu*kvol+i]; 
			///Leaf in the backwards mu and backwards nu direction
			unsigned int din_didm=id[mu*kvol+didn];
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);

			didm = id[mu*kvol+i];
			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})@f$
			a[0]=Leaves[0]*ut[0][din_didm+kvolHalo*mu]-Leaves[1]*conj(ut[1][din_didm+kvolHalo*mu]);
			a[1]=Leaves[0]*ut[1][din_didm+kvolHalo*mu]+Leaves[1]*conj(ut[0][din_didm+kvolHalo*mu]);
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})U_\nu(n-\hat{\nu})@f$
			Leaves[0]=a[0]*ut[0][didm+kvolHalo*nu]-a[1]*conj(ut[1][didm+kvolHalo*nu]);
			Leaves[1]=a[0]*ut[1][didm+kvolHalo*nu]+a[1]*conj(ut[0][didm+kvolHalo*nu]);

			//DEBUG
			//					Leaves[0]=0; Leaves[1]=0;
			break;
	}
	///gen_pos 0 is multiply the entire leaf by the generator from the left
	if(gen_pos==0)
		ByGenLeft(Leaves,gen);
	///gen_pos 4 is multiply the entire leaf by the generator from the left
	if(gen_pos==4)
		ByGenRight(Leaves,gen);
	return 0;
}
/**
 *	@brief	Clover contribution to the Molecular Dynamics force
 *
 *	@param	dSdpi:		Force
 *	@param	ut:			Gauge fields
 *	@param	X1:			@f$\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	X2:			@f$M\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	sigval:		@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$c_sw@f$
 * @param	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	iu,id:		Up/down indices
 * @param	clov:			Clover we're intereted in
 * @param	mu,nu:		Direction of clover we're interested in
 * @param	akappa:		Hopping parameter
 */
void Clover_Force(double *dSdpi, complex<T> *ut[nc], complex<T> *hLeaves[nc], complex<T> *X1, complex<T> *X2,\
		const complex<T> *sigval, const unsigned short *sigin, unsigned int *iu, unsigned int *id,\
		const float akappa){
#ifdef __NVCC__
	cuClover_Force(dSdpi,ut,X1,X1,sigval,sigin,iu,id,akappa);
#else
	Complex_f *hLeaves[ndim][nc];
	//Allocate half-leaf memory. We will have one stream for each direction
	for(unsigned short mu=0;mu<ndim;mu++){
		hLeaves[mu][0]=(Complex_f *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex_f));
		hLeaves[mu][1]=(Complex_f *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex_f));
	}
		for(unsigned short mu=0;mu<ndim-1;mu++)
			for(unsigned short nu=mu+1;nu<ndim;nu++){
				//Clover index
				const unsigned short clov = (mu==0) ? nu-1 :mu+nu;

				//Compute half leaves
				Half_Leaves(hLeaves[mu],ut,iu,id,mu,nu);
				Half_Leaves(hLeaves[nu],ut,iu,id,nu,mu);

				//Compute force for @f$\mu\nu@f$ and @f$\nu\mu@f$
#pragma omp parallel for
				for(unsigned int i=0;i<kvol;i++){
					//Two of these since we have the mu and nu contribut[1]ions
					float dSdpis[3]={0,0,0}; 
					const unsigned int ipm=iu[i+kvol*mu];
					for(unsigned short fclov=0;fclov<(ndim-1)*(ndim-2);fclov++){
						Complex_f fleaf[nadj][nc];
						unsigned int site;
						for(unsigned short gen=0;gen<nadj;gen++){
							//This stores the half-leaf initially, then the out[1]put[1] from Force_Leaves
							Complex_f tmp[nc];
							switch(fclov){
								case(0): //Clover at site
									site=i;
									tmp[0]=hLeaves0[site+0*kvol]; tmp[1]=hLeaves1[site+0*kvol];
									//Get leaf 0 with the correct generator in the initial position
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,0,gen,0);
									fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];

									//Get leaf 2 with the correct generator in the final position
									tmp[0]=hLeaves0[site+2*kvol]; tmp[1]=hLeaves1[site+2*kvol];
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,2,gen,4);
									//-= here as the contribut[1]ion is from @f$Q_{\nu\mu}@f$!!!
									//Conjugate too.
									fleaf[gen][0]-=conjf(tmp[0]); fleaf[gen][1]-=-tmp[1];
									break;
								case(1): //Clover at i+mu
									site=ipm;
									//Get leaf 1 with the correct generator between links 3 and 4
									tmp[0]=hLeaves0[site+1*kvol]; tmp[1]=hLeaves1[site+1*kvol];
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,1,gen,3);
									fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
									//Get leaf 3 with the correct generator between links 1 and 2
									tmp[0]=hLeaves0[site+3*kvol]; tmp[1]=hLeaves1[site+3*kvol];
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,3,gen,1);
									//-= here as the contribut[1]ion is from @f$Q_{\nu\mu}@f$!!!
									//Conjugate too
									fleaf[gen][0]-=conjf(tmp[0]); fleaf[gen][1]-=-tmp[1];
									break;
								case(2): //Clover at i+nu
									site=iu[i+kvol*nu];
									//Get leaf 2 with the correct generator between links 1 and 2
									tmp[0]=hLeaves0[site+2*kvol]; tmp[1]=hLeaves1[site+2*kvol];
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,2,gen,1);
									fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
									break;
								case(3): //Clover at i-nu
									site=id[i+kvol*nu];
									//Get leaf 0 with the correct generator between links 3 and 4
									tmp[0]=hLeaves0[site+0*kvol]; tmp[1]=hLeaves1[site+0*kvol];
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,0,gen,3);
									//- here as the contribut[1]ion is from @f$Q_{\nu\mu}@f$!!!
									//Conjugate too
									fleaf[gen][0]=-conjf(tmp[0]); fleaf[gen][1]=tmp[1];
									break;
								case(4): //Clover at i+mu+nu
									site=iu[ipm+kvol*nu];
									//Get leaf 3 with the correct generator between links 2 and 3
									tmp[0]=hLeaves0[site+3*kvol]; tmp[1]=hLeaves1[site+3*kvol];
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,3,gen,2);
									fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
									break;
								case(5): //Clover at i+mu-nu
									site=id[ipm+kvol*nu];
									//Get leaf 1 with the correct generator between links 2 and 3
									tmp[0]=hLeaves0[site+1*kvol]; tmp[1]=hLeaves1[site+1*kvol];
									Force_Leaf(ut,tmp,iu,id,site,mu,nu,1,gen,2);
									//- here as the contribut[1]ion is from @f$Q_{\nu\mu}@f$!!!
									//Conjugate too
									fleaf[gen][0]=-conjf(tmp[0]); fleaf[gen][1]=tmp[1];
									break;
							}
							//				fleaf[gen][0]=(-I_f/8.0f)*(fleaf[gen][0]+conjf(fleaf[gen][0]));
							//				fleaf[gen][0]=(-I_f/4.0f)*fleaf[gen][0].real();
							fleaf[gen][0]=Complex_f(0,-fleaf[gen][0].real()/4);
							//				fleaf[gen][1]=(-I_f/8.0f)*(fleaf[gen][1]-fleaf[gen][1]);
							fleaf[gen][1]=0;
						}
						for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
							const unsigned short sind = sigin[clov*ndirac+(idirac>>1)]<<(nc-1);	
							//Calculate the index. For the next colour we add kvol
							unsigned int ind = site+kvolHalo*idirac;
							//Prefetching. Might not be needed here though
							Complex_f X1sc[nc];
							//X1 is always conjfugated. So do it once here instead of twice and be done with it.	
							X1sc[0]=conjf(X1[ind]); X1sc[1]=conjf(X1[indi+kvolHalo]);
							ind = site+kvolHalo*sind;
							Complex_f X2s[nc];
							X2s[0]=X2[ind]; X2s[1]=X2[indi+kvolHalo];

							for(unsigned short gen=0;gen<nadj;gen++){
								//					Complex_f fleaf1c=conjf(fleaf[gen][1]);
								float force = (sigval[clov*ndirac+idirac]*(X1sc[0]*(fleaf[gen][0]*X2s[0]+fleaf[gen][1]*X2s[1])+\
											X1sc[1]*(fleaf[gen][0]*X2s[1]-fleaf[gen][1]*X2s[0]))).real();
								//mu direction contribut[1]ion
								dSdpis[gen]+=force;
							}
						}
					}
					for(unsigned short gen=0;gen<nadj;gen++){
						dSdpi[i+kvol*(gen*ndim+mu)]-=akappa*dSdpis[gen];
					}
				}
			}

	for(unsigned short mu=0;mu<ndim;mu++){
		free(hLeaves[mu][0]); free(hLeaves[mu][1]);
	}
#endif
	return;
}
