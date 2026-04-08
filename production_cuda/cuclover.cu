/**
 * @file 
 *
 * @brief	CUDA routines related to clover improved wilson fermions
 *
 * @author	D. Lawlor
 */
#include <clover.h>

//CUDA Device code
/**
 * @brief Multiply leaf (or part of one) by generator from left
 *
 *	The leaves contributing to each force term need to be scaled by the generator, but the generator appears at
 *	different points in each leaf.  This routine multiples by the generator from the left side.
 *
 *	@param	a:		The leaf or partial leaf
 *	@param	gen:	What generator are we multiplying by?
 */
template <typename T>
__device__ void cuByGenLeft(T a[nc],const unsigned short gen){
	T tmp = a[0];
	switch(gen){
		///@f$i\sigma_x@f$
		case(0):
			a[0] = T(-a[1].imag(), -a[1].real());
			a[1] = T( tmp.imag(),  tmp.real());
			break;
			///@f$i\sigma_y@f$
		case(1):
			a[0] = a[1];
			a[1] = -tmp;
			break;
			///@f$i\sigma_z@f$
		case(2):
			a[0] = T(-a[0].imag(), a[0].real());
			a[1] = T(-a[1].imag(), a[1].real());
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
template <typename T>
__device__ void cuByGenRight(T a[nc],const unsigned short gen){
	T tmp = a[0];
	switch(gen){
		///@f$i\sigma_x@f$
		case(0):
			a[0] = T(-a[1].imag(), a[1].real());
			a[1] = T(-tmp.imag(),  tmp.real());
			break;
			///@f$i\sigma_y@f$
		case(1):
			a[0]=-a[1]; a[1]=tmp;
			break;
			///@f$i\sigma_z@f$
		case(2):
			a[0] = T(-a[0].imag(),  a[0].real());
			a[1] = T( a[1].imag(), -a[1].real());
			break;
	}
	return;
}
/**
 *	@brief	Calculates the first half of the leaf for a clover term. We split it so that the force term can reuse the
 *				first half of the leaf
 *
 *	@param	u11t,u12t:			Gauge fields
 *	@param	Leaves:				Leaf
 *	@param	a:						Buffer array
 *	@param	iu,id:				Upper and lower site indices
 *	@param	i:						Lattice index of the clover in question
 *	@param	mu,nu:				Direction in which we're evaluating the leaf
 *	@param	leaf:					Which leaf of the clover is being calculated
 *	
 */
template <typename T>
__device__ int Half_Leaf(complex<T> Leaves[nc], complex<T> *u11t, complex<T> *u12t, complex<T> a[nc], unsigned int *iu,\
		unsigned int *id, const unsigned int i, const unsigned short mu, const unsigned short nu, const unsigned short leaf){
	unsigned int uidm;
	switch(leaf){
		case(0):
			///Both positive is just a standard plaquette
			a[0]=u11t[i+kvolHalo*mu]; a[1]=u12t[i+kvolHalo*mu];
			uidm = iu[mu*kvol+i]; 

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})@f$
			Leaves[0]=a[0]*u11t[uidm+kvolHalo*nu]-a[1]*conj(u12t[uidm+kvolHalo*nu]);
			Leaves[1]=a[0]*u12t[uidm+kvolHalo*nu]+a[1]*conj(u11t[uidm+kvolHalo*nu]);
			break;
		case(1):
			///Leaf in the forward nu and backwards mu direction
			//Should really read didm, but I've already declared this 
			uidm = id[mu*kvol+i];
			a[0]=u11t[i+kvolHalo*nu]; a[1]=u12t[i+kvolHalo*nu];
			//Awkward index...
			const unsigned int uin_didm=iu[nu*kvol+uidm];
			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})@f$
			Leaves[0]=a[0]*conj(u11t[uin_didm+kvolHalo*mu])+a[1]*conj(u12t[uin_didm+kvolHalo*mu]);
			Leaves[1]=-a[0]*u12t[uin_didm+kvolHalo*mu]+a[1]*u11t[uin_didm+kvolHalo*mu];
			break;
		case(2):
			///Leaf in the backwards nu and forwards mu direction
			//Should really read didn, but I've already declared this 
			uidm = id[nu*kvol+i];
			//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
			a[0]=conj(u11t[uidm+kvolHalo*nu]); a[1]=-u12t[uidm+kvolHalo*nu];

			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})@f$
			Leaves[0]=a[0]*u11t[uidm+kvolHalo*mu]-a[1]*conj(u12t[uidm+kvolHalo*mu]);
			//Don't forget negatiion of second term was handled earlier!
			Leaves[1]=a[0]*u12t[uidm+kvolHalo*mu]+a[1]*conj(u11t[uidm+kvolHalo*mu]);
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			//Should really read didm, but I've already declared this 
			uidm  =  id[i+kvol*mu];
			//Daggered. So Conj what goes into a[0] and negate what goes into a[1]
			a[0]=conj(u11t[uidm+kvolHalo*mu]); a[1]=-u12t[uidm+kvolHalo*mu];
			//Another awkward index
			const unsigned int din_didm=id[nu*kvol+uidm];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})@f$
			Leaves[0]=a[0]*conj(u11t[din_didm+kvolHalo*nu])+a[1]*u12t[din_didm+kvolHalo*nu];
			Leaves[1]=-a[0]*conj(u12t[din_didm+kvolHalo*nu])+a[1]*u11t[din_didm+kvolHalo*nu];
			break;
	}
	return 0;
}
/**
 *	@brief	Calculates a leaf for a clover term.
 *
 *	@param	u11t,u12t:	Gauge fields
 *	@param	Leaves:		Array of leaves
 *	@param	iu,id:		Upper and lower site indices
 *	@param	i:				Lattice index of the clover in question
 *	@param	mu,nu:		Direction in which we're evaluating the leaf
 *	@param	leaf:			Which leaf of the clover is being calculated
 *	
 */
template <typename T>
__device__ int Leaf(complex<T> *u11t, complex<T> *u12t, complex<T> Leaves[nc],\
		unsigned int *iu, unsigned int *id, unsigned int i,const unsigned short mu,\
		const unsigned short nu,const unsigned short leaf){
	complex<T> a[nc];
	Half_Leaf(Leaves,u11t,u12t,a,iu,id,i,mu,nu,leaf);
	unsigned int didm,didn,uidm;
	switch(leaf){
		case(0):
			unsigned int uidn = iu[nu*kvol+i]; 
			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})@f$
			a[0]=Leaves[0]*conj(u11t[uidn+kvolHalo*mu])+Leaves[1]*conj(u12t[uidn+kvolHalo*mu]);
			a[1]=-Leaves[0]*u12t[uidn+kvolHalo*mu]+Leaves[1]*u11t[uidn+kvolHalo*mu];

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})U^\dagger_\nu(x)@f$
			Leaves[0]=a[0]*conj(u11t[i+kvolHalo*nu])+a[1]*conj(u12t[i+kvolHalo*nu]);
			Leaves[1]=-a[0]*u12t[i+kvolHalo*nu]+a[1]*u11t[i+kvolHalo*nu];

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(1):
			didm = id[mu*kvol+i];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})@f$
			a[0]=Leaves[0]*conj(u11t[didm+kvolHalo*nu])+Leaves[1]*conj(u12t[didm+kvolHalo*nu]);
			a[1]=-Leaves[0]*u12t[didm+kvolHalo*nu]+Leaves[1]*u11t[didm+kvolHalo*nu];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*u11t[didm+kvolHalo*mu]-a[1]*conj(u12t[didm+kvolHalo*mu]);
			Leaves[1]=a[0]*u12t[didm+kvolHalo*mu]+a[1]*conj(u11t[didm+kvolHalo*mu]);
			//DEBUG
			//			Leaves[0]=0; Leaves[1]=0;
			break;
		case(2):
			///Leaf in the forwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			unsigned int uim_didn=iu[mu*kvol+didn];
			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})@f$
			a[0]=Leaves[0]*u11t[uim_didn+kvolHalo*nu]-Leaves[1]*conj(u12t[uim_didn+kvolHalo*nu]);
			a[1]=Leaves[0]*u12t[uim_didn+kvolHalo*nu]+Leaves[1]*conj(u11t[uim_didn+kvolHalo*nu]);

			/// @f$U^\dagger_\nu(x-\hat{\nu})U_\mu(x-\hat{\nu})U_\nu(x-\hat{\nu}+\hat{\mu})U^\dagger_\mu(x)@f$
			Leaves[0]=a[0]*conj(u11t[i+kvolHalo*mu])+a[1]*u12t[i+kvolHalo*mu];
			Leaves[1]=-a[0]*conj(u12t[i+kvolHalo*mu])+a[1]*u11t[i+kvolHalo*mu];

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			unsigned int din_didm=id[mu*kvol+didn];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})@f$
			a[0]=Leaves[0]*u11t[din_didm+kvolHalo*mu]-Leaves[1]*conj(u12t[din_didm+kvolHalo*mu]);
			a[1]=Leaves[0]*u12t[din_didm+kvolHalo*mu]+Leaves[1]*conj(u11t[din_didm+kvolHalo*mu]);

			didm = id[mu*kvol+i];
			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(n-\hat{\nu}-\hat{\mu})U_\nu(n-\hat{\nu})@f$
			Leaves[0]=a[0]*u11t[didm+kvolHalo*nu]-a[1]*conj(u12t[didm+kvolHalo*nu]);
			Leaves[1]=a[0]*u12t[didm+kvolHalo*nu]+a[1]*conj(u11t[didm+kvolHalo*nu]);

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
	}
	return 0;
}

/**
 *	@brief	Multiplies @f$ X_{\mu\nu}@f$ by a gauge field from the left
 *
 *	@param	out:	Result
 *	@param	X:		@f$X_{\mu\nu}(x)@f$
 *	@param	G:		Gauge field
 */
__device__ void cuGLeft(Complex_f out[4],const Complex_f G[2], const Complex_f X[4]){
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
__device__ void cuGRight(Complex_f out[4],const Complex_f G[2], const Complex_f X[4]){
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
__device__ void cuGSandwich(Complex_f out[4],Complex_f tmp[4], const Complex_f Gl[2], const Complex_f X[4],const Complex_f Gr[2]){
	cuGRight(tmp,Gr,X);
	cuGLeft(out,Gl,tmp);
	return;
}

///CUDA Kernels
/**
 *	@brief Calculates the products of the first two links in a plaquette
 *
 *	@param	hleaves0,hleaves1:	Product of first two links in
 *	@param	u11t,u12t:				Gauge fields
 *	@param	iu,id:					Upper and lower indices
 *	@param	mu,nu:					Clover direction
 */
template <typename T>
__global__ void Half_Leaves(complex<T> *hLeaves0,complex<T> *hLeaves1,complex<T> *u11t,complex<T> *u12t,\
		unsigned int *iu,unsigned int *id,const unsigned short mu,const unsigned short nu){
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	complex<T> Leaves[nc], a[nc];
	for(unsigned short leaf=0;leaf<ndim;leaf++)
		for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
			Half_Leaf(Leaves,u11t,u12t,a,iu,id,i,mu,nu,leaf);
			hLeaves0[i+kvol*leaf]=Leaves[0]; hLeaves1[i+kvol*leaf]=Leaves[1];
		}
	return;
}
/**
 *	@brief Calculates the clovers in all directions at all sites
 *	@f$ F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-Q_{\nu\mu}(n)\right)@f$
 *
 *	@param	clover1,clover2:	Array of clovers
 *	@param	u11t,u12t:			Gauge fields
 *	@param	iu,id:				Upper and lower indices
 *	@param	mu,nu:				Clover direction
 */
template <typename T>
__global__  void Full_Clover(complex<T> *clover1, complex<T> *clover2,\
		complex<T> *u11t, complex<T> *u12t, unsigned int *iu, unsigned int *id, int mu, int nu){
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	complex<T> Leaves[2];
	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		clover1[i]=0;clover2[i]=0;
		for(unsigned short leaf=0;leaf<ndim;leaf++)
		{
			//Pointer arithemetic on the leaves.
			Leaf(u11t,u12t,Leaves,iu,id,i,mu,nu,leaf);
			clover1[i]+=Leaves[0]; clover2[i]+=Leaves[1];
		}
		///The clover is given by @f$F_{\mu\nu}=\frac{-i}{8}\left(Q_{\mu\nu}-Q_{\nu\mu}\right)@f$. We do that
		///manually below.

		///The @f$\alpha@f$ component. Only the imaginary part survives. And since it is multiplied by @f$-i@f$ it is real.
		///Need to be extra cautious here though .imag() returns a real value. So we multiply by I_f manually 
		///The 8.0f becomes a 4.0f to account for the factor of two
		clover1[i]=clover1[i].imag();		clover1[i]*=(1.0f/4.0f);
		//		clover1[i]=clover1[1].imag()/4.0f;

		///The @f$\beta@f$ component. Both real and imaginary components survive. It ends up getting doubled.
		clover2[i]+=clover2[i]; 				clover2[i]*=(-I_f/8.0f);
	}
	return;
}

/**
 *	@brief	Gets @f$X_munu@f$ for the clover force
 *
 *	@param	Xmunu:	All Xmunu values
 *	@param	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param	mu,nu:	Lattice directions
 */
template <typename T>
__global__ void cuCalcXmunu(T *Xmunu, const T *X1, const T *X2, const T *sigval, const unsigned short *sigin,const unsigned short clov){
	const char funcname[] = "Xmunu";
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;
	unsigned short clov;
	//Get sign and index of @f$\sigma_{\mu\nu}@f correct
	if(mu<nu)
		clov = (mu==0) ? nu-1 : mu+nu;
	else
		clov = (nu==0) ? mu-1 : nu+mu;
	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Buffer. Eight registers...
		T Xmn[4]={0,0,0,0};
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
			const unsigned short sind = sigin[clov*ndirac+(idirac>>1)]<<1;
			const T sig = sigval[clov*ndirac+(idirac>>1)];
			for(unsigned short c1=0;c1<nc;c1++){
				//Spinors (rows) So we only load from memory once.
				const T X1s = X1[i+kvolHalo*(sind+c1)];
				const T X2s = X2[i+kvolHalo*(sind+c1)];
				for(unsigned short c2=0;c2<nc;c2++){
					//Conjugated spinor (columns).
					const T X1c = conj(X1[i+kvolHalo*(idirac+c2)]);
					const T X2c = conj(X2[i+kvolHalo*(idirac+c2)]);
					Xmn[(c1*nc+c2)]+=sig*(X2s*X1c+X1s*X2c);
				}
			}
		}
		//And write back to global memory.
		for(unsigned short c=0;c<nc*nc;c++)
			Xmunu[i+kvol*c]=Xmn[c];
	}
	return;
}
/**
 *	@brief cuGets the clover contribution to the force
 *
 *	@param	dSdpi:	Force
 *	@param	ut:		Gauge fields
 *	@param	X1:		Congrad output @f$\left(M^\dagger M\right)\Phi@f$
 *	@param	X2:		@f$M\left(M^\dagger M\right)^{-1}\Phi@f$
 *	@param	sigval:	@f$\sigma_{\mu\nu}@f$ scaled by @f$\frac{c_\text{SW}}{2}@f$
 *	@param	sigin:	Dirac index of @f$\sigma_{\mu\nu}@f$
 *	@param	iu,id:	Neighbouring sites
 */
template <typename T>
__global__ void Clov_Force(double *dSdpi, const T *u11t, const T *u12t, const T *Xmn, const T *sigval, const unsigned short *sigin,\
		const unsigned int *iu, const unsigned int *id, const float akappa,const unsigned short mu, const unsigned short nu){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;
	//Allocate the @f$X_{\mu\nu}@f$ array
	short nclov=6;
	//And get the @f$X_{\mu\nu}@f$ values
	//Loop over @f$\mu@f$ and @f$\nu@f$,
	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Buffer for intermediate force calculation. One for each generator.
		float dSdpis[3] = {0,0,0};
		//This is where it gets messy. Using HiRep/OpenQCD labelling for different intermediate values
		//But recycling to reduce register pressure on GPU
		//First up, W0, W1 and W6 match their Documentation values
		T W0[2]; T W1[2]; T W6[2];	
		//Get the correct site. Originally uid and did stood for up and down. Then I realised only one was needed
		//at a time and am too lazy to change it everywhere.
		unsigned int uid = id[i+kvol*nu];
		//Gauge field @f$U_\nu\left(i-\hat{\nu}\right)
		W1[0]=u11t[uid+kvolHalo*nu]; W1[1]=u12t[uid+kvolHalo*nu];

		//@f$Z_2=X_{\mu\nu}\left(i-\hat{\nu}\right)@f$
		T Z[nc*nc];
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Z[c]=Xmn[uid+kvol*c];

		//W0 is @f$U^\dagger_\mu@f(x-\hat{nu}\right)@f$
		W0[0]=conj(u11t[uid+kvolHalo*mu]); W0[1]=-u12t[uid+kvolHalo*mu];

		//Need a temporary Z buffers for the intermediate result
		T Zbuff1[nc*nc];T Zbuff2[nc*nc];
		cuGSandwich(Zbuff1,Zbuff2,W0,Z,W1);

		//@f$W_6=W_0 W_1@f$
		W6[0]=W0[0]*W1[0]-W0[1]*conj(W1[1]); W6[1]=W0[0]*W1[1]+W0[1]*conj(W1[0]);

		//Z3 is the @f$X_{\mu\nu}\left(x+\hat{\mu}-\hat{\nu}\right)@f$. Store in Z
		uid=iu[uid+kvol*mu];
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Z[c]=Xmn[uid+kvol*c];

		//Need a second Zbuffer for another intermediate result.
		cuGRight(Zbuff2,W6,Z);
		//Sum the two results into Zbuff1. Then scale by -W5
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Zbuff1[c]+=Zbuff2[c];
		//W5 is @f$U^\dagger_\nu\left(x+\hat{\mu}-\hat{\nu}\right)@f$
		T W5[2];
		W5[0]=conj(u11t[uid+kvolHalo*nu]); W5[1]=-u12t[uid+kvolHalo*nu];
		//Now multiply by @f$W_5@f$ from the left into Zbuff2
		cuGLeft(Zbuff2,W5,Zbuff1);

		//Intermediate results from the four parts of the sum.
		T F_int[4];
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			//Negative as it is @f$-W_5@f$
			F_int[c]=-Zbuff2[c];

		//Now we repeat for the last term in the sum. Recycling along the way.
		//First store @f$W_2=U_\nu\left(x+\hat{\mu}\right)@f$ into W0.
		uid=iu[i+kvol*mu];
		W0[0]=u11t[uid+kvolHalo*nu]; W0[1]=u12t[uid+kvolHalo*nu];
		//@f$W_3=U^\dagger_\mu\left(x+\hat{\nu}\right). Storing it in W1
		uid=iu[i+kvol*nu];
		W1[0]=conj(u11t[uid+kvolHalo*mu]); W1[1]=-u12t[uid+kvolHalo*mu];
		//@f$Z_4=X_{\mu\nu}\left(x+\hat{\mu}+\hat{\nu}\right)@f$. Storing in Z
		uid=iu[uid+kvol*mu];
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Z[c]=Xmn[uid+kvol*c];
		//Calculate and write into Zbuff1
		cuGSandwich(Zbuff1,Zbuff2,W0,Z,W1);

		//@f$W_7=W_0 W_1@f$
		T W7[2];
		W7[0]=W0[0]*W1[0]-W0[1]*conj(W1[1]); W7[1]=W0[0]*W1[1]+W0[1]*conj(W1[0]);
		//@f$Z_5=X_{\mu\nu}\left(x+\hat{\nu}\right)@f$
		uid=iu[i+kvol*nu]; 
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Z[c]=Xmn[uid+kvol*c];
		//And calculate the second term
		cuGLeft(Zbuff2,W7,Z);
		//Sum the two results into Zbuff1.
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Zbuff1[c]+=Zbuff2[c];
		//W4 is @f$U^\dagger_\nu\left(x\right)@f$
		T W4[2];
		W4[0]=conj(u11t[i+kvolHalo*nu]); W4[1]=-u12t[i+kvolHalo*nu];
		//Now multiply by @f$W_4@f$ from the right into Zbuff2
		cuGRight(Zbuff2,W4,Zbuff1);

		//Intermediate results from the four parts of the sum.
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			F_int[c]+=Zbuff2[c];
		//The last thing we need is @f$W_8=W_7W_4-W_5W_6@f$. Do it in parts and store intermediates in W0 and W1
		W0[0]=W7[0]*W4[0]-W7[1]*conj(W4[1]); W0[1]=W7[0]*W4[1]+W7[1]*conj(W4[0]);
		W1[0]=W5[0]*W6[0]-W5[1]*conj(W6[1]); W1[1]=W5[0]*W6[1]+W5[1]*conj(W6[0]);
		//Store W8 in W0
		W0[0]-=W1[0]; W0[1]-=W1[1];

		//Now load @f$@Z_0=X_{\mu\nu}(x)@f$
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Z[c]=Xmn[i+kvol*c];
		cuGLeft(Zbuff1,W0,Z);
		//And sum intermediate
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			F_int[c]+=Zbuff1[c];

		//Now load @f$@Z_1=X_{\mu\nu}(x)@f$
		uid=iu[i+kvol*mu];
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			Z[c]=Xmn[uid+kvol*c];
		cuGRight(Zbuff1,W0,Z);
		//And sum intermediate
#pragma unroll
		for(unsigned short c=0;c<nc*nc;c++)
			F_int[c]+=Zbuff1[c];

		//Excellent. Now we just need to multiply by the derivative term
		W0[0]=u11t[i+kvolHalo*mu]; W0[1]=u12t[i+kvolHalo*mu];
		for(unsigned short gen=0;gen<nadj;gen++){
			W1[0]=W0[0]; W1[1]=W0[1];
			cuByGenLeft(W1,gen);
			cuGLeft(Zbuff1,W1,F_int);
			//Sum of the real part of the trace.
			dSdpis[gen]=creal(Zbuff1[0])+creal(Zbuff1[3]);
			if(mu<nu)
				dSdpi[i+kvol*(gen*ndim+mu)]-=akappa*dSdpis[gen]/8.0f;
			else
				dSdpi[i+kvol*(gen*ndim+mu)]+=akappa*dSdpis[gen]/8.0f;
		}
	}
	return;
}

//Clover multiplication
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover1,clover2:	Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered output has no MPI halo, but undaggered does.
 */
template <typename T>
__global__ void ByClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2, complex<T> *sigval, const float akappa, unsigned short *sigin, bool dag){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;

	for(unsigned int i=gthreadId;i<kvol;i+=bsize*gsize){
		//Prefetched r and Phi array
		complex<T> phi_s[ngorkov][nc];
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ngorkov; igorkov++)
			for(unsigned short c=0; c<nc; c++){
				phi_s[igorkov][c]=0;
			}
		complex<T> r_s[nc];
		complex<T> clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover1[clov*kvol+i]; clov_s[1]=clover2[clov*kvol+i];
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
	return;
}
/**
 *	@brief Clover analogue of the Dslash operation. The H in front is for half, as we only act on the fermions of flavour
 *	1
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover1,clover2:	Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 *	@param	akappa:				Hopping Parameter
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	dag:					Daggered has no MPI halo, but undaggered does.
 */
template <typename T>
__global__ void HbyClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2,complex<T> *sigval, const float akappa, unsigned short *sigin,bool dag){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;

	for(unsigned int i=gthreadId;i<kvol;i+=bsize*gsize){
		//Prefetched r and Phi array
		complex<T> phi_s[ndirac*nc];
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc)
			for(unsigned short c=0; c<nc; c++){
				phi_s[idirac+c]=0;
			}
		complex<T> r_s[nc]; complex<T> clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover1[clov*kvol+i]; clov_s[1]=clover2[clov*kvol+i];
			for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
				const unsigned short sind = sigin[clov*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0; c<nc; c++){
					r_s[c]= r[i+kvolHalo*(sind+c)];
				}
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				const complex<T> sig=sigval[clov*ndirac+(idirac>>1)];
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
	return;
}

//Calling Wrappers
//This gets called by C so cannot be templated...
int cuClover(Complex_f *clover[nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id){
	const char funcname[]="cuClover";
#ifdef _DEBUG
	cudaMallocManaged((void **)&clover[0],6*kvol*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&clover[1],6*kvol*sizeof(Complex_f),cudaMemAttachGlobal);
#else
	cudaMallocAsync((void **)&clover[0],6*kvol*sizeof(Complex_f),streams[0]);
	cudaMallocAsync((void **)&clover[1],6*kvol*sizeof(Complex_f),streams[1]);
#endif
	for(unsigned short mu=0;mu<ndim-1;mu++)
		for(unsigned short nu=mu+1;nu<ndim;nu++)
			if(mu!=nu){
				//Clover index
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
				//Allocate clover memory
				//Note that the clover is completely local, so doesn't need a halo for MPI
				Full_Clover<<<dimGrid,dimBlock,0,streams[clov]>>>(clover[0]+clov*kvol,clover[1]+clov*kvol,\
						ut[0],ut[1],iu,id,mu,nu);
			}
	cudaDeviceSynchronise();
	return 0;
}
void cuByClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval,const float akappa, unsigned short *sigin, bool dag){
	ByClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,akappa,sigin,dag);
}
void cuHbyClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval, const float akappa, unsigned short *sigin, bool dag){
	HbyClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,akappa,sigin,dag);
}	
void cuByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float akappa, unsigned short *sigin,bool dag){
	ByClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,akappa,sigin,dag);
}
void cuHbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float akappa, unsigned short *sigin,bool dag){
	HbyClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,akappa,sigin,dag);
}

void cuCalcXmunu(Complex_f *Xmunu, Complex_f *X1, Complex_f *X2, const Complex_f *sigval, const short *sigin,const short mu, const short nu){
	//Get sign and index of @f$\sigma_{\mu\nu}@f correct
	short clov;
	if(mu<nu)
		clov = (mu==0) ? nu-1 : mu+nu;
	else
		clov = (nu==0) ? mu-1 : nu+mu;
	cuCalcXmunu<<<dimGrid,dimBlock,0,mu>>>(Xmunu,X1,X2,sigval,sigin,clov);
	return;
}
int cuClov_Force(double *dSdpi, Complex_f *ut[nc], Complex_f *X1, Complex_f *X2, const Complex_f *sigval,\
		const unsigned short *sigin, const unsigned int *iu, const unsigned int *id, const float akappa){
	const char funcname[]="Clov_Force";

	//Too many pointers here but not bothered doing it correctly. Overhead is basically zero.
	complex<float> *Xmn[ndim][ndim];
	//Allocate half-leaf memory. We will have one stream for each direction
	for(unsigned short mu=0;mu<ndim;mu++)
		for(unsigned short nu=0;nu<ndim;nu++)
			if(mu!=nu){
			short clov;
	//Get sign and index of @f$\sigma_{\mu\nu}@f correct
	if(mu<nu)
		clov = (mu==0) ? nu-1 : mu+nu;
	else
		clov = (nu==0) ? mu-1 : nu+mu;
				//Allocate and evaluate @f$X_{\mu\nu}@f$ terms
				cudaMallocAsync((void **)&Xmn[mu][nu],ndim*kvol*sizeof(complex<float>),streams[mu]);
				cuCalcXmunu<<<dimGrid,dimBlock,0,streams[mu]>>>(Xmn[mu][nu],X1,X2,sigval,sigin,clov);
				cudaMallocAsync((void **)&Xmn[nu][mu],ndim*kvol*sizeof(complex<float>),streams[nu]);
				cuCalcXmunu<<<dimGrid,dimBlock,0,streams[nu]>>>(Xmn[nu][mu],X1,X2,sigval,sigin,clov);

				//Compute force for @f$\mu\nu@f$ and @f$\nu\mu@f$
				Clov_Force<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,ut[0],ut[1],Xmn[mu][nu],\
						sigval,sigin,iu,id,akappa,mu,nu);
				Clov_Force<<<dimGrid,dimBlock,0,streams[nu]>>>(dSdpi,ut[0],ut[1],Xmn[nu][mu],\
						sigval,sigin,iu,id,akappa,nu,mu);

				//Free @f$X_{\mu\nu}@f$ terms
				cudaFreeAsync(Xmn[mu][nu],streams[mu]); cudaFreeAsync(Xmn[nu][mu],streams[nu]);
			}
	cudaDeviceSynchronise();
	return 0;
}
