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
__device__ void ByGenLeft(T a[nc],const unsigned short gen){
	T tmp = a[0];
	switch(gen){
		///@f$i\sigma_x@f$
		case(0):
			//a[0]=-I_f*conj(a[1]); 
			a[0] = T(-a[1].imag(), -a[1].real());
			//a[1]=I_f*conj(tmp);
			a[1] = T( tmp.imag(),  tmp.real());
			break;
			///@f$i\sigma_y@f$
		case(1):
			//a[0]=conj(a[1]); 
			a[0] = T( a[1].real(), -a[1].imag());
			//a[1]=-conj(tmp);
			a[1] = T(-tmp.real(),  tmp.imag());
			break;
			///@f$i\sigma_z@f$
		case(2):
			//			a[0]*=I_f; 
			a[0] = T(-a[0].imag(), a[0].real());
			//			a[1]*=I_f;
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
__device__ void ByGenRight(T a[nc],const unsigned short gen){
	T tmp = a[0];
	switch(gen){
		///@f$i\sigma_x@f$
		case(0):
			//a[0]=I_f*a[1];
			a[0] = T(-a[1].imag(), a[1].real());
			//a[1]=I_f*tmp;
			a[1] = T(-tmp.imag(),  tmp.real());
			///@f$i\sigma_y@f$
			break;
		case(1):
			a[0]=a[1]; a[1]=-tmp;
			///@f$i\sigma_z@f$
			break;
		case(2):
			//a[0]*=I_f;
			a[0] = T(-a[0].imag(),  a[0].real());
			//a[1]*=-I_f;
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
			a[0]=u11t[i+kvol*mu]; a[1]=u12t[i+kvol*mu];
			uidm = iu[mu*kvol+i]; 

			/// @f$U_\mu(x)U^\nu(x+\hat{\mu})@f$
			Leaves[0]=a[0]*u11t[uidm+kvol*nu]-a[1]*conj(u12t[uidm+kvol*nu]);
			Leaves[1]=a[0]*u12t[uidm+kvol*nu]+a[1]*conj(u11t[uidm+kvol*nu]);
			break;
		case(1):
			///Leaf in the forward nu and backwards mu direction
			const unsigned int didm = id[mu*kvol+i];
			a[0]=u11t[i+kvol*nu]; a[1]=u12t[i+kvol*nu];
			const unsigned int uin_didm=iu[nu*kvol+didm];
			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})@f$
			Leaves[0]=a[0]*conj(u11t[uin_didm+kvol*mu])+a[1]*conj(u12t[uin_didm+kvol*mu]);
			Leaves[1]=-a[0]*u12t[uin_didm+kvol*mu]+a[1]*u11t[uin_didm+kvol*mu];
			break;
		case(2):
			///Leaf in the forwards mu and backwards nu direction
			//Another awkward index
			uidm = iu[mu*kvol+i];
			a[0]=u11t[i+kvol*mu]; a[1]=u12t[i+kvol*mu];
			const unsigned int din_uidm=id[nu*kvol+uidm];

			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{\mu}-\hat{\nu})@f$
			Leaves[0]=a[0]*conj(u11t[din_uidm+kvol*nu])+a[1]*conj(u12t[din_uidm+kvol*nu]);
			Leaves[1]=-a[0]*u12t[din_uidm+kvol*nu]+a[1]*u11t[din_uidm+kvol*nu];
			break;
		case(3):
			///Leaf in the forwards mu and backwards nu direction
			//Another awkward index
			const unsigned int didn = id[nu*kvol+i];
			a[0]=u11t[didn+kvol*nu]; a[1]=u12t[didn+kvol*nu];
			const unsigned int dim_didn=id[mu*kvol+didn];

			/// @f$U_\nu^\dagger(x-\hat{\nu})U_\mu^\dagger(x-\hat{\mu}-\hat{\nu})@f$
			Leaves[0]=conj(a[0])*conj(u11t[dim_didn+kvol*mu])-conj(a[1])*u12t[dim_didn+kvol*mu];
			Leaves[1]=-conj(a[0])*u12t[dim_didn+kvol*mu]-a[1]*u11t[dim_didn+kvol*mu];
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
 *	@param	gen:			Which generator do we multiply the leaves by. Used for the force terms
 *	@param	gen_pos:		Where does the generator appear in the multiplication. Used for the force terms.
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
			a[0]=Leaves[0]*conj(u11t[uidn+kvol*mu])+Leaves[1]*conj(u12t[uidn+kvol*mu]);
			a[1]=-Leaves[0]*u12t[uidn+kvol*mu]+Leaves[1]*u11t[uidn+kvol*mu];

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})U^\dagger_\nu(x)@f$
			Leaves[0]=a[0]*conj(u11t[i+kvol*nu])+a[1]*conj(u12t[i+kvol*nu]);
			Leaves[1]=-a[0]*u12t[i+kvol*nu]+a[1]*u11t[i+kvol*nu];

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(1):
			didm = id[mu*kvol+i];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})@f$
			a[0]=Leaves[0]*conj(u11t[didm+kvol*nu])+Leaves[1]*conj(u12t[didm+kvol*nu]);
			a[1]=-Leaves[0]*u12t[didm+kvol*nu]+Leaves[1]*u11t[didm+kvol*nu];

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*u11t[didm+kvol*mu]-a[1]*conj(u12t[didm+kvol*mu]);
			Leaves[1]=a[0]*u12t[didm+kvol*mu]+a[1]*conj(u11t[didm+kvol*mu]);
			//DEBUG
			//			Leaves[0]=0; Leaves[1]=0;
			break;
		case(2):
			///Leaf in the forwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{\mu}-\hat{\nu})U_\mu^\dagger(x-\hat{\nu})@f$
			a[0]=Leaves[0]*conj(u11t[didn+kvol*mu])+Leaves[1]*conj(u12t[didn+kvol*mu]);
			a[1]=-Leaves[0]*u12t[didn+kvol*mu]+Leaves[1]*u11t[didn+kvol*mu];

			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{\mu}-\hat{\nu})U_\mu^\dagger(x-\hat{\nu})U_\nu(x-\hat{\nu})@f$
			Leaves[0]=a[0]*u11t[didn+kvol*nu]-a[1]*conj(u12t[didn+kvol*nu]);
			Leaves[1]=a[0]*u12t[didn+kvol*nu]+a[1]*conj(u11t[didn+kvol*nu]);

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			unsigned int din_didm=id[mu*kvol+didn];

			/// @f$U_\nu^\dagger(x-\hat{\nu})U_\mu^\dagger(x-\hat{\mu}-\hat{\nu})U_\nu(x-\hat{\mu}-\hat{\nu})@f$
			//a[0]=Leaves[0]*u11t[din_didm+kvol*mu]-Leaves[1]*conj(u12t[din_didm+kvol*mu]);
			a[0]=Leaves[0]*u11t[din_didm+kvol*nu]-Leaves[1]*conj(u12t[din_didm+kvol*nu]);
			//a[1]=Leaves[0]*u12t[din_didm+kvol*mu]+Leaves[1]*conj(u11t[din_didm+kvol*mu]);
			a[1]=Leaves[0]*u12t[din_didm+kvol*nu]+Leaves[1]*conj(u11t[din_didm+kvol*nu]);

			didm = id[mu*kvol+i]; 
			/// @f$U_\nu^\dagger(x-\hat{\nu})U_\mu^\dagger(x-\hat{\mu}-\hat{\nu})U_\nu(x-\hat{\mu}-\hat{\nu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*u11t[didm+kvol*mu]-a[1]*conj(u12t[didm+kvol*mu]);
			Leaves[1]=a[0]*u12t[didm+kvol*mu]+a[1]*conj(u11t[didm+kvol*mu]);

			//DEBUG
			//						Leaves[0]=0; Leaves[1]=0;
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
 *	@param	gen:			Which generator do we multiply the leaves by. Used for the force terms
 *	@param	gen_pos:		Where does the generator appear in the multiplication. Used for the force terms.
 *	
 */
template <typename T>
__device__ int Force_Leaf(complex<T> *u11t, complex<T> *u12t, complex<T> Leaves[nc],\
		unsigned int *iu, unsigned int *id, unsigned int i,const unsigned short mu,const unsigned short nu,\
		const unsigned short leaf,short gen,short gen_pos){
	complex<T> a[nc];
	unsigned int didm,didn,uidm;
	switch(leaf){
		case(0):
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);

			unsigned int uidn = iu[nu*kvol+i]; 
			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})@f$
			a[0]=Leaves[0]*conj(u11t[uidn+kvol*mu])+Leaves[1]*conj(u12t[uidn+kvol*mu]);
			a[1]=-Leaves[0]*u12t[uidn+kvol*mu]+Leaves[1]*u11t[uidn+kvol*mu];
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			/// @f$U_\mu(x)U_\nu(x+\hat{\mu})U^\dagger_\mu(x+\hat{\nu})U^\dagger_\nu(x)@f$
			Leaves[0]=a[0]*conj(u11t[i+kvol*nu])+a[1]*conj(u12t[i+kvol*nu]);
			Leaves[1]=-a[0]*u12t[i+kvol*nu]+a[1]*u11t[i+kvol*nu];

			//DEBUG
			//					Leaves[0]=0; Leaves[1]=0;
			break;
		case(1):
			didm = id[mu*kvol+i];
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})@f$
			a[0]=Leaves[0]*conj(u11t[didm+kvol*nu])+Leaves[1]*conj(u12t[didm+kvol*nu]);
			a[1]=-Leaves[0]*u12t[didm+kvol*nu]+Leaves[1]*u11t[didm+kvol*nu];
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			/// @f$U_\nu(x)U^\dagger_\mu(x-\hat{\mu}+\hat{\nu})U^\dagger_\nu(x-\hat{\mu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*u11t[didm+kvol*mu]-a[1]*conj(u12t[didm+kvol*mu]);
			Leaves[1]=a[0]*u12t[didm+kvol*mu]+a[1]*conj(u11t[didm+kvol*mu]);
			//DEBUG
			//		Leaves[0]=0; Leaves[1]=0;
			break;
		case(2):
			///Leaf in the forwards mu and backwards nu direction
			didn = id[nu*kvol+i]; 
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);
			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{\mu}-\hat{\nu})U_\mu^\dagger(x-\hat{\nu})@f$
			a[0]=Leaves[0]*conj(u11t[didn+kvol*mu])+Leaves[1]*conj(u12t[didn+kvol*mu]);
			a[1]=-Leaves[0]*u12t[didn+kvol*mu]+Leaves[1]*u11t[didn+kvol*mu];
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{\mu}-\hat{\nu})U_\mu^\dagger(x-\hat{\nu})U_\nu(x-\hat{\nu})@f$
			Leaves[0]=a[0]*u11t[didn+kvol*nu]-a[1]*conj(u12t[didn+kvol*nu]);
			Leaves[1]=a[0]*u12t[didn+kvol*nu]+a[1]*conj(u11t[didn+kvol*nu]);

			//DEBUG
			//					Leaves[0]=0; Leaves[1]=0;
			break;
		case(3):
			///Leaf in the backwards mu and backwards nu direction
			unsigned int din_didm=id[mu*kvol+didn];
			//Multiply by generator from the right after the first two links
			if(gen_pos==2)
				ByGenRight(Leaves,gen);

			/// @f$U_\nu^\dagger(x-\hat{\nu})U_\mu^\dagger(x-\hat{\mu}-\hat{\nu})U_\nu(x-\hat{\mu}-\hat{\nu})@f$
			//a[0]=Leaves[0]*u11t[din_didm+kvol*mu]-Leaves[1]*conj(u12t[din_didm+kvol*mu]);
			a[0]=Leaves[0]*u11t[din_didm+kvol*nu]-Leaves[1]*conj(u12t[din_didm+kvol*nu]);
			//a[1]=Leaves[0]*u12t[din_didm+kvol*mu]+Leaves[1]*conj(u11t[din_didm+kvol*mu]);
			a[1]=Leaves[0]*u12t[din_didm+kvol*nu]+Leaves[1]*conj(u11t[din_didm+kvol*nu]);
			//Multiply by generator from the right after the first three links
			if(gen_pos==3)
				ByGenRight(a,gen);

			didm = id[mu*kvol+i]; 
			/// @f$U_\nu^\dagger(x-\hat{\nu})U_\mu^\dagger(x-\hat{\mu}-\hat{\nu})U_\nu(x-\hat{\mu}-\hat{\nu})U_\mu(x-\hat{\mu})@f$
			Leaves[0]=a[0]*u11t[didm+kvol*mu]-a[1]*conj(u12t[didm+kvol*mu]);
			Leaves[1]=a[0]*u12t[didm+kvol*mu]+a[1]*conj(u11t[didm+kvol*mu]);

			//DEBUG
			//					Leaves[0]=0; Leaves[1]=0;
			break;
	}
	///gen_pos 4 is multiply the entire leaf by the generator from the left
	if(gen_pos==4){
		ByGenLeft(Leaves,gen);
	}
	return 0;
}
///CUDA Kernels
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
		complex<T> *ut[nc], unsigned int *iu, unsigned int *id, int mu, int nu){
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
			Leaf(ut[0],ut[1],Leaves,iu,id,i,mu,nu,leaf);
			clover1[i]+=Leaves[0]; clover2[i]+=Leaves[1];
		}
		///The clover is given by @f$F_{\mu\nu}=\frac{-i}{8}\left(Q_{\mu\nu}-Q_{\mu\nu}\right)^\dagger@f$. We do that
		///manually below.

		///The @f$\alpha@f$ component. Only the imaginary part survives. And since it is multiplied by @f$-i@f$ it is real.
		clover1[i]=2*clover1[i].imag();		clover1[i]*=(-I/8.0);

		///The @f$\beta@f$ component. Both real and imaginary components survive. It ends up getting doubled.
		clover2[i]+=clover2[i]; 				clover2[i]*=(-I/8.0);
	}
	return;
}

/**
 *	@brief	Clover contribution to the Molecular Dynamics force
 *
 *	@param	dSdpi:		Force
 *	@param	u11t,u12t:	Gauge fields
 *	@param	X1:			@f$\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	X2:			@f$M\left(M^\dagger M\right)^{-1} \Psi@f$
 *	@param	sigval:		@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$c_sw@f$
 * @param	sigin:		What element of the spinor is multiplied by row idirac each sigma matrix?
 * @param	iu,id:		Up/down indices
 * @param	clov:			Clover we're intereted in
 * @param	mu,nu:		Direction of clover we're interested in
 * @param	kappa:		Hopping parameter
 */
template <typename T>
__global__ void Clover_Force(double *dSdpi, complex<T> *ut[nc], complex<T> *hLeaves[nc], complex<T> *X1,\
		complex<T> *X2, const complex<T> *sigval, const unsigned short *sigin, unsigned int *iu, unsigned int *id,\
		const unsigned short clov,const unsigned short mu, const unsigned short nu, const float kappa){
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Two of these since we have the mu and nu contributions
		T dSdpis[2][3]={0,0,0}; 
		const unsigned int ipm=iu[i+kvol*mu];
		for(unsigned short fclov=0;fclov<(ndim-1)*(ndim-2);fclov++){
			complex<T> fleaf[nadj][nc];
			unsigned int site;
			for(unsigned short gen=0;gen<nadj;gen++){
				//This stores the half-leaf initially, then the output from Force_Leaves
				complex<T> tmp[nc];
				switch(fclov){
					case(0): //Clover at site
						site=i;
						tmp[0]=hLeaves[0][site+0*kvol]; tmp[1]=hLeaves[1][site+0*kvol];
						//Get leaf 0 with the correct generator in the initial position
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,0,gen,4);
						fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];

						//Get leaf 2 with the correct generator in the initial position
						tmp[0]=hLeaves[0][site+2*kvol]; tmp[1]=hLeaves[1][site+2*kvol];
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,2,gen,4);
						//+= here!!!
						fleaf[gen][0]+=tmp[0]; fleaf[gen][1]+=tmp[1];
					case(1): //Clover at i+mu
						site=ipm;
						//Get leaf 1 with the correct generator between links 3 and 4
						tmp[0]=hLeaves[0][site+1*kvol]; tmp[1]=hLeaves[1][site+1*kvol];
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,0,gen,3);
						fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
						//Get leaf 3 with the correct generator between links 3 and 4
						tmp[0]=hLeaves[0][site+3*kvol]; tmp[1]=hLeaves[1][site+3*kvol];
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,2,gen,3);
						//+= here!!!
						fleaf[gen][0]+=tmp[0]; fleaf[gen][1]+=tmp[1];
					case(2): //Clover at i+nu
						site=iu[i+kvol*nu];
						//Get leaf 2 with the correct generator between links 3 and 4
						tmp[0]=hLeaves[0][site+2*kvol]; tmp[1]=hLeaves[1][site+2*kvol];
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,2,gen,3);
						fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
					case(3): //Clover at i-nu
						site=id[i+kvol*nu];
						//Get leaf 0 with the correct generator between links 3 and 4
						tmp[0]=hLeaves[0][site+0*kvol]; tmp[1]=hLeaves[1][site+0*kvol];
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,0,gen,3);
						fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
					case(4): //Clover at i+mu+nu
						site=iu[ipm+kvol*nu];
						//Get leaf 3 with the correct generator between links 2 and 3
						tmp[0]=hLeaves[0][site+3*kvol]; tmp[1]=hLeaves[1][site+3*kvol];
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,3,gen,2);
						fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
					case(5): //Clover at i+mu-nu
						site=id[ipm+kvol*nu];
						//Get leaf 1 with the correct generator between links 2 and 3
						tmp[0]=hLeaves[0][site+1*kvol]; tmp[1]=hLeaves[1][site+1*kvol];
						Force_Leaf(ut[0],ut[1],tmp,iu,id,site,mu,nu,1,gen,2);
						fleaf[gen][0]=tmp[0]; fleaf[gen][1]=tmp[1];
				}
			}

			for(unsigned short idirac=0; idirac<ndirac*nc; idirac+=nc){
				const unsigned short sind = sigin[clov*ndirac+(idirac>>1)]<<(nc-1);	
				//Calculate the index. For the next colour we add kvol
				unsigned int ind = site+kvol*idirac;
				//Prefetching. Might not be needed here though
				complex<T> X1sc[nc];
				//X1 is always conjugated. So do it once here instead of twice and be done with it.	
				X1sc[0]=conj(X1[ind]); X1sc[1]=conj(X1[ind+kvol]);
				ind = site+kvol*sind;
				complex<T> X2s[nc];
				X2s[0]=X2[ind]; X2s[1]=X2[ind+kvol];

				for(unsigned short gen=0;gen<nadj;gen++){
					complex<T> fleafc[2];
					fleafc[0]=conj(fleaf[gen][0]); fleafc[1]=conj(fleaf[gen][1]);
					//mu contribution
					dSdpis[0][gen]+=(sigval[clov*ndirac+idirac]*(X1sc[0]*(fleaf[gen][0]*X2s[0]+fleaf[gen][1]*X2s[1])+\
								X1sc[1]*(-fleafc[0]*X2s[0]+fleafc[1]*X2s[1]))).real();
					//nu contribution
					dSdpis[1][gen]+=(sigval[clov*ndirac+idirac]*(X1sc[0]*(fleafc[0]*X2s[0]-fleaf[gen][1]*X2s[1])+\
								X1sc[1]*(fleafc[1]*X2s[0]+fleaf[gen][0]*X2s[1]))).real();
				}
			}
		}
		for(unsigned short gen=0;gen<nadj;gen++){
			dSdpi[i+kvol*(gen*ndim+mu)]-=kappa*dSdpis[0][gen];
			dSdpi[i+kvol*(gen*ndim+nu)]-=kappa*dSdpis[1][gen];
		}
	}
	return;
}
/**
 *	@brief Clover analogue of the Dslash operation. This version acts on all flavours simiilar to Dslash and Dslash_d
 *	
 *
 *	@param	phi:					Final pseudofermion field. This is almost always multiplied by Dslash before calling this function
 *	@param	r:						Pseudofermion field before multiplication. The thing we want to multiply by the clover
 *	@param	clover1,clover2:	Array of clovers
 *	@param	sigval:				@f$ \sigma_{\mu\nu}@f$ entries scaled by @f$ c_{sw}@f$
 * @param	sigin:				What element of the spinor is multiplied by row idirac each sigma matrix?
 */
template <typename T>
__global__ void ByClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2, complex<T> *sigval, unsigned short *sigin){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;

	for(int i=gthreadId;i<kvol;i+=bsize*gsize){
		//Prefetched r and Phi array
		complex<T> phi_s[ndirac][nc];
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
					r_s[c]=r[(i*ngorkov+sind)*nc+c];
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				phi_s[igorkov][0]+=sigval[clov*ndirac+idirac]*(creal(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				phi_s[igorkov][1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]+creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short igorkov=0; igorkov<ndirac; igorkov++)
			for(unsigned short c=0; c<nc; c++)
				///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				phi[i+kvol*(c+nc*igorkov)]+=2*phi_s[igorkov][c];
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
 */
template <typename T>
__global__ void HbyClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2,complex<T> *sigval, const float kappa, unsigned short *sigin){
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
					r_s[c]=r[i+kvol*(sind+c)];
				}
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				const complex<T> sig=sigval[clov*ndirac+(idirac>>1)];
				phi_s[idirac+0]+=kappa*sig*(creal(clov_s[0])*r_s[0]+clov_s[1]*r_s[1]);
				phi_s[idirac+1]+=kappa*sig*(conj(clov_s[1])*r_s[0]+creal(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac; idirac+=nc)
			for(unsigned short c=0; c<nc; c++)
				///@f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				///But then we multiply by @f$-\frac{1}{2}@f$ so the @f$2@f$ disappears
				phi[i+kvol*(c+idirac)]-=phi_s[idirac+c];
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
						ut,iu,id,mu,nu);
			}
	cudaDeviceSynchronise();
	return 0;
}
void cuByClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval, unsigned short *sigin){
	ByClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,sigin);
}
void cuHbyClover(Complex *phi, Complex *r, Complex *clover[nc],Complex *sigval, const float kappa, unsigned short *sigin){
	HbyClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,kappa,sigin);
}
void cuByClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, unsigned short *sigin){
	ByClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,sigin);
}
void cuHbyClover_f(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, const float kappa, unsigned short *sigin){
	HbyClover<<<dimGrid,dimBlock>>>(phi,r,clover[0],clover[1],sigval,kappa,sigin);
}

int cuClover_Force(double *dSdpi, Complex_f *ut[nc], Complex_f *X1, Complex_f *X2, Complex_f *sigval,\
		unsigned short *sigin, unsigned int *iu, unsigned int *id, const float kappa){
	const char funcname[]="Clover_Force";
	Complex_f *hLeaves[6][nc];
	for(unsigned int mu=0;mu<ndim-1;mu++)
		for(unsigned int nu=mu+1;nu<ndim;nu++){
			//Clover index
			const unsigned short clov = (mu==0) ? nu-1 :mu+nu;
			//Allocate half-leaf memory
			cudaMallocAsync((void **)&hLeaves[clov][0],ndim*kvol,streams[mu]);
			cudaMallocAsync((void **)&hLeaves[clov][1],ndim*kvol,streams[mu]);
			Half_Leaves<<<dimGrid,dimBlock,0,streams[mu]>>>(hLeaves[clov][0],hLeaves[clov][1],ut[0],ut[1],iu,id,mu,nu);
		}
	cudaDeviceSynchronise();
	//Cannot stream the actual force calculation. We have a mu-nu and a nu-mu contribution. Streams will create a potential race condition.
	for(unsigned int mu=0;mu<ndim-1;mu++)
		for(unsigned int nu=mu+1;nu<ndim;nu++){
			//Clover index
			const unsigned short clov = (mu==0) ? nu-1 :mu+nu;
			//Allocate half-leaf memory
			Clover_Force<<<dimGrid,dimBlock>>>(dSdpi,hLeaves[clov],ut,X1,X2,sigval,sigin,iu,id,clov,mu,nu,kappa);
		}
	cudaDeviceSynchronise();
	//Free half leaves
	for(unsigned int mu=0;mu<ndim-1;mu++)
		for(unsigned int nu=mu+1;nu<ndim;nu++){
			const unsigned short clov = (mu==0) ? nu-1 :mu+nu;
			cudaFreeAsync(hLeaves[clov][0],streams[mu]); cudaFreeAsync(hLeaves[clov][1],streams[mu]);
		}
	cudaDeviceSynchronise();
	return 0;
}
