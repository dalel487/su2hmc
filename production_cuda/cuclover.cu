/**
 * @file 	cuclover.cu
 *
 * @brief	CUDA routines related to clover improved wilson fermions
 *
 * @author	D. Lawlor
 */
#include <clover.h>

//CUDA Device code
template <typename T>
__device__ int Clover_SU2plaq(complex<T> *u11t, complex<T> *u12t, complex<T> *Leaves1, complex<T> *Leaves2,\
		unsigned int *iu,  int i, int mu, int nu){
	const char *funcname = "SU2plaq";
	unsigned int uidm = iu[mu*kvol+i]; 
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
	Leaves1[i]=u11t[i+kvol*mu]*u11t[uidm+kvol*nu]-u12t[i+kvol*mu]*conj(u12t[uidm+kvol*nu]);
	Leaves2[i]=u11t[i+kvol*mu]*u12t[uidm+kvol*nu]+u12t[i+kvol*mu]*conj(u11t[uidm+kvol*nu]);

	unsigned int uidn = iu[nu*kvol+i]; 
	complex<T> a11=Leaves1[i]*conj(u11t[uidn+kvol*mu])+Leaves2[i]*conj(u12t[uidn+kvol*mu]);
	complex<T> a12=-Leaves1[i]*u12t[uidn+kvol*mu]+Leaves2[i]*u11t[uidn+kvol*mu];

	Leaves1[i]=a11*conj(u11t[i+kvol*nu])+a12*conj(u12t[i+kvol*nu]);
	Leaves2[i]=-a11*u12t[i+kvol*nu]+a12*u11t[i+kvol*nu];
	return 0;
}
template <typename T>
__device__ int Leaf(complex<T> *u11t, complex<T> *u12t, complex<T> *Leaves1, complex<T> *Leaves2,\
		unsigned int *iu, unsigned int *id, unsigned int i, int mu, int nu, short leaf){
	char *funcname="Leaf";
	complex<T> a[nc];
	unsigned int didm,didn,uidn,uidm;
	switch(leaf){
		case(0):
			//Both positive is just a standard plaquette
			Clover_SU2plaq(u11t,u12t,Leaves1,Leaves2,iu,i,mu,nu);
			break;
		case(1):
			//\mu<0 and \nu>=0
			didm = id[mu*kvol+i];
			/// @f$U_\mu^\dagger\(x-\hat{\mu})U_\nu(x-\hat{\mu}\)@f$
			Leaves1[i+kvol*leaf]=conj(u11t[didm+kvol*mu])*u11t[didm+kvol*nu]+u12t[didm+kvol*mu]*conj(u12t[didm+kvol*nu]);
			Leaves2[i+kvol*leaf]=conj(u11t[didm+kvol*mu])*u12t[didm+kvol*nu]-u12t[didm+kvol*mu]*conj(u11t[didm+kvol*nu]);

			int uin_didm=id[nu*kvol+didm];
			/// @f$U_\mu^\dagger\(x-\hat{\mu})U_\nu(x+-hat{\mu}\)U_\mu(x-\hat{mu}+\hat{nu})@f$
			//a[0]=Leaves1[i+kvol*leaf]*conj(u11t[didm+kvol*nu])+conj(Leaves2[i+kvol*leaf])*u12t[didm+kvol*nu];
			a[0]=Leaves1[i+kvol*leaf]*u11t[uin_didm+kvol*mu]-Leaves2[i+kvol*leaf]*conj(u12t[uin_didm+kvol*mu]);
			//a[1]=Leaves2[i+kvol*leaf]*conj(u11t[didm+kvol*nu])-conj(Leaves1[i+kvol*leaf])*u12t[didm+kvol*nu];
			a[1]=Leaves1[i+kvol*leaf]*u12t[uin_didm+kvol*mu]+Leaves2[i+kvol*leaf]*conj(u11t[uin_didm+kvol*mu]);

			/// @f$U_\mu^\dagger\(x)U_\nu^\dagger(x+\hat{\mu}-\hat{\nu}\)U_\mu(x+\hat{mu}-\hat{nu})U_\nu^\dagger(x-\hat{\nu})@f$
			//Leaves1[i+kvol*leaf]=a[0]*u11t[didm+kvol*mu]-conj(a[1])*u12t[didm+kvol*mu];
			Leaves1[i+kvol*leaf]=a[0]*conj(u11t[i+kvol*nu])+a[1]*conj(u12t[i+kvol*nu]);
			Leaves2[i+kvol*leaf]=-a[0]*u12t[i+kvol*nu]+a[1]*u11t[i+kvol*nu];
			break;
		case(2):
			//\mu>=0 and \nu<0
			//TODO: Figure out down site index
			//Another awkward index
			uidm = iu[mu*kvol+i]; int din_uidm=id[nu*kvol+uidm];
			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{mu}-\hat{\nu})@f$
			Leaves1[i+kvol*leaf]=u11t[i+kvol*mu]*conj(u11t[din_uidm+kvol*nu])+u12t[i+kvol*mu]*conj(u12t[din_uidm+kvol*nu]);
			Leaves2[i+kvol*leaf]=-u11t[i+kvol*mu]*u12t[din_uidm+kvol*nu]+u12t[i+kvol*mu]*u11t[din_uidm+kvol*nu];

			didn = id[nu*kvol+i]; 
			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{mu}-\hat{\nu})U_\mu^\dagger(x-\hat{nu}\)@f$
			a[0]=Leaves1[i+kvol*leaf]*conj(u11t[didn+kvol*mu])+Leaves2[i+kvol*leaf]*conj(u12t[didn+kvol*mu]);
			a[1]=-Leaves1[i+kvol*leaf]*u12t[didn+kvol*mu]+Leaves2[i+kvol*leaf]*u11t[didn+kvol*mu];

			/// @f$U_\mu(x)U_\nu^\dagger(x+\hat{mu}-\hat{\nu})U_\mu^\dagger(x-\hat{nu}\)U_\nu(x-\hat{\nu})@f$
			Leaves1[i+kvol*leaf]=a[0]*u11t[didn+kvol*nu]-a[1]*conj(u12t[didn+kvol*nu]);
			Leaves2[i+kvol*leaf]=a[0]*u12t[didn+kvol*nu]+a[1]*conj(u11t[didn+kvol*nu]);

			break;
		case(3):
			//\mu<0 and \nu<0
			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu})@f$
			didm = id[mu*kvol+i];int dim_didn=id[nu*kvol+didm];
			Leaves1[i+kvol*leaf]=conj(u11t[didm+kvol*mu])*conj(u11t[dim_didn+kvol*nu])-u12t[didm+kvol*mu]*conj(u12t[dim_didn+kvol*nu]);
			Leaves2[i+kvol*leaf]=-conj(u11t[didm+kvol*mu])*u12t[dim_didn+kvol*nu]-u12t[didm+kvol*mu]*u11t[dim_didn+kvol*nu];

			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(x-\hat{\mu}-\hat{\nu})@f$
			a[0]=Leaves1[i+kvol*leaf]*u11t[dim_didn+kvol*mu]-Leaves2[i+kvol*leaf]*conj(u12t[dim_didn+kvol*mu]);
			a[1]=Leaves1[i+kvol*leaf]*u12t[dim_didn+kvol*mu]+Leaves2[i+kvol*leaf]*conj(u11t[dim_didn+kvol*mu]);

			didn = id[nu*kvol+i]; 
			/// @f$U_\mu^\dagger(x-\hat{\mu})U_\nu^\dagger(x-\hat{\mu}-\hat{\nu})U_\mu(x-\hat{\mu}-\hat{\nu})U_\nu(x-\hat{\nu})@f$
			Leaves1[i+kvol*leaf]=a[0]*u11t[didn+kvol*nu]-a[1]*conj(u12t[didn+kvol*nu]);
			Leaves2[i+kvol*leaf]=a[0]*u12t[didn+kvol*nu]+a[1]*conj(u11t[didn+kvol*nu]);
			break;
	}
	return 0;
}
//Generator by Leaf
//Fleaf doesn't need to be changed thankfully!
template <typename T>
__device__ int GenLeaf(complex<T> *Fleaf1,complex<T> *Fleaf2, complex<T> *Leaves1, complex<T> *Leaves2,
		const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm){
	const char funcname[] = "GenLeaf";
	//Adding or subtracting this term
	const short sign = (pm) ? 1 : -1;
	//Which generator are we multiplying by? Zero indexed so subtract one from your usual index in textbooks
	switch(adj){
		case(0):
			Fleaf1[i]+=sign*Leaves2[i+kvol*leaf];		Fleaf2[i]+=sign*Leaves1[i+kvol*leaf];
		case(1):
			Fleaf1[i]+=-sign*I*Leaves2[i+kvol*leaf];	Fleaf2[i]+=sign*I*Leaves1[i+kvol*leaf];
		case(2):
			Fleaf1[i]+=sign*Leaves1[i+kvol*leaf];		Fleaf2[i]+=-sign*Leaves2[i+kvol*leaf];
	}
	return 0;
}
template <typename T>
__device__ int GenLeafd(complex<T> *Fleaf1,complex<T> *Fleaf2, complex<T> *Leaves1, complex<T> *Leaves2,
		const unsigned int i,const unsigned short leaf,const unsigned short adj,const bool pm){
	const char funcname[] = "GenLeafd";
	//Adding or subtracting this term
	const short sign = (pm) ? 1 : -1;
	//Which generator are we multiplying by? Zero indexed so subtract one from your usual index in textbooks
	switch(adj){
		case(0):
			Fleaf1[i]+=-sign*Leaves2[i+kvol*leaf]; 		Fleaf2[i]+=sign*conj(Leaves1[i+kvol*leaf]);
		case(1):
			Fleaf1[i]+=sign*I*Leaves2[i+kvol*leaf];		Fleaf2[i]+=sign*I*conj(Leaves1[i+kvol*leaf]);
		case(2):
			Fleaf1[i]+=sign*conj(Leaves1[i+kvol*leaf]); 	Fleaf2[i]+=sign*conj(Leaves2[i+kvol*leaf]);
	}
	return 0;
}

///CUDA Kernels
template <typename T>
__global__  void Half_Clover(complex<T> *clover1, complex<T> *clover2, complex<T> *Leaves1, complex<T> *Leaves2, complex<T> *u11t, complex<T> *u12t, unsigned int *iu, unsigned int *id, int mu, int nu){
	const char funcname[] ="Half_Clover";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		clover1[i]=0;clover2[i]=0;
		for(unsigned short leaf=0;leaf<ndim;leaf++)
		{
			Leaf(u11t,u12t,Leaves1,Leaves2,iu,id,i,mu,nu,leaf);
			clover1[i]+=Leaves1[i+kvol*leaf]; clover2[i]+=Leaves2[i+kvol*leaf];
		}
	}
	return;
}
template <typename T>
__global__  void Full_Clover(complex<T> *clover1, complex<T> *clover2){
	const char funcname[] ="Full_Clover";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		//creal(clover1) drops so we are traceless. And everything else just gets doubled
		clover1[i]-=conj(clover1[i]);	clover1[i]*=(-I/8.0);
		clover2[i]+=clover2[i]; 			clover2[i]*=(-I/8.0);
	}
	return;
}

//Generator by Leaf
template <typename T>
__device__  void GenLeaf(complex<T> *fleaf1,complex<T> *fleaf2,const unsigned short adj){
	const char funcname[] = "GenLeaf";
	//TODO: Check for factors of a half
	//Which generator are we multiplying by?
	//A buffer to store the leaf since we need to swap them
	complex<T> buffer=*fleaf1;
	switch(adj){
		case(0): //Sigma_x
					//Minus is from @f$-\bar{beta}@f$
			*fleaf1=-conj(*fleaf2);
			*fleaf2=buffer;
		case(1): //Sigma_y
					//Minus from Sigma_y and from @f$-\bar{\beta}@f$ gives an overall plus
			*fleaf1=-I*conj(*fleaf2);
			*fleaf2=I*buffer;
		case(2): //Sigma_z
					//No buffer here needed here.
					// *fleaf1=*fleaf1;
					//Minus from Sigma_z and from @f$-\bar{\beta}@f$ gives an overall plus
			*fleaf2=conj(*fleaf2);
	}
	return;
}
template <typename T> 
__device__ void Force_Leaves(complex<T> *fleaf1, complex<T> *fleaf2,complex<T> *Leaves1, complex<T> *Leaves2,\
		unsigned int *iu, unsigned int *id, const unsigned int i){
	///Fleaf consists of the sum of the @f$\mu\nu@f$ and @f$\mu,-\nu@f$ leaves, minus their hermitian conjugates
	///This can be expressed in terms of the imaginary part of Leaves1 and all of Leaves2 doubled
	//Bracket ordering here is just for optimisation. It skips the @f$2\times0@f$ multiplications from the real part.

	//fleaf_c is a register to hold each leaf whilst we gather the terms.
	//We start with the clover at the lattice site in question
	*fleaf1=I*(2*(Leaves1[i].imag()+Leaves1[i+kvol*2].imag()));;

	//Leaves1[i]+Leaves1[i+kvol*2]-conj(Leaves1[i])-conj(Leaves1[i+kvol*2]);

	///NOTE: The clover is scaled by -i/8.0, but the leaves were not. We do that scaling here.
	///		@f$\sigma_{\nu\mu}F_{\nu\mu}=\sigma_{\mu\nu}F_{\mu\nu}@f$ so we can double the final answer
	///		to get 4.0 instead of 8.0
	///
	///		The @f$i@f$ gets dropped since we have a factor of @f$-i@f$ from the derivative term too.
	*fleaf1*=1/4.0f;

	//Second leaf
	*fleaf2=2*(Leaves2[i]+Leaves2[i+kvol*2]);
	//Leaves2[i]+Leaves2[i+kvol*2]-conj(Leaves2[i])-conj(Leaves2[i+kvol*2]);
	*fleaf2*=1/4.0f;
	///Additionally, There are three more clovers with leaves containing @f$U_\mu(x)@f$. This will be a right pain in
	///the arse on the CPU since it means we need to introduce halo exchanges to the leaves...
	return;
}
//Actual force stuff
template <typename T>
__global__ void Clover_Force(double *dSdpi, complex<T> *Leaves[nc], complex<T> *X1, complex<T> *X2,\
		const complex<T> *sigval, const unsigned short *sigin, unsigned int *iu, unsigned int *id,\
		const unsigned short clov,unsigned short mu, const float kappa){
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	complex<T> X1s[nc]={{0,0},{0,0}};
	complex<T> X2s[nc]={{0,0},{0,0}};
	complex<T> fleaf_c[nc] = {{0,0},{0,0}};
	T dSdpis[3]={0,0,0};
	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Pointer arithetic because I'm lazy
		Force_Leaves(fleaf_c,fleaf_c+1,Leaves[0],Leaves[1],iu,id,i);
		for(unsigned short idirac=0;idirac<ndirac;idirac+=nc){
			const unsigned short igork1 = sigin[clov*ndirac+(idirac>>1)]<<(nc-1);	

			//Calculate the index. For the next colour we add kvol
			unsigned int ind = i+kvol*idirac;
			//Prefetching. Might not be needed here though
			X1s[0]=X1[ind]; X1s[1]=X1[ind+kvol];
			ind = i+kvol*igork1;
			X2s[0]=X2[ind]; X2s[1]=X2[ind+kvol];

			//i Sigma_x: Real part of @f$i z@f$ is minus the imaginary part of z
			dSdpis[0]-=(sigval[clov*ndirac+idirac]*(
						conj(X1s[0])*(-conj(fleaf_c[1])*X2s[0]+conj(fleaf_c[0])*X2s[1])+
						conj(X1s[1])*(fleaf_c[0]*X2s[0]+fleaf_c[1]*X2s[1]))).imag();
			//i Sigma_y: Real part of @f$ i i z@f$ is minus the real part of @f$z@f$
			dSdpis[1]-=(sigval[clov*ndirac+idirac]*(
						conj(X1s[0])*(conj(fleaf_c[1])*X2s[0]-conj(fleaf_c[0])*X2s[1])+
						conj(X1s[1])*(fleaf_c[0]*X2s[0]+fleaf_c[1]*X2s[1]))).real();
			//i Sigma_z Real part of @f$i z@f$ is minus the imaginary part of z
			dSdpis[2]-=(sigval[clov*ndirac+idirac]*(
						conj(X1s[0])*(fleaf_c[0]*X2s[0]+fleaf_c[1]*X2s[1])+
						conj(X1s[1])*(conj(fleaf_c[1])*X2s[0]-conj(fleaf_c[0])*X2s[1]))).imag();
		}
		for(unsigned short adj=0;adj<nadj;adj++)
			dSdpi[i+kvol*(adj*ndim+mu)]+=kappa*dSdpis[adj];
	}
	return;
}


template <typename T>
__global__ void ByClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2, complex<T> *sigval, unsigned short *sigin){
	const char funcname[] = "HbyClover";
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
				const unsigned short igork1 = (igorkov<4) ? sigin[clov*ndirac+idirac] : sigin[clov*ndirac+idirac]+4;
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
					r_s[c]=r[(i*ngorkov+igork1)*nc+c];
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
template <typename T>
__global__ void HbyClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2,complex<T> *sigval, const float kappa, unsigned short *sigin){
	const char funcname[] = "HbyClover";
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
				const unsigned short igork1 = sigin[clov*ndirac+(idirac>>1)] << (nc-1);
#pragma unroll
				for(unsigned short c=0; c<nc; c++){
					r_s[c]=r[i+kvol*(igork1+c)];
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
				///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				phi[i+kvol*(c+idirac)]+=2*phi_s[idirac+c];
	}
	return;
}

//Calling Wrappers
//This gets called by C so cannot be templated...
int cuClover(Complex_f *clover[nc],Complex_f *Leaves[6][nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id){
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
#ifdef _DEBUG
				cudaMallocManaged((void **)&Leaves[clov][0],kvol*ndim*sizeof(Complex_f),cudaMemAttachGlobal);
				cudaMallocManaged((void **)&Leaves[clov][1],kvol*ndim*sizeof(Complex_f),cudaMemAttachGlobal);
#else
				cudaMallocAsync((void **)&Leaves[clov][0],kvol*ndim*sizeof(Complex_f),streams[clov]);
				cudaMallocAsync((void **)&Leaves[clov][1],kvol*ndim*sizeof(Complex_f),streams[clov]);
#endif
				Half_Clover<<<dimGrid,dimBlock,0,streams[clov]>>>(clover[0]+clov*kvol,clover[1]+clov*kvol,Leaves[clov][0],Leaves[clov][1],ut[0],ut[1],iu,id,mu,nu);
				Full_Clover<<<dimGrid,dimBlock,0,streams[clov]>>>(clover[0]+clov*kvol,clover[1]+clov*kvol);

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

int cuClover_Force(double *dSdpi, Complex_f *Leaves[6][nc], Complex_f *X1, Complex_f *X2, Complex_f *sigval,\
		unsigned short *sigin, unsigned int *iu, unsigned int *id, const float kappa){
	const char funcname[]="Clover_Force";
	//dSdpi depends on the three values of @f$\mu@f$. So we use that for the streams instead of clover
	for(unsigned int mu=0;mu<ndim-1;mu++){
		for(unsigned int nu=mu+1;nu<ndim;nu++){
			//Clover index
			unsigned short clov = (mu==0) ? nu-1 :mu+nu;
			//Allocate clover memory
			Clover_Force<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,Leaves[clov],X1,X2,sigval,sigin,iu,id,clov,mu,kappa);
		}
	}
	cudaDeviceSynchronise();
	return 0;
}
