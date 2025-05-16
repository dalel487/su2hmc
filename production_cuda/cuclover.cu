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
template <typename T> 
__global__ void Force_Leaves(complex<T> *Fleaf1, complex<T> *Fleaf2,complex<T> *Leaves1, complex<T> *Leaves2,unsigned short adj,unsigned short clov){
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	for(unsigned int i=gthreadId;i<kvol;i+=bsize*gsize){
		Fleaf1[i]=0; Fleaf2[i]=0;
		///Only three clovers @f$\mu\ne\nu@f$ contribute to the force term
		///Out of these clovers only two leaves contribute (containing @f$U_\mu@f$). Daggered and undaggered
		///Additionally, @f$\sigma_{\nu\mu}F_{\nu\mu}=\sigma_{\mu\nu}F_{\mu\nu}@f$ so we can double the final answer

		///Contribution from @f$(f_{\mu\nu})@f$
		///Contribution 1 is the normal plaquette so leaf 0
		GenLeaf(Fleaf1,Fleaf2,Leaves1,Leaves2,i,0,adj,true);
		///Contribution 3 is the normal plaquette daggered so leaf 0
		GenLeafd(Fleaf1,Fleaf2,Leaves1,Leaves2,i,0,adj,false);

		///Contribution 2 is the leaf containing @f$ U_\mu@f$ link and @f$ \nu<0@f$)
		GenLeaf(Fleaf1,Fleaf2,Leaves1,Leaves2,i,2,adj,true);
		///Contribution 4 is the leaf containing @f$ U^dagger_\mu@f$ link and @f$ \nu<0@f$)
		GenLeafd(Fleaf1,Fleaf2,Leaves1,Leaves2,i,2,adj,false);

		///NOTE: The clover is scaled by -i/8.0, but the leaves were not. We do that scaling here.
		///		The 4.0 instead of 8.0 is due to the second contribution from @f$\nu\mu@f$
		Fleaf1[i]*=-I/4.0f; Fleaf2[i]*=-I/4.0f;
	}
	return;
}
//Actual force stuff
template <typename T>
__global__ void Clover_Force(double *dSdpi,complex<T> *Fleaf1,complex<T> *Fleaf2, complex<T> *X1, complex<T> *X2,\
		const complex<T> *sigval, const unsigned short *sigin, const unsigned short adj,const unsigned short clov,unsigned short mu){
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	complex<T> X1s[nc],X2s[nc];
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		complex<T> dSdpis=0;
		for(unsigned short idirac=0;idirac<ndirac;idirac++){
			const unsigned short igork1 = sigin[clov*ndirac+idirac];	

			//Prefetching. Might not be needed here though
			X1s[0]=X1[i+kvol*(nc*idirac)]; X1s[1]=X1[i+kvol*(1+nc*idirac)];
			X2s[0]=X2[i+kvol*(nc*igork1)]; X2s[1]=X2[i+kvol*(1+nc*igork1)];

			dSdpis-=cimag(sigval[clov*ndirac+idirac]*(
						conj(X1s[0])*(Fleaf1[i]*X2s[0]+Fleaf2[i]*X2s[1])+
						conj(X1s[1])*(-conj(Fleaf2[i])*X2s[0]+conj(Fleaf1[i])*X2s[1])));
		}
		dSdpi[i+kvol*(adj*ndim+mu)]=(double)creal(dSdpis);
	}
	return;
}


template <typename T>
__global__ void ByClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2, complex<T> *sigval, unsigned short *sigin){
	const char funcname[] = "HbyClover";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	for(int i=gthreadId;i<kvol;i+=bsize*gsize){
		//Prefetched r and Phi array
		complex<T> phi_s[ndirac][nc];
#pragma unroll
			for(int igorkov=0; igorkov<ngorkov; igorkov++)
		for(unsigned short c=0; c<nc; c++){
			phi_s[igorkov][c]=0;
		}
		complex<T> r_s[nc];
		complex<T> clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover1[clov*kvol+i]; clov_s[1]=clover2[clov*kvol+i];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
			//Mod 4 done bitwise. In general n mod 2^m = n & (2^m-1)
					const unsigned short idirac = igorkov&3;
					const unsigned short igork1 = (igorkov<4) ? sigin[clov*ndirac+idirac] : sigin[clov*ndirac+idirac]+4;
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
					r_s[c]=r[(i*ngorkov+igork1)*nc+c];
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				phi_s[igorkov][0]+=sigval[clov*ndirac+idirac]*(clov_s[0]*r_s[0]+clov_s[1]*r_s[1]);
				phi_s[igorkov][1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]+conj(clov_s[0])*r_s[1]);
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
__global__ void HbyClover(complex<T> *phi, complex<T> *r, complex<T> *clover1, complex<T> *clover2,complex<T> *sigval, unsigned short *sigin){
	const char funcname[] = "HbyClover";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	for(int i=gthreadId;i<kvol;i+=bsize*gsize){
		//Prefetched r and Phi array
		complex<T> phi_s[ndirac][nc];
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac; idirac++)
		for(unsigned short c=0; c<nc; c++){
			phi_s[idirac][c]=0;
		}
		complex<T> r_s[nc];
		complex<T> clov_s[nc];
#pragma unroll
		for(unsigned short clov=0;clov<6;clov++){
			clov_s[0]=clover1[clov*kvol+i]; clov_s[1]=clover2[clov*kvol+i];
			for(int idirac=0; idirac<ndirac; idirac++){
				const unsigned short igork1 = sigin[clov*ndirac+idirac];	
#pragma unroll
				for(unsigned short c=0; c<nc; c++)
					r_s[c]=r[(i*ndirac+igork1)*nc+c];
				///Note that @f$\sigma_{\mu\nu}@f$ was scaled by @f$\frac{c_\text{SW}}{2}@f$ when we defined it.
				phi_s[idirac][0]+=sigval[clov*ndirac+idirac]*(clov_s[0]*r_s[0]+clov_s[1]*r_s[1]);
				phi_s[idirac][1]+=sigval[clov*ndirac+idirac]*(conj(clov_s[1])*r_s[0]+conj(clov_s[0])*r_s[1]);
			}
		}
#pragma unroll
		for(unsigned short idirac=0; idirac<ndirac; idirac++)
			for(unsigned short c=0; c<nc; c++)
				///Also @f$\sigma_{\mu\nu}F_{\mu\nu}=\sigma_{\nu\mu}F_{\nu\mu}@f$ so we double it to take account of that
				phi[i+kvol*(c+nc*idirac)]+=2*phi_s[idirac][c];
	}
	return;
}

//Calling Wrappers
//This gets called by C so cannot be templated...
int cuClover(Complex_f *clover[nc],Complex_f *Leaves[6][nc],Complex_f *ut[nc], unsigned int *iu, unsigned int *id){
	const char funcname[]="cuClover";
	cudaMallocAsync((void **)&clover[0],6*kvol*sizeof(Complex_f),streams[0]);
	cudaMallocAsync((void **)&clover[1],6*kvol*sizeof(Complex_f),streams[1]);
	for(unsigned short mu=0;mu<ndim-1;mu++)
		for(unsigned short nu=mu+1;nu<ndim;nu++)
			if(mu!=nu){
				//Clover index
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
				//Allocate clover memory
				//Note that the clover is completely local, so doesn't need a halo for MPI
				cudaMallocAsync((void **)&Leaves[clov][0],kvol*ndim*sizeof(Complex_f),streams[clov]);
				cudaMallocAsync((void **)&Leaves[clov][1],kvol*ndim*sizeof(Complex_f),streams[clov]);
				Half_Clover<<<dimGrid,dimBlock,0,streams[clov]>>>(clover[0]+clov*kvol,clover[1]+clov*kvol,Leaves[clov][0],Leaves[clov][1],ut[0],ut[1],iu,id,mu,nu);
				Full_Clover<<<dimGrid,dimBlock,0,streams[clov]>>>(clover[0]+clov*kvol,clover[1]+clov*kvol);

			}
	cudaDeviceSynchronise();
	return 0;
}
void cuByClover(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, unsigned short *sigin){
	ByClover<<<dimGrid,dimBlock,0,0>>>(phi,r,clover[0],clover[1],sigval,sigin);
}
void cuHbyClover(Complex_f *phi, Complex_f *r, Complex_f *clover[nc],Complex_f *sigval, unsigned short *sigin){
	HbyClover<<<dimGrid,dimBlock,0,0>>>(phi,r,clover[0],clover[1],sigval,sigin);
}
int cuClover_Force(double *dSdpi, Complex_f *Leaves[6][nc], Complex_f *X1, Complex_f *X2, Complex_f *sigval,unsigned short *sigin){
	const char funcname[]="Clover_Force";
	for(unsigned short adj=0;adj>nadj;adj++){
		Complex_f *Fleaf[nc];
		cudaMallocAsync((void **)&Fleaf[0],kvol*sizeof(Complex_f),streams[adj]);
		cudaMallocAsync((void **)&Fleaf[1],kvol*sizeof(Complex_f),streams[adj]);
		for(unsigned int mu=0;mu<ndim-1;mu++)
			for(unsigned int nu=mu+1;nu<ndim;nu++){
				//Clover index
				unsigned short clov = (mu==0) ? nu-1 :mu+nu;
				//Allocate clover memory
				Force_Leaves<<<dimGrid,dimBlock,0,streams[adj]>>>(Fleaf[0],Fleaf[1],Leaves[clov][0],Leaves[clov][1],adj,clov);
				Clover_Force<<<dimGrid,dimBlock,0,streams[adj]>>>(dSdpi,Fleaf[0],Fleaf[1],X1,X2,sigval,sigin,adj,clov,mu);
			}
		cudaFreeAsync(Fleaf[0],streams[adj]); cudaFreeAsync(Fleaf[1],streams[adj]); 
	}
	cudaDeviceSynchronise();
	return 0;
}
/*
#pragma omp parallel for
for(unsigned int i=0;i<kvol;i++)
{
clover[0][clov*kvol+i]=0;clover[1][clov*kvol+i]=0;
Half_Clover(clover[clov],Leaves[clov],ut,iu,id,i,mu,nu);	
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
}
*/
