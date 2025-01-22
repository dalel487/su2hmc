/*
 * Code for bosonic observables
 * Basically polyakov loop and Plaquette routines
 */
#include	<par_mpi.h>
#include	<su2hmc.h>
#include <matrices.h>
#include <thrust/reduce.h>
//#include <thrust/execution_policy.h>

__host__ void cuAverage_Plaquette(double *hgs, double *hgt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock){
	//	float *hgs_d, *hgt_d;
	int device=-1;
	cudaGetDevice(&device);
	float *hgs_d, *hgt_d;
	//Thrust want things in a weird format for the reduction, thus we oblige
	cudaMallocAsync((void **)&hgs_d,kvol*sizeof(float),NULL);
	thrust::device_ptr<float> hgs_T = thrust::device_pointer_cast(hgs_d);
	cudaMallocAsync((void **)&hgt_d,kvol*sizeof(float),NULL);
	thrust::device_ptr<float> hgt_T = thrust::device_pointer_cast(hgt_d);

	cuAverage_Plaquette<<<dimGrid,dimBlock,0,NULL>>>(hgs_d, hgt_d, u11t, u12t, iu);
	cudaDeviceSynchronise();

	*hgs= (double)thrust::reduce(hgs_T,hgs_T+kvol,(float)0);
	*hgt= (double)thrust::reduce(hgt_T,hgt_T+kvol,(float)0);
	//Temporary holders to keep OMP happy.
	/*
		double hgs_t=0; double hgt_t=0;
#pragma omp parallel for simd reduction(+:hgs_t,hgt_t)
for(int i=0;i<kvol;i++){
hgs_t+=hgs_d[i]; hgt_t+=hgt_d[i];
}
	 *hgs=hgs_t; *hgt=hgt_t;
	 */
	cudaFreeAsync(hgs_d,streams[0]); cudaFreeAsync(hgt_d,streams[1]);
	}
void cuPolyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f *u11t, Complex_f *u12t, dim3 dimGrid, dim3 dimBlock){
	Transpose_c(u11t,ndim,kvol);
	Transpose_c(u12t,ndim,kvol);
	cuPolyakov<<<dimGrid,dimBlock>>>(Sigma11,Sigma12,u11t,u12t);
	Transpose_c(u11t,kvol,ndim);
	Transpose_c(u12t,kvol,ndim);
}
//CUDA Kernels
__global__ void cuAverage_Plaquette(float *hgs_d, float *hgt_d, Complex_f *u11t, Complex_f *u12t, unsigned int *iu){
	const char *funcname = "cuSU2plaq";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	Complex_f Sigma11=0; Complex_f Sigma12=0;
	//TODO: Check if μ and ν loops inside of site loop is faster. I suspect it is due to memory locality.
	for(int i=threadId;i<kvol;i+=bsize*gsize){
		hgt_d[i]=0; hgs_d[i]=0;

		for(int mu=1;mu<ndim;mu++)
			for(int nu=0;nu<mu;nu++){
				//This is threadsafe as the μ and ν loops are not distributed across threads
				cuSU2plaq(u11t,u12t,&Sigma11,&Sigma12,iu,i,mu,nu);
				switch(mu){
					//Time component
					case(ndim-1):
					hgt_d[i] -= creal(Sigma11);
					break;
					//Space component
					default:
					hgs_d[i] -=	creal(Sigma11);
					break;
				}
			}
	}
}
__device__  void cuSU2plaq(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12, unsigned int *iu, int i, int mu, int nu){
	/*
	 * Calculates the plaquette at site i in the μ-ν direction
	 *
	 * Parameters:
	 * ==========
	 * Complex u11t, u12t:	Trial fields
	 * Comples Sigma11, Sigma12: Plaquette components
	 * unsigned int *iu:	Upper halo indices
	 * int mu, nu:				Plaquette direction. Note that mu and nu can be negative
	 * 							to facilitate calculating plaquettes for Clover terms. No
	 * 							sanity checks are conducted on them in this routine.
	 *
	 */
	const char *funcname = "SU2plaq";
	int uidm = iu[i+kvol*mu]; 

	*Sigma11=u11t[i+kvol*mu]*u11t[uidm+kvol*nu]-u12t[i+kvol*mu]*conj(u12t[uidm+kvol*nu]);
	*Sigma12=u11t[i+kvol*mu]*u12t[uidm+kvol*nu]+u12t[i+kvol*mu]*conj(u11t[uidm+kvol*nu]);

	int uidn = iu[i+kvol*nu]; 
	Complex_f a11=*Sigma11*conj(u11t[uidn+kvol*mu])+*Sigma12*conj(u12t[uidn+kvol*mu]);
	Complex_f a12=-*Sigma11*u12t[uidn+kvol*mu]+*Sigma12*u11t[uidn+kvol*mu];

	*Sigma11=a11*conj(u11t[i+kvol*nu])+a12*conj(u12t[i+kvol*nu]);
	*Sigma12=-a11*u12t[i+kvol*nu]+a12*u11t[i+kvol*mu];
}
__global__ void cuPolyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f * u11t,Complex_f *u12t){
	char * funcname = "cuPolyakov";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	//RACE CONDITION? gsize*bsize?
	for(int i=threadId;i<kvol3;i+=gsize*bsize){
		Complex_f Sig[2]; Sig[0]=Sigma11[i]; Sig[1]=Sigma12[i];
		Complex_f u[2];
		for(int it=1;it<ksizet;it++){
			int indexu=it*kvol3+i;
			u[0]=u11t[indexu+3*kvol];u[1]=u12t[indexu+3*kvol];
			Complex_f a11=Sig[0]*u[0]-Sig[1]*conj(u[1]);
			//Instead of having to store a second buffer just assign it directly
			Sig[1]=Sig[0]*u[1]+Sig[1]*conj(u[0]);
			Sig[0]=a11;
		}
		Sigma11[i]=Sig[0]; Sigma12[i]=Sig[1];
	}
}


// __device__ int Leaf(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12,
// 		unsigned int *iu, unsigned int *id, int i, int mu, int nu, short leaf){
// 	/** @brief Evaluates the required clover leaf
// 	 *
// 	 * @param u11t, u12t:			Trial fields
// 	 * @param Sigma11, Sigma12:	Plaquette terms
// 	 * @param iu, id:					Upper/lower halo indices
// 	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
// 	 *										to facilitate calculating plaquettes for Clover terms. No
// 	 *										sanity checks are conducted on them in this routine.
// 	 *	@param i:						Centre of plaquette
// 	 * @param leaf:					Which leaf of the halo are we looking for. Based on the
// 	 * 									signs of μ and ν
// 	 *
// 	 * Calls:
// 	 * ======
// 	 * SU2plaq()
// 	 *
// 	 * @return Zero on success, integer error code otherwise
// 	 */
// 	char *funcname="Leaf";
// 	Complex_f a11,a12;
// 	unsigned int didm,didn,uidn,uidm;
// 	switch(leaf){
// 		case(0):
// 			//Both positive is just a standard plaquette
// 			SU2plaq(u11t,u12t,Sigma11,Sigma12,iu,i,mu,nu);
// 			return 0;
// 		case(1):
// 			//μ<0 and ν>=0
// 			didm = id[mu+ndim*i]; uidn = iu[nu+ndim*i]; 
// 			//U_ν(x)*U_-μ(x+ν)=U_ν(x)*U^† _μ(x-μ+ν)
// 			//Awkward index here unfortunately. Seems safer than trying to find -μ
// 			int uin_didm=id[mu+ndim*uidn];
// 			*Sigma11=u11t[i*ndim+nu]*conj(u11t[uin_didm*ndim+mu])+u12t[i*ndim+nu]*conj(u12t[uin_didm*ndim+mu]);
// 			*Sigma12=-u11t[i*ndim+nu]*conj(u12t[uin_didm*ndim+mu])+u12t[i*ndim+nu]*u11t[uin_didm*ndim+mu];
// 
// 			//(U_ν(x)*U_-μ(x+ν))*U_-ν(x-μ+ν)=(U_ν(x)*U^† _μ(x-μ+ν))*U^†_ν(x-μ)
// 			a11=*Sigma11*conj(u11t[didm*ndim+nu])+*Sigma12*conj(u12t[didm*ndim+nu]);
// 			a12=-*Sigma11*u12t[didm*ndim+nu]+*Sigma12*u11t[didm*ndim+nu];
// 
// 			//((U_ν(x)*U_-μ(x+ν))*U_-ν(x-μ+ν))*U_μ(x-μ)=((U_ν(x)*U^† _μ(x-μ_ν))*U^† _ν(x-μ))*U_μ(x-μ)
// 			*Sigma11=a11*u11t[didm*ndim+mu]-a12*conj(u12t[didm*ndim+mu]);
// 			*Sigma12=a11*u12t[didm*ndim+mu]+a12*conj(u11t[didm*ndim+mu]);
// 			return 0;
// 		case(2):
// 			//μ>=0 and ν<0
// 			//TODO: Figure out down site index
// 			uidm = iu[mu+ndim*i]; didn = id[nu+ndim*i]; 
// 			//U_-ν(x)*U_μ(x-ν)=U^†_ν(x-ν)*U_μ(x-ν)
// 			*Sigma11=conj(u11t[didn*ndim+nu])*u11t[didn*ndim+mu]+conj(u12t[didn*ndim+nu])*u12t[didn*ndim+mu];
// 			*Sigma12=conj(u11t[didn*ndim+nu])*u12t[didn*ndim+mu]-u12t[didn*ndim+mu]*conj(u11t[didn*ndim+nu]);
// 
// 			//(U_-ν(x)*U_μ(x-ν))*U_ν(x+μ-ν)=(U^†_ν(x-ν)*U_μ(x-ν))*U_ν(x+μ-ν)
// 			//Another awkward index
// 			int uim_didn=id[nu+ndim*uidm];
// 			a11=*Sigma11*u11t[uim_didn*ndim+nu]-*Sigma12*conj(u12t[uim_didn*ndim+nu]);
// 			a12=*Sigma11*u12t[uim_didn*ndim+nu]+*Sigma12*conj(u11t[uim_didn*ndim+nu]);
// 
// 			//((U_-ν(x)*U_μ(x-ν))*U_ν(x+μ-ν))*U_-μ(x+μ)=(U^†_ν(x-ν)*U_μ(x-ν))*U_ν(x+μ-ν)
// 			*Sigma11=a11*conj(u11t[i*ndim+mu])+a12*conj(u12t[i*ndim+mu]);
// 			*Sigma12=-a11*u12t[i*ndim+mu]+a12*u11t[i*ndim+mu];
// 			return 0;
// 		case(3):
// 			//μ<0 and ν<0
// 			didm = id[mu+ndim*i]; didn = id[nu+ndim*i]; 
// 			//U_-μ(x)*U_-ν(x-μ)=U^†_μ(x-μ)*U^†_ν(x-μ-ν)
// 			int dim_didn=id[nu+ndim*didm];
// 			*Sigma11=conj(u11t[didm*ndim+mu])*conj(u11t[dim_didn*ndim+nu])+conj(u12t[didm*ndim+mu])*conj(u12t[dim_didn*ndim+nu]);
// 
// 			//(U_-μ(x)*U_-ν(x-μ))*(U_μ(x-μ-ν))
// 			a11=*Sigma11*u11t[dim_didn*ndim+mu]-*Sigma12*conj(u12t[dim_didn*ndim+mu]);
// 			a12=*Sigma11*u12t[dim_didn*ndim+mu]+*Sigma12*conj(u11t[dim_didn*ndim+mu]);
// 
// 			//[(U_-μ(x)*U_-ν(x-μ))*(U_μ(x-μ-ν))]*U_ν(x-ν)
// 			*Sigma11=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
// 			*Sigma12=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+nu]);
// 			return 0;
// 	}
// }
// __device__ int Half_Clover(Complex_f *u11t, Complex_f *u12t, Complex_f *clover11, Complex_f *clover12,
// 		unsigned int *iu, unsigned int *id, int i, int mu, int nu){
// 	/** @brief Calculate one clover leaf \f(Q_{μν}\f), which is half the full clover term
// 	 *
// 	 * @param u11t, u12t:			Trial fields
// 	 * @param clover11, clover12:	Clover fields
// 	 * @param *iu, *id:				Upper/lower halo indices
// 	 *	@param i:						Centre of plaquette
// 	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
// 	 * 									to facilitate calculating plaquettes for Clover terms. No
// 	 * 									sanity checks are conducted on them in this routine.
// 	 *
// 	 * Calls:
// 	 * ======
// 	 * Leaf()
// 	 *
// 	 * @return Zero on success, integer error code otherwise
// 	 */
// 	char *funcname ="Half_Clover";
// //#pragma omp simd reduction(+:*clover11,*clover12)
// 	for(short leaf=0;i<ndim;leaf++)
// 	{
// 		Complex_f Sigma11, Sigma12;
// 		Leaf(u11t,u12t,&Sigma11,&Sigma12,iu,id,i,mu,nu,leaf);
// 		*clover11+=Sigma11; *clover12+=Sigma12;
// 	}
// 	return 0;
// }
// __device__ int Clover(Complex_f *u11t, Complex_f *u12t, Complex_f *clover11, Complex_f *clover12,
// 		unsigned int *iu, unsigned int *id, int i, int mu, int nu){
// 	/** @brief Calculate the clover term in the μ-ν direction
// 	 *	\f$F_{\mu\nu}(n)=\frac{-i}{8a^2}\left(Q_{\mu\nu}(n)-{Q_{\nu\mu}(n)\right)\f$
// 	 *	
// 	 * @param u11t, u12t:			Trial fields
// 	 * @param clover11, clover12:	Clover fields
// 	 * @param *iu, *id:				Upper/lower halo indices
// 	 *	@param i:						Centre of plaquette
// 	 * @param mu, nu:					Plaquette direction. Note that mu and nu can be negative
// 	 * 									to facilitate calculating plaquettes for Clover terms. No
// 	 * 									sanity checks are conducted on them in this routine.
// 	 *
// 	 * Calls:
// 	 * =====
// 	 * Half_Clover()
// 	 *
// 	 * @return Zero on success, integer error code otherwise
// 	 */
// 	char *funcname="Clover";
// 	Half_Clover(u11t,u12t,clover11,clover12,iu,id,i,mu,nu);	
// 	//Hmm, creal(Clover11) drops out then?
// 	*clover11-=conj(*clover11); *clover12-=-(*clover12);
// 	*clover11*=(-I/8.0); *clover12*=(-I/8.0);
// }
