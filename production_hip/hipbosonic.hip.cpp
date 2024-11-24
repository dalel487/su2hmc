#include "hip/hip_runtime.h"
/*
 * Code for bosonic observables
 * Basically polyakov loop and Plaquette routines
 */
#include	<par_mpi.h>
#include	<su2hmc.h>
#include <thrust/reduce.h>
//#include <thrust/execution_policy.h>

///Reduction routines from Nvidia guide https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
void cuPolyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f *u11t, Complex_f *u12t, dim3 dimGrid, dim3 dimBlock){
	cuPolyakov<<<dimGrid,dimBlock>>>(Sigma11,Sigma12,u11t,u12t);
}
__host__ void cuAverage_Plaquette(double *hgs, double *hgt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock){
	//	float *hgs_d, *hgt_d;
	int device=-1;
	hipGetDevice(&device);
	float *hgs_d, *hgt_d;
	//Thrust want things in a weird format for the reduction, thus we oblige
	hipMallocAsync((void **)&hgs_d,kvol*sizeof(float),streams[0]);
	hipMallocAsync((void **)&hgt_d,kvol*sizeof(float),streams[1]);
	cudaDeviceSynchronise();
	cuAverage_Plaquette<<<dimGrid,dimBlock>>>(hgs_d, hgt_d, u11t, u12t, iu);
	//	*hgs= (double)thrust::reduce(hgs_T,hgs_T+kvol,(float)0);
	//	*hgt= (double)thrust::reduce(hgt_T,hgt_T+kvol,(float)0);

	cudaDeviceSynchronise();
	float tmp1=0; float tmp2=0;
	unsigned int bsize = dimBlock.x*dimBlock.y*dimBlock.z;
	reduce6<<<dimGrid,dimBlock,bsize*sizeof(float),streams[0]>>>(hgs_d,&tmp1,kvol);
	reduce6<<<dimGrid,dimBlock,bsize*sizeof(float),streams[1]>>>(hgt_d,&tmp2,kvol);
	cudaDeviceSynchronise();
	*hgs=tmp1; *hgt=tmp2;
	//Temporary holders to keep OMP happy.
	/*
		double hgs_t=0; double hgt_t=0;
#pragma omp parallel for simd reduction(+:hgs_t,hgt_t)
for(int i=0;i<kvol;i++){
hgs_t+=hgs_d[i]; hgt_t+=hgt_d[i];
}
	 *hgs=hgs_t; *hgt=hgt_t;
	 */
	hipFreeAsync(hgs_d,streams[0]); hipFreeAsync(hgt_d,streams[1]);
	}
//CUDA Kernels
__global__ void cuPolyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f * u11t,Complex_f *u12t){
	char * funcname = "cuPolyakov";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int it=1;it<ksizet;it++)
		//RACE CONDITION? gsize*bsize?
		for(int i=threadId;i<kvol3;i+=gsize*bsize){
			int indexu=it*kvol3+i;
			Complex_f a11=Sigma11[i]*u11t[indexu*ndim+3]-Sigma12[i]*conj(u12t[indexu*ndim+3]);
			//Instead of having to store a second buffer just assign it directly
			Sigma12[i]=Sigma11[i]*u12t[indexu*ndim+3]+Sigma12[i]*conj(u11t[indexu*ndim+3]);
			Sigma11[i]=a11;
		}
}

template <typename T>
__device__ void warpReduce(volatile T *sdata, const unsigned int tid, const unsigned int blockSize ) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <typename T>
__global__ void reduce6(T *g_idata, T *g_odata, const unsigned int n) {
	extern __shared__ T sdata[];
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	unsigned int i = blockId*(bsize*2) + threadId;
	sdata[threadId] = 0;
	while (i < n) { sdata[threadId] += g_idata[i] + g_idata[i+bsize]; i += gsize; }
	__syncthreads();
	if (bsize >= 512) { if (threadId < 256) { sdata[threadId] += sdata[threadId + 256]; } __syncthreads(); }
	if (bsize >= 256) { if (threadId < 128) { sdata[threadId] += sdata[threadId + 128]; } __syncthreads(); }
	if (bsize >= 128) { if (threadId < 64) { sdata[threadId] += sdata[threadId + 64]; } __syncthreads(); }
	if (threadId < 32) warpReduce(sdata, threadId, bsize);
	if (threadId == 0) *g_odata = sdata[0];
}

__global__ void cuAverage_Plaquette(float *hgs_d, float *hgt_d, Complex_f *u11t, Complex_f *u12t, unsigned int *iu){
	const char *funcname = "cuSU2plaq";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	//TODO: Check if μ and ν loops inside of site loop is faster. I suspect it is due to memory locality.
	for(int i=threadId;i<kvol;i+=bsize*gsize){
		hgt_d[i]=0; hgs_d[i]=0;

		for(int mu=1;mu<ndim;mu++)
			for(int nu=0;nu<mu;nu++){
				//This is threadsafe as the μ and ν loops are not distributed across threads
				switch(mu){
					//Time component
					case(ndim-1):
						hgt_d[i] -= SU2plaq(u11t,u12t,iu,i,mu,nu);
						break;
						//Space component
					default:
						hgs_d[i] -=	SU2plaq(u11t,u12t,iu,i,mu,nu);
						break;
				}
			}
	}
}
__device__ float SU2plaq(Complex_f *u11t, Complex_f *u12t, unsigned int *iu, int i, int mu, int nu){
	/*
	 * Calculates the plaquette at site i in the μ-ν direction
	 *
	 * Parameters:
	 * ==========
	 * Complex u11t, u12t:	Trial fields
	 * unsignedi int *iu:	Upper halo indices
	 * int mu, nu:				Plaquette direction. Note that mu and nu can be negative
	 * 							to facilitate calculating plaquettes for Clover terms. No
	 * 							sanity checks are conducted on them in this routine.
	 *
	 * Returns:
	 * ========
	 * double corresponding to the plaquette value
	 *
	 */
	const char *funcname = "SU2plaq";
	int uidm = iu[mu+ndim*i]; 

	Complex_f Sigma11=u11t[i*ndim+mu]*u11t[uidm*ndim+nu]-u12t[i*ndim+mu]*conj(u12t[uidm*ndim+nu]);
	Complex_f Sigma12=u11t[i*ndim+mu]*u12t[uidm*ndim+nu]+u12t[i*ndim+mu]*conj(u11t[uidm*ndim+nu]);

	int uidn = iu[nu+ndim*i]; 
	Complex_f a11=Sigma11*conj(u11t[uidn*ndim+mu])+Sigma12*conj(u12t[uidn*ndim+mu]);
	Complex_f a12=-Sigma11*u12t[uidn*ndim+mu]+Sigma12*u11t[uidn*ndim+mu];

	Sigma11=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
	//Not needed in final result as it traces out
	//Sigma12[i]=-a11[i]*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
	return creal(Sigma11);
}
template __device__ void warpReduce<float>(volatile float *sdata, const unsigned int tid, const unsigned int blockSize ) ;
template __global__ void reduce6<float>(float *g_idata, float *g_odata, const unsigned int n) ;
