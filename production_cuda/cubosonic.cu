/**
 * @file 
 * @brief Code for bosonic observables. Basically polyakov loop and Plaquette routines
 *
 * @author D. Lawlor
 */
#include	<su2hmc.h>
#include <matrices.h>
#include <thrust/reduce.h>
//#include <thrust/execution_policy.h>

//CUDA Device code
/**
 * @brief	Calculates the SU2 plaquette
 *
 * @param	u11t, u12t:			Gauge fields
 * @param	Sigma11,Sigma12:	Plaquette entries
 * @param	iu:					Site indices in the up direction
 * @param	i:						Site
 * @param	mu,nu:				Plaquette direction
 */
__device__  void cuSU2plaq(Complex_f *u11t, Complex_f *u12t, Complex_f *Sigma11, Complex_f *Sigma12, unsigned int *iu,\
									const unsigned int i, const unsigned short mu, const unsigned short nu){
	const unsigned int uidm = iu[i+kvol*mu]; 
	unsigned int ind=i+kvol*mu;
	//Need a second index in the nu direction for the first step
	unsigned int indn=uidm+kvol*nu;
	*Sigma11=u11t[ind]*u11t[indn]-u12t[ind]*conj(u12t[indn]);
	*Sigma12=u11t[ind]*u12t[indn]+u12t[ind]*conj(u11t[indn]);

	const int uidn = iu[i+kvol*nu]; 
	ind=uidn+kvol*mu;
	Complex_f a11=*Sigma11*conj(u11t[ind])+*Sigma12*conj(u12t[ind]);
	Complex_f a12=-*Sigma11*u12t[ind]+*Sigma12*u11t[ind];

	ind=i+kvol*nu;
	*Sigma11=a11*conj(u11t[ind])+a12*conj(u12t[ind]);
	*Sigma12=-a11*u12t[ind]+a12*u11t[ind];
	return;
}
//CUDA Kernels
	/** 
	 * @brief	Calculates the gauge action using new (how new?) lookup table
	 * @brief	Follows a routine called qedplaq in some QED3 code
	 *
	 * @param	hgs_d,hgt_d		Gauge component of Hamilton
	 * @param	u11t,u12t		Gauge fields
	 * @param	iu					Upper halo indices
	 *
	 */
__global__ void Average_Plaquette(float *hgs_d, float *hgt_d, Complex_f *u11t, Complex_f *u12t, unsigned int *iu){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	Complex_f Sigma11=0; Complex_f Sigma12=0;
	//TODO: Check if μ and ν loops inside of site loop is faster. I suspect it is due to memory locality.
	for(unsigned int i=threadId;i<kvol;i+=bsize*gsize){
		float hg_c[2];
		hg_c[0]=0; hg_c[1]=0;

		for(unsigned short mu=1;mu<ndim;mu++)
			for(unsigned short nu=0;nu<mu;nu++){
				//This is threadsafe as the μ and ν loops are not distributed across threads
				cuSU2plaq(u11t,u12t,&Sigma11,&Sigma12,iu,i,mu,nu);
				switch(mu){
					//Time component
					case(ndim-1):
					hg_c[0] -= creal(Sigma11);
					break;
					//Space component
					default:
					hg_c[1] -=	creal(Sigma11);
					break;
				}
			}
			hgt_d[i]=hg_c[0]; hgs_d[i]=hg_c[1];
	}
}

	/**
	 * @brief Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * @param	Sigma11,Sigma12	Components of the Polyakov loop
	 * @param	u11t,u12t:	The gauge fields
	 * 
	 */
__global__ void Polyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f * u11t,Complex_f *u12t){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(unsigned int i=threadId;i<kvol3;i+=gsize*bsize){
		Complex_f Sig[2]; Sig[0]=Sigma11[i]; Sig[1]=Sigma12[i];
		Complex_f u[2];
		for(unsigned int it=1;it<ksizet;it++){
			const unsigned int indexu=it*kvol3+i;
			u[0]=u11t[indexu+3*kvol];u[1]=u12t[indexu+3*kvol];
			Complex_f a11=Sig[0]*u[0]-Sig[1]*conj(u[1]);
			//Instead of having to store a second buffer just assign it directly
			Sig[1]=Sig[0]*u[1]+Sig[1]*conj(u[0]);
			Sig[0]=a11;
		}
		Sigma11[i]=Sig[0]; Sigma12[i]=Sig[1];
	}
}

//Calling wrappers

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

	Average_Plaquette<<<dimGrid,dimBlock,0,NULL>>>(hgs_d, hgt_d, u11t, u12t, iu);
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
void cuPolyakov(Complex_f *Sigma[2], Complex_f *ut[2], dim3 dimGrid, dim3 dimBlock){
	int device=-1;
	cudaGetDevice(&device);
	cudaMallocManaged((void **)&Sigma[0],kvol3*sizeof(Complex_f),cudaMemAttachGlobal);
#ifdef _DEBUG
	cudaMallocManaged((void **)&Sigma[1],kvol3*sizeof(Complex_f),cudaMemAttachGlobal);
#else
	cudaMallocAsync((void **)&Sigma[1],kvol3*sizeof(Complex_f),streams[0]);
#endif
	//Extract the time component from each site and save in corresponding Sigma
	cublasCcopy(cublas_handle,kvol3, (cuComplex *)(ut[0])+3*kvol, 1, (cuComplex *)Sigma[0], 1);
	cublasCcopy(cublas_handle,kvol3, (cuComplex *)(ut[1])+3*kvol, 1, (cuComplex *)Sigma[1], 1);

	cudaDeviceSynchronise();
	Polyakov<<<dimGrid,dimBlock>>>(Sigma[0],Sigma[1],ut[0],ut[1]);
	//cudaMemPrefetchAsync(Sigma[0],kvol3*sizeof(Complex_f),cudaCpuDeviceId,streams[0]);
#ifdef _DEBUG
	cudaFree(Sigma[1]);
#else
	cudaFreeAsync(Sigma[1],streams[1]);
#endif
	cudaDeviceSynchronise();
}
