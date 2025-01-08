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
	cudaMallocAsync((void **)&hgs_d,kvol*sizeof(float),streams[0]);
	thrust::device_ptr<float> hgs_T = thrust::device_pointer_cast(hgs_d);
	cudaMallocAsync((void **)&hgt_d,kvol*sizeof(float),streams[1]);
	thrust::device_ptr<float> hgt_T = thrust::device_pointer_cast(hgt_d);

	cuAverage_Plaquette<<<dimGrid,dimBlock>>>(hgs_d, hgt_d, u11t, u12t, iu);

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
