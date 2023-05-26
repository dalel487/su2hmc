/*
 * Code for bosonic observables
 * Basically polyakov loop and Plaquette routines
 */
#include	<par_mpi.h>
#include	<su2hmc.h>
void cuAverage_Plaquette(double *hgs, double *hgt, Complex_f *u11t, Complex_f *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock){
	float *hgs_d, *hgt_d;
	int device=-1;
	cudaGetDevice(&device);
	cudaMallocManaged((void **)&hgs_d,kvol*sizeof(float),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&hgt_d,kvol*sizeof(float),cudaMemAttachGlobal);

	cuAverage_Plaquette<<<dimGrid,dimBlock>>>(hgs_d, hgt_d, u11t, u12t, iu);
	cudaDeviceSynchronise();
	cudaMemPrefetchAsync(hgs_d,kvol*sizeof(float),device,streams[0]);
	cudaMemPrefetchAsync(hgt_d,kvol*sizeof(float),device,streams[1]);
	cudaDeviceSynchronise();
	/*
	*hgs= thrust::reduce(thrust::host,hgs_d,hgt_d+kvol);
	*hgt= thrust::reduce(thrust::host,hgt_d,hgt_d+kvol);
	*/
	//Temporary holders to keep OMP happy.
	double hgs_t=0; double hgt_t=0;
#pragma omp parallel for simd reduction(+:hgs_t,hgt_t)
	for(int i=0;i<kvol;i++){
		hgs_t+=hgs_d[i]; hgt_t+=hgt_d[i];
	}
	*hgs=hgs_t; *hgt=hgt_t;

	cudaFree(hgs_d); cudaFree(hgt_d);
}
void cuPolyakov(Complex *Sigma11, Complex * Sigma12, Complex *u11t, Complex *u12t, dim3 dimGrid, dim3 dimBlock){
	cuPolyakov<<<dimGrid,dimBlock>>>(Sigma11,Sigma12,u11t,u12t);
}
//CUDA Kernels
__global__ void cuAverage_Plaquette(float *hgs_d, float *hgt_d, Complex_f *u11t, Complex_f *u12t, unsigned int *iu){
	char *funcname = "cuSU2plaq";
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
	return Sigma11.real();
}
__global__ void cuPolyakov(Complex *Sigma11, Complex * Sigma12, Complex * u11t,Complex *u12t){
	char * funcname = "cuPolyakov";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int it=1;it<ksizet;it++)
	//RACE CONDITION? gsize*bsize?
		for(int i=threadId;i<kvol3;i+=gsize*bsize){
			int indexu=it*kvol3+i;
			Complex a11=Sigma11[i]*u11t[indexu*ndim+3]-Sigma12[i]*conj(u12t[indexu*ndim+3]);
			//Instead of having to store a second buffer just assign it directly
			Sigma12[i]=Sigma11[i]*u12t[indexu*ndim+3]+Sigma12[i]*conj(u11t[indexu*ndim+3]);
			Sigma11[i]=a11;
		}
}
