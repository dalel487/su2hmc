/*
 * Code for bosonic observables
 * Basically polyakov loop and Plaquette routines
 */
#include	<par_mpi.h>
#include	<su2hmc.h>
void SU2plaq(double *hgs, double *hgt, Complex *u11t, Complex *u12t, unsigned int *iu,dim3 dimGrid, dim3 dimBlock){
	cuSU2plaq<<<dimGrid,dimBlock>>>(hgs, hgt, u11t, u12t, iu);
	}
void Polyakov(Complex *Sigma11, Complex * Sigma12, Complex *u11t, Complex *u12t, dim3 dimGrid, dim3 dimBlock){
	cuPolyakov<<<dimGrid,dimBlock>>>(Sigma11,Sigma12,u11t,u12t);
}
//CUDA Kernels
__global__ void cuSU2plaq(double *hgs, double *hgt, Complex *u11t, Complex *u12t, unsigned int *iu){
	char *funcname = "cuSU2plaq";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int mu=1;mu<ndim;mu++)
		for(int nu=0;nu<mu;nu++)
			for(int i=threadId;i<kvol;i+=gsize){
				//Save us from typing iu[mu+ndim*i] everywhere
				int uidm = iu[mu+ndim*i]; 

				Complex Sigma11=u11t[i*ndim+mu]*u11t[uidm*ndim+nu]-u12t[i*ndim+mu]*conj(u12t[uidm*ndim+nu]);
				Complex Sigma12=u11t[i*ndim+mu]*u12t[uidm*ndim+nu]+u12t[i*ndim+mu]*conj(u11t[uidm*ndim+nu]);

				int uidn = iu[nu+ndim*i]; 
				Complex a11=Sigma11*conj(u11t[uidn*ndim+mu])+Sigma12*conj(u12t[uidn*ndim+mu]);
				Complex a12=-Sigma11*u12t[uidn*ndim+mu]+Sigma12*u11t[uidn*ndim+mu];

				Sigma11=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
				//				Sigma12[i]=-a11[i]*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
				//				Not needed in final result as it traces out

				switch(mu){
					//Time component
					case(ndim-1):	atomicAdd(hgt, -Sigma11.real());
										break;
										//Space component
					default:	atomicAdd(hgs, -Sigma11.real());
								break;
				}
			}
}
__global__ void cuPolyakov(Complex *Sigma11, Complex * Sigma12, Complex * u11t,Complex *u12t){
	char * funcname = "cuPolyakov";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
#pragma unroll
	for(int it=1;it<ksizet;it++)
		for(int i=threadId;i<kvol3;i+=gsize){
			//Seems a bit more efficient to increment indexu instead of reassigning
			//it every single loop
			int indexu=it*kvol3+i;
			Complex a11=Sigma11[i]*u11t[indexu*ndim+3]-Sigma12[i]*conj(u12t[indexu*ndim+3]);
			//Instead of having to store a second buffer just assign it directly
			Sigma12[i]=Sigma11[i]*u12t[indexu*ndim+3]+Sigma12[i]*conj(u11t[indexu*ndim+3]);
			Sigma11[i]=a11;
		}
}
