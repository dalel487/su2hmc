#include <Complex.h>
#include <cuda.h>
int Cuda_init(){
//From Init()
	cudaMallocManaged(&dk4m,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4p,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
//Also from Init()	
	cudaMallocManaged(&u11,ndim*(kvol+halo)*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12,ndim*(kvol+halo)*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t,ndim*(kvol+halo)*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t,ndim*(kvol+halo)*sizeof(complex),cudaMemAttachGlobal);

//From just before the main loop
	cudaMallocManaged(&R1, kfermHalo*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&xi, kfermHalo*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi, nf*kfermHalo*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X0, nf*kfermHalo*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X1, kferm2Halo*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&dSdpi, kmomHalo*sizeof(complex),cudaMemAttachGlobal);
	cudaMallocManaged(&pp, kmomHalo*sizeof(complex),cudaMemAttachGlobal);
}
