#include <cuda.h>
#include <complex>
#include <par_mpi.h>
#include <su2hmc.h>
__host__ int Cuda_init(){
//From Init()
	cudaMallocManaged(&dk4m,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4p,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
//Also from Init()	
	cudaMallocManaged(&u11,ndim*(kvol+halo)*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&u12,ndim*(kvol+halo)*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t,ndim*(kvol+halo)*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t,ndim*(kvol+halo)*sizeof(complex<double>),cudaMemAttachGlobal);

//From just before the main loop
	cudaMallocManaged(&R1, kfermHalo*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&xi, kfermHalo*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi, nf*kfermHalo*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&X0, nf*kfermHalo*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&X1, kferm2Halo*sizeof(complex<double>),cudaMemAttachGlobal);
	cudaMallocManaged(&pp, kmomHalo*sizeof(complex<double>),cudaMemAttachGlobal);
	return 0;
}
