#include	<cuda.h>
#include	<cuda_runtime.h>
#include	<su2hmc.h>
#define MIN(x,y) (x<y?x:y)
#define MAX(x,y) (x>y?x:y)
//Worst case scenario, each block contains 256 threads. This should be tuned later
dim3 dimBlock = dim3(MAX(8*(ksizez/8),8),MAX(16*(ksizey/16),16));
//dim3 dimBlock = 1;
dim3 dimGrid= dim3(ksizex,ksizet);
//dim3 dimGrid= 1;
dim3 dimBlockOne = dim3(1,1,1);
dim3 dimGridOne= dim3(1,1,1);
cudaStream_t streams[ndirac*ndim*nadj];
void	Init_CUDA(Complex *u11t, Complex *u12t,Complex *gamval, Complex_f *gamval_f, int *gamin, double*dk4m,\
					double *dk4p, unsigned int *iu, unsigned int *id){
	/*
	 * Initialises the GPU Components of the system
	 *
	 * Calls:
	 * ======
	 * Addrc, Check_addr, ran2, DHalo_swap_dir, Par_sread, Par_ranset, Reunitarise
	 *
	 * Globals:
	 * =======
	 * Complex gamval:		Gamma Matrices
	 * Complex_f gamval_f:	Float Gamma matrices:
	 *
	 * Parameters:
	 * ==========
	 * int istart:				Zero for cold, >1 for hot, <1 for none
	 * int ibound:				Periodic boundary conditions
	 * int iread:				Read configuration from file
	 * float beta:				beta
	 * float fmu:				Chemical potential
	 * float akappa:			
	 * Complex_f ajq:			Diquark source
	 * Complex *u11:			First colour field
	 * Complex *u12:			Second colour field
	 * Complex *u11t:			First colour trial field
	 * Complex *u11t:			Second colour trial field
	 * Complex_f *u11t_f:	First float trial field
	 * Complex_f *u12t_f:	Second float trial field
	 * double	*dk4m:
	 * double	*dk4p:
	 * float		*dk4m_f:
	 * float		*dk4p_f:
	 * unsigned int *iu:		Up halo indices
	 * unsigned int *id:		Down halo indices
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Init_CUDA";
	int device=-1;
	cudaGetDevice(&device);
	//Initialise streams for concurrent kernels
	for(int i=0;i<(ndirac*nadj*ndim);i++)
		cudaStreamCreate(&streams[i]);
	//Set iu and id to mainly read in CUDA and prefetch them to the GPU
	cudaMemPrefetchAsync(iu,ndim*kvol*sizeof(int),device,streams[0]);
	cudaMemPrefetchAsync(id,ndim*kvol*sizeof(int),device,streams[1]);
	cudaMemAdvise(iu,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(id,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);

	//Gamma matrices and indices on the GPU
	//	cudaMemcpy(gamin_d,gamin,4*4*sizeof(int),cudaMemcpyHostToDevice);
	//	cudaMemcpy(gamval_d,gamval,5*4*sizeof(Complex),cudaMemcpyHostToDevice);
	//	cudaMemcpy(gamval_f_d,gamval_f,5*4*sizeof(Complex_f),cudaMemcpyHostToDevice);
	cudaMemAdvise(gamin,4*4*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(gamval,5*4*sizeof(Complex),cudaMemAdviseSetReadMostly,device);

	//More prefetching and marking as read-only (mostly)
	//Prefetching Momentum Fields and Trial Fields to GPU
	cudaMemAdvise(dk4p,(kvol+halo)*sizeof(double),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(dk4m,(kvol+halo)*sizeof(double),cudaMemAdviseSetReadMostly,device);

	cudaMemPrefetchAsync(dk4p,(kvol+halo)*sizeof(double),device,streams[2]);
	cudaMemPrefetchAsync(dk4m,(kvol+halo)*sizeof(double),device,streams[3]);

	cudaMemPrefetchAsync(u11t, ndim*kvol*sizeof(Complex),device,streams[4]);
	cudaMemPrefetchAsync(u12t, ndim*kvol*sizeof(Complex),device,streams[5]);
}
void cuReal_convert(float *a, double *b, int len, bool dtof, dim3 dimBlock, dim3 dimGrid){
	/* 
	 * Kernel wrapper for conversion between sp and dp complex on the GPU.
	 */
	char *funcname = "cuComplex_convert";
	cuReal_convert<<<dimBlock,dimGrid>>>(a,b,len,dtof);
}
void cuComplex_convert(Complex_f *a, Complex *b, int len, bool dtof, dim3 dimBlock, dim3 dimGrid){
	/* 
	 * Kernel wrapper for conversion between sp and dp complex on the GPU.
	 */
	char *funcname = "cuComplex_convert";
	cuReal_convert<<<dimBlock,dimGrid>>>((float *)a,(double *)b,2*len,dtof);
}
void cuFill_Small_Phi(int na, Complex *smallPhi, Complex *Phi, dim3 dimBlock, dim3 dimGrid){
	cuFill_Small_Phi<<<dimBlock,dimGrid>>>(na,smallPhi,Phi);
}
void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu,dim3 dimBlock, dim3 dimGrid)
{
	char *funcname = "cuZ_gather";
	cuC_gather<<<dimBlock,dimGrid>>>(x,y,n,table,mu);
}
void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu,dim3 dimBlock, dim3 dimGrid)
{
	char *funcname = "cuZ_gather";
	cuZ_gather<<<dimBlock,dimGrid>>>(x,y,n,table,mu);
}
void cuUpDownPart(int na, Complex *X0, Complex *R1,dim3 dimBlock, dim3 dimGrid){
	cuUpDownPart<<<dimBlock,dimGrid>>>(na,X0,R1);	
}
//CUDA Kernels
__global__ void cuReal_convert(float *a, double *b, int len, bool dtof){
	char *funcname = "cuReal_convert";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	//Double to float
	if(dtof)
		for(int i = threadId; i<len;i+=gsize*bsize)
			a[i]=(float)b[i];
	//Float to double
	else
		for(int i = threadId; i<len;i+=gsize*bsize)
			b[i]=(double)a[i];
}
__global__ void cuFill_Small_Phi(int na, Complex *smallPhi, Complex *Phi)
{
	/*Copies necessary (2*4*kvol) elements of Phi into a vector variable
	 *
	 * Globals:
	 * =======
	 * Phi:	  The source array
	 * 
	 * Parameters:
	 * ==========
	 * int na: flavour index
	 * Complex *smallPhi:	  The target array
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "cuFill_Small_Phi";
	//BIG and small phi index
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i = threadId; i<kvol;i+=gsize*bsize)
		for(int idirac = 0; idirac<ndirac; idirac++)
			for(int ic= 0; ic<nc; ic++)
				//	  PHI_index=i*16+j*2+k;
				smallPhi[(i*ndirac+idirac)*nc+ic]=Phi[((na*kvol+i)*ngorkov+idirac)*nc+ic];
}
__global__ void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu)
{
	char *funcname = "cuZ_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i = threadId; i<n;i+=gsize*bsize)
		x[i]=y[table[i*ndim+mu]*ndim+mu];
}
__global__ void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu)
{
	char *funcname = "cuZ_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i = threadId; i<n;i+=gsize*bsize)
		x[i]=y[table[i*ndim+mu]*ndim+mu];
}
__global__ void cuUpDownPart(int na, Complex *X0, Complex *R1){

	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	//Up/down partitioning (using only pseudofermions of flavour 1)
	for(int i = threadId; i<kvol;i+=gsize*bsize)
		for(int idirac = 0; idirac < ndirac; idirac++){
			X0[((na*kvol+i)*ndirac+idirac)*nc]=R1[(i*ngorkov+idirac)*nc];
			X0[((na*kvol+i)*ndirac+idirac)*nc+1]=R1[(i*ngorkov+idirac)*nc+1];
		}
}
