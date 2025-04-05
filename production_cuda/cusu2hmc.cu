#include	<cuda.h>
#include	<cuda_runtime.h>
#include	<su2hmc.h>
#define MIN(x,y) (x<y?x:y)
#define MAX(x,y) (x>y?x:y)
dim3 dimBlockOne = dim3(1,1,1);
dim3 dimGridOne= dim3(1,1,1);
//Worst case scenario, each block contains 256 threads. This should be tuned later
dim3 dimBlock = dim3(1,1,1);
//dim3 dimBlock = 1;
dim3 dimGrid= dim3(1,1,1);
//dim3	dimBlock=dimBlockOne; dim3 dimGrid=dimGridOne;
cudaStream_t streams[ndirac*ndim*nadj];
void blockInit(int x, int y, int z, int t, dim3 *dimBlock, dim3 *dimGrid){

	const char *funcname = "blockInit";

	int device=-1;	cudaGetDevice(&device);
	cudaDeviceProp prop;	cudaGetDeviceProperties(&prop, device);
	//Threads per block
	int tpb=prop.maxThreadsPerBlock/8;
	//Warp size
	int tpw=prop.warpSize;
	int bx=1;
	//Set bx to be the largest power of 2 less than x that fits in a block
	while(bx<=x/2 && bx<tpb)
		bx*=2;
	int by=1;
	//Set by to be the largest power of 2 less than y such that bx*by fits in a block
	while(by<=y/2 && bx*by<tpb)
		by*=2;

	if(bx*by>=128){
		*dimBlock=dim3(bx,by);
		//If the block size neatly divides the lattice size we can create
		//extra blocks safely
		int res= ((nx*ny)/(bx*by) > 1) ? (nx*ny)/(bx*by) :1;
		//		int res = 1;
		*dimGrid=dim3(nz,nt,res);
	}
	else{
		int bz=1;
		//Set by to be the largest power of 2 less than y such that bx*by fits in an optimal block
		while(bz<z/2 && bx*by*bz<tpb)
			bz*=2;
		*dimBlock=dim3(bx,by,bz);

		//If we have an awkward block size then flag it.
		if(bx*by*bz%tpw!=0)
			fprintf(stderr,"Alert %i in %s: Suboptimal block size for warp size %d. bx=%d by=%d bz=%d\n",
					BLOCKALERT,	funcname, tpw, bx, by,bz);
		int res= ((nx*ny)/(bx*by) > 1) ? (nx*ny)/(bx*by) :1;
		*dimGrid=dim3(z/bz,nt,res);
	}
	printf("Block: (%d,%d,%d)\tGrid: (%d,%d,%d)\n",dimBlock->x,dimBlock->y,dimBlock->z,dimGrid->x,dimGrid->y,dimGrid->z);
}
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
	const char *funcname = "Init_CUDA";
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
	const char *funcname = "cuComplex_convert";
	cuReal_convert<<<dimGrid,dimBlock>>>(a,b,len,dtof);
}
void cuComplex_convert(Complex_f *a, Complex *b, int len, bool dtof, dim3 dimBlock, dim3 dimGrid){
	/* 
	 * Kernel wrapper for conversion between sp and dp complex on the GPU.
	 */
	const char *funcname = "cuComplex_convert";
	cuReal_convert<<<dimGrid,dimBlock>>>((float *)a,(double *)b,2*len,dtof);
}
void cuFill_Small_Phi(int na, Complex *smallPhi, Complex *Phi, dim3 dimBlock, dim3 dimGrid){
	cuFill_Small_Phi<<<dimGrid,dimBlock>>>(na,smallPhi,Phi);
}
void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu,dim3 dimBlock, dim3 dimGrid)
{
	const char *funcname = "cuZ_gather";
	cuC_gather<<<dimGrid,dimBlock>>>(x,y,n,table,mu);
}
void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu,dim3 dimBlock, dim3 dimGrid)
{
	const char *funcname = "cuZ_gather";
	cuZ_gather<<<dimGrid,dimBlock>>>(x,y,n,table,mu);
}
void cuUpDownPart(int na, Complex *X0, Complex *R1,dim3 dimBlock, dim3 dimGrid){
	cuUpDownPart<<<dimGrid,dimBlock>>>(na,X0,R1);	
}
void cuReunitarise(Complex *u11t, Complex *u12t, dim3 dimGrid, dim3 dimBlock){
	cuReunitarise<<<dimGrid,dimBlock>>>(u11t,u12t);
	cudaDeviceSynchronise();
}
void cuGauge_Update(const double d, double *pp, Complex *u11t, Complex *u12t, dim3 dimGrid, dim3 dimBlock){
	for(int mu=0;mu<ndim;mu++)
		cuGauge_Update<<<dimGrid,dimBlock,0,streams[mu]>>>(d,pp,u11t,u12t,mu);
}
//CUDA Kernels
__global__ void cuReal_convert(float *a, double *b, int len, bool dtof){
	const char *funcname = "cuReal_convert";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	//Double to float
	if(dtof)
		for(int i = gthreadId; i<len;i+=gsize*bsize)
			a[i]=(float)b[i];
	//Float to double
	else
		for(int i = gthreadId; i<len;i+=gsize*bsize)
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
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	for(int i = gthreadId; i<kvol;i+=gsize*bsize)
		for(int idirac = 0; idirac<ndirac; idirac++)
			for(int ic= 0; ic<nc; ic++)
				//	  PHI_index=i*16+j*2+k;
				smallPhi[i + kvol * (ic + nc * idirac)] = Phi[i + kvol * (ic + idirac * (nc + ngorkov * na))];
}
__global__ void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu)
{
	const char *funcname = "cuC_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	for(int i = gthreadId; i<kvol;i+=gsize*bsize)
		x[i]=y[table[i+kvol*mu]+kvol*mu];
}
__global__ void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu)
{
	const char *funcname = "cuZ_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	for(int i = gthreadId; i<kvol;i+=gsize*bsize)
		x[i]=y[table[i+kvol*mu]+kvol*mu];
}
__global__ void cuUpDownPart(int na, Complex *X0, Complex *R1){

	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	//Up/down partitioning (using only pseudofermions of flavour 1)
	for(int i = gthreadId; i<kvol;i+=gsize*bsize)
		for(int idirac = 0; idirac < ndirac; idirac++){
			X0[i + kvol * (0 + nc * (idirac + ndirac * na))] = R1[i + kvol * (0 + nc * idirac)];
			X0[i + kvol * (1 + nc * (idirac + ndirac * na))] = R1[i + kvol * (1 + nc * idirac)];
		}
}

__global__ void cuReunitarise(Complex *u11t, Complex * u12t){
	/*
	 * Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * Globals:
	 * =======
	 * u11t, u12t
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Reunitarise";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	for(int i=gthreadId; i<kvol*ndim; i+=gsize*bsize){
		//Declaring anorm inside the loop will hopefully let the compiler know it
		//is safe to vectorise aggessively
		double anorm=sqrt(conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]).real();
		//		Exception handling code. May be faster to leave out as the exit prevents vectorisation.
		//		if(anorm==0){
		//			fprintf(stderr, "Error %i in %s on rank %i: anorm = 0 for μ=%i and i=%i.\nExiting...\n\n",
		//					DIVZERO, funcname, rank, mu, i);
		//			MPI_Finalise();
		//			exit(DIVZERO);
		//		}
		u11t[i]/=anorm;
		u12t[i]/=anorm;
	}
}
__global__ void cuGauge_Update(const double d, double *pp, Complex *u11t, Complex *u12t,int mu){
	char *funcname = "Gauge_Update";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Sticking to what was in the FORTRAN for variable names.
		//CCC for cosine SSS for sine AAA for...
		//Re-exponentiating the force field. Can be done analytically in SU(2)
		//using sine and cosine which is nice
		double AAA = d*sqrt(pp[i+kvol*(mu)]*pp[i+kvol*(mu)]\
				+pp[i+kvol*(1*ndim+mu)]*pp[i+kvol*(1*ndim+mu)]\
				+pp[i+kvol*(2*ndim+mu)]*pp[i+kvol*(2*ndim+mu)]);
		double CCC = cos(AAA);
		double SSS = d*sin(AAA)/AAA;
		Complex a11 = CCC+I*SSS*pp[i+kvol*(2*ndim+mu)];
		Complex a12 = pp[i+kvol*(1*ndim+mu)]*SSS + I*SSS*pp[i+kvol*(mu)];
		//b11 and b12 are u11t and u12t terms, so we'll use u12t directly
		//but use b11 for u11t to prevent RAW dependency
		Complex b11 = u11t[i+kvol*mu];
		u11t[i+kvol*mu] = a11*b11-a12*conj(u12t[i+kvol*mu]);
		u12t[i+kvol*mu] = a11*u12t[i+kvol*mu]+a12*conj(b11);
	}
}
