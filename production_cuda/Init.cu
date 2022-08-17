#include	<cuda.h>
#include	<cuda_runtime.h>
#include	<su2hmc.h>
void	Init(Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, Complex gamval[5][4],\
		Complex_f gamval_f[5][4], int gamin[4][4],Complex *gamval_d, Complex_f *gamval_f_d, int *gamin_d,\
		double *dk4m, double *dk4p, float *dk4m_f, float *dk4p_f, unsigned int *iu, unsigned int *id,\
		dim3 *dimBlock, dim3 *dimGrid){
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
	//Set iu and id to mainly read in CUDA and prefetch them to the GPU
	int device=-1;
	cudaGetDevice(&device);
	cudaMemAdvise(iu,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(id,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemPrefetchAsync(iu,ndim*kvol*sizeof(int),device,NULL);
	cudaMemPrefetchAsync(id,ndim*kvol*sizeof(int),device,NULL);

	//Gamma matrices and indices on the GPU
	cudaMemcpy(gamin_d,gamin,4*4*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(gamval_d,gamval,5*4*sizeof(Complex),cudaMemcpyHostToDevice);
	cudaMemcpy(gamval_f_d,gamval_f,5*4*sizeof(Complex_f),cudaMemcpyHostToDevice);

	//More prefetching and marking as read-only (mostly)
	//Prefetching Momentum Fields and Trial Fields to GPU
	cudaMemAdvise(dk4p,(kvol+halo)*sizeof(double),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(dk4m,(kvol+halo)*sizeof(double),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(dk4p_f,(kvol+halo)*sizeof(float),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(dk4m_f,(kvol+halo)*sizeof(float),cudaMemAdviseSetReadMostly,device);

	cudaMemPrefetchAsync(dk4p,(kvol+halo)*sizeof(double),device,NULL);
	cudaMemPrefetchAsync(dk4m,(kvol+halo)*sizeof(double),device,NULL);
	cudaMemPrefetchAsync(dk4p_f,(kvol+halo)*sizeof(float),device,NULL);
	cudaMemPrefetchAsync(dk4m_f,(kvol+halo)*sizeof(float),device,NULL);

	cudaMemPrefetchAsync(u11t, ndim*kvol*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(u12t, ndim*kvol*sizeof(Complex),device,NULL);
	cudaMemPrefetchAsync(u11t_f, ndim*kvol*sizeof(Complex_f),device,NULL);
	cudaMemPrefetchAsync(u12t_f, ndim*kvol*sizeof(Complex_f),device,NULL);

	*dimBlock = dim3(ksizez,ksizet,1);
	*dimGrid= dim3(ksizex,ksizey,1);
}
