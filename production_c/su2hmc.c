#include	<assert.h>
#include	<coord.h>
#ifdef	__NVCC__
#include	<cuda.h>
#include	<cuda_runtime.h>
//Fix this later
#endif
#include	<math.h>
#include	<matrices.h>
#include	<par_mpi.h>
#include	<random.h>
#include	<string.h>
#include	<su2hmc.h>

int Init(int istart, int ibound, int iread, float beta, float fmu, float akappa, Complex_f ajq,\
		Complex *u11, Complex *u12, Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
		Complex gamval[5][4], Complex_f gamval_f[5][4], int gamin[4][4], double *dk4m, double *dk4p, float *dk4m_f, float *dk4p_f,\
		unsigned int *iu, unsigned int *id){
	/*
	 * Initialises the system
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
	const char *funcname = "Init";

#ifdef _OPENMP
	omp_set_num_threads(nthreads);
	omp_get_default_device();
	//Comment out to keep the threads spinning even when there's no work to do
	//Commenting out decrease runtime but increases total CPU time dramatically
	//This can throw of some profilers
	//kmp_set_defaults("KMP_BLOCKTIME=0");
#ifdef __INTEL_MKL__
	mkl_set_num_threads(nthreads);
#endif
#endif
	//First things first, calculate a few constants for coordinates
	Addrc(iu, id);
	//And confirm they're legit
	Check_addr(iu, ksize, ksizet, 0, kvol+halo);
	Check_addr(id, ksize, ksizet, 0, kvol+halo);
#ifdef _OPENACC
#pragma acc enter data copyin(iu[0:ndim*kvol],id[0:ndim*kvol])
#else
#pragma omp target enter data map(to:iu[0:ndim*kvol],id[0:ndim*kvol]) nowait
#endif
#ifdef _DEBUG
	printf("Checked addresses\n");
#endif
	double chem1=exp(fmu); double chem2 = 1/chem1;
	//CUDA this. Only limit will be the bus speed
#pragma omp parallel for simd aligned(dk4m,dk4p:AVX)
	for(int i = 0; i<kvol; i++){
		dk4p[i]=akappa*chem1;
		dk4m[i]=akappa*chem2;
	}
	//Anti periodic Boundary Conditions. Flip the terms at the edge of the time
	//direction
	if(ibound == -1 && pcoord[3+ndim*rank]==npt -1){
#ifdef _DEBUG
		printf("Implimenting antiperiodic boundary conditions on rank %i\n", rank);
#endif
		//Also CUDA this? By the looks of it it should saturate the GPU as is
#pragma omp parallel for simd aligned(dk4m,dk4p:AVX)
		for(int i= 0; i<kvol3; i++){
			int k = kvol - kvol3 + i;
			dk4p[k]*=-1;
			dk4m[k]*=-1;
		}
	}
	//These are constant so swap the halos when initialising and be done with it
	//May need to add a synchronisation statement here first
#if(npt>1)
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);
#endif
	//Float versions
#pragma omp parallel for simd aligned(dk4m,dk4p,dk4m_f,dk4p_f:AVX)
	for(int i=0;i<kvol+halo;i++){
		dk4p_f[i]=(float)dk4p[i];
		dk4m_f[i]=(float)dk4m[i];
	}
#ifdef _OPENACC
#pragma acc data copyin(dk4p[0:kvol+halo], dk4m_f[0:kvol+halo],\
		dk4p_f[0:kvol+halo],dk4m[0:kvol+halo])
#else
#pragma omp target enter data map(to:dk4p[0:kvol+halo], dk4m_f[0:kvol+halo],\
		dk4p_f[0:kvol+halo],dk4m[0:kvol+halo]) nowait
#endif

	//Each gamma matrix is rescaled by akappa by flattening the gamval array
#if defined USE_BLAS
	//Don't cuBLAS this. It is small and won't saturate the GPU. Let the CPU handle
	//it and just copy it later
	cblas_zdscal(5*4, akappa, gamval, 1);
#else
	for(int i=0;i<5;i++)
		for(int j=0;j<4;j++)
			gamval[i][j]*=akappa;
#endif
#pragma omp parallel for simd collapse(2) aligned(gamval,gamval_f:AVX)
	for(int i=0;i<5;i++)
		for(int j=0;j<4;j++)
			gamval_f[i][j]=(Complex_f)gamval[i][j];
#ifdef _OPENACC
#pragma acc enter data copyin(gamval[0:5][0:4], gamval_f[0:5][0:4], gamin[0:4][0:4])
#else
#pragma omp target enter data map(to:gamval[0:5][0:4], gamval_f[0:5][0:4], gamin[0:4][0:4]) nowait
#endif
	if(iread){
		if(!rank) printf("Calling Par_sread() for configuration: %i\n", iread);
		Par_sread(iread, beta, fmu, akappa, ajq,u11,u12,u11t,u12t);
		Par_ranset(&seed,iread);
	}
	else{
		Par_ranset(&seed,iread);
#ifdef _OPENACC
#pragma acc enter data create(u11t[0:ndim*(kvol+halo)],u12t[0:ndim*(kvol+halo)],\
		u11t_f[0:ndim*(kvol+halo)],u12t_f[0:ndim*(kvol+halo)])
#else
#pragma omp target enter data map(alloc:u11t[0:ndim*(kvol+halo)],u12t[0:ndim*(kvol+halo)],\
		u11t_f[0:ndim*(kvol+halo)],u12t_f[0:ndim*(kvol+halo)]) nowait
#endif
		if(istart==0){
			//Initialise a cold start to zero
			//memset is safe to use here because zero is zero 
#pragma omp parallel for simd aligned(u11t:AVX) 
			//Leave it to the GPU?
			for(int i=0; i<kvol*ndim;i++){
				u11t[i]=1;	u12t[i]=0;
			}
		}
		else if(istart>0){
			//Ideally, we can use gsl_ranlux as the PRNG
#ifdef __RANLUX__
			for(int i=0; i<kvol*ndim;i++){
				u11t[i]=2*(gsl_rng_uniform(ranlux_instd)-0.5+I*(gsl_rng_uniform(ranlux_instd)-0.5));
				u12t[i]=2*(gsl_rng_uniform(ranlux_instd)-0.5+I*(gsl_rng_uniform(ranlux_instd)-0.5));
			}
			//If not, the Intel Vectorise Mersenne Twister
#elif (defined __INTEL_MKL__&&!defined USE_RAN2)
			//Good news, casting works for using a double to create random complex numbers
			vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*kvol, u11t, -1, 1);
			vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*kvol, u12t, -1, 1);
			//Last resort, Numerical Recipes' Ran2
#else
			for(int i=0; i<kvol*ndim;i++){
				u11t[i]=2*(ran2(&seed)-0.5+I*(ran2(&seed)-0.5));
				u12t[i]=2*(ran2(&seed)-0.5+I*(ran2(&seed)-0.5));
			}
#endif
		}
		else
			fprintf(stderr,"Warning %i in %s: Gauge fields are not initialised.\n", NOINIT, funcname);

#ifdef __NVCC__
		cudaMemPrefetchAsync(u11t, ndim*kvol*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(u12t, ndim*kvol*sizeof(Complex),device,NULL);
#endif
		//Send trials to accelerator for reunitarisation
#pragma omp taskwait
#ifdef _OPENACC
#pragma acc update device(u11t[0:ndim*kvol],u12t[0:ndim*kvol])
#else
#pragma omp target update to(u11t[0:ndim*kvol],u12t[0:ndim*kvol])
#endif
		Reunitarise(u11t,u12t);
		//Get trials back
		//#pragma omp target update from(u11t[0:ndim*kvol],u12t[0:ndim*kvol]) 
		memcpy(u11, u11t, ndim*kvol*sizeof(Complex));
		memcpy(u12, u12t, ndim*kvol*sizeof(Complex));
	}
#ifdef _DEBUG
	printf("Initialisation Complete\n");
#endif
	return 0;
}
int Hamilton(double *h, double *s, double res2, double *pp, Complex *X0, Complex *X1, Complex *Phi,\
		Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f, unsigned int * iu, unsigned int *id,\
		Complex_f gamval_f[5][4], int gamin[4][4], float *dk4m_f, float * dk4p_f, Complex_f jqq,\
		float akappa, float beta, double *ancgh){
	/* Evaluates the Hamiltonian function
	 * 
	 * Calls:
	 * =====
	 * Average_Plaquette, Par_dsum, Congradq, Fill_Small_Phi
	 *
	 * Globals:
	 * =======
	 * pp, rank, ancgh, X0, X1, Phi
	 *
	 * Parameters:
	 * ===========
	 * double *h: Hamiltonian
	 * double *s: Action
	 * double res2: Limit for conjugate gradient
	 *
	 * Returns:
	 * =======
	 * Zero on success. Integer Error code otherwise.
	 */	
	const char *funcname = "Hamilton";
	double hp;
	//Iterate over momentum terms.
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(pp,kmom*sizeof(double),device,NULL);
	cublasDnrm2(cublas_handle, kmom, pp, 1,&hp);
	hp*=hp;
#elif defined USE_BLAS
	hp = cblas_dnrm2(kmom, pp, 1);
	hp*=hp;
#else
	hp=0;
	for(int i = 0; i<kmom; i++)
		hp+=pp[i]*pp[i]; 
#endif
	hp*=0.5;
	double avplaqs, avplaqt;
	double hg = 0;
	//avplaq? isn't seen again here.
	Average_Plaquette(&hg,&avplaqs,&avplaqt,u11t,u12t,iu,beta);

	double hf = 0; int itercg = 0;
#ifdef __NVCC__
	Complex *smallPhi;
	cudaMallocManaged(&smallPhi,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
	Complex *smallPhi = mkl_malloc(kferm2Halo*sizeof(Complex),AVX);
#else
	Complex *smallPhi = aligned_alloc(AVX,kferm2Halo*sizeof(Complex));
#endif
	//Iterating over flavours
	for(int na=0;na<nf;na++){
		memcpy(X1,X0+na*kferm2,kferm2*sizeof(Complex));
		Fill_Small_Phi(na, smallPhi, Phi);
		Congradq(na,res2,X1,smallPhi,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa,&itercg);
		//Congradq(na,res2,smallPhi, &itercg );
		*ancgh+=itercg;
		Fill_Small_Phi(na, smallPhi,Phi);
		memcpy(X0+na*kferm2,X1,kferm2*sizeof(Complex));
#ifdef __NVCC__
		Complex dot;
		cublasZdotc(cublas_handle,kferm2,(cuDoubleComplex *)smallPhi,1,(cuDoubleComplex *) X1,1,(cuDoubleComplex *) &dot);
		hf+=creal(dot);
#elif defined USE_BLAS
		Complex dot;
		cblas_zdotc_sub(kferm2, smallPhi, 1, X1, 1, &dot);
		hf+=creal(dot);
#else
		//It is a dot product of the flattened arrays, could use
		//a module to convert index to coordinate array...
		for(int j=0;j<kferm2;j++)
			hf+= conj(smallPhi[j])*X1[j];
#endif
	}
#ifdef __INTEL_MKL__
	mkl_free(smallPhi);
#else
	free(smallPhi);
#endif
	//hg was summed over inside of Average_Plaquette.
	#if(nproc>1)
	Par_dsum(&hp); Par_dsum(&hf);
	#endif
	*s=hg+hf; *h=*s+hp;
#ifdef _DEBUG
	if(!rank)
		printf("hg=%e; hf=%e; hp=%e; h=%e\n", hg, hf, hp, *h);
#endif

	return 0;
}
inline int Z_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu)
{
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
#pragma omp parallel for simd aligned (x,y,table:AVX)
	for(int i=0; i<n; i++)
		x[i]=y[table[i*ndim+mu]*ndim+mu];
	return 0;
}
inline int Fill_Small_Phi(int na, Complex *smallPhi, Complex *Phi)
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
	const char *funcname = "Fill_Small_Phi";
	//BIG and small phi index
#pragma omp parallel for simd aligned(smallPhi,Phi:AVX) collapse(3)
	for(int i = 0; i<kvol;i++)
		for(int idirac = 0; idirac<ndirac; idirac++)
			for(int ic= 0; ic<nc; ic++)
				//	  PHI_index=i*16+j*2+k;
				smallPhi[(i*ndirac+idirac)*nc+ic]=Phi[((na*kvol+i)*ngorkov+idirac)*nc+ic];
	return 0;
}
