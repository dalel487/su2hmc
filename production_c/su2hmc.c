/**
 * @file su2hmc.c
 *
 * @brief	An ecclectic collection of functions used in the HMC
 */
#include	<assert.h>
#include <clover.h>
#include	<matrices.h>

int Init(int istart, int ibound, int iread, float beta, float fmu, float akappa, Complex_f ajq,\
		Complex *u[2], Complex *ut[2], Complex_f *ut_f[2], Complex gamval[20], Complex_f gamval_f[20],
		unsigned short gamin[16], double *dk[2], float *dk_f[2], unsigned int *iu, unsigned int *id){
	const char funcname[] = "Init";

#ifdef _OPENMP
	omp_set_num_threads(nthreads);
#ifdef __USE_MKL__
	mkl_set_num_threads(nthreads);
#endif
#endif
	//First things first, calculate a few constants for coordinates
	Addrc(iu, id);
	//And confirm they're legit
	Check_addr(iu, ksize, ksizet, 0, kvolHalo);
	Check_addr(id, ksize, ksizet, 0, kvolHalo);
#ifdef _DEBUG
	printf("Checked addresses\n");
#endif
	double chem1=exp(-fmu); double chem2 = 1/chem1;
	//CUDA this. Only limit will be the bus speed
#pragma omp parallel for simd //aligned(dk[0],dk[1]:AVX)
	for(unsigned int i = 0; i<kvol; i++){
		dk[0][i]=akappa*chem1; dk[1][i]=akappa*chem2;
	}
	//Anti periodic Boundary Conditions. Flip the terms at the edge of the time
	//direction
	if(ibound == -1 && pcoord[3+ndim*rank]==npt-1){
#ifdef _DEBUG
		printf("Implementing antiperiodic boundary conditions on rank %i\n", rank);
#endif
#pragma omp parallel for simd //aligned(dk[0],dk[1]:AVX)
		for(unsigned int k= kvol-1; k>=kvol-kvol3; k--){
			//int k = kvol - kvol3 + i;
			dk[1][k]*=-1;
			dk[0][k]*=-1;
		}
	}
	//These are constant so swap the halos when initialising and be done with it
	//May need to add a synchronisation statement here first
#if(npt>1)
	DHalo_swap_dir(dk[1], 1, 3, UP);
	DHalo_swap_dir(dk[0], 1, 3, UP);
#endif
	//Float versions
#ifdef __NVCC__
	cuReal_convert(dk_f[1],dk[1],kvol+halo,true,dimBlock,dimGrid);
	cuReal_convert(dk_f[0],dk[0],kvol+halo,true,dimBlock,dimGrid);
#else
#pragma omp parallel for simd //aligned(dk[0],dk[1],dk_f[0],dk_f[1]:AVX)
	for(unsigned int i=0;i<kvol+halo;i++){
		dk_f[1][i]=(float)dk[1][i];
		dk_f[0][i]=(float)dk[0][i];
	}
#endif
	//What row of each dirac/sigma matrix contains the entry acting on element i of the spinor
	unsigned short __attribute__((aligned(AVX))) gamin_t[4][4] =	{{3,2,1,0},{3,2,1,0},{2,3,0,1},{2,3,0,1}};
	//Gamma Matrices in Chiral Representation
	//See Appendix 8.1.2 of Montvay and Munster
	//_t is for temp. We copy these into the real gamvals later
#ifdef __NVCC__
	cudaMemcpy(gamin,gamin_t,4*4*sizeof(short),cudaMemcpyHostToDevice);
#else
	memcpy(gamin,gamin_t,4*4*sizeof(short));
#endif
	//Each row of the dirac matrix contains only one non-zero entry, so that's all we encode here
	Complex	__attribute__((aligned(AVX)))	gamval_t[5][4] =	{{-I,-I,I,I},{-1,1,1,-1},{-I,I,I,-I},{1,1,1,1},{1,1,-1,-1}};
	//Each gamma matrix is rescaled by akappa by flattening the gamval array
#if defined USE_BLAS
	//Don't cuBLAS this. It is small and won't saturate the GPU. Let the CPU handle
	//it and just copy it later
	cblas_zdscal(5*4, akappa, gamval_t, 1);
#else
#pragma omp parallel for simd collapse(2) aligned(gamval,gamval_f:AVX)
	for(unsigned short i=0;i<5;i++)
		for(unsigned short j=0;j<4;j++)
			gamval_t[i][j]*=akappa;
#endif


#ifdef __NVCC__
	cudaMemcpy(gamval,gamval_t,5*4*sizeof(Complex),cudaMemcpyHostToDevice);
	cuComplex_convert(gamval_f,gamval,20,true,dimBlockOne,dimGridOne);	
#else
	memcpy(gamval,gamval_t,5*4*sizeof(Complex));
	for(unsigned short i=0;i<5*4;i++)
		gamval_f[i]=(Complex_f)gamval[i];
#endif

	if(iread){
		if(!rank) printf("Calling Par_sread() for configuration: %i\n", iread);
		Par_sread(iread, beta, fmu, akappa, ajq,u[0],u[1],ut[0],ut[1]);
		Par_ranset(&seed,iread);
	}
	else{
		Par_ranset(&seed,iread);
		if(istart==0){
			//Initialise a cold start to zero
			//memset is safe to use here because zero is zero 
#pragma omp parallel for simd //aligned(ut[0]:AVX) 
										//Leave it to the GPU?
			for(unsigned int i=0; i<kvol;i++)
				for(unsigned short mu=0;mu<ndim;mu++){
					ut[0][i+kvoHalol*mu]=1;	ut[1][i+kvolHalo*mu]=0;
				}
		}
		else if(istart>0){
			//Ideally, we can use gsl_ranlux as the PRNG
#ifdef __RANLUX__
			for(unsigned int i=0; i<kvol;i++)
				for(unsigned short mu=0;mu<ndim;mu++){
					ut[0][i+kvolHalo*mu]=2*(gsl_rng_uniform(ranlux_instd)-0.5+I*(gsl_rng_uniform(ranlux_instd)-0.5));
					ut[1][i+kvolHalo*mu]=2*(gsl_rng_uniform(ranlux_instd)-0.5+I*(gsl_rng_uniform(ranlux_instd)-0.5));
				}
			//Last resort, Numerical Recipes' Ran2
#else
			for(unsigned int i=0; i<kvol;i++)
				for(unsigned short mu=0;mu<ndim;mu++){
					ut[0][i+kvolHalo*mu]=2*(ran2(&seed)-0.5+I*(ran2(&seed)-0.5));
					ut[1][i+kvolHalo*mu]=2*(ran2(&seed)-0.5+I*(ran2(&seed)-0.5));
				}
#endif
		}
		else
			fprintf(stderr,"Warning %i in %s: Gauge fields are not initialised.\n", NOINIT, funcname);

#ifdef __NVCC__
		int device=-1;
		cudaGetDevice(&device);
		//cudaMemPrefetchAsync(ut[0], ndim*kvol*sizeof(Complex),device,streams[0]);
		//cudaMemPrefetchAsync(ut[1], ndim*kvol*sizeof(Complex),device,streams[1]);
#endif
		//Send trials to accelerator for reunitarisation
		Reunitarise(ut);
		//Get trials back
#ifdef __NVCC__
#if (nproc>1) //Strided for multi-GPU
		for(unsigned short mu=0;mu<ndim;mu++){
			cudaMemcpy(u[0]+kvol*mu, ut[0]+kvolHalo*mu, kvol*sizeof(Complex),cudaMemcpyDefault);
			cudaMemcpy(u[1]+kvol*mu, ut[1]+kvolHalo*mu, kvol*sizeof(Complex),cudaMemcpyDefault);
		}
#else
		cudaMemcpy(u[0], ut[0], ndim*kvol*sizeof(Complex),cudaMemcpyDefault);
		cudaMemcpy(u[1], ut[1], ndim*kvol*sizeof(Complex),cudaMemcpyDefault);
#endif
#else
		for(unsigned short mu=0;mu<ndim;mu++){
			memcpy(u[0]+kvol*mu, ut[0]+kvolHalo*mu, kvol*sizeof(Complex));
			memcpy(u[1]+kvol*mu, ut[1]+kvolHalo*mu, kvol*sizeof(Complex));
		}
#endif
	}
#ifdef _DEBUG
	printf("Initialisation Complete\n");
#endif
	return 0;
}
int Hamilton(double *h,double *s,double res2,double *pp,Complex *X0,Complex *X1,Complex *Phi, Complex *ud[2],Complex_f *ut[2],
		unsigned int *iu,unsigned int *id, Complex gamval[20], Complex_f gamval_f[20],const unsigned short gamin[16], Complex *sigval, Complex_f *sigval_f,
		unsigned short *sigin, double *dk[2],float *dk_f[2],Complex_f jqq,float akappa,float beta,float c_sw, double *ancgh,
		int traj){
	const char funcname[] = "Hamilton";
	//Iterate over momentum terms.
#ifdef __NVCC__
	double hp;
	int device=-1;
	cudaGetDevice(&device);
	//cudaMemPrefetchAsync(pp,kmom*sizeof(double),device,NULL);
	cublasDnrm2(cublas_handle, kmom, pp, 1,&hp);
	hp*=hp;
#elif defined USE_BLAS
	double hp = cblas_dnrm2(kmom, pp, 1);
	hp*=hp;
#else
	double hp=0;
	for(unsigned int i = 0; i<kmom; i++)
		hp+=pp[i]*pp[i]; 
#endif
	hp*=0.5;
	double avplaqs, avplaqt;
	double hg = 0;
	//avplaq? isn't seen again here.
	Average_Plaquette(&hg,&avplaqs,&avplaqt,ut,iu,beta);

	alignas(8) double hf = 0; int itercg = 0;
#ifdef __NVCC__
	Complex *smallPhi;
#ifdef _DEBUG
	cudaMallocManaged((void **)&smallPhi,kferm2*sizeof(Complex),cudaMemAttachGlobal);
#else
	cudaMallocAsync((void **)&smallPhi,kferm2*sizeof(Complex),NULL);
#endif
#else
	Complex *smallPhi = aligned_alloc(AVX,kferm2*sizeof(Complex));
#endif
	Complex_f *clover[nc];
	if(c_sw)
		Clover(clover,ut,iu,id);
	//Iterating over flavours
	for(unsigned short na=0;na<nf;na++){
#ifdef __NVCC__
#if (nproc>1) //strided for multi-GPU
		for(unsigned short j=0;j<nc*ndirac;j++)
			cudaMemcpyAsync(X1+j*kvolHalo,X0+na*kferm2+j*kvol,kvol*sizeof(Complex),cudaMemcpyDeviceToDevice,streams[j]);
#else
		cudaMemcpyAsync(X1,X0+na*kferm2,kferm2*sizeof(Complex),cudaMemcpyDeviceToDevice,streams[0]);
#endif
#else
		for(unsigned short j=0;j<nc*ndirac;j++)
			memcpy(X1+j*kvolHalo,X0+na*kferm2+j*kvol,kvol*sizeof(Complex));
#endif
		Fill_Small_Phi(na, smallPhi, Phi);
		if(Congradq(na,res2,X1,smallPhi,ud,ut,clover,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,\
					jqq,akappa,c_sw,&itercg))
			fprintf(stderr,"Trajectory %d\n", traj);

		*ancgh+=itercg;
#ifdef __NVCC__
#if (nproc>1) //strided for multi-GPU
		for(unsigned short j=0;j<nc*ndirac;j++)
			cudaMemcpyAsync(X0+na*kferm2+j*kvol,X1+j*kvolHalo,kvol*sizeof(Complex),cudaMemcpyDeviceToDevice,streams[j]);
#else
		cudaMemcpyAsync(X0+na*kferm2,X1,kferm2*sizeof(Complex),cudaMemcpyDeviceToDevice,streams[0]);
#endif
#else
		for(unsigned short j=0;j<nc*ndirac;j++)
			memcpy(X0+na*kferm2+j*kvol,X1+j*kvolHalo,kvol*sizeof(Complex));
#endif
		Fill_Small_Phi(na, smallPhi,Phi);
#ifdef __NVCC__
		alignas(16) Complex dot=0;
#if (nproc>1)
		for(unsigned short j=0;j<nc*ndirac;j++){
			alignas(16) Complex buff;
			cublasZdotc(cublas_handle,kvol,(cuDoubleComplex *)smallPhi+j*kvol,1,(cuDoubleComplex *) X1+j*kvolHalo,1,(cuDoubleComplex *) &buff);
			dot+=buff;
		}
#else
		cublasZdotc(cublas_handle,kferm2,(cuDoubleComplex *)smallPhi,1,(cuDoubleComplex *) X1,1,(cuDoubleComplex *) &dot);
#endif
		hf+=creal(dot);
#elif defined USE_BLAS
		Complex dot=0;
		for(unsigned short j=0;j<nc*ndirac;j++){
			alignas(16) Complex buff=0;
			cblas_zdotc_sub(kvol, smallPhi+j*kvol, 1, X1+j*kvolHalo, 1, &buff);
			dot+=buff;
		}
		hf+=creal(dot);
#else
		//It is a dot product of the flattened arrays, could use
		//a module to convert index to coordinate array...
#pragma omp parallel for simd collapse(2) aligned(smallPhi,X1:AVX)
		for(unsigned short j=0;j<nc*ndirac;j++)
			for(unsigned int i=0;i<kvol;i++)
				hf+=creal(conj(smallPhi[i+j*kvol])*X1[i+j*kvolHalo]);
#endif
	}
	if(c_sw)
		Clover_free(clover);
#ifdef __NVCC__
#ifdef _DEBUG
	cudaFree(smallPhi);
#else
	cudaFreeAsync(smallPhi,NULL);
#endif
#else
	free(smallPhi);
#endif
	//hg was summed over inside of Average_Plaquette.
#if(nproc>1)
	Par_dsum(&hp); Par_dsum(&hf);
#endif
	*s=hg+hf; *h=(*s)+hp;
#ifdef _DEBUG
	if(!rank)
		printf("hg=%.5e; hf=%.5e; hp=%.5e; h=%.5e\n", hg, hf, hp, *h);
#endif
	return 0;
}
inline int C_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu)
{
	const char funcname[] = "C_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
#pragma omp parallel for simd aligned (x,y,table:AVX)
	for(unsigned int i=0; i<n; i++)
		x[i]=y[table[i+kvol*mu]+kvol*mu];
	return 0;
}
inline int Z_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu)
{
	const char funcname[] = "Z_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
#pragma omp parallel for simd aligned (x,y,table:AVX)
	for(unsigned int i=0; i<n; i++)
		x[i]=y[table[i+kvol*mu]+kvol*mu];
	return 0;
}
inline int Fill_Small_Phi(int na, Complex *smallPhi, Complex *Phi)
{
	const char funcname[] = "Fill_Small_Phi";
	//BIG and small phi index
#ifdef __NVCC__
	cuFill_Small_Phi(na,smallPhi,Phi,dimBlock,dimGrid);
#else
#pragma omp parallel for simd aligned(smallPhi,Phi:AVX) collapse(3)
	for(unsigned int i = 0; i<kvol;i++)
		for(unsigned short idirac = 0; idirac<ndirac; idirac++)
			for(unsigned short ic= 0; ic<nc; ic++)
				//	  PHI_index=i*16+j*2+k;
				smallPhi[i + kvol * (ic + nc * idirac)] = Phi[i + kvol * (ic + nc * (idirac + ngorkov * na))];
#endif
	return 0;
}
inline int UpDownPart(const unsigned int na, Complex *X0, Complex *R1){
#ifdef __NVCC__
	cuUpDownPart(na,X0,R1,dimBlock,dimGrid);
	cudaDeviceSynchronise();
#else
#pragma omp parallel for simd collapse(2) aligned(X0,R1:AVX)
	for(unsigned int i=0; i<kvol; i++)
		for(unsigned short idirac = 0; idirac < ndirac; idirac++){
			X0[i + kvol * (0 + nc * (idirac + ndirac * na))] = R1[i + kvol * (0 + nc * idirac)];
			X0[i + kvol * (1 + nc * (idirac + ndirac * na))] = R1[i + kvol * (1 + nc * idirac)];
		}
#endif
	return 0;
}
inline int Reunitarise(Complex *ut[2]){
	const char funcname[] = "Reunitarise";
#ifdef __NVCC__
	cuReunitarise(ut,dimGrid,dimBlock);
#else
#pragma omp parallel for simd
	for(unsigned short mu=0;mu<ndim;mu++)
		for(unsigned int i=0; i<kvol; i++){
			//Declaring anorm inside the loop will hopefully let the compiler know it
			//is safe to vectorise aggressively
			double anorm=sqrt(conj(ut[0][i+kvolHalo*mu])*ut[0][i+kvolHalo*mu]+conj(ut[1][i+kvolHalo*mu])*ut[1][i+kvolHalo*mu]);
			ut[0][i+kvolHalo*mu]/=anorm; ut[1][i+kvolHalo*mu]/=anorm;
		}
#endif
	return 0;
}
int ComplexConvert(Complex_f *a, Complex *b, const unsigned int len, const bool dtof, const unsigned short stride){
	const char funcname[] = "ComplexConvert";
	switch(stride){
		case(0):
			fprintf(stderr,"Error %i in %s: Stride of %d is not valid.\nExiting...\n\n",STRDERROR,funcname,stride);
#if (nproc>1)
			MPI_Abort(comm,STRDERROR);
#else
			exit(STRDERROR);
#endif
			break;
		case(1):
#ifdef __NVCC__
			cuComplex_convert(a,b,len*stride,dtof,dimBlock,dimGrid);
#else
			if(dtof)
#pragma omp parallel for simd aligned(a,b:AVX)
				for(unsigned int i=0;i<len*stride;i++)
					a[i]=(Complex_f)b[i];
			else
#pragma omp parallel for simd aligned(a,b:AVX)
				for(unsigned int i=0;i<len*stride;i++)
					b[i]=(Complex)a[i];
#endif
			break;
		default:
			for(unsigned short j=0;j<stride;j++){
#ifdef __NVCC__
				cuComplex_convert(a+j*(len+halo),b+j*(len+halo),len,dtof,dimBlock,dimGrid);
#else
				if(dtof)
#pragma omp parallel for simd aligned(a,b:AVX)
					for(unsigned int i=0;i<len;i++)
						a[i+j*(len+halo)]=(Complex_f)b[i+j*(len+halo)];
				else
#pragma omp parallel for simd aligned(a,b:AVX)
					for(unsigned int i=0;i<len;i++)
						b[i+j*(len+halo)]=(Complex)a[i+j*(len+halo)];
#endif
			}
			break;
	}
	return 0;
}
