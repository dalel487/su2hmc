#ifdef DIAGNOSTIC
#include <assert.h>
#include <complex.h>
#include <matrices.h>
#include <string.h>

int Diagnostics(int istart, Complex *u11, Complex *u12,Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
		unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk4m, double *dk4p,\
		float *dk4m_f, float *dk4p_f, int *gamin, Complex *gamval, Complex_f *gamval_f,\
		Complex_f jqq,float akappa,float beta, double ancg){
	/*
	 * Routine to check if the multiplication routines are working or not
	 * How I hope this will work is that
	 * 1)	Initialise the system
	 * 2) Just after the initialisation of the system but before anything
	 * 	else call this routine using the C Preprocessor.
	 * 3) Give dummy values for the fields and then do work with them
	 * Caveats? Well this could get messy if we call something we didn't
	 * realise was being called and hadn't initialised it properly (Congradq
	 * springs to mind straight away)
	 */
	const char *funcname = "Diagnostics";

	//Initialise the arrays being used. Just going to assume MKL is being
	//used here will also assert the number of flavours for now to avoid issues
	//later
	assert(nf==1);
#include<float.h>
	printf("FLT_EVAL_METHOD is %i. Check online for what this means\n", FLT_EVAL_METHOD);

#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	Complex *xi,*R1,*Phi,*X0,*X1;
	Complex_f *X0_f, *X1_f, *xi_f, *R1_f, *Phi_f;
	double *dSdpi,*pp;
	cudaMallocManaged(&R1,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&xi,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&R1_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&xi_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&X0,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X1,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X0_f,kferm2Halo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&X1_f,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&pp,kmomHalo*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dSdpi,kmom*sizeof(double),cudaMemAttachGlobal);
#else
	Complex *R1= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex *xi= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex_f *R1_f= aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *xi_f= aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex *Phi= aligned_alloc(AVX,nf*kfermHalo*sizeof(Complex)); 
	Complex_f *Phi_f= aligned_alloc(AVX,nf*kfermHalo*sizeof(Complex_f)); 
	Complex *X0= aligned_alloc(AVX,nf*kferm2Halo*sizeof(Complex)); 
	Complex *X1= aligned_alloc(AVX,kferm2Halo*sizeof(Complex)); 
	double *pp = aligned_alloc(AVX,kmomHalo*sizeof(double));
	Complex_f *X0_f= aligned_alloc(AVX,nf*kferm2Halo*sizeof(Complex_f)); 
	Complex_f *X1_f= aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f)); 
	double *dSdpi = aligned_alloc(AVX,kmom*sizeof(double));
#endif
	//pp is the momentum field

#pragma omp parallel sections
	{
#pragma omp section
		{
			FILE *dk4m_File = fopen("dk4m","w");
			for(int i=0;i<kvol;i+=4)
				fprintf(dk4m_File,"%f\t%f\t%f\t%f\n",dk4m[i],dk4m[i+1],dk4m[i+2],dk4m[i+3]);
		}
#pragma omp section
		{
			FILE *dk4p_File = fopen("dk4p","w");
			for(int i=0;i<kvol;i+=4)
				fprintf(dk4p_File,"%f\t%f\t%f\t%f\n",dk4p[i],dk4p[i+1],dk4p[i+2],dk4p[i+3]);
		}
	}
	for(int test = 0; test<=8; test++){
		//Trial fields shouldn't get modified so were previously set up outside
		switch(istart){
			case(1):
#pragma omp parallel sections num_threads(4)
				{
#pragma omp section
					{
						FILE *trial_out = fopen("u11t", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(u11t[i]),cimag(u11t[i]),creal(u11t[i+1]),cimag(u11t[i+1]),
									creal(u11t[2+i]),cimag(u11t[2+i]),creal(u11t[i+3]),cimag(u11t[i+3]));
						fclose(trial_out);
					}
#pragma omp section
					{
						FILE *trial_out = fopen("u12t", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(u12t[i]),cimag(u12t[i]),creal(u12t[i+1]),cimag(u12t[i+1]),
									creal(u12t[2+i]),cimag(u12t[2+i]),creal(u12t[i+3]),cimag(u12t[i+3]));
						fclose(trial_out);
					}
#pragma omp section
					{
						FILE *trial_out = fopen("u11t_f", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(u11t_f[i]),cimag(u11t_f[i]),creal(u11t_f[i+1]),cimag(u11t_f[i+1]),
									creal(u11t_f[2+i]),cimag(u11t_f[2+i]),creal(u11t_f[i+3]),cimag(u11t_f[i+3]));
						fclose(trial_out);
					}
#pragma omp section
					{
						FILE *trial_out = fopen("u12t_f", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(u12t_f[i]),cimag(u12t_f[i]),creal(u12t_f[i+1]),cimag(u12t_f[i+1]),
									creal(u12t_f[2+i]),cimag(u12t_f[2+i]),creal(u12t_f[i+3]),cimag(u12t_f[i+3]));
						fclose(trial_out);
					}
				}
				break;
			case(2):
#pragma omp parallel for simd aligned(u11t,u12t:AVX)
				for(int i =0; i<ndim*kvol; i+=8){
					u11t[i+0]=1+I; u12t[i]=1+I;
					u11t[i+1]=1-I; u12t[i+1]=1+I;
					u11t[i+2]=1+I; u12t[i+2]=1-I;
					u11t[i+3]=1-I; u12t[i+3]=1-I;
					u11t[i+4]=-1+I; u12t[i+4]=1+I;
					u11t[i+5]=1+I; u12t[i+5]=-1+I;
					u11t[i+6]=-1+I; u12t[i+6]=-1+I;
					u11t[i+7]=-1-I; u12t[i+7]=-1-I;
				}
				break;
			default:
				//Cold start as a default
				memcpy(u11t,u11,kvol*ndim*sizeof(Complex));
				memcpy(u12t,u12,kvol*ndim*sizeof(Complex));
				break;
		}
		Reunitarise(u11t,u12t);
		Trial_Exchange(u11t,u12t,u11t_f,u12t_f);

		//We reset all the random fields between each test. It's one way of ensuring that errors don't propegate from one
		//test to another. Since we start from the same seed each time this should give the same results for each test. If
		//it does not, there's a bug
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
		Gauss_d(pp,kmomHalo,0,1);
		Gauss_z(R1, kferm, 0, 1/sqrt(2));
		Gauss_z(Phi, kferm, 0, 1/sqrt(2));
		Gauss_z(xi, kferm, 0, 1/sqrt(2));
		Gauss_c(R1_f, kferm, 0, 1/sqrt(2));
		Gauss_c(Phi_f, kferm, 0, 1/sqrt(2));
		Gauss_c(xi_f, kferm, 0, 1/sqrt(2));
#else
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R1_f, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, Phi_f, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, xi_f, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R1, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, Phi, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, xi, 0, 1/sqrt(2));
#endif
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
		Gauss_z(X0, kferm2, 0, 1/sqrt(2));
		Gauss_z(X1, kferm2, 0, 1/sqrt(2));
		Gauss_c(X0_f, kferm2, 0, 1/sqrt(2));
		Gauss_c(X1_f, kferm2, 0, 1/sqrt(2));
#else
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X0, 0, 1/sqrt(2));
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X1, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X0_f, 0, 1/sqrt(2));
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm2, X1_f, 0, 1/sqrt(2));
#endif

		//Random nomalised momentum field
		Gauss_d(dSdpi,kmom,0,1/sqrt(2));
#pragma omp for simd aligned(dSdpi:AVX) nowait
		for(int i=0; i<kmom; i+=4){
			double norm = sqrt(dSdpi[i]*dSdpi[i]+dSdpi[i+1]*dSdpi[i+1]+dSdpi[i+2]*dSdpi[i+2]+dSdpi[i+3]*dSdpi[i+3]);
			dSdpi[i]/=norm; dSdpi[i+1]/=norm; dSdpi[i+2]/=norm;dSdpi[i+3]/=norm;
		}
		FILE *output_old, *output;
		FILE *output_f_old, *output_f;
		switch(test){
			case(0):
#pragma omp parallel for simd
				for(int i = 0; i< kferm; i++){
					R1_f[i]=(Complex_f)R1[i];
					xi_f[i]=(Complex_f)xi[i];
				}
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				output_old = fopen("dslash_old", "w");
				output_f_old = fopen("dslash_f_old", "w");
#ifdef __NVCC__
				cudaMemPrefetchAsync(R1,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(xi,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(R1_f,kferm*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(xi_f,kferm*sizeof(Complex_f),device,NULL);
				cudaDeviceSynchronise();
#endif
				for(int i = 0; i< kferm; i+=8){
					fprintf(output_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]),cimag(R1[i]),creal(R1[i+1]),cimag(R1[i+1]),
							creal(R1[i+2]),cimag(R1[i+2]),creal(R1[i+3]),cimag(R1[i+3]),
							creal(R1[i+4]),cimag(R1[i+4]),creal(R1[i+5]),cimag(R1[i+5]),
							creal(R1[i+6]),cimag(R1[i+6]),creal(R1[i+7]),cimag(R1[i+7])	);
					fprintf(output_f_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1_f[i]),cimag(R1_f[i]),creal(R1_f[i+1]),cimag(R1_f[i+1]),
							creal(R1_f[i+2]),cimag(R1_f[i+2]),creal(R1_f[i+3]),cimag(R1_f[i+3]),
							creal(R1_f[i+4]),cimag(R1_f[i+4]),creal(R1_f[i+5]),cimag(R1_f[i+5]),
							creal(R1_f[i+6]),cimag(R1_f[i+6]),creal(R1_f[i+7]),cimag(R1_f[i+7]));
					printf("Difference in dslash double and float R1[%d] to R1[%d+7]:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]-R1_f[i]),cimag(R1[i]-R1_f[i]),creal(R1[i+1]-R1_f[i+1]),cimag(R1[i+1]-R1_f[i+1]),
							creal(R1[i+2]-R1_f[i+2]),cimag(R1[i+2]-R1_f[i+2]),creal(R1[i+3]-R1_f[i+3]),cimag(R1[i+3]-R1_f[i+3]),
							creal(R1[i+4]-R1_f[i+4]),cimag(R1[i+4]-R1_f[i+4]),creal(R1[i+5]-R1_f[i+5]),cimag(R1[i+5]-R1_f[i+5]),
							creal(R1[i+6]-R1_f[i+6]),cimag(R1[i+6]-R1_f[i+6]),creal(R1[i+7]-R1_f[i+7]),cimag(R1[i+7]-R1_f[i+7]));
				}
				fclose(output_old); fclose(output_f_old);
				Dslash(xi,R1,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
#ifdef __NVCC__
				cudaMemPrefetchAsync(xi,kferm*sizeof(Complex),device,NULL);
				cudaDeviceSynchronise();
#endif
				Dslash_f(xi_f,R1_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
#ifdef __NVCC__
				cudaMemPrefetchAsync(xi_f,kferm*sizeof(Complex_f),device,NULL);
				cudaDeviceSynchronise();
#endif
				output = fopen("dslash", "w");
				output_f = fopen("dslash_f", "w");
				for(int i = 0; i< kferm; i+=8){
					fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
					fprintf(output_f, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi_f[i]),cimag(xi_f[i]),creal(xi_f[i+1]),cimag(xi_f[i+1]),
							creal(xi_f[i+2]),cimag(xi_f[i+2]),creal(xi_f[i+3]),cimag(xi_f[i+3]),
							creal(xi_f[i+4]),cimag(xi_f[i+4]),creal(xi_f[i+5]),cimag(xi_f[i+5]),
							creal(xi_f[i+6]),cimag(xi_f[i+6]),creal(xi_f[i+7]),cimag(xi_f[i+7]));
					printf("Difference in dslash double and float xi[%d] to xi[%d+7] after mult:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]-xi_f[i]),cimag(xi[i]-xi_f[i]),creal(xi[i+1]-xi_f[i+1]),cimag(xi[i+1]-xi_f[i+1]),
							creal(xi[i+2]-xi_f[i+2]),cimag(xi[i+2]-xi_f[i+2]),creal(xi[i+3]-xi_f[i+3]),cimag(xi[i+3]-xi_f[i+3]),
							creal(xi[i+4]-xi_f[i+4]),cimag(xi[i+4]-xi_f[i+4]),creal(xi[i+5]-xi_f[i+5]),cimag(xi[i+5]-xi_f[i+5]),
							creal(xi[i+6]-xi_f[i+6]),cimag(xi[i+6]-xi_f[i+6]),creal(xi[i+7]-xi_f[i+7]),cimag(xi[i+7]-xi_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(1):
#pragma omp parallel for simd
				for(int i = 0; i< kferm; i++){
					R1_f[i]=(Complex_f)R1[i];
					xi_f[i]=(Complex_f)xi[i];
				}
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				output_old = fopen("dslashd_old", "w");
				output_f_old = fopen("dslashd_f_old", "w");
#ifdef __NVCC__
				cudaMemPrefetchAsync(R1,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(xi,kferm*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(R1_f,kferm*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(xi_f,kferm*sizeof(Complex_f),device,NULL);
				cudaDeviceSynchronise();
#endif
				for(int i = 0; i< kferm; i+=8){
					fprintf(output_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]),cimag(R1[i]),creal(R1[i+1]),cimag(R1[i+1]),
							creal(R1[i+2]),cimag(R1[i+2]),creal(R1[i+3]),cimag(R1[i+3]),
							creal(R1[i+4]),cimag(R1[i+4]),creal(R1[i+5]),cimag(R1[i+5]),
							creal(R1[i+6]),cimag(R1[i+6]),creal(R1[i+7]),cimag(R1[i+7])	);
					fprintf(output_f_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1_f[i]),cimag(R1_f[i]),creal(R1_f[i+1]),cimag(R1_f[i+1]),
							creal(R1_f[i+2]),cimag(R1_f[i+2]),creal(R1_f[i+3]),cimag(R1_f[i+3]),
							creal(R1_f[i+4]),cimag(R1_f[i+4]),creal(R1_f[i+5]),cimag(R1_f[i+5]),
							creal(R1_f[i+6]),cimag(R1_f[i+6]),creal(R1_f[i+7]),cimag(R1_f[i+7]));
					printf("Difference in dslashd double and float R1[%d] to R1[%d+7]:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(R1[i]-R1_f[i]),cimag(R1[i]-R1_f[i]),creal(R1[i+1]-R1_f[i+1]),cimag(R1[i+1]-R1_f[i+1]),
							creal(R1[i+2]-R1_f[i+2]),cimag(R1[i+2]-R1_f[i+2]),creal(R1[i+3]-R1_f[i+3]),cimag(R1[i+3]-R1_f[i+3]),
							creal(R1[i+4]-R1_f[i+4]),cimag(R1[i+4]-R1_f[i+4]),creal(R1[i+5]-R1_f[i+5]),cimag(R1[i+5]-R1_f[i+5]),
							creal(R1[i+6]-R1_f[i+6]),cimag(R1[i+6]-R1_f[i+6]),creal(R1[i+7]-R1_f[i+7]),cimag(R1[i+7]-R1_f[i+7]));
				}
				fclose(output_old); fclose(output_f_old);
				Dslashd(xi,R1,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
#ifdef __NVCC__
				cudaMemPrefetchAsync(xi,kferm*sizeof(Complex),device,NULL);
				cudaDeviceSynchronise();
#endif
				Dslashd_f(xi_f,R1_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
#ifdef __NVCC__
				cudaMemPrefetchAsync(xi_f,kferm*sizeof(Complex_f),device,NULL);
				cudaDeviceSynchronise();
#endif
				output = fopen("dslashd", "w");
				output_f = fopen("dslashd_f", "w");
				for(int i = 0; i< kferm; i+=8){
					fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
					fprintf(output_f, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi_f[i]),cimag(xi_f[i]),creal(xi_f[i+1]),cimag(xi_f[i+1]),
							creal(xi_f[i+2]),cimag(xi_f[i+2]),creal(xi_f[i+3]),cimag(xi_f[i+3]),
							creal(xi_f[i+4]),cimag(xi_f[i+4]),creal(xi_f[i+5]),cimag(xi_f[i+5]),
							creal(xi_f[i+6]),cimag(xi_f[i+6]),creal(xi_f[i+7]),cimag(xi_f[i+7]));
					printf("Difference in dslashd double and float xi[%d] to xi[%d+7] after mult:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(xi[i]-xi_f[i]),cimag(xi[i]-xi_f[i]),creal(xi[i+1]-xi_f[i+1]),cimag(xi[i+1]-xi_f[i+1]),
							creal(xi[i+2]-xi_f[i+2]),cimag(xi[i+2]-xi_f[i+2]),creal(xi[i+3]-xi_f[i+3]),cimag(xi[i+3]-xi_f[i+3]),
							creal(xi[i+4]-xi_f[i+4]),cimag(xi[i+4]-xi_f[i+4]),creal(xi[i+5]-xi_f[i+5]),cimag(xi[i+5]-xi_f[i+5]),
							creal(xi[i+6]-xi_f[i+6]),cimag(xi[i+6]-xi_f[i+6]),creal(xi[i+7]-xi_f[i+7]),cimag(xi[i+7]-xi_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(2):	
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
#pragma omp parallel for simd
				for(int i = 0; i< kferm2; i++){
					X0_f[i]=(Complex_f)X0[i];
					X1_f[i]=(Complex_f)X1[i];
				}
#ifdef __NVCC__
				cudaMemPrefetchAsync(X0,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X1,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X0_f,kferm2*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(X1_f,kferm2*sizeof(Complex_f),device,NULL);
#endif
				output_old = fopen("hdslash_old", "w");output_f_old = fopen("hdslash_f_old", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X0[i]),cimag(X0[i]),creal(X0[i+1]),cimag(X0[i+1]),
							creal(X0[i+2]),cimag(X0[i+2]),creal(X0[i+3]),cimag(X0[i+3]),
							creal(X0[i+4]),cimag(X0[i+4]),creal(X0[i+5]),cimag(X0[i+5]),
							creal(X0[i+6]),cimag(X0[i+6]),creal(X0[i+7]),cimag(X0[i+7]));
					fprintf(output_f_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X0_f[i]),cimag(X0_f[i]),creal(X0_f[i+1]),cimag(X0_f[i+1]),
							creal(X0_f[i+2]),cimag(X0_f[i+2]),creal(X0_f[i+3]),cimag(X0_f[i+3]),
							creal(X0_f[i+4]),cimag(X0_f[i+4]),creal(X0_f[i+5]),cimag(X0_f[i+5]),
							creal(X0_f[i+6]),cimag(X0_f[i+6]),creal(X0_f[i+7]),cimag(X0_f[i+7]));
					printf("Difference in hdslash double and float X0[%d] to X0[%d+7]:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X0[i]-X0_f[i]),cimag(X0[i]-X0_f[i]),creal(X0[i+1]-X0_f[i+1]),cimag(X0[i+1]-X0_f[i+1]),
							creal(X0[i+2]-X0_f[i+2]),cimag(X0[i+2]-X0_f[i+2]),creal(X0[i+3]-X0_f[i+3]),cimag(X0[i+3]-X0_f[i+3]),
							creal(X0[i+4]-X0_f[i+4]),cimag(X0[i+4]-X0_f[i+4]),creal(X0[i+5]-X0_f[i+5]),cimag(X0[i+5]-X0_f[i+5]),
							creal(X0[i+6]-X0_f[i+6]),cimag(X0[i+6]-X0_f[i+6]),creal(X0[i+7]-X0_f[i+7]),cimag(X0[i+7]-X0_f[i+7]));
				}
				fclose(output_old);fclose(output_f_old);
				Hdslash(X1,X0,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
				Hdslash_f(X1_f,X0_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslash", "w");	output_f = fopen("hdslash_f", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
					fprintf(output_f, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1_f[i]),cimag(X1_f[i]),creal(X1_f[i+1]),cimag(X1_f[i+1]),
							creal(X1_f[i+2]),cimag(X1_f[i+2]),creal(X1_f[i+3]),cimag(X1_f[i+3]),
							creal(X1_f[i+4]),cimag(X1_f[i+4]),creal(X1_f[i+5]),cimag(X1_f[i+5]),
							creal(X1_f[i+6]),cimag(X1_f[i+6]),creal(X1_f[i+7]),cimag(X1_f[i+7]));
					printf("Difference in hdslash double and float X1[%d] to X1[%d+7] after mult.:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1[i]-X1_f[i]),cimag(X1[i]-X1_f[i]),creal(X1[i+1]-X1_f[i+1]),cimag(X1[i+1]-X1_f[i+1]),
							creal(X1[i+2]-X1_f[i+2]),cimag(X1[i+2]-X1_f[i+2]),creal(X1[i+3]-X1_f[i+3]),cimag(X1[i+3]-X1_f[i+3]),
							creal(X1[i+4]-X1_f[i+4]),cimag(X1[i+4]-X1_f[i+4]),creal(X1[i+5]-X1_f[i+5]),cimag(X1[i+5]-X1_f[i+5]),
							creal(X1[i+6]-X1_f[i+6]),cimag(X1[i+6]-X1_f[i+6]),creal(X1[i+7]-X1_f[i+7]),cimag(X1[i+7]-X1_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(3):	
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				for(int i = 0; i< kferm2; i++){
					X0_f[i]=(Complex_f)X0[i];
					X1_f[i]=(Complex_f)X1[i];
				}
#ifdef __NVCC__
				cudaMemPrefetchAsync(X0,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X1,kferm2*sizeof(Complex),device,NULL);
				cudaMemPrefetchAsync(X0_f,kferm2*sizeof(Complex_f),device,NULL);
				cudaMemPrefetchAsync(X1_f,kferm2*sizeof(Complex_f),device,NULL);
#endif
				output_old = fopen("hdslashd_old", "w");output_f_old = fopen("hdslashd_f_old", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X0[i]),cimag(X0[i]),creal(X0[i+1]),cimag(X0[i+1]),
							creal(X0[i+2]),cimag(X0[i+2]),creal(X0[i+3]),cimag(X0[i+3]),
							creal(X0[i+4]),cimag(X0[i+4]),creal(X0[i+5]),cimag(X0[i+5]),
							creal(X0[i+6]),cimag(X0[i+6]),creal(X0[i+7]),cimag(X0[i+7]));
					fprintf(output_f_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X0_f[i]),cimag(X0_f[i]),creal(X0_f[i+1]),cimag(X0_f[i+1]),
							creal(X0_f[i+2]),cimag(X0_f[i+2]),creal(X0_f[i+3]),cimag(X0_f[i+3]),
							creal(X0_f[i+4]),cimag(X0_f[i+4]),creal(X0_f[i+5]),cimag(X0_f[i+5]),
							creal(X0_f[i+6]),cimag(X0_f[i+6]),creal(X0_f[i+7]),cimag(X0_f[i+7]));
					printf("Difference in hdslashd double and float X0[%d] to X0[%d+7]:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X0[i]-X0_f[i]),cimag(X0[i]-X0_f[i]),creal(X0[i+1]-X0_f[i+1]),cimag(X0[i+1]-X0_f[i+1]),
							creal(X0[i+2]-X0_f[i+2]),cimag(X0[i+2]-X0_f[i+2]),creal(X0[i+3]-X0_f[i+3]),cimag(X0[i+3]-X0_f[i+3]),
							creal(X0[i+4]-X0_f[i+4]),cimag(X0[i+4]-X0_f[i+4]),creal(X0[i+5]-X0_f[i+5]),cimag(X0[i+5]-X0_f[i+5]),
							creal(X0[i+6]-X0_f[i+6]),cimag(X0[i+6]-X0_f[i+6]),creal(X0[i+7]-X0_f[i+7]),cimag(X0[i+7]-X0_f[i+7]));
				}
				Hdslashd(X1,X0,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
				Hdslashd_f(X1_f,X0_f,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslashd", "w");	output_f = fopen("hdslashd_f", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
					fprintf(output_f, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1_f[i]),cimag(X1_f[i]),creal(X1_f[i+1]),cimag(X1_f[i+1]),
							creal(X1_f[i+2]),cimag(X1_f[i+2]),creal(X1_f[i+3]),cimag(X1_f[i+3]),
							creal(X1_f[i+4]),cimag(X1_f[i+4]),creal(X1_f[i+5]),cimag(X1_f[i+5]),
							creal(X1_f[i+6]),cimag(X1_f[i+6]),creal(X1_f[i+7]),cimag(X1_f[i+7]));
					printf("Difference in hdslashd double and float X1[%d] to X1[%d+7] after mult.:\n",i,i);
					printf("%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1[i]-X1_f[i]),cimag(X1[i]-X1_f[i]),creal(X1[i+1]-X1_f[i+1]),cimag(X1[i+1]-X1_f[i+1]),
							creal(X1[i+2]-X1_f[i+2]),cimag(X1[i+2]-X1_f[i+2]),creal(X1[i+3]-X1_f[i+3]),cimag(X1[i+3]-X1_f[i+3]),
							creal(X1[i+4]-X1_f[i+4]),cimag(X1[i+4]-X1_f[i+4]),creal(X1[i+5]-X1_f[i+5]),cimag(X1[i+5]-X1_f[i+5]),
							creal(X1[i+6]-X1_f[i+6]),cimag(X1[i+6]-X1_f[i+6]),creal(X1[i+7]-X1_f[i+7]),cimag(X1[i+7]-X1_f[i+7]));
				}
				fclose(output);fclose(output_f);
				break;
			case(4):	
				output_old = fopen("hamiltonian_old", "w");
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output_old, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				}
				fclose(output_old);
				output = fopen("hamiltonian", "w");
				double h,s,ancgh;  h=s=ancgh=0;
				Hamilton(&h,&s,rescgg,pp,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,\
						akappa,beta,&ancgh);
				for(int i = 0; i< kferm2; i+=8){
					fprintf(output, "%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n%.5f+%.5fI\t%.5f+%.5fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				}
				fclose(output);
				break;
				//Two force cases because of the flag. This also tests the conjugate gradient works okay
			case(5):	
				output_old = fopen("force_0_old", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output_old);
				Force(dSdpi, 1, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
						dk4m_f,dk4p_f,jqq,akappa,beta,&ancg);
				output = fopen("force_0", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
			case(6):	
				output_old = fopen("force_1_old", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output_old);
				output = fopen("force_1", "w");
				Force(dSdpi, 0, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
						dk4m_f,dk4p_f,jqq,akappa,beta,&ancg);
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
			case(7):
				output_old = fopen("Gauge_Force_old","w");
				for(int i = 0; i< kmom; i+=12){
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i+4], dSdpi[i+5], dSdpi[i+6], dSdpi[i+7]);
					fprintf(output_old, "%.5f\t%.5f\t%.5f\t%.5f\n\n", dSdpi[i+8], dSdpi[i+9], dSdpi[i+10], dSdpi[i+11]);
				}
				fclose(output_old);	
#ifdef __NVCC__
				cudaMemPrefetchAsync(dSdpi,kmom*sizeof(double),device,NULL);
#endif
				Gauge_force(dSdpi,u11t_f,u12t_f,iu,id,beta);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("Gauge_Force","w");
				for(int i = 0; i< kmom; i+=12){
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i+4], dSdpi[i+5], dSdpi[i+6], dSdpi[i+7]);
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n\n", dSdpi[i+8], dSdpi[i+9], dSdpi[i+10], dSdpi[i+11]);
				}
				fclose(output);	
				break;
			case(8):
				int na=0;
				output_old = fopen("PreUpDownPart","w");
				for(int i=0; i<kferm; i+=2)
					fprintf(output_old,"R1[%d]:\t%.5e+%.5ei\tR1[%d]:\t%.5e+%.5ei\n",\
							i,creal(R1[i]),cimag(R1[i]),i+1,creal(R1[i+1]),cimag(R1[i+1]));
				UpDownPart(na,X0,R1);
				fclose(output_old);
				output = fopen("UpDownPart","w");
				for(int i=0; i<kferm2; i+=2)
					fprintf(output,"X0[%d]:\t%.5e+%.5ei\tX0[%d]:\t%.5e+%.5ei\n",\
							i,creal(X0[i]),cimag(X0[i]),i+1,creal(X0[i+1]),cimag(X0[i+1]));

				fclose(output);
				break;

		}
	}
	//George Michael's favourite bit of the code
#ifdef __NVCC__
	//Make a routine that does this for us
	cudaFree(dk4m); cudaFree(dk4p); cudaFree(R1); cudaFree(dSdpi); cudaFree(pp);
	cudaFree(Phi); cudaFree(u11t); cudaFree(u12t);
	cudaFree(X0); cudaFree(X1); cudaFree(u11); cudaFree(u12);
	cudaFree(X0_f); cudaFree(X1_f); cudaFree(u11t_f); cudaFree(u12t_f);
	cudaFree(id); cudaFree(iu); cudaFree(hd); cudaFree(hu);
#else
	free(dk4m); free(dk4p); free(R1); free(dSdpi); free(pp);
	free(Phi); free(u11t); free(u12t); free(xi);
	free(X0); free(X1); free(u11); free(u12);
	free(id); free(iu); free(hd); free(hu);
	free(pcoord);
#endif

#if(nproc>1)
	MPI_Finalise();
#endif
	exit(0);
}
#endif
