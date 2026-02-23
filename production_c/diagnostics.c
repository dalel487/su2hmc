#ifdef DIAGNOSTIC
#include <assert.h>
#include <complex.h>
#include <matrices.h>
#include <string.h>

int Diagnostics(int istart, Complex *u[2], Complex *ut[2],Complex_f *ut_f[2],\
		unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk[0], double *dk[1],\
		float *dk_f[0], float *dk_f[1], int *gamin, Complex *gamval, Complex_f *gamval_f,\
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
	//Some of these strictly do not have a halo. To make things easier I'm giving them one anyway and adjusting the
	//output compared to what might be expected in the main code (kvol vs kvolHalo mainly)
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
			FILE *dk4m_File = fopen("dk0","w");
			for(int i=0;i<kvol;i+=4)
				fprintf(dk4m_File,"%f\t%f\t%f\t%f\n",dk[0][i],dk[0][i+1],dk[0][i+2],dk[0][i+3]);
		}
#pragma omp section
		{
			FILE *dk4p_File = fopen("dk1","w");
			for(int i=0;i<kvol;i+=4)
				fprintf(dk4p_File,"%f\t%f\t%f\t%f\n",dk[1][i],dk[1][i+1],dk[1][i+2],dk[1][i+3]);
		}
	}
	for(int test = 0; test<=9; test++){
		//Trial fields shouldn't get modified so were previously set up outside
		switch(istart){
			//Got gauge fields from file or random so print them
			case(1):
#pragma omp parallel sections num_threads(4)
				{
#pragma omp section
					{
						FILE *trial_out = fopen("ut0", "w");
						for(int i=0;i<(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(ut[0][i]),cimag(ut[0][i]),creal(ut[0][i+kvolHalo]),cimag(ut[0][i+kvolHalo]),
									creal(ut[0][i+2*kvolHalo]),cimag(ut[0][i+2*kvolHalo]),creal(ut[0][i+3*kvolHalo]),cimag(ut[0][i+3*kvolHalo]));
						fclose(trial_out);
					}
#pragma omp section
					{
						FILE *trial_out = fopen("ut1", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(ut[1][i]),cimag(ut[1][i]),creal(ut[1][i+kvolHalo]),cimag(ut[1][i+kvolHalo]),
									creal(ut[1][i+2*kvolHalo]),cimag(ut[1][i+2*kvolHalo]),creal(ut[1][i+3*kvolHalo]),cimag(ut[1][i+3*kvolHalo]));
						fclose(trial_out);
					}
#pragma omp section
					{
						FILE *trial_out = fopen("ut0", "w");
						for(int i=0;i<(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(ut_f[0][i]),cimag(ut_f[0][i]),creal(ut_f[0][i+kvolHalo]),cimag(ut_f[0][i+kvolHalo]),
									creal(ut_f[0][i+2*kvolHalo]),cimag(ut_f[0][i+2*kvolHalo]),creal(ut_f[0][i+3*kvolHalo]),cimag(ut_f[0][i+3*kvolHalo]));
						fclose(trial_out);
					}
#pragma omp section
					{
						FILE *trial_out = fopen("ut1", "w");
						for(int i=0;i<ndim*(kvol+halo);i+=4)
							fprintf(trial_out,"%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\t%.5f+%.5fI\n",
									creal(ut_f[1][i]),cimag(ut_f[1][i]),creal(ut_f[1][i+kvolHalo]),cimag(ut_f[1][i+kvolHalo]),
									creal(ut_f[1][i+2*kvolHalo]),cimag(ut_f[1][i+2*kvolHalo]),creal(ut_f[1][i+3*kvolHalo]),cimag(ut_f[1][i+3*kvolHalo]));
						fclose(trial_out);
					}
				}
				break;
			default:
				//Cold start as a default. Don't need to print
#pragma omp parallel for
				for(unsigned short mu=0;mu<ndim;mu++){
					memcpy(ut[0]+j*kvolHalo,u[0]+j*kvol,kvol*sizeof(Complex));
					memcpy(ut[1]+j*kvolHalo,u[1]+j*kvol,kvol*sizeof(Complex));
				}
				break;
		}
		Reunitarise(ut[0],ut[1]);
		Trial_Exchange(ut[0],ut[1],ut_f[0],ut_f[1]);

		//We reset all the random fields between each test. It's one way of ensuring that errors don't propegate from one
		//test to another. Since we start from the same seed each time this should give the same results for each test. If
		//it does not, there's a bug
		Gauss_d(pp,kmomHalo,0,1); Gauss_z(R1, kferm, 0, 1/sqrt(2));
		Gauss_z(Phi, kferm, 0, 1/sqrt(2)); Gauss_z(xi, kferm, 0, 1/sqrt(2));
		Gauss_c(R1_f, kferm, 0, 1/sqrt(2)); Gauss_c(Phi_f, kferm, 0, 1/sqrt(2));
		Gauss_c(xi_f, kferm, 0, 1/sqrt(2));

		Gauss_z(X0, kferm2, 0, 1/sqrt(2)); Gauss_z(X1, kferm2, 0, 1/sqrt(2));
		Gauss_c(X0_f, kferm2, 0, 1/sqrt(2)); Gauss_c(X1_f, kferm2, 0, 1/sqrt(2));

		//Random nomalised momentum field
		Gauss_d(dSdpi,kmom,0,1/sqrt(2));
#pragma omp for simd aligned(dSdpi:AVX) nowait
		for(int i=0; i<kmom; i+=4){
			double norm = sqrt(dSdpi[i]*dSdpi[i]+dSdpi[i+1]*dSdpi[i+1]+dSdpi[i+2]*dSdpi[i+2]+dSdpi[i+3]*dSdpi[i+3]);
			dSdpi[i]/=norm; dSdpi[i+1]/=norm; dSdpi[i+2]/=norm;dSdpi[i+3]/=norm;
		}
		FILE *input, *output;
		FILE *input_f, *output_f;
		FILE *input_diff, *output_diff;
		switch(test){
			case(0):
				ComplexConvert(R1_f,R1,kvol,false,nc*ngorkov);
				memset(xi,0,kfermHalo*sizeof(Complex)); memset(xi_f,0,kfermHalo*sizeof(Complex_f));
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				input = fopen("dslash_in", "w"); input_f = fopen("dslash_f_in", "w"); input_diff = fopen("dslash_diff_in", "w");
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(input, "%.5f+%.5fI\t",creal(R1[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]));
						fprintf(input_f, "%.5f+%.5fI\t", creal(R1_f[i+j*kvolHalo]),cimag(R1_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.5f+%.5fI\t", creal(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input); fclose(input_f); fclose(input_diff);
				Dslash(xi,R1,ut,iu,id,gamval,gamin,dk,jqq,akappa);
				Dslash_f(xi_f,R1_f,ut_f[0],ut_f[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("dslash", "w"); output_f = fopen("dslash_f", "w"); output_f = fopen("dslash_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(output, "%.5f+%.5fI\t",creal(xi[i+j*kvolHalo]),cimag(xi[i+j*kvolHalo]));
						fprintf(output_f, "%.5f+%.5fI\t", creal(xi_f[i+j*kvolHalo]),cimag(xi_f[i+j*kvolHalo]));
						fprintf(output_diff,"%.5f+%.5fI\t", creal(xi[i+j*kvolHalo]-xi_f[i+j*kvolHalo]),cimag(xi[i+j*kvolHalo]-xi_f[i+j*kvolHalo]));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output); fclose(output_f); fclose(output_diff);
				break;
			case(1):
				ComplexConvert(R1_f,R1,kvol,false,nc*ngorkov);
				memset(xi,0,kfermHalo*sizeof(Complex)); memset(xi_f,0,kfermHalo*sizeof(Complex_f));
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				input = fopen("dslashd_in", "w"); input_f = fopen("dslashd_f_in", "w"); input_diff = fopen("dslashd_diff_in", "w");
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(input, "%.5f+%.5fI\t",creal(R1[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]));
						fprintf(input_f, "%.5f+%.5fI\t", creal(R1_f[i+j*kvolHalo]),cimag(R1_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.5f+%.5fI\t", creal(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input); fclose(input_f);
				Dslashd(xi,R1,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],jqq,akappa);
				Dslashd_f(xi_f,R1_f,ut_f[0],ut_f[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("dslashd", "w"); output_f = fopen("dslashd_f", "w"); output_f = fopen("dslashd_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					//Note. The output of Dslashd should not have a halo. Whilst xi is defined with one we do not use it here
					//so stride is kvol, not kvolHalo
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(output, "%.5f+%.5fI\t",creal(xi[i+j*kvol]),cimag(xi[i+j*kvol]));
						fprintf(output_f, "%.5f+%.5fI\t", creal(xi_f[i+j*kvol]),cimag(xi_f[i+j*kvol]));
						fprintf(output_diff,"%.5f+%.5fI\t", creal(xi[i+j*kvol]-xi_f[i+j*kvol]),cimag(xi[i+j*kvol]-xi_f[i+j*kvol]));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				input = fopen("dslashd_in", "w"); input_f = fopen("dslashd_f_in", "w"); input_diff = fopen("dslashd_diff_in", "w");
				break;
			case(2):	
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				ComplexConvert(X0_f,X0,kvol,false,nc*ndirac);
				memset(X1,0,kferm2Halo*sizeof(Complex)); memset(X1_f,0,kferm2Halo*sizeof(Complex_f));
				input = fopen("hdslash_in", "w"); input_f = fopen("hdslash_f_in", "w"); input_f = fopen("hdslash_diff_in", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input, "%.5f+%.5fI\t",creal(X0[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]));
						fprintf(input_f, "%.5f+%.5fI\t", creal(X0_f[i+j*kvolHalo]),cimag(X0_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.5f+%.5fI\t", creal(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input);fclose(input_f);fclose(input_diff);
				Hdslash(X1,X0,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
				Hdslash_f(X1_f,X0_f,ut_f[0],ut_f[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslash", "w");	output_f = fopen("hdslash_f", "w"); output_diff = fopen("hdslash_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					//Note. The output of Dslashd should not have a halo. Whilst xi is defined with one we do not use it here
					//so stride is kvol, not kvolHalo
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output, "%.5f+%.5fI\t",creal(X1[i+j*kvolHalo]),cimag(X1[i+j*kvolHalo]));
						fprintf(output_f, "%.5f+%.5fI\t", creal(X1_f[i+j*kvolHalo]),cimag(X1_f[i+j*kvolHalo]));
						fprintf(output_diff,"%.5f+%.5fI\t", creal(X1[i+j*kvolHalo]-X1_f[i+j*kvolHalo]),cimag(X1[i+j*kvolHalo]-X1_f[i+j*kvolHalo]));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output);fclose(output_f);fclose(output_diff);
				break;
			case(3):	
				ComplexConvert(X0_f,X0,kvol,false,nc*ndirac);
				memset(X1,0,kferm2Halo*sizeof(Complex)); memset(X1_f,0,kferm2Halo*sizeof(Complex_f));
				input = fopen("hdslash_in", "w"); input_f = fopen("hdslash_f_in", "w"); input_f = fopen("hdslash_diff_in", "w");
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input, "%.5f+%.5fI\t",creal(X0[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]));
						fprintf(input_f, "%.5f+%.5fI\t", creal(X0_f[i+j*kvolHalo]),cimag(X0_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.5f+%.5fI\t", creal(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input);fclose(input_f);fclose(input_diff);
				Hdslashd(X1,X0,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
				Hdslashd_f(X1_f,X0_f,ut_f[0],ut_f[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslashd", "w");	output_f = fopen("hdslashd_f", "w"); output_diff = fopen("hdslashd_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					//Note. The output of Dslashd should not have a halo. Whilst xi is defined with one we do not use it here
					//so stride is kvol, not kvolHalo
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output, "%.5f+%.5fI\t",creal(X1[i+j*kvol]),cimag(X1[i+j*kvol]));
						fprintf(output_f, "%.5f+%.5fI\t", creal(X1_f[i+j*kvol]),cimag(X1_f[i+j*kvol]));
						fprintf(output_diff,"%.5f+%.5fI\t", creal(X1[i+j*kvol]-X1_f[i+j*kvol]),cimag(X1[i+j*kvol]-X1_f[i+j*kvol]));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output);fclose(output_f);fclose(output_diff);
				break;
			case(4):	
				input = fopen("hamiltonian_in", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); 
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input, "%.5f+%.5fI\t",creal(X1[i+j*kvolHalo]),cimag(X1[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); 
				}
				fclose(input);
				double h,s,ancgh;  h=s=ancgh=0;
				Hamilton(&h,&s,rescgg,pp,X0,X1,Phi,ut[0],ut[1],ut_f[0],ut_f[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,\
						akappa,beta,&ancgh,0);
				output = fopen("hamiltonian_out", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); 
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output, "%.5f+%.5fI\t",creal(X1[i+j*kvolHalo]),cimag(X1[i+j*kvolHalo]));
					}
					fprintf(output, "\n\n"); 
				}
				fclose(output);
				break;
				//Two force cases because of the flag. This also tests the conjugate gradient works okay
			case(5):	
				input = fopen("force_0_in", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(input, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(input);
				Force(dSdpi, 1, rescgg,X0,X1,Phi,ut[0],ut[1],ut_f[0],ut_f[1],iu,id,gamval,gamval_f,gamin,dk[0],dk[1],\
						dk_f[0],dk_f[1],jqq,akappa,beta,&ancg);
				output = fopen("force_0", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
			case(6):	
				input = fopen("force_1_in", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(input, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(input);
				output = fopen("force_1", "w");
				Force(dSdpi, 0, rescgg,X0,X1,Phi,ut[0],ut[1],ut_f[0],ut_f[1],iu,id,gamval,gamval_f,gamin,dk[0],dk[1],\
						dk_f[0],dk_f[1],jqq,akappa,beta,&ancg);
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
			case(7):
				input = fopen("Gauge_Force_in","w");
				for(int i = 0; i< kmom; i+=12){
					fprintf(input, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
					fprintf(input, "%.5f\t%.5f\t%.5f\t%.5f\n", dSdpi[i+4], dSdpi[i+5], dSdpi[i+6], dSdpi[i+7]);
					fprintf(input, "%.5f\t%.5f\t%.5f\t%.5f\n\n", dSdpi[i+8], dSdpi[i+9], dSdpi[i+10], dSdpi[i+11]);
				}
				fclose(input);	
#ifdef __NVCC__
				//cudaMemPrefetchAsync(dSdpi,kmom*sizeof(double),device,NULL);
#endif
				Gauge_force(dSdpi,ut_f[0],ut_f[1],iu,id,beta);
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
				input = fopen("PreUpDownPart","w");
				for(int i=0; i<kferm; i+=2)
					fprintf(input,"R1[%d]:\t%.5e+%.5ei\tR1[%d]:\t%.5e+%.5ei\n",\
							i,creal(R1[i]),cimag(R1[i]),i+1,creal(R1[i+1]),cimag(R1[i+1]));
				UpDownPart(na,X0,R1);
				fclose(input);
				output = fopen("UpDownPart","w");
				for(int i=0; i<kferm2; i+=2)
					fprintf(output,"X0[%d]:\t%.5e+%.5ei\tX0[%d]:\t%.5e+%.5ei\n",\
							i,creal(X0[i]),cimag(X0[i]),i+1,creal(X0[i+1]),cimag(X0[i+1]));

				fclose(output);
				break;
			case(9):
				int itercg=0;
				Congradp(0, respbp, Phi, R1,ut_f[0],ut_f[1],iu,id,gamval_f,gamin,dk_f[0],dk_f[1],jqq,akappa,&itercg);

		}
	}
	//George Michael's favourite bit of the code
#ifdef __NVCC__
	//Make a routine that does this for us
	cudaFree(dk[0]); cudaFree(dk[1]); cudaFree(R1); cudaFree(dSdpi); cudaFree(pp);
	cudaFree(Phi); cudaFree(ut[0]); cudaFree(ut[1]);
	cudaFree(X0); cudaFree(X1); cudaFree(u[0]); cudaFree(u[1]);
	cudaFree(X0_f); cudaFree(X1_f); cudaFree(ut_f[0]); cudaFree(ut_f[1]);
	cudaFree(id); cudaFree(iu); cudaFree(hd); cudaFree(hu);
#else
	free(dk[0]); free(dk[1]); free(R1); free(dSdpi); free(pp);
	free(Phi); free(ut[0]); free(ut[1]); free(xi);
	free(X0); free(X1); free(u[0]); free(u[1]);
	free(id); free(iu); free(hd); free(hu);
	free(pcoord);
#endif

#if(nproc>1)
	MPI_Finalise();
#endif
	exit(0);
}
#endif
