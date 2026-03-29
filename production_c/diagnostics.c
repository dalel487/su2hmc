#ifdef DIAGNOSTIC
#include <assert.h>
#include <complex.h>
#include <float.h>
#include <clover.h>
#include <matrices.h>
#include <su2hmc.h>
#include <string.h>

int Diagnostics(int istart, Complex *u[2], Complex *ut[2],Complex_f *ut_f[2],\
		unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk[2], float *dk_f[2],\
		const unsigned short gamin[16], const Complex gamval[20], const Complex_f gamval_f[20],\
		const Complex *sigval, const Complex_f *sigval_f, const short *sigin,
		Complex_f jqq,float akappa,float beta, float c_sw, double ancg){
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
	printf("FLT_EVAL_METHOD is %i. Check online for what this means\n", FLT_EVAL_METHOD);

	unsigned int itercg=0;
	Complex_f *clover_f[nc], *hLeaves[ndim][nc]; Complex *clover[nc];
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	Complex *xi,*R1,*Phi,*X0,*X1, *smallPhi;
	Complex_f *X0_f, *X1_f, *xi_f, *R1_f, *Phi_f;
	double *dSdpi,*pp;
	//Some of these strictly do not have a halo. To make things easier I'm giving them one anyway and adjusting the
	//output compared to what might be expected in the main code ((void **)kvol vs kvolHalo mainly)
	cudaMallocManaged((void **)clover+0,6*kvol*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)clover+1,6*kvol*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)clover_f+0,6*kvol*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)clover_f+1,6*kvol*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	for(unsigned short i=0;i<ndim;i++){
		cudaMallocManaged((void **)&hLeaves[i][0],ndim*kvol*sizeof(Complex_f),cudaMemAttachGlobal);
		cudaMallocManaged((void **)&hLeaves[i][1],ndim*kvol*sizeof(Complex_f),cudaMemAttachGlobal);
	}
	cudaMallocManaged((void **)&R1,kfermHalo*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&xi,kfermHalo*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&R1_f,kfermHalo*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&xi_f,kfermHalo*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&Phi,nf*kferm*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&smallPhi,kferm2*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&Phi_f,nf*kferm*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&X0,kferm2Halo*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&X1,kferm2Halo*sizeof((void **)Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&X0_f,kferm2Halo*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&X1_f,kferm2Halo*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&X2_f,kferm2Halo*sizeof((void **)Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&pp,kmom*sizeof((void **)double),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&dSdpi,kmom*sizeof((void **)double),cudaMemAttachGlobal);
#else
	clover[0]=aligned_alloc(AVX,6*kvol*sizeof(Complex));
	clover[1]=aligned_alloc(AVX,6*kvol*sizeof(Complex));
	for(unsigned short mu=0;mu<ndim;mu++){
		hLeaves[mu][0]=(Complex_f *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex_f));
		hLeaves[mu][1]=(Complex_f *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex_f));
	}
	Complex *R1= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex *xi= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Complex_f *R1_f= aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex_f *xi_f= aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
	Complex *smallPhi= aligned_alloc(AVX,kferm2*sizeof(Complex)); 
	Complex *Phi= aligned_alloc(AVX,nf*kferm*sizeof(Complex)); 
	Complex_f *Phi_f= aligned_alloc(AVX,nf*kferm*sizeof(Complex_f)); 
	Complex *X0= aligned_alloc(AVX,nf*kferm2Halo*sizeof(Complex)); 
	Complex *X1= aligned_alloc(AVX,kferm2Halo*sizeof(Complex)); 
	double *pp = aligned_alloc(AVX,kmom*sizeof(double));
	Complex_f *X0_f= aligned_alloc(AVX,nf*kferm2Halo*sizeof(Complex_f)); 
	Complex_f *X1_f= aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f)); 
	Complex_f *X2_f= (Complex_f *)aligned_alloc(AVX,kferm2Halo*sizeof(Complex_f));
	double *dSdpi = aligned_alloc(AVX,kmom*sizeof(double));
#endif
	//Trial fields shouldn't get modified (except for gauge_update
	switch(istart){
		//Got gauge fields from file or random so print them
		case(1):
#pragma omp parallel sections
			{
#pragma omp section
				{
					FILE *trial_out = fopen("gauge_t", "w");
					for(unsigned int i=0;i<(kvol+halo);i++){
						if(i<kvol)
							fprintf(trial_out,"Site %d:\n",i);
						else
							fprintf(trial_out,"Halo site %d:\n",i);
						for(unsigned short mu=0;mu<ndim;mu++)
							fprintf(trial_out,"Dir %d:\t%.3f+%.3fI\t%.3f+%.3fI\n", mu,\
									creal(ut[0][i+mu*kvolHalo]),cimag(ut[0][i+mu*kvolHalo]),\
									creal(ut[1][i+mu*kvolHalo]),cimag(ut[1][i+mu*kvolHalo]));
						fprintf(trial_out,"\n");
					}
					fclose(trial_out);
				}
#pragma omp section
				{
					FILE *trial_out_f = fopen("gauge_t_f", "w");
					for(unsigned int i=0;i<(kvol+halo);i++){
						if(i<kvol)
							fprintf(trial_out_f,"Site %d:\n",i);
						else
							fprintf(trial_out_f,"Halo site %d:\n",i);
						for(unsigned short mu=0;mu<ndim;mu++)
							fprintf(trial_out_f,"Dir %d:\t%.3f+%.3fI\t%.3f+%.3fI\n", mu,\
									creal(ut_f[0][i+mu*kvolHalo]),cimag(ut_f[0][i+mu*kvolHalo]),\
									creal(ut_f[1][i+mu*kvolHalo]),cimag(ut_f[1][i+mu*kvolHalo]));
						fprintf(trial_out_f,"\n");
					}
					fclose(trial_out_f);
				}
			}
			break;
		default:
			//Cold start as a default. Don't need to print
			//NOTE: Single link set non unity
			if(!rank)
				printf("Cold Start\n");
			u[0][0]=1+0*I; u[1][0]=0+0*I;
			u[0][1+kvolHalo]=0+0*I; u[1][1+kvolHalo]=0+1*I;
#pragma omp parallel for
			for(unsigned short mu=0;mu<ndim;mu++){
				memcpy(ut[0]+mu*kvolHalo,u[0]+mu*kvol,kvol*sizeof(Complex));
				memcpy(ut[1]+mu*kvolHalo,u[1]+mu*kvol,kvol*sizeof(Complex));
			}
			break;
	}
	//Ensure reunitarisation is working
	Reunitarise(ut);
	for(unsigned short mu=0;mu<ndim;mu++)
		for(unsigned int i=0;i<kvol;i++){
			double diff = 1-fabs(creal(ut[0][i+mu*kvolHalo]*conj(ut[0][i+mu*kvolHalo])+ut[1][i+mu*kvolHalo]*conj(ut[1][i+mu*kvolHalo])));
			if(diff >1e-6){
				fprintf(stderr,"Error %i in %s: Gauge links not correctly reuniterised for site %i and direction %d. Diff %e"\
						"\nExiting...\n\n",REUNIERR,funcname,i,mu,diff);
				exit(REUNIERR);
			}
		}
	//Check precision change works
	ComplexConvert(ut_f[0],ut[0],kvol,true,ndim);
	for(unsigned short mu=0;mu<ndim;mu++)
		for(unsigned int i=0;i<kvol;i++){
			Complex diff =ut_f[0][i+kvolHalo*mu]-ut[0][i+kvolHalo*mu];
			if(fabs(creal(diff))>1e-6||fabs(cimag(diff))>1e-6){
				fprintf(stderr,"Error %i in %s: Gauge links not correctly converted to float for site %i and direction %d. Diff %e+I%e"\
						"\nExiting...\n\n",CONVERR,funcname,i,mu,creal(diff),cimag(diff));
				exit(CONVERR);
			}
		}
	//Repeat in the opposite direction. 
	ComplexConvert(ut_f[0],ut[0],kvol,false,ndim);
	for(unsigned short mu=0;mu<ndim;mu++)
		for(unsigned int i=0;i<kvol;i++){
			Complex diff =ut_f[0][i+kvolHalo*mu]-ut[0][i+kvolHalo*mu];
			if(fabs(creal(diff))>1e-6||fabs(cimag(diff))>1e-6){
				fprintf(stderr,"Error %i in %s: Gauge links not correctly converted to double for site %i and direction %d. Diff %e+I%e"\
						"\nExiting...\n\n",CONVERR,funcname,i,mu,creal(diff),cimag(diff));
				exit(CONVERR);
			}
		}
	//Gauge halo exchange.
	Trial_Exchange(ut,ut_f);
	//TODO: Figure out a test for this. May require a second lattice to be copied over in full...

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
	for(int test = 0; test<=15; test++){

		//We reset all the random fields between each test. It's one way of ensuring that errors don't propegate from one
		//test to another. Since we start from the same seed each time this should give the same results for each test. If
		//it does not, there's a bug
		Gauss_d(pp,kmom,0,1); Gauss_z(R1, kfermHalo, 0, 1/sqrt(2));
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
		int na=0;
		switch(test){
			case(0): //UpDownPart
				input = fopen("PreUpDownPart","w");
				for(int i=0; i<kvol; i++){
					fprintf(input,"Site %d:\t",i);
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input,"%.5e+%.5ei\t", creal(R1[i+j*kvol]),cimag(R1[i+j*kvol]));
					}
					fprintf(input,"\n");
				}
				fclose(input);
				UpDownPart(na,X0,R1);
				output = fopen("UpDownPart","w");
				for(unsigned int i=0; i<kvol; i++){
					fprintf(output,"Site %d:\t",i);
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output,"%.5e+%.5ei\t", creal(X0[i+j*kvol]),cimag(X0[i+j*kvol]));
					}
					fprintf(output,"\n");
				}
				fclose(output);
				for(unsigned short idirac=0;idirac<ndirac;idirac++)
					for(unsigned short ic=0;ic<nc;ic++)
						for(unsigned int i=0;i<kvol;i++){
							if(X0[i+kvol*(ic+nc*(idirac+ndirac*na))]!=R1[i+kvol*(ic+nc*idirac)]){
								fprintf(stderr,"Error %i in %s: Up/down partitioning failed for site %d colour %d and dirac spinor %d."
										"\nExiting...\n\n",UDPERR,funcname,i,ic,idirac);
								exit(UDPERR);
							}
						}
				break;
			case(1): //Dslash
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
						fprintf(input, "%.3f+%.3fI\t",creal(R1[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]));
						fprintf(input_f, "%.3f+%.3fI\t", creal(R1_f[i+j*kvolHalo]),cimag(R1_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.3f+%.3fI\t", creal(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input); fclose(input_f); fclose(input_diff);
				Dslash(xi,R1,ut,iu,id,gamval,gamin,dk,jqq,akappa);
				Dslash_f(xi_f,R1_f,ut_f,iu,id,gamval_f,gamin,dk_f,jqq,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("dslash", "w"); output_f = fopen("dslash_f", "w"); output_diff = fopen("dslash_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(output, "%.3f+%.3fI\t",creal(xi[i+j*kvolHalo]),cimag(xi[i+j*kvolHalo]));
						fprintf(output_f, "%.3f+%.3fI\t", creal(xi_f[i+j*kvolHalo]),cimag(xi_f[i+j*kvolHalo]));
						Complex diff = xi[i+j*kvolHalo]-xi_f[i+j*kvolHalo];
						if(fabs(creal(diff))>1e-6 || fabs(cimag(diff))>1e-6){
							fprintf(stderr,"Error %i in %s: Single and double disagree for Dslash site %i and spinor/color %d. Difference %e+%ei"\
									"\nExiting...\n\n",CONVERR,funcname,i,j,creal(diff),cimag(diff));
							fclose(output);fclose(output_f);fclose(output_diff);
							exit(CONVERR);
						}
						else
							fprintf(output_diff,"%.3f+%.3fI\t", creal(diff),cimag(diff));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output); fclose(output_f); fclose(output_diff);
				break;
			case(2): //Dslashd
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
						fprintf(input, "%.3f+%.3fI\t",creal(R1[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]));
						fprintf(input_f, "%.3f+%.3fI\t", creal(R1_f[i+j*kvolHalo]),cimag(R1_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.3f+%.3fI\t", creal(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input); fclose(input_f);fclose(input_diff);
				Dslashd(xi,R1,ut,iu,id,gamval,gamin,dk,jqq,akappa);
				Dslashd_f(xi_f,R1_f,ut_f,iu,id,gamval_f,gamin,dk_f,jqq,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("dslashd", "w"); output_f = fopen("dslashd_f", "w"); output_diff = fopen("dslashd_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					//Note. The output of Dslashd should not have a halo. Whilst xi is defined with one we do not use it here
					//so stride is kvol, not kvolHalo
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(output, "%.3f+%.3fI\t",creal(xi[i+j*kvol]),cimag(xi[i+j*kvol]));
						fprintf(output_f, "%.3f+%.3fI\t", creal(xi_f[i+j*kvol]),cimag(xi_f[i+j*kvol]));
						Complex diff = xi[i+j*kvol]-xi_f[i+j*kvol];
						if(fabs(creal(diff))>1e-6 || fabs(cimag(diff))>1e-6){
							fprintf(stderr,"Error %i in %s: Single and double disagree for Dslashd site %i and spinor/color %d. Difference %e+%ei"\
									"\nExiting...\n\n",CONVERR,funcname,i,j,creal(diff),cimag(diff));
							fclose(output);fclose(output_f);fclose(output_diff);
							exit(CONVERR);
						}
						else
							fprintf(output_diff,"%.3f+%.3fI\t", creal(diff),cimag(diff));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				input = fopen("dslashd_in", "w"); input_f = fopen("dslashd_f_in", "w"); input_diff = fopen("dslashd_diff_in", "w");
				break;
			case(3):	//Hdslash
						//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
						//Each block to one lattice site
				ComplexConvert(X0_f,X0,kvol,false,nc*ndirac);
				memset(X1,0,kferm2Halo*sizeof(Complex)); memset(X1_f,0,kferm2Halo*sizeof(Complex_f));
				input = fopen("hdslash_in", "w"); input_f = fopen("hdslash_f_in", "w"); input_diff = fopen("hdslash_diff_in", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input, "%.3f+%.3fI\t",creal(X0[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]));
						fprintf(input_f, "%.3f+%.3fI\t", creal(X0_f[i+j*kvolHalo]),cimag(X0_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.3f+%.3fI\t", creal(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input);fclose(input_f);fclose(input_diff);
				Hdslash(X1,X0,ut,iu,id,gamval,gamin,dk,akappa);
				Hdslash_f(X1_f,X0_f,ut_f,iu,id,gamval_f,gamin,dk_f,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslash", "w");	output_f = fopen("hdslash_f", "w"); output_diff = fopen("hdslash_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					//Note. The output of Dslashd should not have a halo. Whilst xi is defined with one we do not use it here
					//so stride is kvol, not kvolHalo
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output, "%.3f+%.3fI\t",creal(X1[i+j*kvolHalo]),cimag(X1[i+j*kvolHalo]));
						fprintf(output_f, "%.3f+%.3fI\t", creal(X1_f[i+j*kvolHalo]),cimag(X1_f[i+j*kvolHalo]));
						Complex diff = X1[i+j*kvolHalo]-X1_f[i+j*kvolHalo];
						if(fabs(creal(diff))>1e-6 || fabs(cimag(diff))>1e-6){
							fprintf(stderr,"Error %i in %s: Single and double disagree for Hdslash site %i and spinor/color %d. Difference %e+%ei"\
									"\nExiting...\n\n",CONVERR,funcname,i,j,creal(diff),cimag(diff));
							fclose(output);fclose(output_f);fclose(output_diff);
							exit(CONVERR);
						}
						else
							fprintf(output_diff,"%.3f+%.3fI\t", creal(diff),cimag(diff));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output);fclose(output_f);fclose(output_diff);
				break;
			case(4):	//Hdslashd
				ComplexConvert(X0_f,X0,kvol,false,nc*ndirac);
				memset(X1,0,kferm2Halo*sizeof(Complex)); memset(X1_f,0,kferm2Halo*sizeof(Complex_f));
				input = fopen("hdslashd_in", "w"); input_f = fopen("hdslashd_f_in", "w"); input_diff = fopen("hdslashd_diff_in", "w");
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input, "%.3f+%.3fI\t",creal(X0[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]));
						fprintf(input_f, "%.3f+%.3fI\t", creal(X0_f[i+j*kvolHalo]),cimag(X0_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.3f+%.3fI\t", creal(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input);fclose(input_f);fclose(input_diff);
				Hdslashd(X1,X0,ut,iu,id,gamval,gamin,dk,akappa);
				Hdslashd_f(X1_f,X0_f,ut_f,iu,id,gamval_f,gamin,dk_f,akappa);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hdslashd", "w");	output_f = fopen("hdslashd_f", "w"); output_diff = fopen("hdslashd_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					//Note. The output of Dslashd should not have a halo. Whilst xi is defined with one we do not use it here
					//so stride is kvol, not kvolHalo
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output, "%.3f+%.3fI\t",creal(X1[i+j*kvol]),cimag(X1[i+j*kvol]));
						fprintf(output_f, "%.3f+%.3fI\t", creal(X1_f[i+j*kvol]),cimag(X1_f[i+j*kvol]));
						Complex diff = X1[i+j*kvol]-X1_f[i+j*kvol];
						if(fabs(creal(diff))>1e-6 || fabs(cimag(diff))>1e-6){
							fprintf(stderr,"Error %i in %s: Single and double disagree for Hdslashd site %i and spinor/color %d. Difference %e+%ei"\
									"\nExiting...\n\n",CONVERR,funcname,i,j,creal(diff),cimag(diff));
							fclose(output);fclose(output_f);fclose(output_diff);
							exit(CONVERR);
						}
						else
							fprintf(output_diff,"%.3f+%.3fI\t", creal(diff),cimag(diff));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output);fclose(output_f);fclose(output_diff);
				break;
			case(5): //Clover
				if(c_sw==0)
					break;
				//Should really make Leaves a seperate case. But too much effort for now
				output = fopen("Leaves","w");
				for(unsigned int i=0;i<kvol;i++){
					fprintf(output,"Site %d\n",i);
					for(unsigned short mu=0;mu<ndim-1;mu++)
						for(unsigned short nu=mu+1;nu<ndim;nu++)
							if(mu!=nu){
								unsigned short clov = (mu==0) ? nu-1 :mu+nu;
								fprintf(output,"Clover %d\n",clov);
								Complex_f Leaves[nc];
								for(unsigned short leaf =0;leaf<ndim;leaf++){
									Leaf(Leaves,ut_f,iu,id,i,mu,nu,leaf);
									fprintf(output,"Leaf %d: Leaf0 = %e+I%e Leaf1=%e+I%e\n",leaf,\
											crealf(Leaves[0]),cimagf(Leaves[0]),crealf(Leaves[1]),cimagf(Leaves[1]));
								}
							}
					fprintf(output,"\n");
				}
				fclose(output);
				Clover(clover_f,ut_f,iu,id);
				output=fopen("Clover","w");
				for(unsigned int i=0;i<kvol;i++){
					fprintf(output,"Site %d\n",i);
					for(unsigned short mu=0;mu<ndim-1;mu++)
						for(unsigned short nu=mu+1;nu<ndim;nu++)
							if(mu!=nu){
								unsigned short clov = (mu==0) ? nu-1 :mu+nu;
								fprintf(output,"mu %d nu %d Clover1 %e+i%e Clover2 %e+i%e\n",mu,nu,\
										crealf(clover_f[0][i+kvol*clov]), cimagf(clover_f[0][i+kvol*clov]), crealf(clover_f[1][i+kvol*clov]),\
										cimagf(clover_f[1][i+kvol*clov]));
							}
					fprintf(output,"\n");
				}
				fclose(output);
				//Clover correct, Convert works so get it in double here for everywhere else
				ComplexConvert(clover_f[0],clover[0],6*kvol,false,1);
				ComplexConvert(clover_f[1],clover[1],6*kvol,false,1);
				break;
			case(6): //ByClover
				if(c_sw==0)
					break;
				ComplexConvert(R1_f,R1,kvol,false,nc*ngorkov);
				memset(xi,0,kfermHalo*sizeof(Complex)); memset(xi_f,0,kfermHalo*sizeof(Complex_f));
				//NOTE: Each line corresponds to one lattice direction, in the form of colour 0, colour 1.
				//Each block to one lattice site
				input = fopen("byclover_in", "w"); input_f = fopen("byclover_f_in", "w"); input_diff = fopen("byclover_diff_in", "w");
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(input, "%.3f+%.3fI\t",creal(R1[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]));
						fprintf(input_f, "%.3f+%.3fI\t", creal(R1_f[i+j*kvolHalo]),cimag(R1_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.3f+%.3fI\t", creal(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]),cimag(R1[i+j*kvolHalo]-R1_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input); fclose(input_f); fclose(input_diff);
				ByClover(xi,R1,clover,sigval,akappa,sigin,false);
				ByClover_f(xi_f,R1_f,clover_f,sigval_f,akappa,sigin,false);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("byclover", "w"); output_f = fopen("byclover_f", "w"); output_diff = fopen("byclover_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ngorkov;j++){
						fprintf(output, "%.3f+%.3fI\t",creal(xi[i+j*kvolHalo]),cimag(xi[i+j*kvolHalo]));
						fprintf(output_f, "%.3f+%.3fI\t", creal(xi_f[i+j*kvolHalo]),cimag(xi_f[i+j*kvolHalo]));
						Complex diff = xi[i+j*kvolHalo]-xi_f[i+j*kvolHalo];
						if(fabs(creal(diff))>1e-6 || fabs(cimag(diff))>1e-6){
							fprintf(stderr,"Error %i in %s: Single and double disagree for ByClover site %i and spinor/color %d. Difference %e+%ei"\
									"\nExiting...\n\n",CONVERR,funcname,i,j,creal(diff),cimag(diff));
							fclose(output);fclose(output_f);fclose(output_diff);
							exit(CONVERR);
						}
						else
							fprintf(output_diff,"%.3f+%.3fI\t", creal(diff),cimag(diff));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output); fclose(output_f); fclose(output_diff);
				break;
			case(7):	//HbyClover
				if(c_sw==0)
					break;
				ComplexConvert(X0_f,X0,kvol,false,nc*ndirac);
				memset(X1,0,kferm2Halo*sizeof(Complex)); memset(X1_f,0,kferm2Halo*sizeof(Complex_f));
				input = fopen("hbyclover_in", "w"); input_f = fopen("hbyclover_f_in", "w"); input_diff = fopen("hbyclover_diff_in", "w");
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); fprintf(input_f, "Site %d:\n",i); fprintf(input_diff, "Site %d:\n",i);
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input, "%.3f+%.3fI\t",creal(X0[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]));
						fprintf(input_f, "%.3f+%.3fI\t", creal(X0_f[i+j*kvolHalo]),cimag(X0_f[i+j*kvolHalo]));
						fprintf(input_diff,"%.3f+%.3fI\t", creal(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]),cimag(X0[i+j*kvolHalo]-X0_f[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); fprintf(input_f,"\n\n"); fprintf(input_diff,"\n\n");
				}
				fclose(input);fclose(input_f);fclose(input_diff);
				HbyClover(X1,X0,clover,sigval,akappa,sigin,false);
				HbyClover_f(X1_f,X0_f,clover_f,sigval_f,akappa,sigin,false);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("hbyclover", "w");	output_f = fopen("hbyclover_f", "w"); output_diff = fopen("hbyclover_diff", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); fprintf(output_f, "Site %d:\n",i); fprintf(output_diff, "Site %d:\n",i);
					//Note. The output of Dslashd should not have a halo. Whilst xi is defined with one we do not use it here
					//so stride is kvol, not kvolHalo
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output, "%.3f+%.3fI\t",creal(X1[i+j*kvol]),cimag(X1[i+j*kvol]));
						fprintf(output_f, "%.3f+%.3fI\t", creal(X1_f[i+j*kvol]),cimag(X1_f[i+j*kvol]));
						Complex diff = X1[i+j*kvol]-X1_f[i+j*kvol];
						if(fabs(creal(diff))>1e-6 || fabs(cimag(diff))>1e-6){
							fprintf(stderr,"Error %i in %s: Single and double disagree for HbyClover site %i and spinor/color %d. Difference %e+%ei"\
									"\nExiting...\n\n",CONVERR,funcname,i,j,creal(diff),cimag(diff));
							fclose(output);fclose(output_f);fclose(output_diff);
							exit(CONVERR);
						}
						else
							fprintf(output_diff,"%.3f+%.3fI\t", creal(diff),cimag(diff));
					}
					fprintf(output, "\n\n"); fprintf(output_f,"\n\n"); fprintf(output_diff,"\n\n");
				}
				fclose(output);fclose(output_f);fclose(output_diff);
			break;
			case(8): //Filling smallPhi
				memset(smallPhi,0,kferm2*sizeof(Complex));
				Fill_Small_Phi(na,smallPhi,Phi);
				for(unsigned int i = 0; i<kvol;i++)
					for(unsigned short idirac = 0; idirac<ndirac; idirac++)
						for(unsigned short ic= 0; ic<nc; ic++)
							//	  PHI_index=i*16+j*2+k;
							if(cabs(smallPhi[i+kvol*(ic+nc*idirac)]-Phi[i+kvol*(ic+nc*(idirac+ngorkov*na))])>1e-6){
								fprintf(stderr,"Error %i in %s: Failed to fill small phi correctly.\nExiting\n\n.",SPHIERR,funcname);
								exit(SPHIERR);
							}
				break;
			case(9): //Congradq
				memset(X1,0,kferm2Halo*sizeof(Complex));
				itercg=0;
				if(Congradq(0,rescga,X1,smallPhi,ut,ut_f,clover_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,c_sw,&itercg)){
					fprintf(stderr,"Error %i in %s: Congradq failed to converge.\nExiting\n\n",ITERLIM,funcname);
					exit(ITERLIM);
				}
				break;
			case(10):	//Hamilton
				memset(X1,0,kferm2Halo*sizeof(Complex));
				input = fopen("hamiltonian_in", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input, "Site %d:\n",i); 
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(input, "%.3f+%.3fI\t",creal(X1[i+j*kvolHalo]),cimag(X1[i+j*kvolHalo]));
					}
					fprintf(input, "\n\n"); 
				}
				fclose(input);
				double h,s,ancgh;  h=s=ancgh=0;
				Hamilton(&h,&s,rescgg,pp,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,\
						jqq,akappa,beta,c_sw,&ancgh,0);
				output = fopen("hamiltonian_out", "w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output, "Site %d:\n",i); 
					for(unsigned short j=0;j<nc*ndirac;j++){
						fprintf(output, "%.3f+%.3fI\t",creal(X1[i+j*kvolHalo]),cimag(X1[i+j*kvolHalo]));
					}
					fprintf(output, "\n\n"); 
				}
				fclose(output);
				break;
			case(11): //Gauge Force
				memset(dSdpi,0,kmom*sizeof(double));
				input = fopen("Gauge_Force_in","w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(input,"Site %d:\n",i);
					for(unsigned short gen=0;gen<nadj;gen++){
						fprintf(input,"Gen %d:\n",gen);
						for(unsigned int mu=0;mu<ndim;mu++){
							fprintf(input, "%.3f\t", dSdpi[i+kvol*(gen*ndim+mu)]);
						}
						fprintf(input,"\n");
					}
					fprintf(input,"\n");
				}
				fclose(input);	
#ifdef __NVCC__
				//cudaMemPrefetchAsync(dSdpi,kmom*sizeof(double),device,NULL);
#endif
				Gauge_force(dSdpi,ut_f,iu,id,beta);
#ifdef __NVCC__
				cudaDeviceSynchronise();
#endif
				output = fopen("Gauge_Force_out","w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output,"Site %d:\n",i);
					for(unsigned short gen=0;gen<nadj;gen++){
						fprintf(output,"Gen %d:\n",gen);
						for(unsigned int mu=0;mu<ndim;mu++){
							fprintf(output, "%.3f\t", dSdpi[i+kvol*(gen*ndim+mu)]);
						}
						fprintf(output,"\n");
					}
					fprintf(output,"\n");
				}
				fclose(output);	
				break;
				//Two force cases because of the flag. This also tests the conjugate gradient works okay
			case(12):	//Wilson Force
				if(nproc>1){
					fprintf(stderr,"Error %i in %s: MPI force diagnostic not implemented yet.\n\n"\
							"Breaking and moving to next test",NOIMPL,funcname);
					break;
				}
				memset(dSdpi,0,kmom*sizeof(double));
				ComplexConvert(X1_f,X1,kvol,true,nc*ndirac);
				Hdslash_f(X2_f,X1_f,ut_f,iu,id,gamval_f,gamin,dk_f,akappa);

				for(unsigned short mu=0;mu<ndim-1;mu++)
					Force_s(dSdpi,ut_f,X1_f,X2_f,gamval_f,iu,gamin,akappa,mu);
				Force_t(dSdpi,ut_f,X1_f,X2_f,gamval_f,dk_f,iu,gamin,akappa);
				output = fopen("Wilson_Force","w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output,"Site %d:\n",i);
					for(unsigned short gen=0;gen<nadj;gen++){
						fprintf(output,"Gen %d:\n",gen);
						for(unsigned int short mu=0;mu<ndim;mu++){
							fprintf(output, "%.3f\t", dSdpi[i+kvol*(gen*ndim+mu)]);
						}
						fprintf(output,"\n");
					}
					fprintf(output,"\n");
				}
				fclose(output);
				break;
			case(13): //Clover Half Leaves
				if(c_sw==0)
					break;
				unsigned short mu=0; unsigned short nu=1;
				Half_Leaves(hLeaves[mu],ut_f,iu,id,mu,nu);
				Half_Leaves(hLeaves[nu],ut_f,iu,id,nu,mu);
				output=fopen("Half_leaves","w");
				for(unsigned int i=0;i<kvol;i++){
					fprintf(output,"Site %d\n",i);
					fprintf(output,"mu %d nu %d\n",mu,nu);
					unsigned short clov = (mu==0) ? nu-1 :mu+nu;
					fprintf(output,"mu-nu: hLeaf1 %e+i%e hLeaf2 %e+i%e\nnu-mu: hLeaf1 %e+i%e hLeaf2 %e+i%e\n",\
							crealf(hLeaves[mu][0][i+kvol*clov]), cimagf(hLeaves[mu][0][i+kvol*clov]),\
							crealf(hLeaves[mu][1][i+kvol*clov]), cimagf(hLeaves[mu][1][i+kvol*clov]),\
							crealf(hLeaves[nu][0][i+kvol*clov]), cimagf(hLeaves[nu][0][i+kvol*clov]),\
							crealf(hLeaves[nu][1][i+kvol*clov]), cimagf(hLeaves[nu][1][i+kvol*clov]));
					fprintf(output,"\n");
				}
				fclose(output);
				break;
			case(14): //Clover Force
				if(nproc>1){
					fprintf(stderr,"Error %i in %s: MPI clover force not implemented yet.\n\n"\
							"Breaking and moving to next test",NOIMPL,funcname);
					break;
				}
				//Don't test if no clover.
				if(c_sw==0)
					break;
				memset(dSdpi,0,kmom*sizeof(double));
				//Make it easier to keep track of the force. Set pseudeofermion fields to one.
				#pragma omp parallel for simd aligned(X1_f,X2_f:AVX)
				for(unsigned int i=0;i<kferm2Halo;i++){
					X1_f[i]=1; X2_f[i]=1;
				}
				Clover_Force(dSdpi,ut_f,X1_f,X2_f,sigval_f,sigin,iu,id,akappa);
				output = fopen("Clover_Force","w");
				for(unsigned int i = 0; i< kvol; i++){
					fprintf(output,"Site %d:\n",i);
					for(unsigned short gen=0;gen<nadj;gen++){
						fprintf(output,"Gen %d:\n",gen);
						for(unsigned short mu=0;mu<ndim;mu++){
							fprintf(output, "%.3f\t", dSdpi[i+kvol*(gen*ndim+mu)]);
						}
						fprintf(output,"\n");
					}
					fprintf(output,"\n");
				}
				fclose(output);
				break;
			case(15):
				itercg=0;
				if(Congradp(0, respbp, Phi, R1,ut,ut_f,clover_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,c_sw,&itercg)){
					fprintf(stderr,"Error %i in %s: Congradp failed to converge.\nExiting\n\n",ITERLIM,funcname);
					exit(ITERLIM);
				}
				break;

		}
	}
	//George Michael's favourite bit of the code
#ifdef __NVCC__
	//Make a routine that does this for us
	cudaFree(dk[0]); cudaFree(dk[1]); cudaFree(R1); cudaFree(dSdpi); cudaFree(pp);
	cudaFree(Phi); cudaFree(ut[0]); cudaFree(ut[1]);
	cudaFree(clover[0]); cudaFree(clover[1]);
	cudaFree(clover_f[0]); cudaFree(clover_f[1]);
	cudaFree(X0); cudaFree(X1); cudaFree(u[0]); cudaFree(u[1]);
	cudaFree(X0_f); cudaFree(X1_f); cudaFree(ut_f[0]); cudaFree(ut_f[1]);
	cudaFree(X2_f);
	for(unsigned short i=0;i<ndim;i++){
		cudaFree(hLeaves[i][0]); cudaFree(hLeaves[i][1]);
	}
	cudaFree(id); cudaFree(iu); cudaFree(hd); cudaFree(hu);
#else
	free(dk[0]); free(dk[1]); free(R1); free(dSdpi); free(pp);
	free(Phi); free(ut[0]); free(ut[1]); free(xi);
	free(clover[0]); free(clover[1]);
	free(clover_f[0]); free(clover_f[1]);
	for(unsigned short i=0;i<ndim;i++){
		free(hLeaves[i][0]); free(hLeaves[i][1]);
	}
	free(X0); free(X1); free(u[0]); free(u[1]);
	free(X2_f);
	free(id); free(iu); free(hd); free(hu);
	free(pcoord);
#endif

#if(nproc>1)
	MPI_Finalise();
#endif
	exit(0);
}
#endif
