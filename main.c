#include	<assert.h>
#include	<coord.h>
#include	<math.h>
#include	<matrices.h>
#include	<par_mpi.h>
#include	<random.h>
#include	<string.h>
#include	<su2hmc.h>
#ifdef	__NVCC__
#include <cublas_v2.h>
#include	<cuda.h>
#include	<cuda_runtime.h>
cublasHandle_t cublas_handle;
cublasStatus_t cublas_status;
//Fix this later
#endif
/*
 * For the early phases of this translation, I'm going to try and
 * copy the original format as much as possible and keep things
 * in one file. Hopefully this will change as we move through
 * the methods so we can get a more logical structure.
 *
 * Another vestige of the Fortran code that will be implemented here is
 * the frequent flattening of arrays. But while FORTRAN Allows you to write
 * array(i,j) as array(i+M*j) where M is the number of rows, C resorts to 
 * pointers
 *
 * One change I will try and make is the introduction of error-codes (nothing
 * to do with the Irish postal service)
 * These can be found in the file errorcode.h and can help with debugging
 *
 * Lastly, the comment style for the start of a function is based off of 
 * Niall Moran's python style (which may have come from numpy?) It should
 * consist of a description of the function, a list of parameters with a brief
 * explanation and lastly what is returned by the function (on success or failure)
 */
int main(int argc, char *argv[]){
	/*******************************************************************
	 *    Hybrid Monte Carlo algorithm for Two Colour QCD with Wilson-Gor'kov fermions
	 *    based on the algorithm of Duane et al. Phys. Lett. B195 (1987) 216. 
	 *
	 *    There is "up/down partitioning": each update requires
	 *    one operation of congradq on complex*16 vectors to determine
	 *    (Mdagger M)**-1  Phi where Phi has dimension 4*kvol*nc*Nf - 
	 *    The matrix M is the Wilson matrix for a single flavor
	 *    there is no extra species doubling as a result
	 *
	 *    matrix multiplies done using routines hdslash and hdslashd
	 *
	 *    Hence, the number of lattice flavors Nf is related to the
	 *    number of continuum flavors N_f by
	 *                  N_f = 2 * Nf
	 *
	 *    Fermion expectation values are measured using a noisy estimator.
	 *    on the Wilson-Gor'kov matrix, which has dimension 8*kvol*nc*Nf
	 *    inversions done using congradp, and matrix multiplies with dslash,
	 *    dslashd
	 *
	 *    trajectory length is random with mean dt*stepl
	 *    The code runs for a fixed number ntraj of trajectories.
	 *
	 *    Phi: pseudofermion field 
	 *    bmass: bare fermion mass 
	 *    fmu: chemical potential 
	 *    actiona: running average of total action
	 *
	 *    Fermion expectation values are measured using a noisy estimator.
	 *    outputs:
	 *    fort.11   psibarpsi, energy density, baryon density
	 *    fort.12   spatial plaquette, temporal plaquette, Polyakov line
	 *    fort.13   real<qq>, real <qbar qbar>, imag <qq>= imag<qbar qbar>
	 *
	 *                                               SJH March 2005
	 *
	 *     Hybrid code, P.Giudice, May 2013
	 *     Converted from Fortran to C by D. Lawlor March 2021
	 ******************************************************************/
	//Instead of hard coding the function name so the error messages are easier to implement
	const char *funcname = "main";

	Par_begin(argc, argv);
	//Add error catching code...
#if(nproc>1)
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
#endif

	//Input
	//The default values here are straight from the FORTRAN
	//=====================================================
	float beta = 1.7;
	float akappa = 0.1780f;
	Complex_f jqq = 0;
	float fmu = 0.0;
	int iread = 0;
	int istart = 1;
	int ibound = -1;
	int iwrite = 1;
	int iprint = 1; //How often are measurements made
	int icheck = 5; //How often are configurations saved
#ifdef USE_MATH_DEFINES
	const double tpi = 2*M_PI;
#else
	const double tpi = 2*acos(-1.0);
#endif
	float dt=0.004; float ajq = 0.0;
	float delb; //Not used?
	float athq = 0.0;
	int stepl = 250; int ntraj = 10;
	//rank is zero means it must be the "master process"
	if(!rank){
		FILE *midout;
		const char *filename = (argc!=2) ?"midout":argv[1];
		char *fileop = "r";
		if( !(midout = fopen(filename, fileop) ) ){
			fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n"\
					, OPENERROR, funcname, filename, fileop);
#if(nproc>1)
			MPI_Abort(comm,OPENERROR);
#else
			exit(OPENERROR);
#endif
		}
		//See the README for what each entry means
		fscanf(midout, "%f %f %f %f %f %f %f %d %d %d %d %d", &dt, &beta, &akappa,\
				&ajq, &athq, &fmu, &delb, &stepl, &ntraj, &istart, &icheck, &iread);
		fclose(midout);
		assert(stepl>0);	assert(ntraj>0);	  assert(istart>0);  assert(icheck>0);  assert(iread>=0); 
	}
	//Send inputs to other ranks
#if(nproc>1)
	Par_fcopy(&dt); Par_fcopy(&beta); Par_fcopy(&akappa); Par_fcopy(&ajq);
	Par_fcopy(&athq); Par_fcopy(&fmu); Par_fcopy(&delb); //Not used?
	Par_icopy(&stepl); Par_icopy(&ntraj); Par_icopy(&istart); Par_icopy(&icheck);
	Par_icopy(&iread); jqq=ajq*cexp(athq*I);
#endif
	//End of input
	//For CUDA code, device only variables are needed
#ifdef __NVCC__
	//CUBLAS Handle
	cublasCreate(&cublas_handle);
#endif
#ifdef _DEBUG
	printf("jqq=%f+(%f)I\n",creal(jqq),cimag(jqq));
#endif
#ifdef _DEBUG
	seed = 967580161;
#else
	seed = time(NULL);
#endif

	//Gauge, trial and momentum fields 
	//You'll notice that there are three different allocation/free statements
	//One for CUDA, one for MKL and one for everything else depending on what's
	//being used
	Complex *u11, *u12, *u11t, *u12t;
	Complex_f *u11t_f, *u12t_f;
	double *dk4m, *dk4p, *pp;
	float	*dk4m_f, *dk4p_f;
	//Halo index arrays
	unsigned int *iu, *id;
#ifdef __NVCC__
	cudaMallocManaged((void**)&iu,ndim*kvol*sizeof(int),cudaMemAttachGlobal);
	cudaMallocManaged((void**)&id,ndim*kvol*sizeof(int),cudaMemAttachGlobal);

	cudaMallocManaged(&dk4m,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4p,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4m_f,(kvol+halo)*sizeof(float),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4p_f,(kvol+halo)*sizeof(float),cudaMemAttachGlobal);

	int	*gamin;
	Complex	*gamval;
	Complex_f *gamval_f;
	cudaMallocManaged(&gamin,4*4*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&gamval,5*4*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&gamval_f,5*4*sizeof(Complex_f),cudaMemAttachGlobal);

	cudaMallocManaged(&u11,ndim*kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12,ndim*kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t_f,ndim*(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t_f,ndim*(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
	id = (unsigned int*)mkl_malloc(ndim*kvol*sizeof(int),AVX);
	iu = (unsigned int*)mkl_malloc(ndim*kvol*sizeof(int),AVX);

	int	*gamin = (int *)mkl_malloc(4*4*sizeof(int),AVX);
	Complex	*gamval=(Complex *)mkl_malloc(5*4*sizeof(Complex),AVX);
	Complex_f *gamval_f=(Complex_f *)mkl_malloc(5*4*sizeof(Complex_f),AVX);;

	dk4m = (double *)mkl_malloc((kvol+halo)*sizeof(double), AVX);
	dk4p = (double *)mkl_malloc((kvol+halo)*sizeof(double), AVX);
	dk4m_f = (float *)mkl_malloc((kvol+halo)*sizeof(float), AVX);
	dk4p_f = (float *)mkl_malloc((kvol+halo)*sizeof(float), AVX);

	u11 = (Complex *)mkl_malloc(ndim*kvol*sizeof(Complex),AVX);
	u12 = (Complex *)mkl_malloc(ndim*kvol*sizeof(Complex),AVX);
	u11t = (Complex *)mkl_malloc(ndim*(kvol+halo)*sizeof(Complex),AVX);
	u12t = (Complex *)mkl_malloc(ndim*(kvol+halo)*sizeof(Complex),AVX);
	u11t_f = (Complex_f *)mkl_malloc(ndim*(kvol+halo)*sizeof(Complex_f),AVX);
	u12t_f = (Complex_f *)mkl_malloc(ndim*(kvol+halo)*sizeof(Complex_f),AVX);
#else
	id = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));
	iu = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));

	dk4m = (double *)aligned_alloc(AVX,(kvol+halo)*sizeof(double));
	dk4p = (double *)aligned_alloc(AVX,(kvol+halo)*sizeof(double));
	dk4m_f = (float *)aligned_alloc(AVX,(kvol+halo)*sizeof(float));
	dk4p_f = (float *)aligned_alloc(AVX,(kvol+halo)*sizeof(float));

	u11 = (Complex *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex));
	u12 = (Complex *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex));
	u11t = (Complex *)aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex));
	u12t = (Complex *)aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex));
	u11t_f = (Complex_f *)aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex_f));
	u12t_f = (Complex_f *)aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex_f));
#endif
	//Initialisation
	//istart < 0: Start from tape in FORTRAN?!? How old was this code?
	//istart = 0: Ordered/Cold Start
	//			For some reason this leaves the trial fields as zero in the FORTRAN code?
	//istart > 0: Random/Hot Start
	Init(istart,ibound,iread,beta,fmu,akappa,ajq,u11,u12,u11t,u12t,u11t_f,u12t_f,gamval,gamval_f,gamin,dk4m,dk4p,dk4m_f,dk4p_f,iu,id);
#ifdef __NVCC__
	//GPU Initialisation stuff
	Init_CUDA(u11t,u12t,u11t_f,u12t_f,gamval,gamval_f,gamin,\
			dk4m,dk4p,dk4m_f,dk4p_f,iu,id);//&dimBlock,&dimGrid);
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
#ifdef DIAGNOSTIC
	double ancg_diag=0;
	Diagnostics(istart, u11, u12, u11t, u12t, u11t_f, u12t_f, iu, id, hu, hd, dk4m, dk4p,\
			dk4m_f, dk4p_f, gamin, gamval, gamval_f, jqq, akappa, beta, ancg_diag);
#endif

	//Initial Measurements
	//====================
	double poly = Polyakov(u11t,u12t);
#ifdef _DEBUG
	if(!rank) printf("Initial Polyakov loop evaluated as %e\n", poly);
#endif
	double hg, avplaqs, avplaqt;
	//Halo exchange of the trial fields
	Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
	Average_Plaquette(&hg,&avplaqs,&avplaqt,u11t,u12t,iu,beta);
	//Trajectory length
	double traj=stepl*dt;
	//Acceptance probability
	double proby = 2.5/stepl;
	char suffix[FILELEN]="";
	int buffer; char buff2[7];
	//Add script for extracting correct mu, j etc.
	buffer = (int)round(100*beta);
	sprintf(buff2,"b%03d",buffer);
	strcat(suffix,buff2);
	//κ
	buffer = (int)round(10000*akappa);
	sprintf(buff2,"k%04d",buffer);
	strcat(suffix,buff2);
	//μ
	buffer = (int)round(1000*fmu);
	sprintf(buff2,"mu%04d",buffer);
	strcat(suffix,buff2);
	//J
	buffer = (int)round(1000*ajq);
	sprintf(buff2,"j%03d",buffer);
	strcat(suffix,buff2);
	//nx
	sprintf(buff2,"s%02d",nx);
	strcat(suffix,buff2);
	//nt
	sprintf(buff2,"t%02d",nt);
	strcat(suffix,buff2);
	char outname[FILELEN] = "Output."; char *outop="w";
	strcat(outname,suffix);
	FILE *output;
	if(!rank){
		if(!(output=fopen(outname, outop) )){
			fprintf(stderr,"Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",OPENERROR,funcname,outname,outop);
#if(nproc>1)
			MPI_Abort(comm,OPENERROR);
#else
			exit(OPENERROR);
#endif
		}
		printf("hg = %e, <Ps> = %e, <Pt> = %e, <Poly> = %e\n", hg, avplaqs, avplaqt, poly);
		fprintf(output, "ksize = %i ksizet = %i Nf = %i Halo =%i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i β = %e\nκ = %e μ = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %ld\n",
				ksize, ksizet, nf, halo, dt, traj, ntraj, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#ifdef _DEBUG
		//Print to terminal during debugging
		printf("ksize = %i ksizet = %i Nf = %i Halo = %i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i β = %e\nκ = %e μ = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %ld\n",
				ksize, ksizet, nf, halo, dt, traj, ntraj, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#endif
	}
	//Initialise for averages
	//======================
	double actiona = 0.0; double vel2a = 0.0; double pbpa = 0.0; double endenfa = 0.0; double denfa = 0.0;
	double yav = 0.0; double yyav = 0.0; 

	int naccp = 0; int ipbp = 0; int itot = 0;

	//This was originally in the half-step of the FORTRAN code, but it makes more sense to declare
	//it outside the loop. Since it's always being subtracted we'll define it as negative
	const	double d = -dt*0.5;
	//Start of classical evolution
	//===========================
	double pbp;
	Complex qq;
	double *dSdpi;
	//Field and related declarations
	Complex *Phi, *R1, *X0, *X1;
	//Initialise Arrays. Leaving it late for scoping
	//check the sizes in sizes.h
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);

	cudaMallocManaged(&R1, kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi, nf*kferm*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X0, nf*kferm2*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X1, kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&pp, kmom*sizeof(double),cudaMemAttachGlobal);
	cudaMalloc(&dSdpi, kmom*sizeof(double));
#elif defined __INTEL_MKL__
	R1= mkl_malloc(kfermHalo*sizeof(Complex),AVX);
	Phi= mkl_malloc(nf*kferm*sizeof(Complex),AVX); 
	X0= mkl_malloc(nf*kferm2*sizeof(Complex),AVX); 
	X1= mkl_malloc(kferm2Halo*sizeof(Complex),AVX); 
	dSdpi = mkl_malloc(kmom*sizeof(double), AVX);
	//pp is the momentum field
	pp = mkl_malloc(kmom*sizeof(double), AVX);
#else
	R1= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Phi= aligned_alloc(AVX,nf*kferm*sizeof(Complex)); 
	X0= aligned_alloc(AVX,nf*kferm2*sizeof(Complex)); 
	X1= aligned_alloc(AVX,kferm2Halo*sizeof(Complex)); 
	dSdpi = aligned_alloc(AVX,kmom*sizeof(double));
	pp = aligned_alloc(AVX,kmom*sizeof(double));
#endif
	//#pragma omp target enter data map(alloc:pp[0:kmom],dSdpi[0:kmom],X1[0:kferm2Halo]) nowait
	//For offloaded OpenACC Code, make device versions of arrays
#pragma acc enter data create(pp[0:kmom],dSdpi[0:kmom],X1[0:kferm2Halo])
	//Arabic for hour/watch so probably not defined elsewhere like TIME potentially is
#if (defined SA3AT)
	double start_time=0;
	if(!rank){
#if(nproc>1)
		start_time = MPI_Wtime();
#else
		start_time = omp_get_wtime();
#endif
	}
#endif
	double action;
	//Conjugate Gradient iteration counters
	double ancg,ancgh,totancg,totancgh=0;
	for(int itraj = iread+1; itraj <= ntraj+iread; itraj++){
		//Reset conjugate gradient averages
		ancg = 0; ancgh = 0;
#ifdef _DEBUG
		if(!rank)
			printf("Starting itraj %i\n", itraj);
#endif
		for(int na=0; na<nf; na++){
			//Probably makes sense to declare this outside the loop
			//but I do like scoping/don't want to break anything else just teat
			//
			//How do we optimise this for use in CUDA? Do we use CUDA's PRNG
			//or stick with MKL and synchronise/copy over the array
#ifdef __NVCC__
			Complex_f *R,*R1_f;
			cudaMallocManaged(&R,kfermHalo*sizeof(Complex_f),cudaMemAttachGlobal);
			cudaMallocManaged(&R1_f,kferm*sizeof(Complex_f),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
			Complex_f *R=mkl_malloc(kfermHalo*sizeof(Complex_f),AVX);
			Complex_f *R1_f=mkl_malloc(kferm*sizeof(Complex_f),AVX);
#else
			Complex *R=aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
			Complex *R1_f=aligned_alloc(AVX,kferm*sizeof(Complex_f));
#endif
			//Multiply the dimension of R by 2 because R is complex
			//The FORTRAN code had two Gaussian routines.
			//gaussp was the normal Box-Muller and gauss0 didn't have 2 inside the square root
			//Using σ=1/sqrt(2) in these routines has the same effect as gauss0
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
			Gauss_c(R, kferm, 0, 1/sqrt(2));
#else
			vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R, 0, 1/sqrt(2));
#endif
#ifdef __NVCC__
			cudaMemPrefetchAsync(R,kfermHalo*sizeof(Complex_f),device,NULL);
#endif
			Dslashd_f(R1_f, R,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
			for(int i=0;i<kferm;i++)
				R1[i]=R1_f[i];
			memcpy(Phi+na*kferm,R1, kferm*sizeof(Complex));
			//Up/down partitioning (using only pseudofermions of flavour 1)
#pragma omp parallel for simd collapse(2) aligned(X0,R1:AVX)
			for(int i=0; i<kvol; i++)
				for(int idirac = 0; idirac < ndirac; idirac++){
					X0[((na*kvol+i)*ndirac+idirac)*nc]=R1[(i*ngorkov+idirac)*nc];
					X0[((na*kvol+i)*ndirac+idirac)*nc+1]=R1[(i*ngorkov+idirac)*nc+1];
				}
#ifdef __NVCC__
			cudaFree(R);cudaFree(R1_f);
#elif defined __INTEL_MKL__
			mkl_free(R); mkl_free(R1_f);
#else
			free(R); free(R1_f);
#endif
		}	
		//Heatbath
		//========
		//We're going to make the most of the new Gauss_d routine to send a flattened array
		//and do this all in one step.
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
		Gauss_d(pp, kmom, 0, 1);
#else
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, kmom, pp, 0, 1);
#endif
#pragma acc update device(pp[0:kmom])
		//Initialise Trial Fields
		memcpy(u11t, u11, ndim*kvol*sizeof(Complex));
		memcpy(u12t, u12, ndim*kvol*sizeof(Complex));
		Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
#ifdef __NVCC__
		cudaMemPrefetchAsync(u11t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(u12t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(u11t_f, ndim*(kvol+halo)*sizeof(Complex_f),device,NULL);
		cudaMemPrefetchAsync(u12t_f, ndim*(kvol+halo)*sizeof(Complex_f),device,NULL);
#endif
		double H0, S0;
		Hamilton(&H0, &S0, rescga,pp,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval_f,gamin,\
				dk4m_f,dk4p_f,jqq,akappa,beta,&ancgh);
#ifdef _DEBUG
		if(!rank) printf("H0: %e S0: %e\n", H0, S0);
#endif
		if(itraj==1)
			action = S0/gvol;

		//Half step forward for p
		//=======================
#ifdef _DEBUG
		printf("Evaluating force on rank %i\n", rank);
#endif
		Force(dSdpi, 1, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
				dk4m_f,dk4p_f,jqq,akappa,beta,&ancg);
#ifdef __NVCC__
		cublasDaxpy(cublas_handle,kmom, &d, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
		cblas_daxpy(kmom, d, dSdpi, 1, pp, 1);
#else
		for(int i=0;i<kmom;i++)
			//d negated above
			pp[i]+=d*dSdpi[i];
#endif
		//Main loop for classical time evolution
		//======================================
		for(int step = 1; step<=stepmax; step++){
#ifdef __NVCC__
			cudaDeviceSynchronise();
#endif
#ifdef _DEBUG
			if(!rank)
				printf("Traj: %d\tStep: %d\n", itraj, step);
#endif
			//The FORTRAN redefines d=dt here, which makes sense if you have a limited line length.
			//I'll stick to using dt though.
			//step (i) st(t+dt)=st(t)+p(t+dt/2)*dt;
			//Note that we are moving from kernel to kernel within the default streams so don't need a Device_Sync here
			New_trial(dt,pp,u11t,u12t);
			Reunitarise(u11t,u12t);
			//Get trial fields from accelerator for halo exchange
			//Cancel that until we check for double precision flags. It's really bad on Xe since it isn't natively supported
#pragma acc update self(u11t[0:ndim*kvol],u12t[0:ndim*kvol])
			Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
			//Mark trial fields as primarily read only here? Can re-enable writing at the end of each trajectory

			//p(t+3et/2)=p(t+dt/2)-dSds(t+dt)*dt
			//	Force(dSdpi, 0, rescgg);
			Force(dSdpi, 0, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
					dk4m_f,dk4p_f,jqq,akappa,beta,&ancg);
			//The same for loop is given in both the if and else
			//statement but only the value of d changes. This is due to the break in the if part
			if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
#ifdef __NVCC__
				cublasDaxpy(cublas_handle,kmom, &d, dSdpi, 1, pp, 1);
				cudaDeviceSynchronise();
#elif defined USE_BLAS
				cblas_daxpy(kmom, d, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd aligned(pp,dSdpi:AVX)
				for(int i = 0; i<kmom; i++)
					//d negated above
					pp[i]+=d*dSdpi[i];
#endif
#pragma acc update device(pp[0:kmom]) 
				itot+=step;
				break;
				ancg/=step;
				totancg+=ancg;
			}
			else{
#ifdef __NVCC__
				//dt is needed for the trial fields so has to be negated every time.
				double dt_d=-1*dt;
				cublasDaxpy(cublas_handle,kmom, &dt_d, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
				cblas_daxpy(kmom, -dt, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd aligned(pp,dSdpi:AVX)
				for(int i = 0; i<kmom; i++)
					pp[i]-=dt*dSdpi[i];
#endif
#pragma acc update device(pp[0:kmom]) 
			}
		}
		//Monte Carlo step: Accept new fields with the probability of min(1,exp(H0-X0))
		//Kernel Call needed here?
		Reunitarise(u11t,u12t);
#pragma acc update self(u11t[0:kvol*ndim],u12t[0:kvol*ndim])
		double H1, S1;
		Hamilton(&H1, &S1, rescga,pp,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval_f,gamin,\
				dk4m_f,dk4p_f,jqq,akappa,beta,&ancgh);
		totancgh+=ancgh;
#ifdef _DEBUG
		printf("H0-H1=%f-%f",H0,H1);
#endif
		double dH = H0 - H1;
#ifdef _DEBUG
		printf("=%f\n",dH);
#endif
		double dS = S0 - S1;
		if(!rank){
			fprintf(output, "dH = %e dS = %e\n", dH, dS);
#ifdef _DEBUG
			printf("dH = %e dS = %e\n", dH, dS);
#endif
		}
		double y = exp(dH);
		yav+=y;
		yyav+=y*y;
		//The Monte-Carlo
		//Always update  dH is positive (gone from higher to lower energy)
		if(dH>0 || Par_granf()<=y){
			//Step is accepted. Set s=st
			if(!rank)
				printf("New configuration accepted on trajectory %i.\n", itraj);
			//Original FORTRAN Comment:
			//JIS 20100525: write config here to preempt troubles during measurement!
			//JIS 20100525: remove when all is ok....
			memcpy(u11,u11t,ndim*kvol*sizeof(Complex));
			memcpy(u12,u12t,ndim*kvol*sizeof(Complex));
			naccp++;
			//Divide by gvol since we've summed over all lattice sites
			action=S1/gvol;
		}
		else
			if(!rank)
				printf("New configuration rejected on trajectory %i.\n", itraj);
		actiona+=action; 
		double vel2=0.0;
#ifdef __NVCC__
		cublasDnrm2(cublas_handle,kmom, pp, 1,&vel2);
		cudaDeviceSynchronise();
		vel2*=vel2;
#elif defined USE_BLAS
		vel2 = cblas_dnrm2(kmom, pp, 1);
		vel2*=vel2;
#else
#pragma unroll
		for(int i=0; i<kmom; i++)
			vel2+=pp[i]*pp[i];
#endif
#if(nproc>1)
		Par_dsum(&vel2);
#endif
		vel2a+=vel2/(ndim*nadj*gvol);

		if(itraj%iprint==0){
			//If rejected, copy the previously accepted field in for measurements
			memcpy(u11t, u11, ndim*kvol*sizeof(Complex));
			memcpy(u12t, u12, ndim*kvol*sizeof(Complex));
			Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
#ifdef _DEBUG
			if(!rank)
				printf("Starting measurements\n");
#endif
			int itercg=0;
			double endenf, denf;
			Complex qbqb;
			//Stop gap for measurement failure on Kay;
			//If the Congrad in Measure fails, don't measure the Diquark or PBP-Density observables for
			//that trajectory
			int measure_check=0;
			measure_check = Measure(&pbp,&endenf,&denf,&qq,&qbqb,respbp,&itercg,u11t,u12t,u11t_f,u12t_f,iu,id,\
					gamval,gamval_f,gamin,dk4m,dk4p,dk4m_f,dk4p_f,jqq,akappa,Phi,R1);
#ifdef _DEBUG
			if(!rank)
				printf("Finished measurements\n");
#endif
			pbpa+=pbp; endenfa+=endenf; denfa+=denf; ipbp++;
			Average_Plaquette(&hg,&avplaqs,&avplaqt,u11t,u12t,iu,beta);
			poly = Polyakov(u11t,u12t);
			//We have four output files, so may as well get the other ranks to help out
			//and abuse scoping rules while we're at it.
			//Can use either OpenMP or MPI to do this
#if (nproc>=4)
			switch(rank)
#else
				if(!rank)
#pragma omp parallel for
					for(int i=0; i<4; i++)
						switch(i)
#endif
						{
							case(0):	
								//Output code... Some files weren't opened in the main loop of the FORTRAN code 
								//That will need to be looked into for the C version
								//It would explain the weird names like fort.1X that looked like they were somehow
								//FORTRAN related...
								//Not yet implemented
								fprintf(output, "Iter (CG) %i ancg %.1f ancgh %.1f\n", itercg, ancg, ancgh);
								fflush(output);
								break;
							case(1):
								{
									FILE *fortout;
									char fortname[FILELEN] = "fermi.";
									strcat(fortname,suffix);
									const char *fortop= (itraj==1) ? "w" : "a";
									if(!measure_check){
										if(!(fortout=fopen(fortname, fortop) )){
											fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
													OPENERROR, funcname, fortname, fortop);
#if(nproc>1)
											MPI_Abort(comm,OPENERROR);
#else
											exit(OPENERROR);
#endif
										}
										if(itraj==1)
											fprintf(fortout, "pbp\tendenf\tdenf\n");
										fprintf(fortout, "%e\t%e\t%e\n", pbp, endenf, denf);
										fclose(fortout);
										break;
									}
								}
							case(2):
								//The original code implicitly created these files with the name
								//fort.XX where XX is the file label
								//from FORTRAN. This was fort.12
								{
									FILE *fortout;
									char fortname[FILELEN] = "bose."; 
									strcat(fortname,suffix);
									const char *fortop= (itraj==1) ? "w" : "a";
									if(!(fortout=fopen(fortname, fortop) )){
										fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
												OPENERROR, funcname, fortname, fortop);
									}
									if(itraj==1)
										fprintf(fortout, "avplaqs\tavplaqt\tpoly\n");
									fprintf(fortout, "%e\t%e\t%e\n", avplaqs, avplaqt, poly);
									fclose(fortout);
									break;
								}
							case(3):
								{
									if(!measure_check){
										FILE *fortout;
										char fortname[FILELEN] = "diq.";
										strcat(fortname,suffix);
										const char *fortop= (itraj==1) ? "w" : "a";
										if(!(fortout=fopen(fortname, fortop) )){
											fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
													OPENERROR, funcname, fortname, fortop);
#if(nproc>1)
											MPI_Abort(comm,OPENERROR);
#else
											exit(OPENERROR);
#endif
										}
										if(itraj==1)
											fprintf(fortout, "Re(qq)\n");
										fprintf(fortout, "%e\n", creal(qq));
										fclose(fortout);
										break;
									}
								}
							default: break;
						}
		}
		if(itraj%icheck==0){
			Par_swrite(itraj,icheck,beta,fmu,akappa,ajq,u11,u12);
		}
		if(!rank)
			fflush(output);
	}
#if (defined SA3AT)
	double elapsed = 0;
	if(!rank){
#if(nproc>1)
		elapsed = MPI_Wtime()-start_time;
#else
		elapsed = omp_get_wtime()-start_time;
#endif
	}
#endif
	//End of main loop
	//Free arrays
#ifdef _OPENACC
#pragma acc exit data delete(u11t[0:ndim*(kvol+halo)],u12t[0:ndim*(kvol+halo)],\
		u11t_f[0:ndim*(kvol+halo)],u12t_f[0:ndim*(kvol+halo)], dk4m[0:kvol*halo],\
		dk4p[0:kvol+halo], dk4m_f[0:kvol+halo],dk4p_f[0:kvol+halo], iu[0:ndim*kvol],id[0:ndim*kvol],\
		pp[0:kmom],dSdpi[0:kmom],X1[0:kferm2Halo])
#else
#pragma omp target exit data map(delete:u11t[0:ndim*(kvol+halo)],u12t[0:ndim*(kvol+halo)],\
		u11t_f[0:ndim*(kvol+halo)],u12t_f[0:ndim*(kvol+halo)], dk4m[0:kvol*halo],\
		dk4p[0:kvol+halo], dk4m_f[0:kvol+halo],dk4p_f[0:kvol+halo], iu[0:ndim*kvol],id[0:ndim*kvol])
#endif
#ifdef __NVCC__
	//Make a routine that does this for us
	cudaFree(dk4m); cudaFree(dk4p); cudaFree(R1); cudaFree(dSdpi); cudaFree(pp);
	cudaFree(Phi); cudaFree(u11t); cudaFree(u12t);
	cudaFree(X0); cudaFree(X1); cudaFree(u11); cudaFree(u12);
	cudaFree(id); cudaFree(iu); cudaFree(hd); cudaFree(hu);
	cudaFree(dk4m_f); cudaFree(dk4p_f); cudaFree(u11t_f); cudaFree(u12t_f);
	cublasDestroy(cublas_handle);
#elif defined __INTEL_MKL__
	mkl_free_buffers();
	mkl_free(dk4m); mkl_free(dk4p); mkl_free(R1); mkl_free(dSdpi); mkl_free(pp);
	mkl_free(Phi); mkl_free(u11t); mkl_free(u12t);
	mkl_free(X0); mkl_free(X1); mkl_free(u11); mkl_free(u12);
	mkl_free(id); mkl_free(iu); mkl_free(hd); mkl_free(hu);
	mkl_free(dk4m_f); mkl_free(dk4p_f); mkl_free(u11t_f); mkl_free(u12t_f);
	mkl_free(h1u); mkl_free(h1d); mkl_free(halosize);
	mkl_free(pcoord);	mkl_free_buffers();
#if (!defined  __RANLUX__&&!defined USE_RAN2)
	vslDeleteStream(&stream);
#endif
#else
	free(dk4m); free(dk4p); free(R1); free(dSdpi); free(pp); free(Phi);
	free(u11t); free(u12t); free(X0); free(X1);
	free(u11); free(u12); free(id); free(iu); free(hd); free(hu);
	free(dk4m_f); free(dk4p_f); free(u11t_f); free(u12t_f);
	free(h1u); free(h1d); free(halosize);
	free(pcoord);
#endif
#if (defined SA3AT)
	if(!rank){
		FILE *sa3at = fopen("Bench_times.csv", "a");
		fprintf(sa3at, "%s\nβ%0.3f κ:%0.4f μ:%0.4f j:%0.3f s:%i t:%i kvol:%ld\n"
				"npx:%i npt:%i nthread:%i ncore:%i time:%f traj_time:%f\n\n",\
				__VERSION__,beta,akappa,fmu,ajq,nx,nt,kvol,npx,npt,nthreads,npx*npy*npz*npt*nthreads,elapsed,elapsed/ntraj);
		fclose(sa3at);
	}
#endif
	//Get averages for final output
	actiona/=ntraj; vel2a/=ntraj; pbpa/=ipbp; endenfa/=ipbp; denfa/=ipbp;
	totancg/=ntraj; totancgh/=ntraj; yav/=ntraj; yyav=sqrt((yyav/ntraj - yav*yav)/(ntraj-1));
	double atraj=dt*itot/ntraj;

	if(!rank){
		fprintf(output, "Averages for the last %i trajectories\n"\
				"Number of acceptances: %i Average Trajectory Length = %e\n"\
				"<exp(dh)> = %e +/- %e\n"\
				"Average number of congrad iter guidance: %f acceptance %f\n"\
				"psibarpsi = %e\n"\
				"Mean Square Velocity = %e Action Per Site = %e\n"\
				"Energy Density = %e Number Density %e\n",\
				ntraj, naccp, atraj, yav, yyav, totancg, totancgh, pbpa, vel2a, actiona, endenfa, denfa);
		fclose(output);
	}
#if(nproc>1)
	MPI_Finalise();
#endif
	fflush(stdout);
	return 0;
}
