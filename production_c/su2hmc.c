#include	<assert.h>
#include	<coord.h>
#ifdef	__NVCC__
#include	<cuda.h>
#include	<cuda_runtime.h>
cublasHandle_t cublas_handle;
cublasHandle_t cublas_status;
//Fix this later
#endif
#include	<math.h>
#include	<matrices.h>
#include	<par_mpi.h>
#include	<random.h>
#include	<string.h>
#include	<su2hmc.h>

//Extern definitions
//Some of these are only required for GPUs, hence the __NVCC__ preprocessor statements
#ifdef __NVCC__
dim3	dimBlock(ksizex,ksizey,1);
dim3	dimGrid(ksizez,ksizet,1);
#endif
int
#ifndef __NVCC__ 
__attribute__((aligned(AVX)))
#endif
	gamin[4][4] =	{{3,2,1,0},
		{3,2,1,0},
		{2,3,0,1},
		{2,3,0,1}};
//Gamma Matrices in Chiral Representation
//Gattringer and Lang have a nice crash course in appendix A.2 of
//Quantum Chromodynamics on the Lattice (530.14 GAT)
Complex
#ifndef __NVCC__ 
__attribute__((aligned(AVX)))
#endif
	gamval[5][4] =	{{-I,-I,I,I},
		{-1,1,1,-1},
		{-I,I,I,-I},
		{1,1,1,1},
		{1,1,-1,-1}};
Complex_f 
#ifndef __NVCC__ 
__attribute__((aligned(AVX)))
#endif
	gamval_f[5][4] =	{{-I,-I,I,I},
		{-1,1,1,-1},
		{-I,I,I,-I},
		{1,1,1,1},
		{1,1,-1,-1}};

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
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

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
			MPI_Abort(comm,OPENERROR);
		}
		//See the README for what each entry means
		fscanf(midout, "%f %f %f %f %f %f %f %d %d %d %d %d", &dt, &beta, &akappa,\
				&ajq, &athq, &fmu, &delb, &stepl, &ntraj, &istart, &icheck, &iread);
		fclose(midout);
		assert(stepl>0);	assert(ntraj>0);	  assert(istart>0);  assert(icheck>0);  assert(iread>=0); 
	}
	//Send inputs to other ranks
	Par_fcopy(&dt); Par_fcopy(&beta); Par_fcopy(&akappa); Par_fcopy(&ajq);
	Par_fcopy(&athq); Par_fcopy(&fmu); Par_fcopy(&delb); //Not used?
	Par_icopy(&stepl); Par_icopy(&ntraj); Par_icopy(&istart); Par_icopy(&icheck);
	Par_icopy(&iread); jqq=ajq*cexp(athq*I);
	//End of input
	//For CUDA code, device only variables are needed
#ifdef __NVCC__
	Complex *jqq_d, *beta_d, *akappa_d;
	cudaMalloc(&jqq_d,sizeof(Complex));		cudaMalloc(&beta_d,sizeof(Complex));
	cudaMalloc(&akappa_d,sizeof(Complex));	

	cudaMemcpy(jqq_d,&jqq,sizeof(Complex),cudaMemcpyHostToDevice);
	cudaMemcpy(beta_d,&beta,sizeof(Complex),cudaMemcpyHostToDevice);
	cudaMemcpy(akappa_d,&akappa,sizeof(Complex),cudaMemcpyHostToDevice);
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

	int	*gamin_d;
	Complex	*gamval_d;
	Complex_f *gamval_f_d;
	cudaMalloc(&gamin_d,4*4*sizeof(Complex));
	cudaMalloc(&gamval_d,5*4*sizeof(Complex));
	cudaMalloc(&gamval_f_d,5*4*sizeof(Complex_f));

	cudaMallocManaged(&u11,ndim*kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12,ndim*kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t_f,ndim*(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t_f,ndim*(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
	id = (unsigned int*)mkl_malloc(ndim*kvol*sizeof(int),AVX);
	iu = (unsigned int*)mkl_malloc(ndim*kvol*sizeof(int),AVX);

	dk4m = mkl_malloc((kvol+halo)*sizeof(double), AVX);
	dk4p = mkl_malloc((kvol+halo)*sizeof(double), AVX);
	dk4m_f = mkl_malloc((kvol+halo)*sizeof(float), AVX);
	dk4p_f = mkl_malloc((kvol+halo)*sizeof(float), AVX);

	u11 = mkl_malloc(ndim*kvol*sizeof(Complex),AVX);
	u12 = mkl_malloc(ndim*kvol*sizeof(Complex),AVX);
	u11t = mkl_malloc(ndim*(kvol+halo)*sizeof(Complex),AVX);
	u12t = mkl_malloc(ndim*(kvol+halo)*sizeof(Complex),AVX);
	u11t_f = mkl_malloc(ndim*(kvol+halo)*sizeof(Complex_f),AVX);
	u12t_f = mkl_malloc(ndim*(kvol+halo)*sizeof(Complex_f),AVX);
#else
	id = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));
	iu = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));

	dk4m = aligned_alloc(AVX,(kvol+halo)*sizeof(double));
	dk4p = aligned_alloc(AVX,(kvol+halo)*sizeof(double));
	dk4m_f = aligned_alloc(AVX,(kvol+halo)*sizeof(float));
	dk4p_f = aligned_alloc(AVX,(kvol+halo)*sizeof(float));

	u11 = aligned_alloc(AVX,ndim*kvol*sizeof(Complex));
	u12 = aligned_alloc(AVX,ndim*kvol*sizeof(Complex));
	u11t = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex));
	u12t = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex));
	u11t_f = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex_f));
	u12t_f = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex_f));
#endif
	//Initialisation
	//istart < 0: Start from tape in FORTRAN?!? How old was this code?
	//istart = 0: Ordered/Cold Start
	//			For some reason this leaves the trial fields as zero in the FORTRAN code?
	//istart > 0: Random/Hot Start
	Init(istart,ibound,iread,beta,fmu,akappa,ajq,u11,u12,u11t,u12t,u11t_f,u12t_f,dk4m,dk4p,dk4m_f,dk4p_f,iu,id);
#ifdef DIAGNOSTIC
	Diagnostics(istart);
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
	//??
	buffer = (int)round(10000*akappa);
	sprintf(buff2,"k%04d",buffer);
	strcat(suffix,buff2);
	//??
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
			MPI_Abort(comm,OPENERROR);
		}
		printf("hg = %e, <Ps> = %e, <Pt> = %e, <Poly> = %e\n", hg, avplaqs, avplaqt, poly);
		fprintf(output, "ksize = %i ksizet = %i Nf = %i Halo =%i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i ?? = %e\n?? = %e ?? = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %i\n",
				ksize, ksizet, nf, halo, dt, traj, ntraj, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#ifdef _DEBUG
		//Print to terminal during debugging
		printf("ksize = %i ksizet = %i Nf = %i Halo = %i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i ?? = %e\n?? = %e ?? = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %i\n",
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
	memcpy(gamin_d,gamin,4*4*sizeof(int));
	gamval_d=NULL;
	cudaMemcpy(gamval_d,gamval,5*4*sizeof(Complex),cudaMemcpyHostToDevice);
	cudaMemAdvise(gamin_d,16*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemPrefetchAsync(gamin_d,16*sizeof(int),device,NULL);

	cudaMemcpy(gamval_f_d,gamval_f,5*4*sizeof(Complex_f),cudaMemcpyHostToDevice);
	cudaMallocManaged(&R1, kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi, nf*kferm*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X0, nf*kferm2*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X1, kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&pp, kmom*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dSdpi, kmom*sizeof(double),cudaMemAttachGlobal);
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
	if(!rank)
		start_time = MPI_Wtime();
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
			Complex *R;
			cudaMallocManaged(&R,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
#elif defined __INTEL_MKL__
			Complex *R=mkl_malloc(kfermHalo*sizeof(Complex),AVX);
#else
			Complex *R=aligned_alloc(AVX,kfermHalo*sizeof(Complex));
#endif
			//Multiply the dimension of R by 2 because R is complex
			//The FORTRAN code had two Gaussian routines.
			//gaussp was the normal Box-Muller and gauss0 didn't have 2 inside the square root
			//Using ??=1/sqrt(2) in these routines has the same effect as gauss0
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
			Gauss_z(R, kferm, 0, 1/sqrt(2));
#else
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R, 0, 1/sqrt(2));
#endif
#ifdef __NVCC__
			cudaMemPrefetchAsync(R,kfermHalo*sizeof(Complex),device,NULL);
#endif
			Dslashd(R1, R,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
			memcpy(Phi+na*kferm,R1, kferm*sizeof(Complex));
			//Up/down partitioning (using only pseudofermions of flavour 1)
#pragma omp parallel for simd collapse(2) aligned(X0,R1:AVX)
			for(int i=0; i<kvol; i++)
				for(int idirac = 0; idirac < ndirac; idirac++){
					X0[((na*kvol+i)*ndirac+idirac)*nc]=R1[(i*ngorkov+idirac)*nc];
					X0[((na*kvol+i)*ndirac+idirac)*nc+1]=R1[(i*ngorkov+idirac)*nc+1];
				}
#ifdef __NVCC_
			cudaFree(R);
#elif defined __INTEL_MKL__
			mkl_free(R);
#else
			free(R);
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
#ifdef _DEBUG
			if(!rank)
				printf("Traj: %d\tStep: %d\n", itraj, step);
#endif
			//The FORTRAN redefines d=dt here, which makes sense if you have a limited line length.
			//I'll stick to using dt though.
			//step (i) st(t+dt)=st(t)+p(t+dt/2)*dt;
			//Replace with a Kernel call and move trial exchange onto CPU for now
			New_trial(dt,pp,u11t,u12t);
			Reunitarise(u11t,u12t);
			//Get trial fields from accelerator for halo exchange
			//Cancel that until we check for double precision flags. It's really bad on Xe since it isn't natively supported
#pragma acc update self(u11t[0:ndim*kvol],u12t[0:ndim*kvol])
			Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
#ifdef __NVCC__
			//Mark trial fields as primarily read only here? Can re-enable writing at the end of each trajectory
			cudaMemPrefetchAsync(u11t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
			cudaMemPrefetchAsync(u12t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
#endif
			//p(t+3et/2)=p(t+dt/2)-dSds(t+dt)*dt
			//	Force(dSdpi, 0, rescgg);
			Force(dSdpi, 0, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
					dk4m_f,dk4p_f,jqq,akappa,beta,&ancg);
			//The same for loop is given in both the if and else
			//statement but only the value of d changes. This is due to the break in the if part
			if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
#ifdef __NVCC__
				cublasDaxpy(cublas_handle,kmom, &d, dSdpi, 1, pp, 1);
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
		double dH = H0 - H1;
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
		vel2*=vel2;
#elif defined USE_BLAS
		vel2 = cblas_dnrm2(kmom, pp, 1);
		vel2*=vel2;
#else
#pragma unroll
		for(int i=0; i<kmom; i++)
			vel2+=pp[i]*pp[i];
#endif
		Par_dsum(&vel2);
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
								fprintf(output, "Iter (CG) %i ancg %e ancgh %e\n", itercg, ancg, ancgh);
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
											MPI_Abort(comm,OPENERROR);
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
										MPI_Abort(comm,OPENERROR);
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
											MPI_Abort(comm,OPENERROR);
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
	if(!rank)
		elapsed = MPI_Wtime()-start_time;
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
#elif defined __INTEL_MKL__
	mkl_free_buffers();
	mkl_free(dk4m); mkl_free(dk4p); mkl_free(R1); mkl_free(dSdpi); mkl_free(pp);
	mkl_free(Phi); mkl_free(u11t); mkl_free(u12t);
	mkl_free(X0); mkl_free(X1); mkl_free(u11); mkl_free(u12);
	mkl_free(id); mkl_free(iu); mkl_free(hd); mkl_free(hu);
	mkl_free(dk4m_f); mkl_free(dk4p_f); mkl_free(u11t_f); mkl_free(u12t_f);
	mkl_free(h1u); mkl_free(h1d); mkl_free(halosize);
	mkl_free(pcoord);
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
		fprintf(sa3at, "%s\n??%0.3f ??:%0.4f ??:%0.4f j:%0.3f s:%lu t:%lu kvol:%lu\n\
				npx:%lu npt:%lu nthread:%lu ncore:%lu time:%f traj_time:%f\n\n",\
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
	MPI_Finalise();
	fflush(stdout);
	return 0;
}
int Init(int istart, int ibound, int iread, float beta, float fmu, float akappa, Complex_f ajq,\
		Complex *u11, Complex *u12, Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
		double *dk4m, double *dk4p, float *dk4m_f, float *dk4p_f, unsigned int *iu, unsigned int *id){
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
#ifdef __NVCC__
	//Set iu and id to mainly read in CUDA and prefetch them to the GPU
	int device=-1;
	cudaGetDevice(&device);
	cudaMemAdvise(iu,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(id,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemPrefetchAsync(iu,ndim*kvol*sizeof(int),device,NULL);
	cudaMemPrefetchAsync(id,ndim*kvol*sizeof(int),device,NULL);
#endif
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
	DHalo_swap_dir(dk4p, 1, 3, UP);
	DHalo_swap_dir(dk4m, 1, 3, UP);
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
#ifdef __NVCC__
	//Gamma matrices and indices on the GPU
	//More prefetching and marking as read-only (mostly)
	cudaMemAdvise(dk4p,(kvol+halo)*sizeof(double),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(dk4m,(kvol+halo)*sizeof(double),cudaMemAdviseSetReadMostly,device);

	cudaMemPrefetchAsync(dk4p,(kvol+halo)*sizeof(double),device,NULL);
	cudaMemPrefetchAsync(dk4m,(kvol+halo)*sizeof(double),device,NULL);
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
	Par_dsum(&hp); Par_dsum(&hf);
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
