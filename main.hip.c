/** 
 * 	@file main.c
 *
 *   @brief Hybrid Monte Carlo algorithm for Two Colour QCD with Wilson-Gor'kov fermions
 *				based on the algorithm of Duane et al. Phys. Lett. B195 (1987) 216. 
 *
 *    There is "up/down partitioning": each update requires
 *    one operation of Congradq() on complex*16 vectors to determine
 *    @f$(M^{\dagger} M)^{-1}  \Phi@f$ where @f$\Phi@f$ has dimension 4*kvol*nc*Nf - 
 *    The matrix M is the Wilson matrix for a single flavor
 *    there is no extra species doubling as a result
 *
 *    Matrix multiplies done using routines Hdslash() and Hdslashd()
 *
 *    Hence, the number of lattice flavors Nf is related to the
 *    number of continuum flavors @f$N_f@f$ by
 *                 @f$ \text{Nf} = 2 N_f@f$
 *
 *    Fermion expectation values are measured using a noisy estimator.
 *    on the Wilson-Gor'kov matrix, which has dimension 8*kvol*nc*Nf
 *    inversions done using Congradp(), and matrix multiplies with Dslash(),
 *    Dslashd()
 *
 *    Trajectory length is random with mean dt*stepl
 *    The code runs for a fixed number ntraj of trajectories.
 *
 *    @f$\Phi@f$: pseudofermion field <br>
 *    bmass: bare fermion mass  <br>
 *    @f$\mu@f$: chemical potential  <br>
 *    actiona: running average of total action <br>
 *
 *    Fermion expectation values are measured using a noisy estimator.
 *
 *    outputs: <br>
 *    fermi		psibarpsi, energy density, baryon density <br>
 *    bose	   spatial plaquette, temporal plaquette, Polyakov line <br>
 *    diq	   real<qq>
 *
 *     @author SJH			(Original Code, March 2005)
 *     @author P.Giudice	(Hybrid Code, May 2013)
 *     @author D. Lawlor	(Fortran to C Conversion, March 2021. Mixed Precision. GPU, March 2024)
 ******************************************************************/
#include	<assert.h>
#include	<coord.h>
#include	<math.h>
#include	<matrices.h>
#include	<par_mpi.h>
#include	<random.h>
#include	<string.h>
#include	<su2hmc.h>
#ifdef	__GPU__

//Get AMD to convert the CUDA for us
#ifdef __HIPCC__
#include <hipifly.h>
#endif

#include <hipblas.h>
#include	<hip/hip_runtime.h>
#include	<hip/hip_runtime.h>
hipblasHandle_t cublas_handle;
hipblasStatus_t cublas_status;
hipMemPool_t mempool;
//Fix this later
#endif
/**
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
 * doxygen. It should consist of a description of the function, a list of parameters with a brief
 * explanation and lastly what is returned by the function (on success or failure).
 */
int main(int argc, char *argv[]){
	//Instead of hard coding the function name so the error messages are easier to implement
	const char *funcname = "main";

	Par_begin(argc, argv);
	//Add error catching code...
#if(nproc>1)
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
#endif

	/**
	 * @subsection inputs Input Parameters.
	 * The input file format is like the table below, with values sepearated by whitespace
	 *
	 * 0.0100|1.7|0.1780|0.00|0.000|0.0|0.0|100|4|1|5|1|
	 * ------|---|------|----|-----|---|---|---|-|-|-|-|
	 * dt	|beta|akappa|jqq|thetaq|fmu|aNf|stepl|ntraj|istart|icheck|iread|
	 *
	 *	The default values here are straight from the FORTRAN. Note that the bottom line labelling each input is ignored
	 *
	 *	@param dt		Step length for HMC	
	 *	@param beta 	Inverse Gauge Coupling
	 *	@param akappa	Hopping Parameter
	 *	@param jqq		Diquark Source
	 *	@param thetaq	Depericiated/Legacy.
	 *	@param fmu		Chemical Potential
	 *	@param aNf		Depreciated/Legacy
	 *	@param stepl	Mean number of steps per HMC trajectory
	 *	@param istart 	If 0, start from cold start. If one, start from hot start
	 *	@param iprint	How often are measurements made (every iprint trajectories)
	 *	@param icheck	How often are configurations saved (every icheck trajectories)
	 *	@param iread  	Config to read in. If zero, the start based on value of istart
	 */
	float beta = 1.7f;
	float akappa = 0.1780f;
#ifdef __GPU__
	__managed__ 
#endif
		Complex_f jqq = 0;
	float fmu = 0.0f;
	int iread = 0;
	int istart = 1;
	int iprint = 1; //How often are measurements made
	int icheck = 5; //How often are configurations saved
	int ibound = -1;
#ifdef USE_MATH_DEFINES
	const double tpi = 2*M_PI;
#else
	const double tpi = 2*acos(-1.0);
#endif
	float dt=0.004; float ajq = 0.0;
	float delb=0; //Not used?
	float athq = 0.0;
	int stepl = 250; int ntraj = 10;
	//rank is zero means it must be the "master process"
	if(!rank){
		FILE *midout;
		const char *filename = (argc!=2) ?"midout":argv[1];
		char *fileop = "r";
		if( !(midout = fopen(filename, fileop) ) ){
			fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
					OPENERROR, funcname, filename, fileop);
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
		assert(stepl>0);	assert(ntraj>0);	  assert(istart>=0);  assert(icheck>0);  assert(iread>=0); 
	}
	//Send inputs to other ranks
#if(nproc>1)
	Par_fcopy(&dt); Par_fcopy(&beta); Par_fcopy(&akappa); Par_fcopy(&ajq);
	Par_fcopy(&athq); Par_fcopy(&fmu); Par_fcopy(&delb); //Not used?
	Par_icopy(&stepl); Par_icopy(&ntraj); Par_icopy(&istart); Par_icopy(&icheck);
	Par_icopy(&iread); 
#endif
	jqq=ajq*cexp(athq*I);
	//End of input
#ifdef __NVCC__
	//CUBLAS Handle
	hipblasCreate(&cublas_handle);
	//Set up grid and blocks
	blockInit(nx, ny, nz, nt, &dimBlock, &dimGrid);
	//CUDA device
	int device=-1;
	hipGetDevice(&device);
	//For asynchronous memory, when CUDA syncs any unused memory in the pool is released back to the OS
	//unless a threshold is given. We'll base our threshold off of Congradq
	hipDeviceGetDefaultMemPool(&mempool, device);
	int threshold=2*kferm2*sizeof(Complex_f);
	hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &threshold);
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
	//You'll notice that there are two different allocation/free statements
	//One for CUDA and one for everything else depending on what's
	//being used
	Complex *u11, *u12, *u11t, *u12t;
	Complex_f *u11t_f, *u12t_f;
	double *dk4m, *dk4p, *pp;
	float	*dk4m_f, *dk4p_f;
	//Halo index arrays
	unsigned int *iu, *id;
#ifdef __NVCC__
	hipMallocManaged((void**)&iu,ndim*kvol*sizeof(int),hipMemAttachGlobal);
	hipMallocManaged((void**)&id,ndim*kvol*sizeof(int),hipMemAttachGlobal);

	hipMallocManaged(&dk4m,(kvol+halo)*sizeof(double),hipMemAttachGlobal);
	hipMallocManaged(&dk4p,(kvol+halo)*sizeof(double),hipMemAttachGlobal);
#ifdef _DEBUG
	hipMallocManaged(&dk4m_f,(kvol+halo)*sizeof(float),hipMemAttachGlobal);
	hipMallocManaged(&dk4p_f,(kvol+halo)*sizeof(float),hipMemAttachGlobal);
#else
	hipMalloc(&dk4m_f,(kvol+halo)*sizeof(float));
	hipMalloc(&dk4p_f,(kvol+halo)*sizeof(float));
#endif

	int	*gamin;
	Complex	*gamval;
	Complex_f *gamval_f;
	hipMallocManaged(&gamin,4*4*sizeof(Complex),hipMemAttachGlobal);
	hipMallocManaged(&gamval,5*4*sizeof(Complex),hipMemAttachGlobal);
#ifdef _DEBUG
	hipMallocManaged(&gamval_f,5*4*sizeof(Complex_f),hipMemAttachGlobal);
#else
	hipMalloc(&gamval_f,5*4*sizeof(Complex_f));
#endif
	hipMallocManaged(&u11,ndim*kvol*sizeof(Complex),hipMemAttachGlobal);
	hipMallocManaged(&u12,ndim*kvol*sizeof(Complex),hipMemAttachGlobal);
	hipMallocManaged(&u11t,ndim*(kvol+halo)*sizeof(Complex),hipMemAttachGlobal);
	hipMallocManaged(&u12t,ndim*(kvol+halo)*sizeof(Complex),hipMemAttachGlobal);
#ifdef _DEBUG
	hipMallocManaged(&u11t_f,ndim*(kvol+halo)*sizeof(Complex_f),hipMemAttachGlobal);
	hipMallocManaged(&u12t_f,ndim*(kvol+halo)*sizeof(Complex_f),hipMemAttachGlobal);
#else
	hipMalloc(&u11t_f,ndim*(kvol+halo)*sizeof(Complex_f));
	hipMalloc(&u12t_f,ndim*(kvol+halo)*sizeof(Complex_f));
#endif
#else
	id = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));
	iu = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));

	int	*gamin = (int *)aligned_alloc(AVX,4*4*sizeof(int));
	Complex	*gamval=(Complex *)aligned_alloc(AVX,5*4*sizeof(Complex));
	Complex_f *gamval_f=(Complex_f *)aligned_alloc(AVX,5*4*sizeof(Complex_f));;

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
	/**
	 * \subsection initialise Initialisation
	 *
	 * Changing the value of istart in the input parameter file gives us the following start options. These are quoted
	 * from the FORTRAN comments
	 *
	 * istart < 0: Start from tape in FORTRAN?!? How old was this code? (depreciated, replaced with iread)
	 *
	 * istart = 0: Ordered/Cold Start
	 * 			For some reason this leaves the trial fields as zero in the FORTRAN code?
	 *
	 * istart > 0: Random/Hot Start
	 */
	Init(istart,ibound,iread,beta,fmu,akappa,ajq,u11,u12,u11t,u12t,u11t_f,u12t_f,gamval,gamval_f,gamin,dk4m,dk4p,dk4m_f,dk4p_f,iu,id);
#ifdef __NVCC__
	//GPU Initialisation stuff
	Init_CUDA(u11t,u12t,gamval,gamval_f,gamin,dk4m,dk4p,iu,id);//&dimBlock,&dimGrid);
#endif
	//Send trials to accelerator for reunitarisation
	Reunitarise(u11t,u12t);
	//Get trials back
	memcpy(u11, u11t, ndim*kvol*sizeof(Complex));
	memcpy(u12, u12t, ndim*kvol*sizeof(Complex));
#ifdef DIAGNOSTIC
	double ancg_diag=0;
	Diagnostics(istart, u11, u12, u11t, u12t, u11t_f, u12t_f, iu, id, hu, hd, dk4m, dk4p,\
			dk4m_f, dk4p_f, gamin, gamval, gamval_f, jqq, akappa, beta, ancg_diag);
#endif

	//Initial Measurements
	//====================
	Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
	double poly = Polyakov(u11t_f,u12t_f);
#ifdef _DEBUG
	if(!rank) printf("Initial Polyakov loop evaluated as %e\n", poly);
#endif
	double hg, avplaqs, avplaqt;
	//Halo exchange of the trial fields
	Average_Plaquette(&hg,&avplaqs,&avplaqt,u11t_f,u12t_f,iu,beta);
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
	char outname[FILELEN] = "Output."; char *outop="a";
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
	//Expected canged in Hamiltonian
	double e_dH=0; double e_dH_e=0;
	//Expected Metropolis accept probability. Skewed by cases where the hamiltonian decreases.
	double yav = 0.0; double yyav = 0.0; 

	int naccp = 0; int ipbp = 0; int itot = 0;

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
	hipMallocManaged(&R1, kfermHalo*sizeof(Complex),hipMemAttachGlobal);
	hipMalloc(&Phi, nf*kferm*sizeof(Complex));
#ifdef _DEBUG
	hipMallocManaged(&X0, nf*kferm2*sizeof(Complex),hipMemAttachGlobal);
#else
	hipMalloc(&X0, nf*kferm2*sizeof(Complex));
#endif

	hipMallocManaged(&X1, kferm2Halo*sizeof(Complex),hipMemAttachGlobal);
	hipMallocManaged(&pp, kmom*sizeof(double),hipMemAttachGlobal);
	hipMalloc(&dSdpi, kmom*sizeof(double));
	cudaDeviceSynchronise();
#else
	R1= aligned_alloc(AVX,kfermHalo*sizeof(Complex));
	Phi= aligned_alloc(AVX,nf*kferm*sizeof(Complex)); 
	X0= aligned_alloc(AVX,nf*kferm2*sizeof(Complex)); 
	X1= aligned_alloc(AVX,kferm2Halo*sizeof(Complex)); 
	dSdpi = aligned_alloc(AVX,kmom*sizeof(double));
	//pp is the momentum field
	pp = aligned_alloc(AVX,kmom*sizeof(double));
#endif
	/**
	 * @subsection timing Timing
	 * To time the code compile with @verbatim -DSA3AT @endverbatim
	 * This is arabic for hour/watch so is probably not reserved like time is
	 */
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
			Complex_f *R1_f,*R;
			hipMallocManaged(&R,kfermHalo*sizeof(Complex_f),hipMemAttachGlobal);
#ifdef _DEBUG
			hipMallocManaged(&R1_f,kferm*sizeof(Complex_f),hipMemAttachGlobal);
			hipMemset(R1_f,0,kferm*sizeof(Complex_f));
#else
			hipMallocAsync(&R1_f,kferm*sizeof(Complex_f),streams[0]);
			hipMemsetAsync(R1_f,0,kferm*sizeof(Complex_f),streams[0]);
#endif
#else
			Complex_f *R1_f=aligned_alloc(AVX,kferm*sizeof(Complex_f));
			Complex_f *R=aligned_alloc(AVX,kfermHalo*sizeof(Complex_f));
			memset(R1_f,0,kferm*sizeof(Complex_f));
#endif
			//The FORTRAN code had two Gaussian routines.
			//gaussp was the normal Box-Muller and gauss0 didn't have 2 inside the square root
			//Using σ=1/sqrt(2) in these routines has the same effect as gauss0
#if (defined __NVCC__ && defined _DEBUG)
			hipMemPrefetchAsync(R1_f,kferm*sizeof(Complex_f),device,streams[1]);
#endif
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
			Gauss_c(R, kferm, 0, 1/sqrt(2));
#else
			vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R, 0, 1/sqrt(2));
#endif
#ifdef __NVCC__
			hipMemPrefetchAsync(R,kfermHalo*sizeof(Complex_f),device,NULL);
			//Transpose needed here for Dslashd
			Transpose_c(R1_f,ngorkov*nc,kvol,dimGrid,dimBlock);
			Transpose_c(R,ngorkov*nc,kvol,dimGrid,dimBlock);
			//R is random so this techincally isn't required. But it does keep the code output consistent with previous
			//versions.
			//Flip all the gauge fields around so memory is coalesced
			Transpose_c(u11t_f,ndim,kvol,dimGrid,dimBlock);
			Transpose_c(u12t_f,ndim,kvol,dimGrid,dimBlock);
			cudaDeviceSynchronise();
#endif
			Dslashd_f(R1_f,R,u11t_f,u12t_f,iu,id,gamval_f,gamin,dk4m_f,dk4p_f,jqq,akappa);
#ifdef __NVCC__
			//Make sure the multiplication is finished before freeing its input!!
			hipFree(R);//cudaDeviceSynchronise(); 
							//hipFree is blocking so don't need to synchronise
			Transpose_c(R1_f,kvol,ngorkov*nc,dimGrid,dimBlock);
			cuComplex_convert(R1_f,R1,kferm,false,dimBlock,dimGrid);
			Transpose_c(u11t_f,kvol,ndim,dimGrid,dimBlock);
			Transpose_c(u12t_f,kvol,ndim,dimGrid,dimBlock);
			//cudaDeviceSynchronise();
			//hipFreeAsync(R1_f,NULL);
			hipMemcpyAsync(Phi+na*kferm,R1, kferm*sizeof(Complex),hipMemcpyDefault,0);
			//hipMemcpyAsync(Phi+na*kferm,R1, kferm*sizeof(Complex),hipMemcpyDefault,streams[1]);
			cudaDeviceSynchronise();
#ifdef _DEBUG
			hipFree(R1_f);
#else
			hipFreeAsync(R1_f,0);
#endif
			//hipFree is blocking so don't need cudaDeviceSynchronise()
#else
			free(R); 
#pragma omp simd aligned(R1_f,R1:AVX)
			for(int i=0;i<kferm;i++)
				R1[i]=(Complex)R1_f[i];
			free(R1_f);
			memcpy(Phi+na*kferm,R1, kferm*sizeof(Complex));
			//Up/down partitioning (using only pseudofermions of flavour 1)
#endif
			UpDownPart(na, X0, R1);
		}	
		//Heatbath
		//========
		//We're going to make the most of the new Gauss_d routine to send a flattened array
		//and do this all in one step.
#ifdef __NVCC__
		hipMemcpyAsync(u11t, u11, ndim*kvol*sizeof(Complex),hipMemcpyHostToDevice,streams[1]);
		hipMemcpyAsync(u12t, u12, ndim*kvol*sizeof(Complex),hipMemcpyHostToDevice,streams[2]);
#else
		memcpy(u11t, u11, ndim*kvol*sizeof(Complex));
		memcpy(u12t, u12, ndim*kvol*sizeof(Complex));
#endif
#if (defined(USE_RAN2)||defined(__RANLUX__)||!defined(__INTEL_MKL__))
		Gauss_d(pp, kmom, 0, 1);
#else
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, kmom, pp, 0, 1);
#endif
		//Initialise Trial Fields
#ifdef __NVCC__
		hipMemPrefetchAsync(pp,kmom*sizeof(double),device,streams[1]);
		hipMemcpy(u11t, u11, ndim*kvol*sizeof(Complex),hipMemcpyDefault);
		hipMemcpy(u12t, u12, ndim*kvol*sizeof(Complex),hipMemcpyDefault);
#else
		memcpy(u11t, u11, ndim*kvol*sizeof(Complex));
		memcpy(u12t, u12, ndim*kvol*sizeof(Complex));
#endif
		Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
		double H0, S0;
		Hamilton(&H0, &S0, rescga,pp,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval_f,gamin,\
				dk4m_f,dk4p_f,jqq,akappa,beta,&ancgh,itraj);
#ifdef _DEBUG
		if(!rank) printf("H0: %e S0: %e\n", H0, S0);
#endif
		if(itraj==1)
			action = S0/gvol;

		//Integration 
		//TODO: Have this as a runtime parameter.
#if (defined INT_LPFR && defined INT_OMF2) ||(defined INT_LPFR && defined INT_OMF4)||(defined INT_OMF2 && defined INT_OMF4)
#error "Only one integrator may be defined"
#elif defined INT_LPFR
		Leapfrog(u11t, u12t, u11t_f, u12t_f, X0, X1, Phi, dk4m, dk4p, dk4m_f, dk4p_f, dSdpi, pp,iu, id, gamval,
				gamval_f, gamin, jqq, beta,akappa,stepl,dt,&ancg,&itot,proby);
#elif defined INT_OMF2
		OMF2(u11t, u12t, u11t_f, u12t_f, X0, X1, Phi, dk4m, dk4p, dk4m_f, dk4p_f, dSdpi, pp,iu, id, gamval,
				gamval_f, gamin, jqq, beta,akappa,stepl,dt,&ancg,&itot,proby);
#elif defined INT_OMF4
		OMF4(u11t, u12t, u11t_f, u12t_f, X0, X1, Phi, dk4m, dk4p, dk4m_f, dk4p_f, dSdpi, pp,iu, id, gamval,
				gamval_f, gamin, jqq, beta,akappa,stepl,dt,&ancg,&itot,proby);
#else
#error "No integrator defined. Please define {INT_LPFR.INT_OMF2,INT_OMF4}"
#endif

		totancg+=ancg;
		//Monte Carlo step: Accept new fields with the probability of min(1,exp(H0-X0))
		//Kernel Call needed here?
		Reunitarise(u11t,u12t);
		double H1, S1;
		Hamilton(&H1, &S1, rescga,pp,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval_f,gamin,\
				dk4m_f,dk4p_f,jqq,akappa,beta,&ancgh,itraj);
		ancgh/=2.0; //Hamilton is called at start and end of trajectory
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
		e_dH+=dH; e_dH_e+=dH*dH;
		double y = exp(dH);
		yav+=y;
		yyav+=y*y;
		//The Monte-Carlo
		//Always update  dH is positive (gone from higher to lower energy)
		bool acc;
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
			acc=true;
		}
		else{
			if(!rank)
				printf("New configuration rejected on trajectory %i.\n", itraj);
			acc=false;
		}
		actiona+=action; 
		double vel2=0.0;
#ifdef __NVCC__
		hipblasDnrm2(cublas_handle,kmom, pp, 1,&vel2);
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
			if(!acc){
#ifdef __NVCC__
				hipMemcpyAsync(u11t, u11, ndim*kvol*sizeof(Complex),hipMemcpyDefault,streams[0]);
				hipMemcpyAsync(u12t, u12, ndim*kvol*sizeof(Complex),hipMemcpyDefault,streams[1]);
#else
				memcpy(u11t, u11, ndim*kvol*sizeof(Complex));
				memcpy(u12t, u12, ndim*kvol*sizeof(Complex));
#endif
				Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
			}
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
			Average_Plaquette(&hg,&avplaqs,&avplaqt,u11t_f,u12t_f,iu,beta);
			poly = Polyakov(u11t_f,u12t_f);
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
								fprintf(output, "Measure (CG) %i Update (CG) %.3f Hamiltonian (CG) %.3f\n", itercg, ancg, ancgh);
								fflush(output);
								break;
							case(1):
								{
									FILE *fortout;
									char fortname[FILELEN] = "fermi.";
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
										fprintf(fortout, "pbp\tendenf\tdenf\n");
									if(measure_check)
										fprintf(fortout, "%e\t%e\t%e\n", NAN, NAN, NAN);
									else
										fprintf(fortout, "%e\t%e\t%e\n", pbp, endenf, denf);
									fclose(fortout);
									break;
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
									if(measure_check)
										fprintf(fortout, "%e\n", NAN);
									else
										fprintf(fortout, "%e\n", creal(qq));
									fclose(fortout);
									break;
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
#ifdef __NVCC__
	//Make a routine that does this for us
	hipFree(dk4m); hipFree(dk4p); hipFree(R1); hipFree(dSdpi); hipFree(pp);
	hipFree(Phi); hipFree(u11t); hipFree(u12t);
	hipFree(X0); hipFree(X1); hipFree(u11); hipFree(u12);
	hipFree(id); hipFree(iu); 
	hipFree(dk4m_f); hipFree(dk4p_f); hipFree(u11t_f); hipFree(u12t_f);
	hipFree(gamin); hipFree(gamval); hipFree(gamval_f);
	hipblasDestroy(cublas_handle);
#else
	free(dk4m); free(dk4p); free(R1); free(dSdpi); free(pp);
	free(Phi); free(u11t); free(u12t);
	free(X0); free(X1); free(u11); free(u12);
	free(id); free(iu);
	free(dk4m_f); free(dk4p_f); free(u11t_f); free(u12t_f);
	free(gamin); free(gamval); free(gamval_f);
#endif
	free(hd); free(hu);free(h1u); free(h1d); free(halosize); free(pcoord);
#ifdef __RANLUX__
	gsl_rng_free(ranlux_instd);
#elif (defined __INTEL_MKL__ &&!defined USE_RAN2)
	vslDeleteStream(&stream);
#endif
#if (defined SA3AT)
	if(!rank){
		FILE *sa3at = fopen("Bench_times.csv", "a");
#ifdef __NVCC__
		char *version[256];
		int cuversion; hipRuntimeGetVersion(&cuversion);
		sprintf(version,"CUDA %d\tBlock: (%d,%d,%d)\tGrid: (%d,%d,%d)\n%s\n",cuversion,\
					dimBlock.x,dimBlock.y,dimBlock.z,dimGrid.x,dimGrid.y,dimGrid.z,__VERSION__);
#else
		char *version=__VERSION__;
#endif
		fprintf(sa3at, "%s\nβ%0.3f κ:%0.4f μ:%0.4f j:%0.3f s:%i t:%i kvol:%ld\n"
				"npx:%i npt:%i nthread:%i ncore:%i time:%f traj_time:%f\n\n",\
				version,beta,akappa,fmu,ajq,nx,nt,kvol,npx,npt,nthreads,npx*npy*npz*npt*nthreads,elapsed,elapsed/ntraj);
		fclose(sa3at);
	}
#endif
	//Get averages for final output
	actiona/=ntraj; vel2a/=ntraj; pbpa/=ipbp; endenfa/=ipbp; denfa/=ipbp;
	totancg/=ntraj; totancgh/=ntraj; 
	e_dH/=ntraj; e_dH_e=sqrt((e_dH_e/ntraj-e_dH*e_dH)/(ntraj-1));
	yav/=ntraj; yyav=sqrt((yyav/ntraj - yav*yav)/(ntraj-1));
	float traj_cost=totancg/dt;
	double atraj=dt*itot/ntraj;

	if(!rank){
		fprintf(output, "Averages for the last %i trajectories\n"\
				"Number of acceptances: %i\tAverage Trajectory Length = %e\n"\
				"<dH>=%e+/-%e\t<exp(dH)>=%e+/-%e\tTrajectory cost=N_cg/dt =%e\n"\
				"Average number of congrad iter guidance: %.3f acceptance %.3f\n"\
				"psibarpsi = %e\n"\
				"Mean Square Velocity = %e\tAction Per Site = %e\n"\
				"Energy Density = %e\tNumber Density %e\n\n\n",\
				ntraj, naccp, atraj, e_dH,e_dH_e, yav, yyav, traj_cost, totancg, totancgh, pbpa, vel2a, actiona, endenfa, denfa);
		fclose(output);
	}
#if(nproc>1)
	//Ensure writing is done before finalising just in case finalise segfaults and crashes the other ranks mid-write
	MPI_Barrier(comm);
	MPI_Finalise();
#endif
	fflush(stdout);
	return 0;
}
