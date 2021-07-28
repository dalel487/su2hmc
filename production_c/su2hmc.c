#include	<coord.h>
#ifdef	__NVCC__
#include	<cuda.h>
#include	<cuda_runtime.h>
//Fix this later
//#define cudaMemAttachGlobal 0x01
#endif
#include	<math.h>
#include	<par_mpi.h>
#include	<random.h>
#include	<matrices.h>
#include	<stdlib.h>
#include	<stdio.h>
#include	<string.h>
#include	<su2hmc.h>

//Extern definitions, especially default values for fmu, beta and akappa
Complex jqq = 0;
double fmu = 0.0;
double beta = 1.7;
double akappa = 0.1780;
int gamin[4][4] =	{{3,2,1,0},
	{3,2,1,0},
	{2,3,0,1},
	{2,3,0,1}};
Complex gamval[5][4] =	{{-I,-I,I,I},
	{-1,1,1,-1},
	{-I,I,I,-I},
	{1,1,1,1},
	{1,1,-1,-1}};
Complex_f gamval_f[5][4] =	{{-I,-I,I,I},
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
 * Another vestiage of the Fortran code that will be implimented here is
 * the frequent flattening of arrays. But while FORTRAN Allows you to write
 * array(i,j) as array(i+M*j) where M is the number of rows, C resorts to 
 * pointers
 *
 * One change I will try and make is the introdction of error-codes (nothing
 * to do with the Irish postal service)
 * These can be found in the file errorcode.h and can help with debugging
 *
 * Lastly, the comment style for the start of a function is based off of 
 * Niall Moran's python style (which may have come from numpy?) It should
 * consist of a description of the function, a list of parameters with a brief
 * explaination and lastly what is returned by the function (on success or failure)
 */
int main(int argc, char *argv[]){
	/*******************************************************************
	 *    Hybrid Monte Carlo algorithm for Two Color QCD with Wilson-Gor'kov fermions
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
	const char *funcname = "main";
	Par_begin(argc, argv);
	//Add error catching code...
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	//Input
	//The default values here are straight from the FORTRAN
	//=====================================================
	int iread = 0;
	int istart = 1;
	ibound = -1;
	int iwrite = 1;
	int iprint = 1; //For the measures
	int icheck = 5; //Save conf
#ifdef USE_MATH_DEFINES
	const double tpi = 2*M_PI;
#else
	const double tpi = 2*acos(-1.0);
#endif
	//End of input
	//===========
	//rank is zero means it must be the "master process"
	double dt=0.004; double ajq = 0.0;
	double delb; //Not used?
	double athq = 0.0;
	int stepl = 250; int ntraj = 10;
	if(!rank){
		FILE *midout;
		//Instead of hardcoding so the error messages are easier to impliment
		char *filename = "midout";
		char *fileop = "rb";
		if( !(midout = fopen(filename, fileop) ) ){
			fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n"\
					, OPENERROR, funcname, filename, fileop);
			exit(OPENERROR);
		}
		fscanf(midout, "%lf %lf %lf %lf %lf %lf %lf %d %d %d", &dt, &beta, &akappa, &ajq, &athq, &fmu, &delb, &stepl, &ntraj, &istart);
		fclose(midout);
	}
	if(iread){
#ifdef _DEBUG
		if(!rank) printf("Calling Par_sread() with seed: %i\n", seed);
#endif
		Par_sread();
	}
	//Send inputs to other ranks
	Par_dcopy(&dt); Par_dcopy(&beta); Par_dcopy(&akappa); Par_dcopy(&ajq);
	Par_dcopy(&athq); Par_dcopy(&fmu); Par_dcopy(&delb); //Not used?
	Par_icopy(&stepl); Par_icopy(&ntraj); 
	jqq=ajq*cexp(athq*I);
	float akappa_f=(float)akappa;
	float jqq_f=(float)jqq;
#ifdef _DEBUG
	printf("jqq=%f+(%f)I\n",creal(jqq),cimag(jqq));
#endif
	Par_ranset(&seed);

	//Initialisation
	//istart < 0: Start from tape?!? How old is this code?
	//istart = 0: Ordered/Cold Start
	//			For some reason this leaves the trial fields as zero in the FORTRAN code?
	//istart > 0: Random/Hot Start
	Init(istart);
#ifdef DIAGNOSTIC
	Diagnostics(istart);
#endif

	//Initial Measurements
	//====================
	double poly = Polyakov();
#ifdef _DEBUG
	if(!rank) printf("Initial Polyakov loop evaluated as %e\n", poly);
#endif
	double hg, avplaqs, avplaqt;
	Trial_Exchange();
	SU2plaq(&hg,&avplaqs,&avplaqt);
	//Loop on β
	//Print Heading
	double traj=stepl*dt;
	double proby = 2.5/stepl;
	char *outname = "Output"; char *outop="w";
	FILE *output;
	if(!rank){
		if(!(output=fopen(outname, outop) )){
			fprintf(stderr,"Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",OPENERROR,funcname,outname,outop);
			MPI_Finalise();
			exit(OPENERROR);
		}
		printf("hg = %e, <Ps> = %e, <Pt> = %e, <Poly> = %e\n", hg, avplaqs, avplaqt, poly);
		fprintf(output, "ksize = %i ksizet = %i Nf = %i Halo =%i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i β = %e\nκ = %e μ = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %i\n",
				ksize, ksizet, nf, halo, dt, traj, ntraj, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#ifdef _DEBUG
		//Print to terminal during debugging
		printf("ksize = %i ksizet = %i Nf = %i Halo = %i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i β = %e\nκ = %e μ = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %i\n",
				ksize, ksizet, nf, halo, dt, traj, ntraj, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#endif
	}
	//Initialise for averages
	//======================
	double actiona = 0.0; double vel2a = 0.0; double pbpa = 0.0; double endenfa = 0.0; double denfa = 0.0;
	double yav = 0.0; double yyav = 0.0; 

	int naccp = 0; int ipbp = 0; int itot = 0;

	ancg = 0; ancgh = 0;
	//This was originally in the half-step of the fortran code, but it makes more sense to declare
	//it outside the loop. Since it's always being subtracted we'll define it as negative
	const	double d = -dt*0.5;
	//Start of classical evolution
	//===========================
	double pbp;
	complex qq;
	//Initialise Some Arrays. Leaving it late for scoping
	//check the sizes in sizes.h
	double *dSdpi;
	//There is absolutely no reason to keep the cold trial fields as zero now, so I won't
	if(istart==0){
		memcpy(u11t,u11,(kvol+halo)*ndim*sizeof(complex));
		Trial_Exchange();
	}
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	cudaMallocManaged(&R1, kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&xi, kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&Phi, nf*kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X0, nf*kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&X1, kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&pp, kmomHalo*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dSdpi, kmomHalo*sizeof(double),cudaMemAttachGlobal);
#elif defined USE_MKL
	R1= mkl_malloc(kfermHalo*sizeof(complex),AVX);
	xi= mkl_malloc(kfermHalo*sizeof(complex),AVX);
	Phi= mkl_malloc(nf*kfermHalo*sizeof(complex),AVX); 
	X0= mkl_malloc(nf*kferm2Halo*sizeof(complex),AVX); 
	X1= mkl_malloc(kferm2Halo*sizeof(complex),AVX); 
	dSdpi = mkl_malloc(kmomHalo*sizeof(double), AVX);
	//pp is the momentum field
	pp = mkl_malloc(kmomHalo*sizeof(double), AVX);
#else
	R1= aligned_alloc(AVX,kfermHalo*sizeof(complex));
	xi= aligned_alloc(AVX,kfermHalo*sizeof(complex));
	Phi= aligned_alloc(AVX,nf*kfermHalo*sizeof(complex)); 
	X0= aligned_alloc(AVX,nf*kferm2Halo*sizeof(complex)); 
	X1= aligned_alloc(AVX,kferm2Halo*sizeof(complex)); 
	dSdpi = aligned_alloc(AVX,kmomHalo*sizeof(double));
	pp = aligned_alloc(AVX,kmomHalo*sizeof(double));
#endif
	//Arabic for hour/watch so probably not defined elsewhere like TIME potentially is
#if (defined SA3AT)
	double start_time=0;
	if(!rank)
		start_time = MPI_Wtime();
#endif
	for(int itraj = 1; itraj <= ntraj; itraj++){
#ifdef _DEBUG
		if(!rank)
			printf("Starting itraj %i\n", itraj);
#endif
		for(int na=0; na<nf; na++){
			//Probably makes sense to declare this outside the loop
			//but I do like scoping/don't want to break anything else just yeat
			//
			//How do we optimise this for use in CUDA? Do we use CUDA's PRNG
			//or stick with MKL and synchronise/copy over the array
#ifdef __NVCC__
			Complex *R;
			cudaMallocManaged(&R,kfermHalo*sizeof(Complex),cudaMemAttachGlobal);
#elif defined USE_MKL
			complex *R=mkl_malloc(kfermHalo*sizeof(complex),AVX);
#else
			complex *R=aligned_alloc(AVX,kfermHalo*sizeof(complex));
#endif
			//Multiply the dimension of R by 2 because R is complex
			//The FORTRAN code had two gaussian routines.
			//gaussp was the normal box-muller and gauss0 didn't have 2 inside the square root
			//Using σ=1/sqrt(2) in these routines has the same effect as gauss0
#if (defined(USE_RAN2)||!defined(USE_MKL))
			Gauss_z(R, kferm, 0, 1/sqrt(2));
#else
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R, 0, 1/sqrt(2));
#endif
#ifdef __NVCC__
			cudaMemPrefetchAsync(R,kfermHalo*sizeof(Complex),device,NULL);
#endif
			Dslashd(R1, R);
			memcpy(Phi+na*kfermHalo,R1, nc*ngorkov*kvol*sizeof(complex));
			//Up/down partitioning (using only pseudofermions of flavour 1)
			//CUDAFY THIS?
#pragma omp parallel for simd aligned(X0:AVX,R1:AVX)
			for(int i=0; i<kvol; i++)
				for(int idirac = 0; idirac < ndirac; idirac++){
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc]=R1[(i*ngorkov+idirac)*nc];
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1]=R1[(i*ngorkov+idirac)*nc+1];
				}
#ifdef __NVCC_
			cudaFree(R);
#elif defined USE_MKL
			mkl_free(R);
#else
			free(R);
#endif
		}	
		//Heatbath
		//========
		//We're going to make the most of the new Gauss_d routine to send a flattened array
		//and do this all in one step.
#if (defined(USE_RAN2)||!defined(USE_MKL))
		Gauss_d(pp, kmom, 0, 1);
#else
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, kmom, pp, 0, 1);
#endif

		//Initialise Trial Fields
		//Does CUDA like memcpy in this way?
		memcpy(u11t, u11, ndim*kvol*sizeof(complex));
		memcpy(u12t, u12, ndim*kvol*sizeof(complex));

		Trial_Exchange();
#ifdef __NVCC__
		cudaMemPrefetchAsync(u11t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(u12t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(u11t_f, ndim*(kvol+halo)*sizeof(Complex_f),device,NULL);
		cudaMemPrefetchAsync(u12t_f, ndim*(kvol+halo)*sizeof(Complex_f),device,NULL);
#endif
		double H0, S0;
		Hamilton(&H0, &S0, rescga);
#ifdef _DEBUG
		if(!rank) printf("H0: %e S0: %e\n", H0, S0);
#endif
		double action;
		if(itraj==1)
			action = S0/gvol;

		//Half step forward for p
		//=======================
#ifdef _DEBUG
		printf("Evaluating force on rank %i\n", rank);
#endif
		Force(dSdpi, 1, rescgg);
#ifdef _DEBUG
		double av_force=0;
#pragma omp parallel for simd reduction(+:av_force) aligned(dSdpi:AVX)
		for(int i = 0; i<kmom; i++)
			av_force+=dSdpi[i];
		printf("av_force before we do anything= %e\n", av_force/kmom);
#endif
#ifdef __NVCC__
		cublasDaxpy(cublas_handle,nadj*ndim*kvol, &d, dSdpi, 1, pp, 1);
#elif (defined USE_MKL || defined USE_BLAS)
		cblas_daxpy(nadj*ndim*kvol, d, dSdpi, 1, pp, 1);
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
			New_trial(dt);
			Reunitarise();
			Trial_Exchange();

#ifdef __NVCC__
//Mark trial fields as primarily read only here? Can renable writing at the end of each trajectory
			cudaMemPrefetchAsync(u11t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
			cudaMemPrefetchAsync(u12t, ndim*(kvol+halo)*sizeof(Complex),device,NULL);
#endif
			//p(t+3dt/2)=p(t+dt/2)-dSds(t+dt)*dt
			Force(dSdpi, 0, rescgg);
#ifdef _DEBUG
#pragma omp parallel for simd reduction(+:av_force) aligned(dSdpi:AVX)
			for(int i = 0; i<kmom; i++)
				av_force+=dSdpi[i];
			printf("av_force after trial field update = %e\n", av_force/kmom);
#endif
			//Need to check Par_granf again 
			//The same for loop is given in both the if and else
			//statement but only the value of d changes. This is due to the break in the if part
			if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
#ifdef __NVCC__
				cublasDaxpy(cublas_handle,ndim*nadj*kvol, &d, dSdpi, 1, pp, 1);
#elif (defined USE_MKL || defined USE_BLAS)
				//cuBLAS calls from CPU allowed?
				cblas_daxpy(ndim*nadj*kvol, d, dSdpi, 1, pp, 1);
#else
				for(int i = 0; i<kmom; i++)
					//d negated above
					pp[i]+=d*dSdpi[i];
#endif
				itot+=step;
				break;
			}
			else{
#ifdef __NVCC__
				//dt is needed for the trial fields so has to be negated every time.
				dt*=-1;
				cublasDaxpy(cublas_handle,ndim*nadj*kvol, &dt, dSdpi, 1, pp, 1);
				dt*=-1;
#elif (defined USE_MKL || defined USE_BLAS)
				cblas_daxpy(ndim*nadj*kvol, -dt, dSdpi, 1, pp, 1);
#else
				for(int i = 0; i<kvol; i++)
					for(int iadj=0; iadj<nadj; iadj++)
						for(int mu = 0; mu < ndim; mu++)
							pp[(i*nadj+iadj)*nc+mu]-=dt*dSdpi[(i*nadj+iadj)*nc+mu];
#endif

			}
		}
		//Monte Carlo step: Accept new fields with the probability of min(1,exp(H0-X0))
		//Kernel Call needed here?

#ifdef __NVCC__
		cudaMemPrefetchAsync(u11t, ndim*kvol*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(u12t, ndim*kvol*sizeof(Complex),device,NULL);
#endif
		Reunitarise();
		double H1, S1;
		Hamilton(&H1, &S1, rescga);
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
		//x is unassigned in the FORTRAN at declaration, so hopefully that won't be an issue here...
		//Only update x if dH is negative
		if(dH>0 || Par_granf()<=y){
			// We only test x if it is updated (inside the previous if block)
			//But that required a goto in FORTRAN to get around doing the acceptance operations
			//in the case where dH>=0 or x<=y. We'll nest the if statements in C to 
			//get around this using the reverse test to the FORTRAN if (x<=y instead of x>y).
			//Step is accepted. Set s=st
			if(!rank)
				printf("New configuration accepted on trajectory %i.\n", itraj);
			//Original FORTRAN Comment:
			//JIS 20100525: write config here to preempt troubles during measurement!
			//JIS 20100525: remove when all is ok....
			memcpy(u11,u11t,ndim*(kvol+halo)*sizeof(complex));
			memcpy(u12,u12t,ndim*(kvol+halo)*sizeof(complex));
			naccp++;
			//Divide by gvol because of halos?
			action=S1/gvol;
		}
		actiona+=action; 
		double vel2=0.0;
#ifdef __NVCC__
		cublasDnrm2(cublas_handle,kmom, pp, 1,&vel2);
		vel2*=vel2;
#elif (defined USE_MKL || defined USE_BLAS)
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
			//Unified memory and memcpy?
			memcpy(u11t, u11, ndim*(kvol+halo)*sizeof(complex));
			memcpy(u12t, u12, ndim*(kvol+halo)*sizeof(complex));
#ifdef _DEBUG
			if(!rank)
				printf("Starting measurements\n");
#endif
			int itercg;
			double endenf, denf;
			complex qbqb;
			Measure(&pbp,&endenf,&denf,&qq,&qbqb,respbp,&itercg);
#ifdef _DEBUG
			if(!rank)
				printf("Finished measurements\n");
#endif
			pbpa+=pbp; endenfa+=endenf; denfa+=denf; ipbp++;
			SU2plaq(&hg,&avplaqs,&avplaqt); 
			poly = Polyakov();
			//We have four output files, so may as well get the other ranks to help out
			//and abuse scoping rules while we're at it.
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
								//The origninal code implicitly created these files with the name fort.XX where XX
								//is the file label from FORTRAN. We'll stick with that for now.
								{
									FILE *fortout;
									char *fortname = "PBP-Density";
									char *fortop= (itraj==1) ? "w" : "a";
									if(!(fortout=fopen(fortname, fortop) )){
										fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
												OPENERROR, funcname, fortname, fortop);
										MPI_Finalise();
										exit(OPENERROR);
									}
									if(itraj==1)
										fprintf(fortout, "pbp\tendenf\tdenf\n");
									fprintf(fortout, "%e\t%e\t%e\n", pbp, endenf, denf);
									fclose(fortout);
									break;
								}
							case(2):
								//The origninal code implicitly created these files with the name
								//fort.XX where XX is the file label
								//from FORTRAN. This was fort.12
								{
									FILE *fortout;
									char *fortname = "Plaquette"; 
									char *fortop= (itraj==1) ? "w" : "a";
									if(!(fortout=fopen(fortname, fortop) )){
										fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
												OPENERROR, funcname, fortname, fortop);
										MPI_Finalise();
										exit(OPENERROR);
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
									char *fortname = "Diquark";
									char *fortop= (itraj==1) ? "w" : "a";
									if(!(fortout=fopen(fortname, fortop) )){
										fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
												OPENERROR, funcname, fortname, fortop);
										MPI_Finalise();
										exit(OPENERROR);
									}
									if(itraj==1)
										fprintf(fortout, "Re(qq)\n");
									fprintf(fortout, "%e\n", creal(qq));
									fclose(fortout);
									break;
								}
							default: break;
						}
		}
		if(itraj%icheck==0){
			Par_swrite(itraj,icheck,beta,fmu,akappa,ajq);
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
#ifdef __NVCC__
	//Make a routine that does this for us
	cudaFree(dk4m); cudaFree(dk4p); cudaFree(R1); cudaFree(dSdpi); cudaFree(pp);
	cudaFree(Phi); cudaFree(u11t); cudaFree(u12t); cudaFree(xi);
	cudaFree(X0); cudaFree(X1); cudaFree(u11); cudaFree(u12);
	cudaFree(id); cudaFree(iu); cudaFree(hd); cudaFree(hu);
	cudaFree(dk4m_f); cudaFree(dk4p_f); cudaFree(u11t_f); cudaFree(u12t_f);
#elif defined USE_MKL
	mkl_free(dk4m); mkl_free(dk4p); mkl_free(R1); mkl_free(dSdpi); mkl_free(pp);
	mkl_free(Phi); mkl_free(u11t); mkl_free(u12t); mkl_free(xi);
	mkl_free(X0); mkl_free(X1); mkl_free(u11); mkl_free(u12);
	mkl_free(id); mkl_free(iu); mkl_free(hd); mkl_free(hu);
	mkl_free(dk4m_f); mkl_free(dk4p_f); mkl_free(u11t_f); mkl_free(u12t_f);
	mkl_free(pcoord);
#else
	free(dk4m); free(dk4p); free(R1); free(dSdpi); free(pp); free(Phi);
	free(u11t); free(u12t); free(xi); free(X0); free(X1);
	free(u11); free(u12); free(id); free(iu); free(hd); free(hu);
	free(pcoord);
#endif
#if (defined SA3AT)
	if(!rank){
		FILE *sa3at = fopen("Bench_times.csv", "a");
#ifdef __NVCC__
		char *lang = "cuda";
#else 
		char *lang = "C";
#endif
		fprintf(sa3at, "%s,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%f,%f\n",lang,nx,nt,kvol,npx,npt,nthreads,npx*npt*nthreads,elapsed,elapsed/ntraj);
		fclose(sa3at);
	}
#endif
	actiona/=ntraj; vel2a/=ntraj; pbpa/=ipbp; endenfa/=ipbp; denfa/=ipbp;
	ancg/=nf*itot; ancgh/=2*nf*ntraj; yav/=ntraj; yyav=sqrt((yyav/ntraj - yav*yav)/(ntraj-1));
	double atraj=dt*itot/ntraj;

	if(!rank){
		fprintf(output, "Averages for the last %i trajectories\n"\
				"Number of acceptances: %i Average Trajectory Length = %e\n"\
				"<exp(dh)> = %e +/- %e\n"\
				"Average number of congrad iter guidance: %f acceptance %f\n"\
				"psibarpsi = %e\n"\
				"Mean Square Velocity = %e Action Per Site = %e\n"\
				"Energy Density = %e Number Density %e\n",\
				ntraj, naccp, atraj, yav, yyav, ancg, ancgh, pbpa, vel2a, actiona, endenfa, denfa);
		fclose(output);
	}
	MPI_Finalise();
	fflush(stdout);
	return 0;
}
int Init(int istart){
	/*
	 * Initialises the system
	 *
	 * Calls:
	 * ======
	 * Addrc. Rand_init
	 *
	 *
	 * Globals:
	 * ========
	 * u11t, u12t, dk4m, dk4p
	 *
	 * Parameters:
	 * ==========
	 * int istart: Zero for cold, >1 for hot, <1 for none
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Init";

#ifdef _OPENMP
	omp_set_num_threads(nthreads);
	//Comment out to keep the threads spinning even when there's no work to do
	//Commenting out decrease runtime but increases total CPU time dramatically
	//This can throw of some profilers
	//kmp_set_defaults("KMP_BLOCKTIME=0");
#ifdef USE_MKL
	mkl_set_num_threads(nthreads);
#endif
#endif
	//First things first, calculate a few constants
	Addrc();
	//And confirm they're legit
	Check_addr(iu, ksize, ksizet, 0, kvol+halo);
	Check_addr(id, ksize, ksizet, 0, kvol+halo);
#ifdef _DEBUG
	printf("Checked addresses\n");
#endif
	double chem1=exp(fmu); double chem2 = 1/chem1;
#ifdef __NVCC__
	//Set iu and id to mainly read in CUDA and prefetch them to the GPU
	int device=-1;
	cudaGetDevice(&device);
	cudaMemAdvise(iu,ndim*kvol*sizeof(int),..SetReadMostly,device);
	cudaMemAdvise(id,ndim*kvol*sizeof(int),..SetReadMostly,device);
	cudaMemPrefetchAsync(iu,ndim*kvol*sizeof(int),device,NULL);
	cudaMemPrefetchAsync(id,ndim*kvol*sizeof(int),device,NULL);

	cudaMallocManaged(&dk4m,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4p,(kvol+halo)*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4m_f,(kvol+halo)*sizeof(float),cudaMemAttachGlobal);
	cudaMallocManaged(&dk4p_f,(kvol+halo)*sizeof(float),cudaMemAttachGlobal);
#elif defined USE_MKL
	dk4m = mkl_malloc((kvol+halo)*sizeof(double), AVX);
	dk4p = mkl_malloc((kvol+halo)*sizeof(double), AVX);
	dk4m_f = mkl_malloc((kvol+halo)*sizeof(float), AVX);
	dk4p_f = mkl_malloc((kvol+halo)*sizeof(float), AVX);
#else
	dk4m = aligned_alloc(AVX,(kvol+halo)*sizeof(double));
	dk4p = aligned_alloc(AVX,(kvol+halo)*sizeof(double));
	dk4m_f = aligned_alloc(AVX,(kvol+halo)*sizeof(float));
	dk4p_f = aligned_alloc(AVX,(kvol+halo)*sizeof(float));
#endif
#pragma omp parallel for simd aligned(dk4m:AVX,dk4p:AVX)
	//CUDA this. Only limit will be the bus speed
	for(int i = 0; i<kvol; i++){
		dk4p[i]=akappa*chem1;
		dk4m[i]=akappa*chem2;
	}
	//Antiperiodic Boundary Conditions. Flip the terms at the edge of the time
	//direction
	if(ibound == -1 && pcoord[3+ndim*rank]==npt -1){
#ifdef _DEBUG
		printf("Implimenting antiperiodic boundary conditions on rank %i\n", rank);
#endif
#pragma omp parallel for simd aligned(dk4m:AVX,dk4p:AVX)
		//Also CUDA this. By the looks of it it should saturate the GPU
		//as is
		for(int i= 0; i<kvol3; i++){
			int k = kvol - kvol3 + i;
			dk4p[k]*=-1;
			dk4m[k]*=-1;
		}
	}
	//These are constant so swap the halos when initialising and be done with it
	//May need to add a synchronisation statement here first
	for(int mu = 0; mu <ndim; mu++){
		DHalo_swap_dir(dk4p, 1, 3, UP);
		DHalo_swap_dir(dk4m, 1, 3, UP);
	}
#pragma omp parallel for simd aligned(dk4m:AVX,dk4p:AVX,dk4m_f:AVX,dk4p_f:AVX)
	for(int i=0;i<kvol+halo;i++){
		dk4p_f[i]=(float)dk4p[i];
		dk4m_f[i]=(float)dk4m[i];
	}
	//Each gamma matrix is rescaled by akappa by flattening the gamval array
#if (defined USE_MKL || defined USE_BLAS)
	//Don't cuBLAS this. It is small and won't saturate the GPU. Let the CPU handle
	//it and just copy it later
	cblas_zdscal(5*4, akappa, gamval, 1);
#else
	for(int i=0;i<5;i++)
		for(int j=0;j<4;j++)
			gamval[i][j]*=akappa;
#endif
#pragma omp parallel for simd collapse(2) aligned(gamval:AVX,gamval_f:AVX)
	for(int i=0;i<5;i++)
		for(int j=0;j<4;j++)
			gamval_f[i][j]=(Complex_f)gamval[i][j];
#ifdef __NVCC__
	//More prefetching and marking as read-only (mostly)
	cudaMemAdvise(dk4p,(kvol+halo)*sizeof(double),..SetReadMostly,device);
	cudaMemAdvise(dk4m,(kvol+halo)*sizeof(double),..SetReadMostly,device);
	cudaMemAdvise(gamval,20*sizeof(Complex),..SetReadMostly,device);
	cudaMemPrefetchAsync(dk4p,(kvol+halo)*sizeof(double),device,NULL);
	cudaMemPrefetchAsync(dk4m,(kvol+halo)*sizeof(double),device,NULL);
	cudaMemPrefetchAsync(gamval,20*sizeof(Complex),device,NULL);

	cudaMallocManaged(&u11,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t,ndim*(kvol+halo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged(&u11t_f,ndim*(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged(&u12t_f,ndim*(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
#elif defined USE_MKL
	u11 = mkl_calloc(ndim*(kvol+halo),sizeof(complex),AVX);
	u12 = mkl_calloc(ndim*(kvol+halo),sizeof(complex),AVX);
	u11t = mkl_calloc(ndim*(kvol+halo),sizeof(complex),AVX);
	u12t = mkl_calloc(ndim*(kvol+halo),sizeof(complex),AVX);
	u11t_f = mkl_calloc(ndim*(kvol+halo),sizeof(Complex_f),AVX);
	u12t_f = mkl_calloc(ndim*(kvol+halo),sizeof(Complex_f),AVX);
#else
	u11 = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(complex));
	u12 = aligned_alloc(AVX,ndim*(kvol+halo)*,sizeof(complex));
	u11t = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(complex));
	u12t = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(complex));
	u11t_f = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex_f));
	u12t_f = aligned_alloc(AVX,ndim*(kvol+halo)*sizeof(Complex_f));
#endif
	if(istart==0){
		//Initialise a cold start to zero
		//memset is safe to use here because zero is zero 
#pragma omp parallel for simd aligned(u11:AVX) 
		//Leave it to the GPU?
		for(int i=0; i<kvol*ndim;i++)
			u11t[i]=1;
	}
	else if(istart>0){
		//Still thinking about how best to deal with PRNG
#if (defined USE_MKL&&!defined USE_RAN2)
		//Good news, casting works for using a double to create random complex numbers
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*kvol, u11t, -1, 1);
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*kvol, u12t, -1, 1);
#else
		//Depending if we have the RANLUX or SFMT19977 generator.	
		for(int i=0; i<kvol*ndim;i++){
			u11t[i]=2*(ran2(&seed)-0.5+I*(ran2(&seed)-0.5));
			u12t[i]=2*(ran2(&seed)-0.5+I*(ran2(&seed)-0.5));
		}
#endif
#ifdef __NVCC__
		int device=-1;
		cudaGetDevice(&device);
		cudaMemPrefetchAsync(u11t, ndim*kvol*sizeof(Complex),device,NULL);
		cudaMemPrefetchAsync(u12t, ndim*kvol*sizeof(Complex),device,NULL);
#endif
		Reunitarise();
		memcpy(u11, u11t, ndim*kvol*sizeof(complex));
		memcpy(u12, u12t, ndim*kvol*sizeof(complex));
	}
	else
		fprintf(stderr,"Warning %i in %s: Gauge fields are not initialised.\n", NOINIT, funcname);
#ifdef _DEBUG
	printf("Initialisation Complete\n");
#endif
	return 0;
}
int Hamilton(double *h, double *s, double res2){
	/* Evaluates the Hamiltonian function
	 * 
	 * Calls:
	 * =====
	 * SU2plaq
	 * Par_dsum
	 * Congradq
	 * Fill_Small_Phi
	 *
	 * Globals:
	 * =======
	 * pp, kmom, rank, ancgh, X0, Phi
	 *
	 * Parameters:
	 * ===========
	 * double *h:
	 * double *s:
	 * double res2:
	 *
	 * Returns:
	 * =======
	 * Zero on success. Integer Error code otherwise.
	 */	
	const char *funcname = "Hamilton";
	double hp;
	//Itereate over momentum terms.
#ifdef __NVCC__
	int device=-1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(pp,kmom*sizeof(double),device,NULL);
	cublasDnrm2(cublas_handle, kmom, pp, 1,&hp);
	hp*=hp;
#elif (defined USE_MKL || defined USE_BLAS)
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
	SU2plaq(&hg,&avplaqs,&avplaqt);

	double hf = 0; int itercg = 0;
#ifdef __NVCC__
	Complex *smallPhi;
	cudaMallocManaged(&smallPhi,kferm2Halo*sizeof(Complex),cudaMemAttachGlobal);
#elif defined USE_MKL
	complex *smallPhi = mkl_malloc(kferm2Halo*sizeof(complex),AVX);
#else
	complex *smallPhi = aligned_alloc(AVX,kferm2Halo*sizeof(complex));
#endif
	//Iterating over flavours
	for(int na=0;na<nf;na++){
		memcpy(X1,X0+na*kferm2Halo,kferm2*sizeof(complex));
		Congradq(na,res2,smallPhi,&itercg);
		ancgh+=itercg;
		Fill_Small_Phi(na, smallPhi);
		memcpy(X0+na*kferm2Halo,X1,kferm2*sizeof(complex));
#ifdef __NVCC__
		complex dot;
		cublasZdotc(cublas_handle,kferm2, smallPhi, 1, X1, 1, &dot);
		hf+=creal(dot);
#elif (defined USE_MKL || defined USE_BLAS)
		complex dot;
		cblas_zdotc_sub(kferm2, smallPhi, 1, X1, 1, &dot);
		hf+=creal(dot);
#else
		//It is a dot product of the flattend arrays, could use
		//a module to convert index to coordinate array...
		for(int j=0;j<kferm2;j++)
			hf+= conj(smallPhi[j])*X1[j];
#endif
	}
#ifdef USE_MKL
	mkl_free(smallPhi);
#else
	free(smallPhi);
#endif
	//hg was summed over inside of SU2plaq.
	Par_dsum(&hp); Par_dsum(&hf);
	*s=hg+hf; *h=*s+hp;
#ifdef _DEBUG
	if(!rank)
		printf("hg=%e; hf=%e; hp=%e; h=%e\n", hg, hf, hp, *h);
#endif

	return 0;
}
int Congradq(int na, double res, complex *smallPhi, int *itercg){
	/*
	 * Matrix Inversion via Conjugate Gradient
	 * Solves (M^†)Mx=Phi
	 * Impliments up/down partitioning
	 * 
	 * Calls:
	 * =====
	 * Fill_Small_Phi
	 * Hdslash
	 * Hdslashd
	 *
	 * Globals:
	 * =======
	 * Phi, X0, X1, jqq 
	 * WARNING: Due to how the common statement works in FORTRAN X1 here is the X1 in force and Hamilton, but
	 * 		called x in the FORTRAN congradq as so not to clash with the placeholder x1 (FORTRAN is 
	 * 		case insensitive.)
	 *
	 * Parameters:
	 * ==========
	 * int na: Flavour index
	 * double res: Resolution
	 * int itercg: Counts the iterations of the conjugate gradiant?
	 *
	 * Returns:
	 * =======
	 * 0 on success, integer error code otherwise
	 */
	const char *funcname = "Congradq";
	double resid = kferm2*res*res;
	*itercg = 0;
	//The κ^2 factor is needed to normalise the fields correctly
	//jqq is the diquark codensate and is global scope.
#ifdef __NVCC__
	__managed__
#endif
		complex fac = conj(jqq)*jqq*akappa*akappa;
	//These were evaluated only in the first loop of niterx so we'll just do it ouside of the loop.
	//These alpha and beta terms should be double, but that causes issues with BLAS. Instead we declare
	//them complex and work with the real part (especially for α_d)
	complex alphan;
	//Give initial values Will be overwritten if niterx>0
	double betad = 1.0; complex alphad=0; complex alpha = 1;
	//Because we're dealing with flattened arrays here we can call cblas safely without the halo
#ifdef __NVCC__
	complex *p, *r, *x2;
	Complex_f *p_f, *x1_f, *x2_f;
	int device=-1; cudaGetDevice(&device);
	cudaMallocManaged(&p, kferm2Halo*sizeof(complex),cudaMemAttachGlobal);
	cudaMemAdvise(p,kferm2Halo*sizeof(complex),cudaMemAdviseSetPreferredLocation,device);

	cudaMallocManaged(&p_f, kferm2Halo*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMemAdvise(p_f,kferm2Halo*sizeof(Complex_f),cudaMemAdviseSetPreferredLocation,device);

	cudaMallocManaged(&r, kferm2Halo*sizeof(complex),cudaMemAttachGlobal);
	cudaMemAdvise(r,kferm2Halo*sizeof(complex),cudaMemAdviseSetPreferredLocation,device);

	cudaMalloc(&x1_f, kferm2Halo*sizeof(Complex_f));
	cudaMalloc(&x2_f, kferm2Halo*sizeof(Complex_f));

	cudaMallocManaged(&x2, kferm2Halo*sizeof(complex),cudaMemAttachGlobal);
	cudaMemAdvise(x2,kferm2Halo*sizeof(complex),cudaMemAdviseSetPreferredLocation,device);
	cudaMemPrefetchAsync(x2,kferm2Halo*sizeof(Complex).device,NULL);
#elif defined USE_MKL
	complex *p  = mkl_calloc(kferm2Halo,sizeof(complex),AVX);
	complex *r  = mkl_calloc(kferm2,sizeof(complex),AVX);
	complex *x2=mkl_calloc(kferm2Halo, sizeof(complex), AVX);

	Complex_f *p_f  = mkl_calloc(kferm2Halo,sizeof(Complex_f),AVX);
	Complex_f *x2_f=mkl_calloc(kferm2Halo, sizeof(Complex_f), AVX);
	Complex_f *x1_f=mkl_calloc(kferm2Halo, sizeof(Complex_f), AVX);
#else
	complex *p  = calloc(kferm2Halo,sizeof(complex));
	complex *r  = calloc(kferm2,sizeof(complex));
	complex *x1=calloc(kferm2Halo,sizeof(complex));
	complex *x2=calloc(kferm2Halo,sizeof(complex));
#endif
	Fill_Small_Phi(na, smallPhi);
	//Instead of copying elementwise in a loop, use memcpy.
	memcpy(p, X1, kferm2*sizeof(complex));
#ifdef __NVCC__
	cudaMemPrefetchAsync(p,kferm2Halo*sizeof(Complex).device,NULL);
#endif
	memcpy(r, smallPhi, kferm2*sizeof(complex));

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	complex betan;
	for(int niterx=0; niterx<niterc; niterx++){
		(*itercg)++;
#pragma omp parallel for simd
		for(int i=0;i<kferm2;i++)
			p_f[i]=(Complex_f)p[i];
#ifdef	__NVCC__
		cudaMemPrefetchAsync(p_f,kferm2Halo*sizeof(Complex_f).device,NULL);
#endif
		//x2 =  (M^†M)p 
		Hdslash_f(x1_f,p_f); Hdslashd_f(x2_f, x1_f);
#pragma omp parallel for simd
		for(int i=0;i<kferm2;i++)
			x2[i]=(Complex)x2_f[i];
#ifdef	__NVCC__
		//x2 =  (M^†M+J^2)p 
		cublasZaxpy(cublas_handle,kferm2,&fac,p,1,x2,1);
#elif (defined USE_MKL || defined USE_BLAS)
		//x2 =  (M^†M+J^2)p 
		cblas_zaxpy(kferm2, &fac, p, 1, x2, 1);
#else
#pragma omp parallel for simd
		for(int i=0; i<kferm2; i++)
			x2[i]+=fac*p[i];
#endif
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(niterx){
			//α_d= p* (M^†M+J^2)p
#ifdef __NVCC__
			cublasZdotc(cublas_handle,kferm2,p,1,x2,1,&alphad);
#elif (defined USE_MKL || defined USE_BLAS)
			cblas_zdotc_sub(kferm2, p, 1, x2, 1, &alphad);
#else
			alphad=0;
			for(int i=0; i<kferm2; i++)
				alphad+=conj(p[i])*x2[i];
#endif
			//TODO: Implement Par_csum. For now I'll cast it into a double for the reduction.
			//And reduce. α_d does have a complex component but we only care about the real part
			Par_dsum(&alphad);
			//α=α_n/α_d = (r.r)/p(M^†M)p 
			alpha=creal(alphan)/creal(alphad);
			//x-αp, 
#ifdef __NVCC__
			cublasZaxpy(cublas_handle,kferm2,&alpha,p,1,X1,1);
#elif (defined USE_MKL || defined USE_BLAS)
			cblas_zaxpy(kferm2, &alpha, p, 1, X1, 1);
#else
			for(int i=0; i<kferm2; i++)
				X1[i]+=alpha*p[i];
#endif
		}			
		// r_n+1 = r_n-α(M^† M)p_n and β_n=r*.r
#ifdef	__NVCC__
		alpha*=-1;
		cublasZaxpy(cublas_handle, kferm2,&alpha,x2,1,r,1);
		alpha*=-1;
		cublasDznrm2(cublas_handle,kferm2,r,1,&betan);
		betan *= betan;
#elif (defined USE_MKL || defined USE_BLAS)
		alpha *= -1;
		cblas_zaxpy(kferm2, &alpha, x2, 1, r, 1);
		//Undo the negation for the BLAS routine
		alpha*=-1;
		betan = cblas_dznrm2(kferm2, r,1);
		//Gotta square it to "undo" the norm
		betan *= betan;
#else
		betan=0;
		for(int i=0; i<kferm2; i++){
			r[i]-=alpha*x2[i];
			betan += conj(r[i])*r[i];
		}
#endif
		//And... reduce.
		Par_zsum(&betan);
		if(creal(betan)<resid){ 
#ifdef _DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %e toler = %e\n", niterx+1, creal(betan), resid);
#endif
			break;
		}
		else if(niterx==niterc-1){
			if(!rank) fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i β_n=%e\n", ITERLIM, funcname, niterc, creal(betan));
			break;
		}
		//Here we evaluate β=(r_{k+1}.r_{k+1})/(r_k.r_k) and then shuffle our indices down the line.
		//On the first iteration we define beta to be zero.
		complex beta = (niterx) ?  creal(betan)/betad : 0;
		betad=betan; alphan=betan;
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multipyling y by
		//β instead of x.
#if (defined USE_MKL||defined USE_BLAS)
		complex a = 1.0;
		//There is cblas_zaxpby in the MKL and AMD though, set a = 1 and b = β.
		//If we get a small enough β_n before hitting the iteration cap we break
		cblas_zaxpby(kferm2, &a, r, 1, &beta,  p, 1);
#else 
		for(int i=0; i<kferm2; i++)
			p[i]=r[i]+beta*p[i];
#endif
	}
#ifdef __NVCC__
	cudaFree(x2); cudaFree(p); cudaFree(r);
	cudaFree(x1_f);cudaFree(x2_f); cudaFree(p_f);
#elif defined USE_MKL
	mkl_free(x1_f); mkl_free(x2); mkl_free(p); mkl_free(r);
	mkl_free(p_f); mkl_free(x2_f);
#else
	free(x1), free(x2), free(p), free(r);
#endif
	return 0;
}
int Congradp(int na, double res, int *itercg){
	/*
	 * Matrix Inversion via Conjugate Gradient
	 * Solves (M^†)Mx=Phi
	 * No even/odd partitioning
	 *
	 * Calls:
	 * =====
	 * Fill_Small_Phi
	 * Hdslash
	 * Hdslashd
	 *
	 * Globals:
	 * =======
	 * Phi, X0, xi
	 * WARNING: Due to how the FORTRAN common statement works, you can have different names for the same global
	 * 		variable in different functions. It is the order they appear on the list that matters. xi here
	 * 		was called xi in the FORTRAN Measure subroutine and x in the congradp subroutine. We'll use
	 * 		xi for both as it does not appear elsewhere
	 * 		xi stores the result
	 *
	 * Parameters:
	 * ==========
	 * int na: Flavour index
	 * double res:
	 * int itercg:
	 *
	 * Returns:
	 * =======
	 * 0 on success, integer error code otherwise
	 */
	const char *funcname = "Congradp";
	double resid = kferm*res*res;
	*itercg = 0;
	//The κ^2 factor is needed to normalise the fields correctly
	//jqq is the diquark codensate and is global scope.
	complex fac = conj(jqq)*jqq*akappa*akappa;
	//These were evaluated only in the first loop of niterx so we'll just do it ouside of the loop.
	//These alpha and beta terms should be double, but that causes issues with BLAS. Instead we declare
	//them complex and work with the real part (especially for α_d)
	complex alphan;
	//Give initial values Will be overwritten if niterx>0
	double betad = 1.0; double alphad=0; complex alpha = 1;
#ifdef __NVCC__
	complex *p, *r;
	int device; cudaGetDevice(&device);
	cudaMallocManaged(&p, kfermHalo*sizeof(complex),cudaMemAttachGlobal);
	cudaMemAdvise(p,kfermHalo*sizeof(complex),cudaMemAdviseSetPreferredLocation,device);

	cudaMallocManaged(&r, kfermHalo*sizeof(complex),cudaMemAttachGlobal);
	cudaMemAdvise(r,kfermHalo*sizeof(complex),cudaMemAdviseSetPreferredLocation,device);
#elif defined USE_MKL
	complex *p  = mkl_malloc(kfermHalo*sizeof(complex),AVX);
	complex *r  = mkl_malloc(kferm*sizeof(complex),AVX);
#else
	complex *p  = malloc(kfermHalo*sizeof(complex));
	complex *r  = malloc(kferm*sizeof(complex));
#endif
	//Instead of copying elementwise in a loop, use memcpy.
	memcpy(p, xi, kferm*sizeof(complex));
	memcpy(r, Phi+na*kfermHalo, kferm*sizeof(complex));

	// Declaring placeholder arrays 
	// This x1 is NOT related to the /common/vectorp/X1 in the FORTRAN code and should not
	// be confused with X1 the global variable
#ifdef __NVCC__
	complex *x1, *x2;
	cudaMemPrefetchAsync(p,kfermHalo*sizeof(complex),device,NULL);
	cudaMalloc(&x1, kferm2Halo*sizeof(complex));

	cudaMallocManaged(&x2, kferm2Halo*sizeof(complex),cudaMemAttachGlobal);
	cudaMemAdvise(x2,kferm2Halo*sizeof(complex),cudaMemAdviseSetPreferredLocation,device);
#elif defined USE_MKL
	complex *x1=mkl_malloc(kfermHalo*sizeof(complex), AVX);
	complex *x2=mkl_malloc(kfermHalo*sizeof(complex), AVX);
#else
	complex *x1=aligned_alloc(AVX,kfermHalo*sizeof(complex));
	complex *x2=aligned_alloc(AVX,kfermHalo*sizeof(complex));
#endif

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	complex betan;
	for(int niterx=0; niterx<=niterc; niterx++){
		(*itercg)++;
		Dslash(x1,p);
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(niterx){
			//x*.x
#ifdef __NVCC__
			cublasDznrm2(cublas_handle,kferm, x1, 1,&alphad);
			alphad *= alphad;
#elif (defined USE_MKL || defined USE_BLAS)
			alphad = cblas_dznrm2(kferm, x1, 1);
			alphad *= alphad;
#else
			alphad=0;
			for(int i = 0; i<kferm; i++)
				alphad+=conj(x1[i])*x1[i];
#endif
			Par_dsum(&alphad);
			//α=(r.r)/p(M^†)Mp
			alpha=creal(alphan)/alphad;
			//x+αp
#ifdef __NVCC__
			cublasZaxpy(cublas_handle,kferm, &alpha, p, 1, xi, 1);
#elif (defined USE_MKL || defined USE_BLAS)
			cblas_zaxpy(kferm, &alpha, p, 1, xi, 1);
#else
			for(int i = 0; i<kferm; i++)
				xi[i]+=alpha*p[i];
#endif
		}
		//x2=(M^†)x1=(M^†)Mp
		Dslashd(x2,x1);
		//r-α(M^†)Mp and β_n=r*.r
#ifdef __NVCC__
		alpha*=-1;
		cublasZaxpy(cublas_handle,kferm, &alpha, x2, 1, r, 1);
		alpha*=-1;
		//r*.r
		cublasDznrm2(cublas_handle,kferm, r,1,&betan);
		//Gotta square it to "undo" the norm
		betan *= betan;
#elif (defined USE_MKL || defined USE_BLAS)
		alpha*=-1;
		cblas_zaxpy(kferm, &alpha, x2, 1, r, 1);
		alpha*=-1;
		//r*.r
		betan = cblas_dznrm2(kferm, r,1);
		//Gotta square it to "undo" the norm
		betan *= betan;
#else
		//Just like Congradq, this loop could be unrolled but will need a reduction to deal with the betan 
		//addition.
		betan = 0;
		//If we get a small enough β_n before hitting the iteration cap we break
		for(int i = 0; i<kferm;i++){
			r[i]-=alpha*x2[i];
			betan+=conj(r[i])*r[i];
		}
#endif
		//This is basically just congradq at the end. Check there for comments
		Par_zsum(&betan);
		if(creal(betan)<resid){
#ifdef _DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %e toler = %e\n", niterx+1, creal(betan), resid);
#endif
			break;
		}
		else if(niterx==niterc-1){
			if(!rank) fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i β_n=%e\n", ITERLIM, funcname, niterc, creal(betan));
			break;
		}
		complex beta = (niterx) ? betan/betad : 0;
		betad=creal(betan); alphan=betan;
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multipyling y by 
		//β instead of x.
		//There is cblas_zaxpby in the MKL though, set a = 1 and b = β.
#if (defined USE_MKL||defined USE_BLAS)
		complex a = 1;
		cblas_zaxpby(kferm, &a, r, 1, &beta,  p, 1);
#else
		for(int i=0; i<kferm; i++)
			p[i]=r[i]+beta*p[i];
#endif
	}
#ifdef	__NVCC__
	cudaFree(x2); cudaFree(p); cudaFree(r);
	cudaFree(x1);
#elif defined USE_MKL
	mkl_free(p); mkl_free(r); mkl_free(x1); mkl_free(x2);
#else
	free(p); free(r); free(x1); free(x2);
#endif
	return 0;
}
inline int Z_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu)
{
	//FORTRAN had a second parameter m gving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
#pragma omp parallel for simd aligned (x:AVX,y:AVX,table:AVX)
	for(int i=0; i<n; i++)
		x[i]=y[table[i*ndim+mu]*ndim+mu];
	return 0;
}
inline int Fill_Small_Phi(int na, Complex *smallPhi)
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
	 * complex *smallPhi:	  The target array
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Fill_Small_Phi";
	//BIG and small phi index
#pragma omp parallel for simd aligned(smallPhi:AVX,Phi:AVX)
	for(int i = 0; i<kvol;i++)
#pragma unroll
		for(int idirac = 0; idirac<ndirac; idirac++)
#pragma unroll
			for(int ic= 0; ic<nc; ic++){
				//	  PHI_index=i*16+j*2+k;
				smallPhi[(i*ndirac+idirac)*nc+ic]=Phi[((na*kvol+i)*ngorkov+idirac)*nc+ic];
			}
	return 0;
}
double Norm_squared(Complex *z, int n)
{
	/* Called znorm2 in the original FORTRAN.
	 * Takes a complex number vector of length n and finds the square of its 
	 * norm using the formula
	 * 
	 *	    |z(i)|^2 = z(i)Xz*(i)
	 *
	 * Parameters:
	 * ==========
	 *  complex z:	The Number being normalised
	 *  int n:	The length of the vector
	 * 
	 * Returns:
	 * =======
	 *  double: The norm of the complex number
	 * 
	 */
	//BLAS? Use cblas_zdotc instead for vectorisation
	const char *funcname = "Norm_squared";
	double norm = 0;
#pragma omp parallel for simd reduction(+:norm)
	for(int i=0; i<n; i++)
		norm+=z[i]*conj(z[i]);
	return norm;
}
