#include <coord.h>
#include <math.h>
#include <par_mpi.h>
#include <random.h>
#include <multiply.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <su2hmc.h>

//Extern definitions, especially default values for fmu, beta and akappa
complex jqq = 0;
double fmu = 0.0;
double beta = 1.7;
double akappa = 0.1780;
const int gamin[4][4] =	{{3,2,1,0},
	{3,2,1,0},
	{2,3,0,1},
	{2,3,0,1}};
complex gamval[5][4] =	{{-I,-I,I,I},
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
	int istart = 0;
	ibound = 1;
	int iwrite = 1;
	int iprint = 1; //For the measures
	int icheck = 5; //Save conf (ICHEC could be a better name...)
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
		//I'm hoping to use scoping to avoid any accidents.
		FILE *midout;
		//Instead of hardcoding so the error messages are easier to impliment
		char *filename = "midout";
		char *fileop = "rb";
		if( !(midout = fopen(filename, fileop) ) ){
			fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n"\
					, OPENERROR, funcname, filename, fileop);
			exit(OPENERROR);
		}
		fscanf(midout, "%lf %lf %lf %lf %lf %lf %lf %d %d", &dt, &beta, &akappa, &ajq, &athq, &fmu, &delb, &stepl, &ntraj);
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
	Par_dcopy(&athq); Par_dcopy(&fmu); //Par_dcopy(&delb); Not used?
	Par_icopy(&stepl); Par_icopy(&ntraj); 
	jqq=ajq*cexp(athq*I);
	Par_ranset(&seed);

	//Initialisation
	//istart < 0: Start from tape?!? How old is this code?
	//istart = 0: Ordered/Cold Start
	//			For some reason this leaves the trial fields as zero in the FORTRAN code?
	//istart > 0: Random/Hot Start
	Init(istart);

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
	char *outname = "output"; char *outop="w";
	FILE *output;
	if(!rank){
		if(!(output=fopen(outname, outop) )){
			fprintf(stderr,"Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",OPENERROR,funcname,outname,outop);
			MPI_Finalise();
			exit(OPENERROR);
		}
		printf("hg = %e, <Ps> = %e, <Pt> = %e, <Poly> = %e\n", hg, avplaqs, avplaqt, poly);
		fprintf(output, "ksize = %i ksizet = %i Nf = %i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i β = %e\nκ = %e μ = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %i\n",
				ksize, ksizet, nf, dt, traj, ntraj, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#ifdef _DEBUG
		//Print to terminal during debugging
		printf("ksize = %i ksizet = %i Nf = %i\nTime step dt = %e Trajectory length = %e\n"\
				"No. of Trajectories = %i β = %e\nκ = %e μ = %e\nDiquark source = %e Diquark phase angle = %e\n"\
				"Stopping Residuals: Guidance: %e Acceptance: %e, Estimator: %e\nSeed = %i\n",
				ksize, ksizet, nf, dt, traj, ntraj, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#endif
	}
	//Initialise for averages
	//======================
	double actiona = 0.0; double vel2a = 0.0; double pbpa = 0.0; double endenfa = 0.0; double denfa = 0.0;
	double yav = 0.0; double yyav = 0.0; 

	int naccp = 0; int ipbp = 0; int itot = 0;

	ancg = 0; ancgh = 0;
	//This was originally in the half-step of the fortran code, but it makes more sense to declare
	//it outside the loop
	const	double d = dt*0.5;
	//Start of classical evolution
	//===========================
	double pbp;
	complex qq;
	//Initialise Some Arrays. Leaving it late for scoping
	//check the sizes in sizes.h
	double *dSdpi;
	//There is absolutely no reason to keep the trial fields as zero now, so I won't
	memcpy(u11t,u11,kvol*ndim*sizeof(complex));
	Trial_Exchange();
#ifdef __NVCC__
	cudaMallocManaged(&R1, kfermHalo*sizeof(complex));
	cudaMallocManaged(&xi, kfermHalo*sizeof(complex));
	cudaMallocManaged(&Phi, nf*kfermHalo*sizeof(complex));
	cudaMallocManaged(&X0, nf*kfermHalo*sizeof(complex));
	cudaMallocManaged(&X1, kferm2Halo*sizeof(complex));
	cudaMallocManaged(&dSdpi, kmommHalo*sizeof(complex));
	cudaMallocManaged(&pp, kmomHalo*sizeof(complex));
#elif defined USE_MKL
	R1= mkl_malloc(kfermHalo*sizeof(complex),AVX);
	xi= mkl_malloc(kfermHalo*sizeof(complex),AVX);
	Phi= mkl_malloc(nf*kfermHalo*sizeof(complex),AVX); 
	X0= mkl_malloc(nf*kferm2Halo*sizeof(complex),AVX); 
	X1= mkl_malloc(kferm2Halo*sizeof(complex),AVX); 
	dSdpi = mkl_malloc(kmomHalo*sizeof(double), AVX);
	pp = mkl_malloc(kmomHalo*sizeof(double), AVX);
#else
	R1= malloc(kfermHalo*sizeof(complex));
	xi= malloc(kfermHalo*sizeof(complex));
	Phi= malloc(nf*kfermHalo*sizeof(complex)); 
	X0= malloc(nf*kferm2Halo*sizeof(complex)); 
	X1= malloc(kferm2Halo*sizeof(complex)); 
	dSdpi = malloc(kmomHalo*sizeof(double));
	pp = malloc(kmomHalo*sizeof(double));
#endif
	//Arabic for hour/watch so probably not defined elsewhere like TIME potentially is
#if (defined SA3AT && defined _OPENMP)
	double start_time = omp_get_wtime();
#endif
	for(int itraj = 1; itraj <= ntraj; itraj++){
#ifdef _DEBUG
		if(!rank)
			printf("Starting itraj %i\n", itraj);
#endif
		for(int na=0; na<nf; na++){
#ifdef USE_MKL
			//Probably makes sense to declare this outside the loop
			//but I do like scoping/don't want to break anything else just yeat
			complex *R=mkl_malloc(kfermHalo*sizeof(complex),AVX);
			//Multiply the dimension of R by 2 because R is complex
			//The FORTRAN code had two gaussian routines.
			//gaussp was the normal box-muller and gauss0 didn't have 2 inside the square root
			//Using σ=1/sqrt(2) in these routines has the same effect as gauss0
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, R, 0, 1/sqrt(2));
#else
			complex *R=malloc(kfermHalo*sizeof(complex));
			Gauss_z(R, kferm, 0, 1/sqrt(2));
#endif
			Dslashd(R1, R);
			memcpy(Phi+na*kfermHalo,R1, nc*ngorkov*kvol*sizeof(complex));
			//Up/down partitioning (using only pseudofermions of flavour 1)
#pragma omp parallel for simd aligned(X0:AVX,R1:AVX)
			for(int i=0; i<kvol; i++)
				for(int idirac = 0; idirac < ndirac; idirac++){
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc]=R1[(i*ngorkov+idirac)*nc];
					X0[((na*(kvol+halo)+i)*ndirac+idirac)*nc+1]=R1[(i*ngorkov+idirac)*nc+1];
				}
#ifdef USE_MKL
			mkl_free(R);
#else
			free(R);
#endif
		}	
		//Heatbath
		//========
		//We're going to make the most of the new Gauss_d routine to send a flattened array
		//and do this all in one step.
#ifdef USE_MKL
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, kmom, pp, 0, 1);
#else
		Gauss_d(pp, kmom, 0, 1);
#endif

		//Initialise Trial Fields
		memcpy(u11t, u11, ndim*kvol*sizeof(complex));
		memcpy(u12t, u12, ndim*kvol*sizeof(complex));
		Trial_Exchange();
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
#if (defined USE_MKL || defined USE_BLAS)
		cblas_daxpy(nadj*ndim*kvol, -d, dSdpi, 1, pp, 1);
#else
		for(int i=0;i<kmom;i++)
			pp[i]-=d*dSdpi[i];
#endif
		//Main loop for classical time evolution
		//======================================
		for(int step = 1; step<=stepmax; step++){
#ifdef _DEBUG
			if(!rank)
				printf("step: %i\n", step);
#endif
			//The FORTRAN redefines d=dt here, which makes sense if you have a limited line length.
			//I'll stick to using dt though.
#pragma omp parallel for simd collapse(2) aligned(pp:AVX, u11t:AVX, u12t:AVX)
			for(int i=0;i<kvol;i++)
				for(int mu = 0; mu<ndim; mu++){
					//Sticking to what was in the FORTRAN for variable names.
					//CCC for cosine SSS for sine AAA for...
					double AAA = dt*sqrt(pp[i*nadj*ndim+mu]*pp[i*nadj*ndim+mu]\
							+pp[(i*nadj+1)*ndim+mu]*pp[(i*nadj+1)*ndim+mu]\
							+pp[(i*nadj+2)*ndim+mu]*pp[(i*nadj+2)*ndim+mu]);
					double CCC = cos(AAA);
					double SSS = dt*sin(AAA)/AAA;
					complex a11 = CCC+I*SSS*pp[(i*nadj+2)*ndim+mu];
					complex a12 = pp[(i*nadj+1)*ndim+mu]*SSS + I*SSS*pp[i*nadj*ndim+mu];
					//b11 and b12 are u11t and u12t terms, so we'll use u12t directly
					//but use b11 for u11t to prevent RAW dependency
					complex b11 = u11t[i*ndim+mu];
					u11t[i*ndim+mu] = a11*b11-a12*conj(u12t[i*ndim+mu]);
					u12t[i*ndim+mu] = a11*u12t[i*ndim+mu]+a12*conj(b11);
				}
			Trial_Exchange();
			Reunitarise();
			//p(t+3dt/2)=p(t+dt/2)-dSds(t+dt)*dt
			Force(dSdpi, 0, rescgg);
			//Need to check Par_granf again 
			//The same for loop is given in both the if and else
			//statement but only the value of d changes. This is due to the break in the if part
			if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
#if (defined USE_MKL || defined USE_BLAS)
				cblas_daxpy(ndim*nadj*kvol, -d, dSdpi, 1, pp, 1);
#else
				for(int i = 0; i<kmom; i++)
					pp[i]-=d*dSdpi[i];
#endif
				itot+=step;
				break;
			}
			else{
#if (defined USE_MKL || defined USE_BLAS)
				cblas_daxpy(ndim*nadj*kvol, -dt, dSdpi, 1, pp, 1);
#else
				for(int i = 0; i<kvol; i++)
					for(int iadj=0; iadj<nadj; iadj++)
						for(int mu = 0; mu < ndim; mu++)
							pp[(i*nadj+iadj)*nc+mu]-=d*dSdpi[(i*nadj+iadj)*nc+mu];
#endif

			}
		}
		//Monte Carlo step: Accept new fields with the probability of min(1,exp(H0-X0))
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
		if(dH<0 && Par_granf()<=y){
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
			memcpy(u11,u11t,ndim*kvol*sizeof(complex));
			memcpy(u12,u12t,ndim*kvol*sizeof(complex));
			naccp++;
			//Divide by gvol because of halos?
			action=S1/gvol;
		}
		actiona+=action; 
		double vel2=0.0;

#if (defined USE_MKL || defined USE_BLAS)
		vel2 = cblas_dnrm2(kmom, pp, 1);
		vel2*=vel2;
#else
#pragma unroll
		for(int i=0; i<kmom; i++)
			vel2+=pp[i]*pp[(i)];
#endif
		Par_dsum(&vel2);
		vel2a+=vel2/(ndim*nadj*gvol);

		if((itraj/iprint)*iprint==itraj){
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
				if(!rank){
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
							case(1):
								//The origninal code implicitly created these files with the name fort.XX where XX
								//is the file label from FORTRAN. We'll stick with that for now.
								{
									FILE *fortout;
									char *fortname = "fort11";
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
								}
							case(2):
								//The origninal code implicitly created these files with the name
								//fort.XX where XX is the file label
								//from FORTRAN. We'll stick with that for now.
								{
									FILE *fortout;
									char *fortname = "fort12"; 
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
								}

							case(3):
								{
									FILE *fortout;
									char *fortname = "fort13";
									char *fortop= (itraj==1) ? "w" : "a";
									if(!(fortout=fopen(fortname, fortop) )){
										fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
												OPENERROR, funcname, fortname, fortop);
										MPI_Finalise();
										exit(OPENERROR);
									}
									if(itraj==1)
										fprintf(fortout, "Re(qq)\t\n");
									fprintf(fortout, "%e\n", creal(qq));
									fclose(fortout);
								}
							default: continue;
						}
				}
		}
		if((itraj/icheck)*icheck==itraj){
			//ranget(seed);
			Par_swrite(itraj);
		}
		if(!rank)
			fflush(output);
	}
#if (defined SA3AT && defined _OPENMP)
	double elapsed = omp_get_wtime()-start_time;
#endif
	//End of main loop
	//Free arrays
#ifdef __NVCC__
	cudaFree(dk4m); cudaFree(dk4p); cudaFree(R1); cudaFree(dSdpi); cudaFree(pp);
	cudaFree(Phi); cudaFree(u11t); cudaFree(u12t); cudaFree(xi);
	cudaFree(X0); cudaFree(X1); cudaFree(u11); cudaFree(u12);
	cudaFree(id); cudaFree(iu); cudaFree(hd); cudaFree(hu);
#elif defined USE_MKL
	mkl_free(dk4m); mkl_free(dk4p); mkl_free(R1); mkl_free(dSdpi); mkl_free(pp);
	mkl_free(Phi); mkl_free(u11t); mkl_free(u12t); mkl_free(xi);
	mkl_free(X0); mkl_free(X1); mkl_free(u11); mkl_free(u12);
	mkl_free(id); mkl_free(iu); mkl_free(hd); mkl_free(hu);
	mkl_free(pcoord);
#else
	free(dk4m); free(dk4p); free(R1); free(dSdpi); free(pp); free(Phi);
	free(u11t); free(u12t); free(xi); free(X0); free(X1);
	free(u11); free(u12); free(id); free(iu); free(hd); free(hu);
	free(pcoord);
#endif
#if (defined SA3AT && defined _OPENMP)
	FILE *sa3at = fopen("Bench_times.csv", "a");
	fprintf(sa3at, "%lu,%lu,%lu,%lu,%f,%f\n",nx,nt,kvol,nthreads,elapsed,elapsed/ntraj);
	fclose(sa3at);
#endif
	actiona/=ntraj; vel2a/=ntraj; pbpa/=ipbp; endenfa/=ipbp; denfa/=ipbp;
	ancg/=nf*itot; ancgh/=2*nf*ntraj; yav/=ntraj; yyav=yyav/ntraj - yav*yav;
	double atraj=dt*itot/ntraj;

	if(!rank){
		fprintf(output, "Averages for the last %i trajectories\n"\
				"Number of acceptances: %i Average Trajectory Length = %e\n"\
				"exp(dh) = %e +/- %e\n"\
				"Average number of congrad iter guidance: %e acceptance %e\n"\
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
	cudaMallocManaged(&dk4m,(kvol+halo)*sizeof(double));
	cudaMallocManaged(&dk4p,(kvol+halo)*sizeof(double));
#elif defined USE_MKL
	dk4m = mkl_malloc((kvol+halo)*sizeof(double), AVX);
	dk4p = mkl_malloc((kvol+halo)*sizeof(double), AVX);
#else
	dk4m = malloc((kvol+halo)*sizeof(double));
	dk4p = malloc((kvol+halo)*sizeof(double));
#endif
#pragma omp parallel for simd aligned(dk4m:AVX,dk4p:AVX)
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
		for(int i= 0; i<kvol3; i++){
			int k = kvol - kvol3 + i;
			dk4p[k]*=-1;
			dk4m[k]*=-1;
		}
	}
	//These are constant so swap the halos when initialising and be done with it
	for(int mu = 0; mu <ndim; mu++){
		DHalo_swap_dir(dk4p, 1, 3, UP);
		DHalo_swap_dir(dk4m, 1, 3, UP);
	}
	//Each gamma matrix is rescaled by akappa by flattening the gamval array
#if (defined USE_MKL || defined USE_BLAS)
	cblas_zdscal(5*4, akappa, gamval, 1);
#else
	for(int i=0;i<5;i++)
		for(int j=0;j<4;j++)
			gamval[i][j]*=akappa;
#endif
#ifdef __NVCC__
	cudaMallocManaged(u11,ndim*(kvol+halo)*sizeof(complex));
	cudaMallocManaged(u12,ndim*(kvol+halo)*sizeof(complex));
	cudaMallocManaged(u11t,ndim*(kvol+halo)*sizeof(complex));
	cudaMallocManaged(u12t,ndim*(kvol+halo)*sizeof(complex));
#elif defined USE_MKL
	u11 = mkl_malloc(ndim*(kvol+halo)*sizeof(complex),AVX);
	u12 = mkl_calloc(ndim*(kvol+halo),sizeof(complex),AVX);
	u11t = mkl_calloc(ndim*(kvol+halo),sizeof(complex),AVX);
	u12t = mkl_calloc(ndim*(kvol+halo),sizeof(complex),AVX);
#else
	u11 = malloc(ndim*(kvol+halo)*sizeof(complex));
	u12 = calloc(ndim*(kvol+halo),sizeof(complex));
	u11t = calloc(ndim*(kvol+halo),sizeof(complex));
	u12t = calloc(ndim*(kvol+halo),sizeof(complex));
#endif
	if(istart==0){
		//Initialise a cold start to zero
		//memset is safe to use here because zero is zero 
#pragma omp parallel for simd aligned(u11t:AVX) 
		for(int i=0; i<kvol*ndim;i++)
			u11[i]=1;
	}
	else if(istart>0){
		//#ifdef __NVCC__
		//		complex *cu_u1xt;
		//		cudaMallocManaged(&cu_u1xt, ndim*kvol*sizeof(complex));

#if defined USE_MKL
		//Good news, casting works for using a double to create random complex numbers
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*(kvol+halo), u11t, -1, 1);
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*(kvol+halo), u12t, -1, 1);
#else
		//Depending if we have the RANLUX or SFMT19977 generator.	
#pragma unroll
		for(int i=0; i<kvol*ndim;i++){
			u11t[i]=sfmt_genrand_real1(&sfmt)+sfmt_genrand_real1(&sfmt)*I;
			u12t[i]=sfmt_genrand_real1(&sfmt)+sfmt_genrand_real1(&sfmt)*I;
		}
#endif
		Reunitarise();
		memcpy(u11, u11t, ndim*(kvol+halo)*sizeof(complex));
		memcpy(u12, u12t, ndim*(kvol+halo)*sizeof(complex));
	}
	else{
		fprintf(stderr,"Warning %i in %s: Gauge fields are not initialised.\n", NOINIT, funcname);
	}
#ifdef _DEBUG
	printf("Initialisation Complete\n");
#endif
	return 0;
}
int Gauge_force(double *dSdpi){
	/*
	 * Calculates dSdpi due to the Wilson Action at each intermediate time
	 *
	 * Globals:
	 * =======
	 * u11t, u12t, u11, u12, iu, id, beta
	 * Calls:
	 * =====
	 * Z_Halo_swap_all, Z_gather, Z_Halo_swap_dir
	 */
	const char *funcname = "Gauge_force";

	//We define zero halos for debugging
	//	#ifdef _DEBUG
	//		memset(u11t[kvol], 0, ndim*halo*sizeof(complex));	
	//		memset(u12t[kvol], 0, ndim*halo*sizeof(complex));	
	//	#endif
#ifdef USE_MKL
	complex *z = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
#else
	complex *z = malloc((kvol+halo)*sizeof(complex));
#endif
	//Was a trial field halo exchange here at one point.
#ifdef USE_MKL
	complex *Sigma11 = mkl_malloc(kvol*sizeof(complex),AVX); 
	complex *Sigma12= mkl_malloc(kvol*sizeof(complex),AVX); 
	complex *u11sh = mkl_malloc(kvol*sizeof(complex),AVX); 
	complex *u12sh = mkl_malloc(kvol*sizeof(complex),AVX); 
#else
	complex *Sigma11 = malloc(kvol*sizeof(complex)); 
	complex *Sigma12= malloc(kvol*sizeof(complex)); 
	complex *u11sh = malloc(kvol*sizeof(complex)); 
	complex *u12sh = malloc(kvol*sizeof(complex)); 
#endif
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
		memset(Sigma11,0, kvol*sizeof(complex));
		memset(Sigma12,0, kvol*sizeof(complex));
		for(int nu=0; nu<ndim; nu++){
			if(mu!=nu){
				//The +ν Staple
#pragma omp parallel for simd aligned(u11t:AVX,u12t:AVX,Sigma11:AVX,Sigma12:AVX)
				for(int i=0;i<kvol;i++){

					int uidm = iu[mu+ndim*i];
					int uidn = iu[nu+ndim*i];
					complex	a11=u11t[uidm*ndim+nu]*conj(u11t[uidn*ndim+mu])+\
							    u12t[uidm*ndim+nu]*conj(u12t[uidn*ndim+mu]);
					complex	a12=-u11t[uidm*ndim+nu]*u12t[uidn*ndim+mu]+\
							    u12t[uidm*ndim+nu]*u11t[uidn*ndim+mu];

					Sigma11[i]+=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
					Sigma12[i]+=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+nu];
				}
#if (defined USE_MKL || defined USE_BLAS)
				cblas_zcopy(kvol+halo, u11t+nu, 4, z, 1);
#else
#pragma unroll
				for(int i=0; i<kvol+halo;i++)
					z[i]=u11t[i*ndim+nu];
#endif
				Z_gather(u11sh, z, kvol, id+nu);
#if (defined USE_MKL || defined USE_BLAS)
				cblas_zcopy(kvol+halo, u12t+nu, 4, z, 1);
#else
#pragma unroll
				for(int i=0; i<kvol+halo;i++)
					z[i]=u12t[i*ndim+nu];
#endif
				Z_gather(u12sh, z, kvol, id+nu);
				ZHalo_swap_dir(u11sh, 1, mu, DOWN);
				ZHalo_swap_dir(u12sh, 1, mu, DOWN);
				//Next up, the -ν staple
#pragma omp parallel for simd aligned(u11t:AVX,u12t:AVX,Sigma11:AVX,Sigma12:AVX)
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu+ndim*i];
					int didn = id[nu+ndim*i];
					//uidm is correct here
					complex a11=conj(u11sh[uidm])*conj(u11t[didn*ndim+mu])-\
							u12sh[uidm]*conj(u12t[didn*ndim+mu]);
					complex a12=-conj(u11sh[uidm])*u12t[didn*ndim+mu]-\
							u12sh[uidm]*u11t[didn*ndim+mu];

					Sigma11[i]+=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
					Sigma12[i]+=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+nu]);
				}
			}
		}
#pragma omp parallel for simd aligned(u11t:AVX,u12t:AVX,Sigma11:AVX,Sigma12:AVX,dSdpi:AVX)
		for(int i=0;i<kvol;i++){
			complex a11 = u11t[i*ndim+mu]*Sigma12[i]+u12t[i*ndim+mu]*conj(Sigma11[i]);
			complex a12 = u11t[i*ndim+mu]*Sigma11[i]+conj(u12t[i*ndim+mu])*Sigma12[i];

			dSdpi[(i*nadj)*ndim+mu]=beta*cimag(a11);
			dSdpi[(i*nadj+1)*ndim+mu]=beta*creal(a11);
			dSdpi[(i*nadj+2)*ndim+mu]=beta*cimag(a12);
		}
	}
	//MPI was acting funny here for more than one process on Boltzmann
#ifdef USE_MKL
	mkl_free(u11sh); mkl_free(u12sh); mkl_free(Sigma11); mkl_free(Sigma12);
	mkl_free(z);
#else
	free(u11sh); free(u12sh); free(Sigma11); free(Sigma12); free(z);
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
#if (defined USE_MKL || defined USE_BLAS)
	//Can we use BLAS here with the halo?
	//The halo could interfere with things
	hp = cblas_dnrm2(kmom, pp, 1);
	hp*=hp;
#else
	hp=0;
	for(int i = 0; i<kmom; i++)
		//Three dimensions, so three pointers to get down the the actual value
		//What we're effectively doing is
		hp+=(*(pp+i))*(*(pp+i)); 
#endif
	hp*=0.5;
	double avplaqs, avplaqt;
	double hg = 0;
	//avplaq? isn't seen again here.
	SU2plaq(&hg,&avplaqs,&avplaqt);

	double hf = 0; int itercg = 0;
#ifdef USE_MKL
	complex *smallPhi = mkl_malloc(kferm2Halo*sizeof(complex),AVX);
#else
	complex *smallPhi = malloc(kferm2Halo*sizeof(complex));
#endif
	//Iterating over flavours
	for(int na=0;na<nf;na++){
		memcpy(X1,X0+na*kferm2Halo,kferm2*sizeof(complex));
		Congradq(na,res2,smallPhi,&itercg);
		ancgh+=itercg;
		Fill_Small_Phi(na, smallPhi);
		memcpy(X0+na*kferm2Halo,X1,kferm2*sizeof(complex));
#if (defined USE_MKL || defined USE_BLAS)
		complex dot;
		cblas_zdotc_sub(kferm2, smallPhi, 1, X1, 1, &dot);
		hf+=creal(dot);
#else
		//It is a dot product of the flattend arrays, could use
		//a module to convert index to coordinate array...
		for(int j=0;j<kferm2;j++)
			//Cheat using pointer for now
			hf+= X1[j]*conj(smallPhi[j]) ;
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
	complex fac = conj(jqq)*jqq*akappa*akappa;
	//These were evaluated only in the first loop of niterx so we'll just do it ouside of the loop.
	//These alpha and beta terms should be double, but that causes issues with BLAS. Instead we declare
	//them complex and work with the real part (especially for α_d)
	complex alphan;
	//Give initial values Will be overwritten if niterx>0
	double betad = 1.0; complex alphad=0; complex alpha = 1;
	//Because we're dealing with flattened arrays here we can call cblas safely without the halo
#ifdef __NVCC__
	complex *p, *r, *x1, *x2;
	cudaMallocManaged(&p, kferm2Halo*sizeof(complex));
	cudaMallocManaged(&r, kferm2Halo*sizeof(complex));
	cudaMallocManaged(&x1, kferm2Halo*sizeof(complex));
	cudaMallocManaged(&x2, kferm2Halo*sizeof(complex));
#elif defined USE_MKL
	complex *p  = mkl_malloc(kferm2Halo*sizeof(complex),AVX);
	complex *r  = mkl_malloc(kferm2*sizeof(complex),AVX);
	complex *x1=mkl_calloc(kferm2Halo, sizeof(complex), AVX);
	complex *x2=mkl_calloc(kferm2Halo, sizeof(complex), AVX);
#else
	complex *p  = malloc(kferm2Halo*sizeof(complex));
	complex *r  = malloc(kferm2*sizeof(complex));
	complex *x1=calloc(kferm2Halo,sizeof(complex));
	complex *x2=calloc(kferm2Halo,sizeof(complex));
#endif
	Fill_Small_Phi(na, smallPhi);
	//Instead of copying elementwise in a loop, use memcpy.
	memcpy(p, X1, kferm2*sizeof(complex));
	memcpy(r, smallPhi, kferm2*sizeof(complex));

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	complex betan;
	for(int niterx=0; niterx<niterc; niterx++){
		(*itercg)++;
		//x2 =  (M^†M)p 
		Hdslash(x1,p); Hdslashd(x2, x1);
		//x2 =  (M^†M+J^2)p 
#if (defined USE_MKL || defined USE_BLAS)
		cblas_zaxpy(kferm2, &fac, p, 1, x2, 1);
#else
		for(int i=0; i<kferm2; i++)
			x2[i]+=fac*p[i];
#endif
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(niterx){
			//α_d= p* (M^†M+J^2)p
#if (defined USE_MKL || defined USE_BLAS)
			cblas_zdotc_sub(kferm2, p, 1, x2, 1, &alphad);
#else
			alphad=0;
			for(int i=0; i<kferm2; i++)
				alphad+=conj(p[i])*x2[i];
#endif
			//And reduce. α_d does have a complex component but we only care about the real part
			Par_zsum(&alphad);
			//α=α_n/α_d = (r.r)/p(M^†M)p 
			alpha=creal(alphan)/creal(alphad);
			//x-αp, 
#if (defined USE_MKL || defined USE_BLAS)
			cblas_zaxpy(kferm2, &alpha, p, 1, X1, 1);
#else
			for(int i=0; i<kferm2; i++)
				X1[i]+=alpha*p[i];
#endif
		}			
		// r_n+1 = r_n-α(M^† M)p_n and β_n=r*.r
#if (defined USE_MKL || defined USE_BLAS)
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
		//Here we evaluate β=(r_{k+1}.r_{k+1})/(r_k.r_k) and then shuffle our indices down the line.
		//On the first iteration we define beta to be zero.
		complex beta = (niterx) ?  creal(betan)/betad : 0;
		betad=betan; alphan=betan;
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multipyling y by
		//β instead of x.
		//There is cblas_zaxpby in the MKL though, set a = 1 and b = β.
#ifdef USE_MKL
		complex a = 1;
		cblas_zaxpby(kferm2, &a, r, 1, &beta,  p, 1);
#else 
		for(int i=0; i<kferm2; i++)
			p[i]=r[i]+beta*p[i];
#endif
		//If we get a small enough β_n before hitting the iteration cap we break
		if(creal(betan)<resid){ 
#ifdef _DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %e toler = %e\n", niterx, creal(betan), resid);
#endif
			break;
		}
		if(!rank && niterx==niterc-1)
			fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i β_n=%e\n", ITERLIM, funcname, niterc, creal(betan));
	}
#ifdef __NVCC__
	cudaFree(x1), cudaFree(x2), cudaFree(p), cudaFree(r);
#elif defined USE_MKL
	mkl_free(x1), mkl_free(x2), mkl_free(p), mkl_free(r);
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
#ifdef USE_MKL
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
	complex *x1, *x2;
#ifdef USE_MKL
	x1=mkl_calloc(kfermHalo, sizeof(complex), AVX);
	x2=mkl_calloc(kfermHalo, sizeof(complex), AVX);
#else
	x1=calloc(kfermHalo,sizeof(complex));
	x2=calloc(kfermHalo,sizeof(complex));
#endif

	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	complex betan;
	Trial_Exchange();
	for(int niterx=0; niterx<niterc; niterx++){
		(*itercg)++;
		Dslash(x1,p);
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(niterx){
			//x*.x
#if (defined USE_MKL || defined USE_BLAS)
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
#if (defined USE_MKL || defined USE_BLAS)
			cblas_zaxpy(kferm, &alpha, p, 1, xi, 1);
#else
			for(int i = 0; i<kferm; i++)
				xi[i]+=alpha*p[i];
#endif
		}
		//x2=(M^†)x1=(M^†)Mp
		Dslashd(x2,x1);
		//r-α(M^†)Mp and β_n=r*.r
#if (defined USE_MKL || defined USE_BLAS)
		alpha*=-1;
		cblas_zaxpy(kferm, &alpha, x2, 1, r, 1);
		//r*.r
		betan = cblas_dznrm2(kferm, r,1);
		//Gotta square it to "undo" the norm
		betan *= betan;
#else
		//Just like Congradq, this loop could be unrolled but will need a reduction to deal with the betan 
		//addition.
		betan = 0;
		for(int i = 0; i<kferm;i++){
			r[i]-=alpha*x2[i];
			betan+=conj(r[i])*r[i];
		}
#endif
		//This is basically just congradq at the end. Check there for comments
		Par_zsum(&betan);
		complex beta = (niterx) ? betan/betad : 0;
		betad=betan; alphan=betan;
		//BLAS for p=r+βp doesn't exist in standard BLAS. This is NOT an axpy case as we're multipyling y by 
		//β instead of x.
		//There is cblas_zaxpby in the MKL though, set a = 1 and b = β.
#ifdef USE_MKL
		complex a = 1;
		cblas_zaxpby(kferm, &a, r, 1, &beta,  p, 1);
#else
		for(int i=0; i<kferm; i++)
			p[i]=r[i]+beta*p[i];
#endif
		//If we get a small enough β_n before hitting the iteration cap we break
		if(creal(betan)<resid){
#ifdef _DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %e toler = %e\n", niterx, creal(betan), resid);
#endif
			break;
		}
		if(!rank && niterx==niterc-1)
			fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i β_n=%e\n", ITERLIM, funcname, niterc, creal(betan));
	}
#ifdef USE_MKL
	mkl_free(p); mkl_free(r); mkl_free(x1); mkl_free(x2);
#else
	free(p); free(r); free(x1); free(x2);
#endif
	return 0;
}
int Measure(double *pbp, double *endenf, double *denf, complex *qq, complex *qbqb, double res, int *itercg){
	/*
	 * Calculate fermion expectation values via a noisy estimator
	 * -matrix inversion via conjugate gradient algorithm
	 * solves Mx=x1
	 * (Numerical Recipes section 2.10 pp.70-73)   
	 * uses NEW lookup tables **
	 * Implimented in CongradX
	 *
	 * Calls:
	 * =====
	 * Gauss_z
	 * Par_dsum
	 * ZHalo_swap_dir
	 * DHalo_swap_dir
	 *
	 * Globals:
	 * =======
	 * Phi, X0, xi, R1, u11t, u12t 
	 *
	 * Parameters:
	 * ==========
	 * double *pbp:		Pointer to ψ-bar ψ
	 * double endenf:
	 * double denf:
	 * complex qq:
	 * complex qbqb:
	 * double res:
	 * int itercg:
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Measure";
	//This x is just a storage container

#ifdef USE_MKL
	complex *ps = mkl_calloc(kvol, sizeof(complex), AVX);
	complex *x = mkl_malloc(kfermHalo*sizeof(complex), AVX);
#else
	complex *ps = calloc(kvol, sizeof(complex));
	complex *x = malloc(kfermHalo*sizeof(complex));
#endif
	//Setting up noise. I don't see any reason to loop
	//over colour indices as it is a two-colour code.
	//where I do have an issue is the loop ordering.

	//The root two term comes from the fact we called gauss0 in the fortran code instead of gaussp
#ifdef USE_MKL
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*kferm, xi, 0, 1/sqrt(2));
#else
	Gauss_z(xi, kferm, 0, 1/sqrt(2));
#endif
	memcpy(x, xi, kferm*sizeof(complex));

	//R_1= M^† Ξ 
	Dslashd(R1, xi);
	//Copying R1 to the first (zeroth) flavour index of Phi
	//This should be safe with memcpy since the pointer name
	//references the first block of memory for that pointer
	memcpy(Phi, R1, nc*ngorkov*kvol*sizeof(complex));
	memcpy(xi, R1, nc*ngorkov*kvol*sizeof(complex));

	//Evaluate xi = (M^† M)^-1 R_1 
	Congradp(0, res, itercg);
#if (defined USE_MKL || defined USE_BLAS)
	complex buff;
	cblas_zdotc_sub(kferm, x, 1, xi,  1, &buff);
	*pbp=creal(buff);
#else
	*pbp = 0;
#pragma unroll
	for(int i=0;i<kferm;i++)
		*pbp+=creal(conj(x[i])*xi[i]);
#endif
	Par_dsum(pbp);
	*pbp/=4*gvol;

	*qbqb=0; *qq=0;
	for(int idirac = 0; idirac<ndirac; idirac++){
		int igork=idirac+4;
		//Unrolling the colour indices, Then its just (γ_5*x)*Ξ or (γ_5*Ξ)*x 
#if (defined USE_MKL || defined USE_BLAS)
#pragma unroll
		for(int ic = 0; ic<nc; ic++){
			complex dot;
			//Because we have kvol on the outer index and are summing over it, we set the
			//step for BLAS to be ngorkov*nc=16. 
			cblas_zdotc_sub(kvol, &x[idirac*nc+ic], ngorkov*nc, &xi[igork*nc+ic], ngorkov*nc, &dot);
			*qbqb+=dot;
			cblas_zdotc_sub(kvol, &x[igork*nc+ic], ngorkov*nc, &xi[idirac*nc+ic], ngorkov*nc, &dot);
			*qq-=dot;
		}
		*qbqb *= gamval[4][idirac];
		*qq *= gamval[4][idirac];
#else
#pragma unroll
		for(int i=0; i<kvol; i++){
			//What is the optimal order to evaluate these in?
			*qbqb+=gamval[4][idirac]*conj(x[(i*ngorkov+idirac)*nc])*xi[(i*ngorkov+igork)*nc];
			*qq-=gamval[4][idirac]*conj(x[(i*ngorkov+igork)*nc])*xi[(i*ngorkov+idirac)*nc];
			*qbqb+=gamval[4][idirac]*conj(x[(i*ngorkov+idirac)*nc+1])*xi[(i*ngorkov+igork)*nc+1];
			*qq-=gamval[4][idirac]*conj(x[(i*ngorkov+igork)*nc+1])*xi[(i*ngorkov+idirac)*nc+1];
		}
#endif
	}
	//In the FORTRAN Code dsum was used instead despite qq and qbqb being complex
	Par_zsum(qq); Par_zsum(qbqb);
	*qq=(*qq+*qbqb)/(2*gvol);
	double xu, xd, xuu, xdd;

	//Halos
	ZHalo_swap_dir(x,16,3,DOWN);		ZHalo_swap_dir(x,16,3,UP);
	//Pesky halo exchange indices again
	//The halo exchange for the trial fields was done already at the end of the trajectory
	//No point doing it again

	//Instead of typing id[i*ndim+3] a lot, we'll just assign them to variables.
	//Idea. One loop instead of two loops but for xuu and xdd just use ngorkov-(igorkov+1) instead
#pragma omp parallel for reduction(+:xd,xu,xdd,xuu) 
	for(int i = 0; i<kvol; i++){
		int did=id[3+ndim*i];
		int uid=iu[3+ndim*i];
#pragma unroll
#pragma omp simd aligned(u11t:AVX,u12t:AVX,xi:AVX,x:AVX,dk4m:AVX,dk4p:AVX) 
		for(int igorkov=0; igorkov<4; igorkov++){
			int igork1=gamin[3][igorkov];
			//For the C Version I'll try and factorise where possible

			xu+=dk4p[did]*(conj(x[(did*ngorkov+igorkov)*nc])*(\
						u11t[did*ndim+3]*(xi[(i*ngorkov+igork1)*nc]-xi[(i*ngorkov+igorkov)*nc])+\
						u12t[did*ndim+3]*(xi[(i*ngorkov+igork1)*nc+1]-xi[(i*ngorkov+igorkov)*nc+1]) )+\
					conj(x[(did*ngorkov+igorkov)*nc+1])*(\
						conj(u11t[did*ndim+3])*(xi[(i*ngorkov+igork1)*nc+1]-xi[(i*ngorkov+igorkov)*nc+1])+\
						conj(u12t[did*ndim+3])*(xi[(i*ngorkov+igorkov)*nc]-xi[(i*ngorkov+igork1)*nc])));

			xd+=dk4m[i]*(conj(x[(uid*ngorkov+igorkov)*nc])*(\
						conj(u11t[i*ndim+3])*(xi[(i*ngorkov+igork1)*nc]+xi[(i*ngorkov+igorkov)*nc])-\
						u12t[i*ndim+3]*(xi[(i*ngorkov+igork1)*nc+1]+xi[(i*ngorkov+igorkov)*nc+1]) )+\
					conj(x[(uid*ngorkov+igorkov)*nc+1])*(\
						u11t[i*ndim+3]*(xi[(i*ngorkov+igork1)*nc+1]+xi[(i*ngorkov+igorkov)*nc+1])+\
						conj(u12t[i*ndim+3])*(xi[(i*ngorkov+igorkov)*nc]+xi[(i*ngorkov+igork1)*nc]) ) );

			int igorkovPP=igorkov+4;
			int igork1PP=igork1+4;
			xuu-=dk4m[did]*(conj(x[(did*ngorkov+igorkovPP)*nc])*(\
						u11t[did*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc]-xi[(i*ngorkov+igorkovPP)*nc])+\
						u12t[did*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc+1]-xi[(i*ngorkov+igorkovPP)*nc+1]) )+\
					conj(x[(did*ngorkov+igorkovPP)*nc+1])*(\
						conj(u11t[did*ndim+3])*(xi[(i*ngorkov+igork1PP)*nc+1]-xi[(i*ngorkov+igorkovPP)*nc+1])+\
						conj(u12t[did*ndim+3])*(xi[(i*ngorkov+igorkovPP)*nc]-xi[(i*ngorkov+igork1PP)*nc]) ) );

			xdd-=dk4p[i]*(conj(x[(uid*ngorkov+igorkovPP)*nc])*(\
						conj(u11t[i*ndim+3])*(xi[(i*ngorkov+igork1PP)*nc]+xi[(i*ngorkov+igorkovPP)*nc])-\
						u12t[i*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc+1]+xi[(i*ngorkov+igorkovPP)*nc+1]) )+\
					conj(x[(uid*ngorkov+igorkovPP)*nc+1])*(\
						u11t[i*ndim+3]*(xi[(i*ngorkov+igork1PP)*nc+1]+xi[(i*ngorkov+igorkovPP)*nc+1])+\
						conj(u12t[i*ndim+3])*(xi[(i*ngorkov+igorkovPP)*nc]+xi[(i*ngorkov+igork1PP)*nc]) ) );
		}
	}
	*endenf=xu-xd-xuu+xdd;
	*denf=xu+xd+xuu+xdd;

	Par_dsum(endenf); Par_dsum(denf);
	*endenf/=2*gvol; *denf/=2*gvol;
	//Future task. Chiral susceptibility measurements
#ifdef USE_MKL
	mkl_free(ps); mkl_free(x);
#else
	free(ps); free(x);
#endif
	return 0;
}
int SU2plaq(double *hg, double *avplaqs, double *avplaqt){
	/* 
	 * Calculates the gauge action using new (how new?) lookup table
	 * Follows a routine called qedplaq in some QED3 code
	 *
	 * Globals:
	 * =======
	 * 
	 *
	 * Parameters:
	 * ===========
	 * double hg
	 * double avplaqs
	 * double avplaqt
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "SU2plaq";
	//Do equivalent of a halo swap
#ifdef USE_MKL
	complex *z1 = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
	complex *z2 = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
#else
	complex *z1 = malloc((kvol+halo)*sizeof(complex));
	complex *z2 = malloc((kvol+halo)*sizeof(complex));
#endif
	//Was a halo exchange here but moved it outside
#ifdef USE_MKL
	mkl_free(z1); mkl_free(z2);
	complex *Sigma11 = mkl_malloc(kvol*sizeof(complex),AVX);
	complex *Sigma12 = mkl_malloc(kvol*sizeof(complex),AVX);
#else
	free(z1); free(z2);
	complex *Sigma11 = malloc(kvol*sizeof(complex));
	complex *Sigma12 = malloc(kvol*sizeof(complex));
#endif
	//	The fortran code used several consecutive loops to get the plaquette
	//	Instead we'll just make a11 and a12 values and do everything in one loop
	//	complex a11[kvol], a12[kvol]  __attribute__((aligned(AVX)));
	double hgs = 0; double hgt = 0;
	for(int mu=0;mu<ndim;mu++)
		for(int nu=0;nu<mu;nu++){
			//Don't merge into a single loop. Makes vectorisation easier?
			//Or merge into a single loop and dispense with the a arrays?
#pragma omp parallel for simd aligned(Sigma11:AVX,Sigma12:AVX,u11t:AVX,u12t:AVX)
			for(int i=0;i<kvol;i++){
				int uidm = iu[mu+ndim*i]; 

				Sigma11[i]=u11t[i*ndim+mu]*u11t[uidm*ndim+nu]-u12t[i*ndim+mu]*conj(u12t[uidm*ndim+nu]);
				Sigma12[i]=u11t[i*ndim+mu]*u12t[uidm*ndim+nu]+u12t[i*ndim+mu]*conj(u11t[uidm*ndim+nu]);
				//			}
				//			for(i=0;i<kvol;i++){
				int uidn = iu[nu+ndim*i]; 
				complex a11=Sigma11[i]*conj(u11t[uidn*ndim+mu])+Sigma12[i]*conj(u12t[uidn*ndim+mu]);
				complex a12=-Sigma12[i]*u12t[uidn*ndim+mu]+Sigma12[i]*u11t[uidn*ndim+mu];
				//			}
				//			for(i=0;i<kvol;i++){
				Sigma11[i]=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
				//				Sigma12[i]=-a11[i]*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
				//				Not needed in final result as it traces out?
		}
		//Space component
		if(mu<ndim-1)
			//Is there a BLAS routine for this
#pragma omp parallel for simd reduction(+:hgs) aligned(Sigma11:AVX)
			for(int i=0;i<kvol;i++)
				hgs-=creal(Sigma11[i]);
		//Time component
		else
#pragma omp parallel for simd reduction(+:hgt) aligned(Sigma11:AVX)
			for(int i=0;i<kvol;i++)
				hgt-=creal(Sigma11[i]);
		}
#ifdef USE_MKL
	mkl_free(Sigma11); mkl_free(Sigma12);
#else
	free(Sigma11); free(Sigma12);
#endif
	Par_dsum(&hgs); Par_dsum(&hgt);
	*avplaqs=-hgs/(3*gvol); *avplaqt=-hgt/(gvol*3);
	*hg=(hgs+hgt)*beta;
#ifdef _DEBUG
	if(!rank)
		printf("hgs=%e  hgt=%e  hg=%e\n", hgs, hgt, *hg);
#endif
	return 0;
}
double Polyakov(){
	/*
	 * Calculate the Polyakov loop (no prizes for guessing that one...)
	 *
	 * Globals:
	 * =======
	 * u11t, u12t, u11t, u12t
	 *
	 * Calls:
	 * ======
	 * Par_tmul, Par_dsum
	 * 
	 * Parameters:
	 * ==========
	 * double *poly The Polyakov Loop value
	 * 
	 * Returns:
	 * =======
	 * Double corresponding to the polyakov loop
	 */
	const char *funcname = "Polyakov";
	double poly = 0;
	//Originally at the very end before Par_dsum
	//Now all cores have the value for the complete Polyakov line at all spacial sites
	//We need to globally sum over spacial processores but not across time as these
	//are duplicates. So we zero the value for all but t=0
	//This is (according to the FORTRAN code) a bit of a hack
	//I will expand on this hack and completely avoid any work
	//for this case rather than calculating everything just to set it to zero
	if(!pcoord[3+ndim*rank]){
#ifdef USE_MKL
		complex *Sigma11 = mkl_malloc(kvol3*sizeof(complex),AVX);
		complex *Sigma12 = mkl_malloc(kvol3*sizeof(complex),AVX);
#else
		complex *Sigma11 = malloc(kvol3*sizeof(complex));
		complex *Sigma12 = malloc(kvol3*sizeof(complex));
#endif
#if (defined USE_MKL || defined USE_BLAS)
		cblas_zcopy(kvol3, &u11t[3], ndim, Sigma11, 1);
		cblas_zcopy(kvol3, &u12t[3], ndim, Sigma12, 1);
#else
		for(int i=0; i<kvol3; i++){
			Sigma11[i]=u11t[i*ndim+3];
			Sigma12[i]=u12t[i*ndim+3];
		}
#endif
		//	Some Fortran commentary
		//	Changed this routine.
		//	u11t and u12t now defined as normal ie (kvol+halo,4).
		//	Copy of Sigma11 and Sigma12 is changed so that it copies
		//	in blocks of ksizet.
		//	Variable indexu also used to select correct element of u11t and u12t 
		//	in loop 10 below.
		//
		//	Change the order of multiplication so that it can
		//	be done in parallel. Start at t=1 and go up to t=T:
		//	previously started at t+T and looped back to 1, 2, ... T-1
		//Buffers
		complex a11=0;
		//There is a dependency. Can only parallelise the inner loop
#pragma unroll
		for(int it=1;it<ksizet;it++)
			//will be faster for parallel code
#pragma omp parallel for simd private(a11) aligned(u11t:AVX,u12t:AVX,Sigma11:AVX,Sigma12:AVX)
			for(int i=0;i<kvol3;i++){
				//Seems a bit more efficient to increment indexu instead of reassigning
				//it every single loop
				int indexu=it*kvol3+i;
				a11=Sigma11[i]*u11t[indexu*ndim+3]-Sigma12[i]*conj(u12t[indexu*ndim+3]);
				//Instead of having to store a second buffer just assign it directly
				Sigma12[i]=Sigma11[i]*u12t[indexu*ndim+3]+Sigma12[i]*conj(u11t[indexu*ndim+3]);
				Sigma11[i]=a11;
			}

		//Multiply this partial loop with the contributions of the other cores in the
		//timelike dimension
#if (npt>1)
#ifdef _DEBUG
		printf("Multiplying with MPI\n");
#endif
		//Par_tmul does nothing if there is only a single processor in the time direction. So we only compile
		//its call if it is required
		Par_tmul(Sigma11, Sigma12);
#endif
#pragma omp parallel for simd reduction(+:poly) aligned(Sigma11:AVX)
		for(int i=0;i<kvol3;i++)
			poly+=creal(Sigma11[i]);
#ifdef USE_MKL
		mkl_free(Sigma11); mkl_free(Sigma12);
#else
		free(Sigma11); free(Sigma12);
#endif
	}

	Par_dsum(&poly);
	poly/=gvol3;
	return poly;	
}
inline int Reunitarise(){
	/*
	 * Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * Globals:
	 * =======
	 * u11t, u12t
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Reunitarise";
#pragma ivdep
	for(int i=0; i<kvol*ndim; i++){
		//Declaring anorm inside the loop will hopefully let the compiler know it
		//is safe to vectorise aggessively
		double anorm=sqrt(conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]);
		//		Exception handling code. May be faster to leave out as the exit prevents vectorisation.
		//		if(anorm==0){
		//			fprintf(stderr, "Error %i in %s on rank %i: anorm = 0 for μ=%i and i=%i.\nExiting...\n\n",
		//					DIVZERO, funcname, rank, mu, i);
		//			MPI_Finalise();
		//			exit(DIVZERO);
		//		}
		u11t[i]/=anorm;
		u12t[i]/=anorm;
	}
	return 0;
}
inline int Z_gather(complex *x, complex *y, int n, unsigned int *table){
	//FORTRAN had a second parameter m gving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
#pragma ivdep
	for(int i=0; i<n; i++)
		x[i]=y[table[i]];
	return 0;
}
inline int Fill_Small_Phi(int na, complex *smallPhi){
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
				// The original code behaves in this manner, same loop order and same formula for PHI_index
				// We end up hiting the first 8 elements of the Phi array. But because i is multiplied by
				// 2*ndirac*nc we end up skipping the second, fourth, sixth etc. groups of 8 elements.
				// This is not yet clear to me why, but I'll update when it is clarified.
				//	  PHI_index=i*16+j*2+k;
				smallPhi[(i*ndirac+idirac)*nc+ic]=Phi[((na*kvol+i)*ngorkov+idirac)*nc+ic];
			}
	return 0;
}
double Norm_squared(complex *z, int n){
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
