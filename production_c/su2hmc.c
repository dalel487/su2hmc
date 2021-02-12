#include <coord.h>
#ifdef USE_CUDA
#include <curand.h>
#endif
#include <par_mpi.h>
#include <math.h>
#include <random.h>
#include <slash.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <su2hmc.h>

//Extern definitions, especially default values for fmu, beta and akappa
const complex zi = 0.0+1.0*I;
const complex real1 = 1.0+0.0*I;
complex jqq = 0;
double fmu = 0.0;
double beta = 1.7;
double akappa = 0.1780;
const int gamin[4][4] =	{{3,2,1,0},
	{3,2,1,0},
	{2,3,0,1},
	{2,3,0,1}};
complex gamval[5][4] =	{{-1*zi,-1*zi,zi,zi},
	{-1*real1,real1,real1,-1*real1},
	{-1*zi,zi,zi,-1*zi},
	{real1,real1,real1,real1},
	{real1,real1,-1*real1,-1*real1}};

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
int main(int argc, char *argv){
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
	 *    trajectory length is random with mean dt*iterl
	 *    The code runs for a fixed number iter2 of trajectories.
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
	 *     Converted from Fortran to C by D. Lawlor 2020-2021
	 *
	 ******************************************************************/
	const char *funcname = "main";
	Par_begin(argc, &argv);
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
	const double tpi = 2*acos(0.0);
#endif
	//End of input
	//===========
	//rank is zero means it must be the "master process"
	double dt=0.004; double ajq = 0.0;
	double delb; //Not used?
	double athq = 0.0;
	int iterl = 250; int iter2 = 10;
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
		fscanf(midout, "%lf %lf %lf %lf %lf %lf %lf %d %d", &dt, &beta, &akappa, &ajq, &athq, &fmu, &delb, &iterl, &iter2);
		fclose(midout);
	}
	if(iread){
#ifdef DEBUG
		if(!rank) printf("Calling Par_sread() with seed: %i\n", seed);
#endif
		Par_sread();
	}
	//Send inputs to other ranks
	Par_dcopy(&dt); Par_dcopy(&beta); Par_dcopy(&akappa); Par_dcopy(&ajq);
	Par_dcopy(&athq); Par_dcopy(&fmu); //Par_dcopy(&delb); Not used?
	Par_icopy(&iterl); Par_icopy(&iter2); 
	jqq=ajq*cexp(athq*I);
	Par_ranset(&seed);

	//Initialisation
	//istart < 0: Start from tape?!? How old is this code?
	//istart = 0: Ordered/Cold Start
	//istart > 0: Random/Hot Start
	Init(istart);

	//Initial Measurements
	//====================
	poly = Polyakov();
#ifdef DEBUG
	if(!rank) printf("Initial Polyakov loop evaluated as %f\n", poly);
#endif
	double hg, avplaqs, avplaqt;
	SU2plaq(&hg,&avplaqs,&avplaqt);
	//Loop on β
	//Print Heading
	double traj=iterl*dt;
	double proby = 2.5/iterl;
	char *outname = "output"; char *outop="w";
	FILE *output;
	if(!rank){
		if(!(output=fopen(outname, outop) )){
			fprintf(stderr,"Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",OPENERROR,funcname,outname,outop);
			MPI_Finalize();
			exit(OPENERROR);
		}
		printf("hg = %f, <Ps> = %f, <Pt> = %f, <Poly> = %f\n", hg, avplaqs, avplaqt, poly);
		fprintf(output, "ksize = %i ksizet = %i Nf = %i\nTime step dt = %f Trajectory length = %f\n"\
				"No. of Trajectories = %i β = %f\nκ = %f μ = %f\nDiquark source = %f Diquark phase angle = %f\n"\
				"Stopping Residuals: Guidance: %f Acceptance: %f, Estimator: %f\nSeed = %i\n",
				ksize, ksizet, nf, dt, traj, iter2, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);
#ifdef DEBUG
		//Print to terminal during debugging
		printf("ksize = %i ksizet = %i Nf = %i\nTime step dt = %f Trajectory length = %f\n"\
				"No. of Trajectories = %i β = %f\nκ = %f μ = %f\nDiquark source = %f Diquark phase angle = %f\n"\
				"Stopping Residuals: Guidance: %f Acceptance: %f, Estimator: %f\nSeed = %i\n",
				ksize, ksizet, nf, dt, traj, iter2, beta, akappa, fmu, ajq, athq, rescgg, rescga, respbp, seed);

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
	double d = dt/2;
	//Start of classical evolution
	//===========================
	double pbp;
	complex qq;
	//Open the output files that haven't been opened yet
	for(int isweep = 1; isweep <= iter2; isweep++){
#ifdef DEBUG
		if(!rank)
			printf("Starting isweep %i\n", isweep);
#endif
		for(int na=0; na<nf; na++){
			complex R[kvol+halo][ngorkov][nc] __attribute__((aligned(AVX)));
#ifdef USE_MKL
			//Multiply the dimension of R by 2 because R is complex
			//The FORTRAN code had two gaussian routines.
			//gaussp was the normal box-muller and gauss0 didn't have 2 inside the square root
			//Using σ=1/sqrt(2) in these routines has the same effect as gauss0
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*nc*ngorkov*kvol, **R, 0, 1/sqrt(2));
#else
			Gauss_z(**R, nc*ngorkov*kvol, 0, 1/sqrt(2));
#endif
			Dslashd(R1, R);
			memcpy(Phi[na],R1, nc*ngorkov*kvol*sizeof(complex));
			//Slamming on the brakes. Can we simply memcpy here because ndirac<ngorkov?
			//Up/down partitioning (using only pseudofermions of flavour 1) causes this
#pragma omp parallel for collapse(3)
			for(int i=0; i<kvol; i++)
				for(int idirac = 0; idirac < ndirac; idirac++)
					for(int ic = 0; ic <nc; ic++)
						X0[na][i][idirac][ic]=R1[i][idirac][ic];
		}	
		//Heatbath
		//========
		//We're going to make the most of the new Gauss_d routine to send a flattened array
		//and do this all in one step.
#ifdef USE_MKL
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, nadj*ndim*kvol, pp, 0, 1);
#else
		Gauss_d(pp, nadj*ndim*kvol, 0, 1);
#endif

		//Initialise Trial Fields
		memcpy(u11t, u11, ndim*kvol*sizeof(complex));
		memcpy(u12t, u12, ndim*kvol*sizeof(complex));
		double H0, S0;
		Hamilton(&H0, &S0, rescga);
#ifdef DEBUG
		if(!rank) printf("H0: %f S0: %f\n", H0, S0);
#endif
		double action;
		if(isweep==1)
			action = S0/gvol;

		//Half step forward for p
		//=======================
		double dSdpi[kvol+halo][3][ndim];
#ifdef DEBUG
		printf("Evaluating force on rank %i\n", rank);
#endif
		Force(dSdpi, 1, rescgg);
#ifdef USE_MKL
		cblas_daxpy(nadj*ndim*kvol, -d, dSdpi, 1, pp, 1);
#else
		for(int i=0;i<kvol;i++)
			for(int iadj=0;iadj<nadj;iadj++)
				for(int mu=0;mu<ndim;mu++)
					pp[i][iadj][mu]-=d*dSdpi[i][iadj][mu];
#endif
		//Main loop for classical time evolution
		//======================================
		for(int iter = 0; iter<itermax; iter++){
#ifdef DEBUG
			if(!rank)
				printf("iter: %i\n", iter);
#endif
			//The FORTRAN redefines d=dt here, which makes sense if you have a limited line length.
			//I'll stick to using dt though.
#pragma omp parallel for collapse(2)
			for(int i=0;i<kvol;i++)
				for(int mu = 0; mu<ndim; mu++){
					//Sticking to what was in the FORTRAN for variable names.
					//CCC for cosine SSS for sine AAA for...
					double AAA = dt*sqrt(pp[i][0][mu]*pp[i][0][mu]+pp[i][1][mu]*pp[i][1][mu]+pp[i][2][mu]*pp[i][2][mu]);
					double CCC = cos(AAA);
					double SSS = dt*sin(AAA)/AAA;
					complex a11 = CCC+I*SSS*pp[i][2][mu];
					complex a12 = pp[i][1][mu]*SSS + I*SSS*pp[i][0][mu];
					//b11 and b12 are u11t and u12t terms, so we'll use u12t directly
					//but use b11 for u11t to prevent RAW dependency
					complex b11 = u11t[i][mu];
					u11t[i][mu] = a11*b11-a12*conj(u12t[i][mu]);
					u12t[i][mu] = a11*u12t[i][mu]+a12*conj(b11);
				}
			Reunitarise();
			Force(dSdpi, 0, rescgg);
			//Need to check Par_granf again 
			double ytest = Par_granf();
			//This was pretty awkward looking in the original code.
			//The same for loop is given in both the if and else
			//statement but only the value of d changes. This is due to the break in the if part
			//Also, why are we looking for iterl*6/5? What is the point of iterl if we just
			//choose a bigger one?

			if((ytest<proby || iter>=iterl*(6/5)) && iter>=iterl*4/5){
#ifdef USE_MKL
				cblas_daxpy(ndim*nadj*kvol, -d, dSdpi, 1, pp, 1);
#else
				for(int i = 0; i<kvol; i++)
					for(int iadj=0; iadj<nadj; iadj++)
						for(int mu = 0; mu < ndim; mu++)
							pp[i][iadj][mu]-=d*dSdpi[i][iadj][mu];
#endif
				itot+=iter;
				break;
			}
			else{
#ifdef USE_MKL
				cblas_daxpy(ndim*nadj*kvol, -dt, dSdpi, 1, pp, 1);
#else
				for(int i = 0; i<kvol; i++)
					for(int iadj=0; iadj<nadj; iadj++)
						for(int mu = 0; mu < ndim; mu++)
							pp[i][iadj][mu]-=dt*dSdpi[i][iadj][mu];
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
			fprintf(output, "%f %f\n", dH, dS);
#ifdef DEBUG
			printf("dH = %f dS = %f\n", dH, dS);
#endif
		}
		double y = exp(dH);
		yav+=y;
		yyav+=y*y;
		//The Monte-Carlo
		//x is unassigned in the FORTRAN at declaration, so hopefully that won't be an issue here...
		double x;
		//Only update x if dH is negative
		if(dH<0){
			x=Par_granf();

			// We only test x if it is updated (inside the previous if block)
			//But that required a goto in FORTRAN to get around doing the acceptance operations
			//in the case where dH>=0 or x<=y. We'll nest the if statements in C to 
			//get around this using the reverse test to the FORTRAN if (x<=y instead of x>y).
			if(x<=y){
				//Step is accepted. Set s=st
#ifdef DEBUG
				if(!rank)
					printf("New configuration accepted.\n");
#endif
				//Original FORTRAN Comment:
				//JIS 20100525: write config here to preempt troubles during measurement!
				//JIS 20100525: remove when all is ok....
				//On closer inspection, this is more clever than I first thought. Using
				//integer division like that
				if((isweep/icheck)*icheck==isweep){
					//ranget(seed);
					Par_swrite(isweep);
				}
				memcpy(u11,u11t,ndim*kvol*sizeof(complex));
				memcpy(u12,u12t,ndim*kvol*sizeof(complex));
				naccp++;
				//Divide by gvol because of halos?
				action=S1/gvol;
			}
			actiona+=action; 
			double vel2=0.0;

#ifdef USE_MKL
			vel2 += cblas_ddot(kmom, pp[0][0], 1, pp[0][0], 1 );
#else
#pragma unroll
			for(int i=0; i<kvol; i++)
				for(int iadj = 0; iadj<nadj; iadj++)
					for(int mu=0; mu<ndim; mu++)
						vel2+=pp[i][iadj][mu]*pp[i][iadj][mu];
#endif
			Par_dsum(&vel2);
			vel2a+=vel2/(12*gvol);

			if((isweep/iprint)*iprint==isweep){
				memcpy(u11t, u11, ndim*(kvol+halo)*sizeof(complex));
				memcpy(u12t, u12, ndim*(kvol+halo)*sizeof(complex));
#ifdef DEBUG
				if(!rank)
					printf("Starting measurements\n");
#endif
				int itercg;
				double endenf, denf;
				complex qbqb;
				Measure(&pbp,&endenf,&denf,&qq,&qbqb,respbp,&itercg);
#ifdef DEBUG
				if(!rank)
					printf("Finished measurements\n");
#endif
				pbpa+=pbp; endenfa+=endenf; denfa+=denf; ipbp++;
				SU2plaq(&hg,&avplaqs,&avplaqt); 
				poly = Polyakov();
				//We have four output files, so may as well get the other ranks to help out
				//and abuse scoping rules while we're at it.
				if(rank==0)
					//Output code... Some files weren't opened in the main loop of the FORTRAN code 
					//That will need to be looked into for the C version
					//It would explain the weird names like fort.1X that looked like they were somehow
					//FORTRAN related...
					//Not yet implemented
					fprintf(output, "%i %f %f\n", itercg, ancg, ancgh);
				else if(rank==1){
					//The origninal code implicitly created these files with the name fort.XX where XX is the file label
					//from FORTRAN. We'll stick with that for now.
					FILE *fortout;
					char *fortname = "fort11";
					char *fortop= (isweep==0) ? "w" : "a";
					if(!(fortout=fopen(fortname, fortop) )){
						fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n", OPENERROR, funcname, fortname, fortop);
						MPI_Finalize();
						exit(OPENERROR);
					}
					free(fortname); free(fortop);
					fprintf(fortout, "%f %f %f\n", pbp, endenf, denf);
					fclose(fortout);
				}
				else if(rank == 2){
					//The origninal code implicitly created these files with the name fort.XX where XX is the file label
					//from FORTRAN. We'll stick with that for now.
					FILE *fortout;
					char *fortname = "fort12"; 
					char *fortop= (isweep==0) ? "w" : "a";
					if(!(fortout=fopen(fortname, fortop) )){
						fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n", OPENERROR, funcname, fortname, fortop);
						MPI_Finalize();
						exit(OPENERROR);
					}
					free(fortname); free(fortop);
					fprintf(fortout, "%f %f %f\n", avplaqs, avplaqt, poly);
					fclose(fortout);
				}
				else if(rank == 3){
					FILE *fortout;
					char *fortname = "fort13";
					char *fortop= (isweep==0) ? "w" : "a";
					if(!(fortout=fopen(fortname, fortop) )){
						fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n", OPENERROR, funcname, fortname, fortop);
						MPI_Finalize();
						exit(OPENERROR);
					}
					free(fortname); free(fortop);
					fprintf(fortout, "%f\n", creal(qq));
					fclose(fortout);
				}
				if((isweep/icheck)*icheck==isweep){
					//ranget(seed);
					Par_swrite(isweep);
				}
			}
		}
	}
	//End of main loop
	actiona/=iter2; vel2a/=iter2; pbpa/=ipbp; endenfa/=ipbp; denfa/=ipbp;
	ancg/=nf*itot; ancgh/=2*nf*iter2; yav/=iter2; yyav=yyav/iter2 - yav*yav;
	double atraj=dt*itot/iter2;

	if(!rank)
		fprintf(output, "Averages for the last %i trajectories\n"\
				"Number of acceptances: %i Average Trajectory Length = %f\n"\
				"exp(dh) = %f +/- %f\n"\
				"Average number of congrad iter guidance: %f acceptance %f\n"\
				"psibarpsi = %f\n"\
				"Mean Square Velocity = %f Action Per Site = %f\n"\
				"Energy Density = %f Number Density %f\n",\
				iter2, naccp, atraj, yav, yyav, ancg, ancgh, pbpa, vel2a, actiona, endenfa, denfa);
	fclose(output);	
	//Not yet implimented
	MPI_Finalize();
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

	//First things first, calculate a few constants
	Addrc();
	//And confirm they're legit
	Check_addr(iu, ksize, ksizet, 0, kvol+halo);
	Check_addr(id, ksize, ksizet, 0, kvol+halo);
#ifdef DEBUG
	printf("Checked addresses\n");
#endif
	double chem1=exp(fmu); double chem2 = 1/chem1;
#pragma omp parallel for 
	for(int i = 0; i<kvol; i++){
		dk4p[i]=akappa*chem1;
		dk4m[i]=akappa*chem2;
	}
	//Antiperiodic Boundary Conditions. Flip the terms at the edge of the time
	//direction
	if(ibound == -1 && pcoord[3][rank]==npt -1){
#ifdef DEBUG
		printf("Implimenting antiperiodic boundary conditions on rank %i\n", rank);
#endif
#pragma omp parallel for
		for(int i= 0; i<kvol3; i++){
			int k = kvol - kvol3 + i;
			dk4p[k]*=-1;
			dk4m[k]*=-1;
		}
	}
	//Each gamma matrix is rescaled by akappa by flattening the gamval array
#ifdef USE_MKL
	cblas_zdscal(5*4, akappa, gamval, 1);
#else
	for(int i=0;i<5;i++)
		for(int j=0;j<4;j++)
			gamval[i][j]*=akappa;
#endif
	if(istart==0){
		//Initialise a cold start to zero
		//memset is safe to use here because zero is zero 
		for(int i=0; i<kvol;i++)
			for(int mu=0; mu<ndim; mu++){
				u11t[i][mu]=1;
				u12t[i][mu]=0;
			}
	}
	else if(istart>0){
#ifdef USE_CUDA
		complex *cu_u1xt;
		cudaMalloc(&cu_u1xt, ndim*kvol*sizeof(complex));

#elif defined USE_MKL
		//Good news, casting works for using a double to create random complex numbers
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*(kvol+halo), u11t, -1, 1);
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 2*ndim*(kvol+halo), u12t, -1, 1);
#else
		//Depending if we have the RANLUX or SFMT19977 generator.	
#pragma unroll
		for(int i=0; i<kvol;i++)
			for(int j=0; j<ndim; j++){
				u11t[i][j]=sfmt_genrand_real1(&sfmt)+sfmt_genrand_real1(&sfmt)*I;
				u12t[i][j]=sfmt_genrand_real1(&sfmt)+sfmt_genrand_real1(&sfmt)*I;
			}
#endif
	}
	else{
		fprintf(stderr,"Waring %i in %s: Gauge fields are not initialised.\n", NOINIT, funcname);
	}
#ifdef DEBUG
	printf("Initialisation Complete\n");
#endif
	Reunitarise();
	memcpy(u11, u11t, ndim*(kvol+halo)*sizeof(complex));
	memcpy(u12, u12t, ndim*(kvol+halo)*sizeof(complex));
	return 0;
}
int Force(double dSdpi[][3][ndirac], int iflag, double res1){
	/*
	 *	Calculates dSds at each intermediate time
	 *	
	 *	Calls:
	 *	=====
	 *
	 *	Globals:
	 *	=======
	 *	u11t, u12t, X1, Phi
	 *
	 *	This X1 is the one being referred to in the common/vector/ statement in the original FORTRAN
	 *	code. There may subroutines with a different X1 (or use a different common block definition
	 *	for this X1) so keep your wits about you
	 *
	 *	Parameters:
	 *	===========
	 *	double dSdpi[][3][kvol+halo]
	 *	int	iflag
	 *	double	res1;
	 *
	 *	Returns:
	 *	=======
	 *	Zero on success, integer error code otherwise
	 */
	const char *funcname = "Force";

	complex X2[kvol+halo][ndirac][nc] __attribute__((aligned(AVX)));
	Gauge_force(dSdpi);
#ifdef USE_MKL
	complex blasa, blasb;
#endif
	//X1=(M†M)^{1} Phi
	int itercg;
	for(int na = 0; na<nf; na++){
		memcpy(X1, X0[na], nc*ndirac*kvol*sizeof(complex));
		//FORTRAN's logic is backwards due to the implied goto method
		//If iflag is zero we do some initalisation stuff? 
		if(!iflag){
			Congradq(na, res1, &itercg);
			ancg+=itercg;
			//BLASable? If we cheat and flatten the array it is!
			//This is not a general BLAS Routine, just an MKL one
#ifdef USE_MKL
			blasa=2.0; blasb=-1.0;
			cblas_zaxpby(kvol*ndirac*nc, &blasa, X1, 1, &blasb, X0[na], 1); 
#else
			for(int i=0;i<kvol;i++){
#pragma unroll
				for(int idirac=0;idirac<ndirac;idirac++){
					X0[na][i][idirac][0]=2*X1[i][idirac][0]-X0[na][i][idirac][0];
					X0[na][i][idirac][1]=2*X1[i][idirac][1]-X0[na][i][idirac][1];
				}
			}
#endif
		}
		Hdslash(X2,X1);
#ifdef USE_MKL
		double blasd=2.0;
		cblas_zdscal(kvol*ndirac*nc, blasd, X2, 1);
#else
		for(int i=0;i<kvol;i++)
#pragma unroll
			for(int idirac=0;idirac<ndirac;idirac++){
				X2[i][idirac][0]*=2;
				X2[i][idirac][1]*=2;
			}
#endif
#pragma unroll
		for(int mu=0;mu<4;mu++){
			ZHalo_swap_dir(X1,8,mu,DOWN);
			ZHalo_swap_dir(X2,8,mu,DOWN);
		}

		//	The original FORTRAN Comment:
		//    dSdpi=dSdpi-Re(X1*(d(Mdagger)dp)*X2) -- Yikes!
		//   we're gonna need drugs for this one......
		//
		//  Makes references to X1(.,.,iu(i,mu)) AND X2(.,.,iu(i,mu))
		//  as a result, need to swap the DOWN halos in all dirs for
		//  both these arrays, each of which has 8 cpts
		//
		int mu, uid, igork1;
#pragma omp parallel for private(mu,uid,igork1)
		for(int i=0;i<kvol;i++)
			for(int idirac=0;idirac<ndirac;idirac++){
				//Unrolling the loop
#pragma unroll (3) 
				for(mu=0; mu<3; mu++){
					//Long term ambition. I used the diff command on the different
					//spacial components of dSdpi and saw a lot of the values required
					//for them are duplicates (u11(i,mu)*X2(1,idirac,i) is used again with
					//a minus in front for example. Why not evaluate them first /and then plug 
					//them into the equation? Reduce the number of evaluations needed and look
					//a bit neater (although harder to follow as a consequence).

					//Up indices
					uid = iu[mu][i];
					igork1 = gamin[mu][idirac];	
					dSdpi[i][0][mu]+=akappa*creal(zi*
							(conj(X1[i][idirac][0])*
							 (-conj(u12t[i][mu])*X2[uid][idirac][0]
							  +conj(u11t[i][mu])*X2[uid][idirac][1])
							 +conj(X1[uid][idirac][0])*
							 ( u12t[i][mu] *X2[i][idirac][0]
							   -conj(u11t[i][mu])*X2[i][idirac][1])
							 +conj(X1[i][idirac][1])*
							 (u11t[i][mu] *X2[uid][idirac][0]
							  +u12t[i][mu] *X2[uid][idirac][1])
							 +conj(X1[uid][idirac][1])*
							 (-u11t[i][mu] *X2[i][idirac][0]
							  -conj(u12t[i][mu])*X2[i][idirac][1])))
						+creal(zi*gamval[idirac][mu]*
								(conj(X1[i][idirac][0])*
								 (-conj(u12t[i][mu])*X2[uid][igork1][0]
								  +conj(u11t[i][mu])*X2[uid][igork1][1])
								 +conj(X1[uid][idirac][0])*
								 (-u12t[i][mu] *X2[i][igork1][0]
								  +conj(u11t[i][mu])*X2[i][igork1][1])
								 +conj(X1[i][idirac][1])*
								 (u11t[i][mu] *X2[uid][igork1][0]
								  +u12t[i][mu] *X2[uid][igork1][1])
								 +conj(X1[uid][idirac][1])*
								 (u11t[i][mu] *X2[i][igork1][0]
								  +conj(u12t[i][mu])*X2[i][igork1][1])));

					dSdpi[i][1][mu]+=akappa*creal(
							(conj(X1[i][idirac][0])*
							 (-conj(u12t[i][mu])*X2[uid][idirac][0]
							  +conj(u11t[i][mu])*X2[uid][idirac][1])
							 +conj(X1[uid][idirac][0])*
							 (-u12t[i][mu] *X2[i][idirac][0]
							  -conj(u11t[i][mu])*X2[i][idirac][1])
							 +conj(X1[i][idirac][1])*
							 (-u11t[i][mu] *X2[uid][idirac][0]
							  -u12t[i][mu] *X2[uid][idirac][1])
							 +conj(X1[uid][idirac][1])*
							 (u11t[i][mu] *X2[i][idirac][0]
							  -conj(u12t[i][mu])*X2[i][idirac][1])))
						+creal(gamval[idirac][mu]*
								(conj(X1[i][idirac][0])*
								 (-conj(u12t[i][mu])*X2[uid][igork1][0]
								  +conj(u11t[i][mu])*X2[uid][igork1][1])
								 +conj(X1[uid][idirac][0])*
								 (u12t[i][mu] *X2[i][igork1][0]
								  +conj(u11t[i][mu])*X2[i][igork1][1])
								 +conj(X1[i][idirac][1])*
								 (-u11t[i][mu] *X2[uid][igork1][0]
								  -u12t[i][mu] *X2[uid][igork1][1])
								 +conj(X1[uid][idirac][1])*
								 (-u11t[i][mu] *X2[i][igork1][0]
								  +conj(u12t[i][mu])*X2[i][igork1][1])));

					dSdpi[i][2][mu]+=akappa*creal(zi*
							(conj(X1[i][idirac][0])*
							 (u11t[i][mu] *X2[uid][idirac][0]
							  +u12t[i][mu] *X2[uid][idirac][1])
							 +conj(X1[uid][idirac][0])*
							 (-conj(u11t[i][mu])*X2[i][idirac][0]
							  -u12t[i][mu] *X2[i][idirac][1])
							 +conj(X1[i][idirac][1])*
							 (conj(u12t[i][mu])*X2[uid][idirac][0]
							  -conj(u11t[i][mu])*X2[uid][idirac][1])
							 +conj(X1[uid][idirac][1])*
							 (-conj(u12t[i][mu])*X2[i][idirac][0]
							  +u11t[i][mu] *X2[i][idirac][1])))
						+creal(zi*gamval[idirac][mu]*
								(conj(X1[i][idirac][0])*
								 (u11t[i][mu] *X2[uid][igork1][0]
								  +u12t[i][mu] *X2[uid][igork1][1])
								 +conj(X1[uid][idirac][0])*
								 (conj(u11t[i][mu])*X2[i][igork1][0]
								  +u12t[i][mu] *X2[i][igork1][1])
								 +conj(X1[i][idirac][1])*
								 (conj(u12t[i][mu])*X2[uid][igork1][0]
								  -conj(u11t[i][mu])*X2[uid][igork1][1])
								 +conj(X1[uid][idirac][1])*
								 (conj(u12t[i][mu])*X2[i][igork1][0]
								  -u11t[i][mu] *X2[i][igork1][1])));

				}
				//We're not done tripping yet!! Time like term is different. dk4? shows up
				//For consistency we'll leave mu in instead of hard coding.
				mu=3;
				uid = iu[mu][i];
				//We are mutiplying terms by dk4?[i] Also there is no akappa or gamval factor in the time direction		
				//for the "gamval" terms the sign of d4kp flips
				dSdpi[i][0][mu]+=creal(zi*
						(conj(X1[i][idirac][0])*
						 (dk4m[i]*(-conj(u12t[i][mu])*X2[uid][idirac][0]
							     +conj(u11t[i][mu])*X2[uid][idirac][1]))
						 +conj(X1[uid][idirac][0])*
						 (dk4p[i]*      (+u12t[i][mu] *X2[i][idirac][0]
								     -conj(u11t[i][mu])*X2[i][idirac][1]))
						 +conj(X1[i][idirac][1])*
						 (dk4m[i]*       (u11t[i][mu] *X2[uid][idirac][0]
									+u12t[i][mu] *X2[uid][idirac][1]))
						 +conj(X1[uid][idirac][1])*
						 (dk4p[i]*      (-u11t[i][mu] *X2[i][idirac][0]
								     -conj(u12t[i][mu])*X2[i][idirac][1]))))
					+creal(zi*
							(conj(X1[i][idirac][0])*
							 (dk4m[i]*(-conj(u12t[i][mu])*X2[uid][igork1][0]
								     +conj(u11t[i][mu])*X2[uid][igork1][1]))
							 +conj(X1[uid][idirac][0])*
							 (-dk4p[i]*       (u12t[i][mu] *X2[i][igork1][0]
										 -conj(u11t[i][mu])*X2[i][igork1][1]))
							 +conj(X1[i][idirac][1])*
							 (dk4m[i]*       (u11t[i][mu] *X2[uid][igork1][0]
										+u12t[i][mu] *X2[uid][igork1][1]))
							 +conj(X1[uid][idirac][1])*
							 (-dk4p[i]*      (-u11t[i][mu] *X2[i][igork1][0]
										-conj(u12t[i][mu])*X2[i][igork1][1]))));

				dSdpi[i][1][mu]+=creal(
						conj(X1[i][idirac][0])*
						(dk4m[i]*(-conj(u12t[i][mu])*X2[uid][idirac][0]
							    +conj(u11t[i][mu])*X2[uid][idirac][1]))
						+conj(X1[uid][idirac][0])*
						(dk4p[i]*      (-u12t[i][mu] *X2[i][idirac][0]
								    -conj(u11t[i][mu])*X2[i][idirac][1]))
						+conj(X1[i][idirac][1])*
						(dk4m[i]*      (-u11t[i][mu] *X2[uid][idirac][0]
								    -u12t[i][mu] *X2[uid][idirac][1]))
						+conj(X1[uid][idirac][1])*
						(dk4p[i]*      ( u11t[i][mu] *X2[i][idirac][0]
								     -conj(u12t[i][mu])*X2[i][idirac][1])))
					+creal(
							(conj(X1[i][idirac][0])*
							 (dk4m[i]*(-conj(u12t[i][mu])*X2[uid][igork1][0]
								     +conj(u11t[i][mu])*X2[uid][igork1][1]))
							 +conj(X1[uid][idirac][0])*
							 (-dk4p[i]*      (-u12t[i][mu] *X2[i][igork1][0]
										-conj(u11t[i][mu])*X2[i][igork1][1]))
							 +conj(X1[i][idirac][1])*
							 (dk4m[i]*      (-u11t[i][mu] *X2[uid][igork1][0]
									     -u12t[i][mu] *X2[uid][igork1][1]))
							 +conj(X1[uid][idirac][1])*
							 (-dk4p[i]*       (u11t[i][mu] *X2[i][igork1][0]
										 -conj(u12t[i][mu])*X2[i][igork1][1]))));

				dSdpi[i][2][mu]+=creal(zi*
						(conj(X1[i][idirac][0])*
						 (dk4m[i]*       (u11t[i][mu] *X2[uid][idirac][0]
									+u12t[i][mu] *X2[uid][idirac][1]))
						 +conj(X1[uid][idirac][0])*
						 (dk4p[i]*(-conj(u11t[i][mu])*X2[i][idirac][0]
							     -u12t[i][mu] *X2[i][idirac][1]))
						 +conj(X1[i][idirac][1])*
						 (dk4m[i]* (conj(u12t[i][mu])*X2[uid][idirac][0]
								-conj(u11t[i][mu])*X2[uid][idirac][1]))
						 +conj(X1[uid][idirac][1])*
						 (dk4p[i]*(-conj(u12t[i][mu])*X2[i][idirac][0]
							     +u11t[i][mu] *X2[i][idirac][1]))))
					+creal(zi*
							(conj(X1[i][idirac][0])*
							 (dk4m[i]*       (u11t[i][mu] *X2[uid][igork1][0]
										+u12t[i][mu] *X2[uid][igork1][1]))
							 +conj(X1[uid][idirac][0])*
							 (-dk4p[i]*(-conj(u11t[i][mu])*X2[i][igork1][0]
									-u12t[i][mu] *X2[i][igork1][1]))
							 +conj(X1[i][idirac][1])*
							 (dk4m[i]* (conj(u12t[i][mu])*X2[uid][igork1][0]
									-conj(u11t[i][mu])*X2[uid][igork1][1]))
							 +conj(X1[uid][idirac][1])*
							 (-dk4p[i]*(-conj(u12t[i][mu])*X2[i][igork1][0]
									+u11t[i][mu] *X2[i][igork1][1]))));

			}
	}
	return 0;
}
int Gauge_force(double dSdpi[][3][ndirac]){
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
	//	#ifdef DEBUG
	//		u11t=calloc(ndim*(kvol+halo),sizeof(complex));
	//		u12t=calloc(ndim*(kvol+halo),sizeof(complex));
	//	#endif
#pragma unroll
	for(int mu=0; mu<ndim; mu++){
		//Since we've had to swap the rows and columns of u11t and u12t we need to extract the 
		//correct terms for a halo exchange
		//A better approach is clearly needed
		complex z[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, &u11t[0][mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i][mu];
#endif
		ZHalo_swap_all(z,1);
		//And the swap back
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u11t[0][mu], ndim);
#else
		for(int i=0; i<kvol;i++)
			u11t[i][mu]=z[i];
#endif
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, &u12t[0][mu], 4, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u12t[i][mu];
#endif
		ZHalo_swap_all(z,1);
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z, 1, &u12t[0][mu], 4);
#else
		for(int i=0; i<kvol;i++)
			u12t[i][mu]=z[i];
#endif
	}
	complex Sigma11[kvol], Sigma12[kvol] __attribute__((aligned(AVX)));
	complex u11sh[kvol+halo], u12sh[kvol+halo] __attribute__((aligned(AVX)));
	//Holders for directions
	for(int mu=0; mu<ndim; mu++){
		memset(Sigma11,0, kvol*sizeof(complex));
		memset(Sigma12,0, kvol*sizeof(complex));
		complex a11[kvol], a12[kvol] __attribute__((aligned(AVX)));
		for(int nu=0; nu<ndim; nu++){
			if(mu!=nu){
				//The +ν Staple
#pragma omp parallel for
				for(int i=0;i<kvol;i++){

					int uidm = iu[mu][i];
					int uidn = iu[nu][i];
					a11[i]=u11t[uidm][nu]*conj(u11t[uidn][mu])+\
						 u12t[uidm][nu]*conj(u12t[uidn][mu]);
					a12[i]=-u11t[uidm][nu]*u12t[uidn][mu]+\
						 u12t[uidm][nu]*u11t[uidn][mu];

					Sigma11[i]+=a11[i]*conj(u11t[i][nu])+a12[i]*conj(u12t[i][nu]);
					Sigma12[i]+=-a11[i]*u12t[i][nu]+a12[i]*u11t[i][nu];
				}
				complex z[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
				cblas_zcopy(kvol+halo, &u11t[0][nu], 4, z, 1);
#else
#pragma unroll
				for(int i=0; i<kvol+halo;i++)
					z[i]=u11t[i][nu];
#endif
				Z_gather(u11sh, z, kvol, id[nu]);
#ifdef USE_MKL
				cblas_zcopy(kvol+halo, &u12t[0][nu], 4, z, 1);
#else
#pragma unroll
				for(int i=0; i<kvol+halo;i++)
					z[i]=u12t[i][nu];
#endif
				Z_gather(u12sh, z, kvol, id[nu]);
				ZHalo_swap_dir(u11sh, 1, mu, DOWN);
				ZHalo_swap_dir(u12sh, 1, mu, DOWN);
				//Next up, the -ν staple
#pragma omp parallel for
				for(int i=0;i<kvol;i++){
					int uidm = iu[mu][i];	int uidn = iu[nu][i];
					int didm = id[mu][i];	int didn = id[nu][i];
					//uidm is correct here
					a11[i]=conj(u11sh[uidm])*conj(u11t[didn][mu])-\
						 u12sh[uidm]*conj(u12t[didn][mu]);
					a12[i]=-conj(u11sh[uidm])*u12t[didn][mu]-\
						 u12sh[uidm]*u11t[didn][mu];

					Sigma11[i]+=a11[i]*u11t[didn][nu]-a12[i]*conj(u12t[didn][nu]);
					Sigma12[i]+=a11[i]*u12t[didn][nu]+a12[i]*conj(u11t[didn][nu]);
				}
			}
		}
#pragma omp parallel for
		for(int i=0;i<kvol;i++){
			a11[i] = u11t[i][mu]*Sigma12[i]+u12t[i][mu]*conj(Sigma11[i]);
			a12[i] = u11t[i][mu]*Sigma11[i]+conj(u11t[i][mu])*Sigma12[i];

			dSdpi[i][0][mu]=beta*cimag(a11[i]);
			dSdpi[i][1][mu]=beta*creal(a11[i]);
			dSdpi[i][2][mu]=beta*cimag(a12[i]);

		}
	}
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
	 * int isweep:	The current sweep number
	 *
	 * Returns:
	 * =======
	 * Zero on success. Integer Error code otherwise.
	 */	
	const char *funcname = "Hamilton";
	double hp;
	//Itereate over momentum terms.
#ifdef USE_MKL
	//Can we use BLAS here with the halo?
	//The halo could interfere with things
	hp = cblas_dnrm2(kmom, **pp, 1);
	hp*=hp;
#else
	hp=0;
	for(int i = 0; i<kmom; i++)
		//Three dimensions, so three pointers to get down the the actual value
		//What we're effectively doing is
		hp+=(***(pp+i))*(***(pp+i)); 
#endif
	hp*=0.5;
	double avplaqs, avplaqt;
	double hg = 0;
	//avplaq? isn't seen again here.
	SU2plaq(&hg,&avplaqs,&avplaqt);

	double hf = 0; int itercg;
	complex smallPhi[kferm2Halo] __attribute__((aligned(AVX)));
	//Iterating over flavours
	for(int na=0;na<nf;na++){
		memcpy(X1,X0[na],kferm2*sizeof(complex));
		Congradq(na,res2,&itercg);
		ancgh+=itercg;
		Fill_Small_Phi(na, smallPhi);
		memcpy(X0[na],X1,kferm2*sizeof(complex));
#ifdef USE_MKL
		complex dot;
		cblas_zdotc_sub(kferm2, smallPhi, 1, X1, 1, &dot);
		hf+=creal(dot);
#else
		//It is a dot product of the flattend arrays, could use
		//a module to convert index to coordinate array...
		for(int j=0;j<kferm2;j++)
			//Cheat using pointer for now
			hf+= ***(X1+j)*conj(smallPhi[j]) ;
#endif
	}
	//hg was summed over inside of SU2plaq.
	Par_dsum(&hp); Par_dsum(&hf);
	*s=hg+hf; *h=*s+hp;
	//Here the FORTRAN code prints isweep and the values of all the h's.
	//I'm going to use the preprocessor to do that instead, with the isweep
	//outside the function.
#ifdef DEBUG
	if(!rank)
		printf("hg=%f; hp=%f; hf=%f; h=%f\n", hg, hp, hf, *h);
#endif

	return 0;
}
int Congradq(int na, double res, int *itercg){
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
	complex betad = 1.0; complex alphad=0; complex alpha = 1;
	//Because we're dealing with flattened arrays here we can call cblas safely without the halo
	complex p[kferm2Halo], r[kferm2], smallPhi[kferm2Halo] __attribute__((aligned(AVX)));
	Fill_Small_Phi(na, smallPhi);
	//Instead of copying elementwise in a loop, use memcpy.
	memcpy(p, X1, kferm2*sizeof(complex));
	memcpy(r, smallPhi, kferm2*sizeof(complex));

	//	Declaring placeholder vectors
	complex *x1, *x2;
#ifdef USE_MKL
	x1=mkl_calloc(kferm2Halo, sizeof(complex), AVX);
	x2=mkl_calloc(kferm2Halo, sizeof(complex), AVX);
#else
	x1=calloc(kferm2Halo,sizeof(complex));
	x2=calloc(kferm2Halo,sizeof(complex));
#endif
	//niterx isn't called as an index but we'll start from zero with the C code to make the
	//if statements quicker to type
	complex betan;
	for(int niterx=0; niterx<niterc; niterx++){
		(*itercg)++;
		//x2 =  (M^†M)p 
		Hdslash(x1,p); Hdslashd(x2, x1);
		//x2 =  (M^†M+J^2)p 
#ifdef USE_MKL
		cblas_zaxpy(kferm2, &fac, p, 1, x2, 1);
#else
		for(int i=0; i<kferm2; i++)
			x2[i]+=fac*p[i];
#endif
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(niterx){
			//α_d= p* (M^†M+J^2)p
#ifdef USE_MKL
			cblas_zdotc_sub(kferm2, p, 1, x2, 1, &alphad);
#else
			alphad=0;
			for(int i=0; i<kferm2; i++)
				alphad+=conj(p[i])*x2[i];
#endif
			//And reduce. α_d does have a complex component but we only care about the real part
			//Since we pass the address into Par_?sum it will only reduce the first 8 bits of
			//α_d (i.e. the real part)
			Par_dsum(&alphad);
			//α=α_n/α_d = (r.r)/p(M^†M)p 
			alpha=creal(alphan)/creal(alphad);
			//x-αp, 
#ifdef USE_MKL
			cblas_zaxpy(kferm2, &alpha, p, 1, X1, 1);
#else
			for(int i=0; i<kferm2; i++)
				***(X1+i)+=alpha*p[i];
#endif
		}			
		// r_n+1 = r_n-α(M^† M)p_n and β_n=r*.r
#ifdef USE_MKL
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
		Par_dsum(&betan);
		//Here we evaluate β=(r_{k+1}.r_{k+1})/(r_k.r_k) and then shuffle our indices down the line.
		//On the first iteration we define beta to be zero.
		complex beta = (niterx) ?  betan/betad : 0;
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
#ifdef DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %f toler = %f\n", niterx, creal(betan), resid);
#endif
			return 0;
		}
	}
	if(!rank)
		fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i β_n=%f\n", ITERLIM, funcname, niterc, creal(betan));
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
	complex betad = 1.0; complex alphad=0; complex alpha = 1;
	complex p[kfermHalo], r[kferm] __attribute__((aligned(AVX)));
	//Instead of copying elementwise in a loop, use memcpy.
	memcpy(p, xi, kfermHalo*sizeof(complex));
	memcpy(r, &Phi[na], kferm*sizeof(complex));

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
	for(int niterx=0; niterx<niterc; niterx++){
		(*itercg)++;
		Dslash(x1,p);
		//We can't evaluate α on the first niterx because we need to get β_n.
		if(niterx){
			//x*.x
#ifdef USE_MKL
			cblas_zdotc_sub(kferm, x1, 1, x1, 1, &alphad);
#else
			alphad=0;
			for(int i = 0; i<kferm; i++)
				alphad+=conj(x1[i])*x1[i];
#endif
			Par_dsum(&alphad);
			//α=(r.r)/p(M^†)Mp
			alpha=alphan/alphad;
			//x+αp
#ifdef USE_MKL
			cblas_zaxpy(kferm, &alpha, p, 1, xi, 1);
#else
			for(int i = 0; i<kferm; i++)
				***(xi+i)+=alpha*p[i];
#endif
		}
		//x2=(M^†)x1=(M^†)Mp
		Dslashd(x2,x1);
		//r-α(M^†)Mp and β_n=r*.r
		complex betan;
#ifdef USE_MKL
		alpha*=-1;
		cblas_zaxpy(kferm, &alpha, x2, 1, r, 1);
		//Undoing the negation from the BLAS routine
		alpha*=-1;
		//r*.r
		betan = cblas_dznrm2(kferm, r,1);
		//Gotta square it to "undo" the norm
		betan *= betan;
#else
		//Just like Congradq, this loop could be unrolled but will need a reduction to deal with the betan 
		//addition.
		betan = 0;
		for(int i = 0; i<kferm;i++){
			r[i]-=alpha* *(x2+i);
			betan+=conj(r[i])*r[i];
		}
#endif
		//This is basically just congradq at the end. Check there for comments
		Par_dsum(&betan);
		beta = (niterx) ? betan/betad : 0;
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
#ifdef DEBUG
			if(!rank) printf("Iter (CG) = %i resid = %f toler = %f", niterx, betan, resid);
#endif
			return 0;
		}
	}
	if(!rank) fprintf(stderr, "Warning %i in %s: Exceeded iteration limit %i\n", ITERLIM, funcname, niterc);
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
	complex *ps;
	//This x is just a storage container
	complex x[kvol+halo][ngorkov][nc] __attribute__((aligned(64)));
#ifdef USE_MKL
	ps = mkl_calloc(kvol, sizeof(complex), AVX);
#else
	ps = calloc(kvol, sizeof(complex));
#endif
	//Setting up noise. I don't see any reason to loop
	//over colour indices as it is a two-colour code.
	//where I do have an issue is the loop ordering.

	//The root two term comes from the fact we called gauss0 in the fortran code instead of gaussp
#ifdef USE_MKL
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, 2*ngorkov*nc*kvol, xi[0], 0, 1/sqrt(2));
#else
	Gauss_z(xi[0], ngorkov, 0, 1/sqrt(2));
#endif
	memcpy(x, xi, nc*ngorkov*kvol*sizeof(complex));

	//R_1= M^† Ξ 
	Dslashd(R1, xi);
	//Copying R1 to the first (zeroth) flavour index of Phi
	//This should be safe with memcpy since the pointer name
	//references the first block of memory for that pointer
	memcpy(Phi, R1, 2*ngorkov*kvol*sizeof(complex));
	memcpy(xi, R1, 2*ngorkov*kvol*sizeof(complex));

	//Evaluate xi = (M^† M)^-1 R_1 
	Congradp(1, res, itercg);
	*pbp = 0;
#ifdef USE_MKL
	*pbp = cblas_dznrm2(kvol*ngorkov*nc, x, 1);
	*pbp*=*pbp;
#else
	for(int i=0;i<kvol;i++)
#pragma unroll
		for(int igorkov = 0; igorkov<ngorkov; igorkov++){
			*pbp+=creal(conj(x[i][igorkov][0])*x[i][igorkov][0]);
			*pbp+=creal(conj(x[i][igorkov][1])*x[i][igorkov][1]);
		}

#endif
	Par_dsum(pbp);
	*pbp/=4*gvol;

	*qbqb=0; *qq=0;
	for(int idirac = 0; idirac<ndirac; idirac++){
		int igork=idirac+4;
		//Unrolling the colour indices, Then its just (γ_5*x)*Ξ or (γ_5*Ξ)*x 
#ifdef USE_MKL
#pragma unroll
		for(int ic = 0; ic<nc; ic++){
			complex dot;
			//Because we have kvol on the outer index and are summing over it, we set the
			//step for BLAS to be ngorkov*nc=16. 
			cblas_zdotc_sub(kvol, &x[0][idirac][ic], ngorkov*nc, &xi[0][igork][ic], ngorkov*nc, &dot);
			*qbqb+=dot;
			cblas_zdotc_sub(kvol, &x[0][igork][ic], ngorkov*nc, &xi[0][idirac][ic], ngorkov*nc, &dot);
			*qq-=dot;
		}
		*qbqb *= gamval[4][idirac];
		*qq *= gamval[4][idirac];
#else
#pragma unroll
		for(int i=0; i<kvol; i++){
			//What is the optimal order to evaluate these in?
			*qbqb+=gamval[4][idirac]*conj(x[i][idirac][0])*xi[i][igork][0];
			*qq-=gamval[4][idirac]*conj(x[i][igork][0])*xi[i][idirac][0];
			*qbqb+=gamval[4][idirac]*conj(x[i][idirac][1])*xi[i][igork][1];
			*qq-=gamval[4][idirac]*conj(x[i][igork][1])*xi[i][idirac][1];
		}
#endif
	}
	//In the FORTRAN Code dsum was used instead despite qq and qbqb being complex
	//Does that extract the real component?
	Par_dsum(qq); Par_dsum(qbqb);
	*qq=(*qq+*qbqb)/(2*gvol);
	double xu, xd, xuu, xdd;

	//Halos
	ZHalo_swap_dir(x,16,3,DOWN);		ZHalo_swap_dir(x,16,3,UP);
	//Pesky halo exchange indices again
	complex z[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
	cblas_zcopy(kvol, &u11t[0][3], 4, z, 1);
#else
#pragma unroll
	for(int i=0; i<kvol;i++)
		z[i]=u11t[i][3];
#endif
	ZHalo_swap_dir(z,1,3, UP);
#ifdef USE_MKL
	cblas_zcopy(kvol+halo, z, 1, &u11t[0][3], 4);
#else
#pragma unroll
	for(int i=0; i<kvol;i++)
		u11t[i][3]=z[i];
#endif
#ifdef USE_MKL
	cblas_zcopy(kvol, &u12t[0][3], 4, z, 1);
#else
#pragma unroll
	for(int i=0; i<kvol;i++)
		z[i]=u12t[i][3];
#endif
	ZHalo_swap_dir(z,1,3, UP);
#ifdef USE_MKL
	cblas_zcopy(kvol+halo, z, 1, &u12t[0][3], 4);
#else
#pragma unroll
	for(int i=0; i<kvol;i++)
		u12t[i][3]=z[i];
#endif

	DHalo_swap_dir(dk4p, 1, 3, UP);		DHalo_swap_dir(dk4m, 1, 3, UP);	
	//Instead of typing id[i][3] a lot, we'll just assign them to variables.
	int did, uid, igork1;
	//Idea. One loop instead of two loops but for xuu and xdd just use ngorkov-(igorkov+1) instead
#pragma omp parallel for reduction(+:xd,xu) 
	for(int i = 0; i<kvol; i++){
		did=id[3][i];
		uid=iu[3][i];
#pragma unroll
		for(int igorkov=0; igorkov<4; igorkov++){
			igork1=gamin[3][igorkov];
			//For the C Version I'll try and factorise where possible

			xu+=dk4p[did]*(conj(x[did][igorkov][0])*(\
						u11t[did][3]*(xi[i][igork1][0]-xi[i][igorkov][0])+\
						u12t[did][3]*(xi[i][igork1][1]-xi[i][igorkov][1]) )+\
					conj(x[did][igorkov][1])*(\
						conj(u11t[did][3])*(xi[i][igork1][1]-xi[i][igorkov][0])+\
						conj(u12t[did][3])*(xi[i][igorkov][0]-xi[i][igork1][1]) ) );			

			//This looks very BLASable if not for the UID terms. Could BLAS the vector sums?
			//No. That's stupid and won't work.
			xd+=dk4m[i]*(conj(x[uid][igorkov][0])*(\
						conj(u11t[3][i])*(xi[i][igork1][0]+xi[i][igorkov][0])-\
						u12t[i][3]*(xi[i][igork1][1]+xi[i][igorkov][1]) )+\
					conj(x[uid][igorkov][1])*(\
						u11t[3][i]*(xi[i][igork1][1]+xi[i][igorkov][1])+\
						conj(u12t[i][3])*(xi[i][igorkov][0]+xi[i][igork1][0]) ) );
		}
	}
	int idirac;
#pragma omp parallel for reduction(+:xdd,xuu) 
	for(int i = 0; i<kvol; i++){
		did=id[3][i];
		uid=id[3][i];
#pragma unroll
		for(int igorkov = 4; igorkov<ngorkov; igorkov++){
			idirac=igorkov-4;
			igork1=gamin[3][idirac]+4;
			xuu-=dk4m[did]*(conj(x[did][igorkov][0])*(\
						u11t[did][3]*(xi[i][igork1][0]-xi[i][igorkov][0])+\
						u12t[did][3]*(xi[i][igork1][1]-xi[i][igorkov][1]) )+\
					conj(x[1][igorkov][did])*(\
						conj(u11t[did][3])*(xi[i][igork1][1]-xi[i][igorkov][1])+\
						conj(u12t[did][3])*(xi[i][igorkov][0]-xi[i][igork1][0]) ) );

			xdd-=dk4p[uid]*(conj(x[uid][igorkov][0])*(\
						conj(u11t[3][i])*(xi[i][igork1][0]+xi[i][igorkov][0])-\
						u12t[i][3]*(xi[i][igork1][1]+xi[i][igorkov][1]) )+\
					conj(x[i][igorkov][1])*(\
						u11t[3][i]*(xi[i][igork1][1]+xi[i][igorkov][1])+\
						conj(u12t[i][3])*(xi[i][igorkov][0]+xi[i][igork1][0]) ) );
		}
	}
	*endenf=xu-xd-xuu+xdd;
	*denf=xu+xd+xuu+xdd;

	Par_dsum(endenf); Par_dsum(denf);
	*endenf/=2*gvol; *denf/=2*gvol;
	//Future task. Chiral susceptibility measurements
#ifdef USE_MKL
	mkl_free(ps);
#else
	free(ps);
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
	for(int mu=0;mu<ndim;mu++){
		complex z1[kvol+halo], z2[kvol+halo] __attribute__((aligned(AVX)));
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, &u11t[0][mu], ndim, z1, 1);
		cblas_zcopy(kvol+halo, &u12t[0][mu], ndim, z2, 1);
#else
		for(int i=0; i<kvol;i++){
			z1[i]=u11t[i][mu];
			z2[i]=u12t[i][mu];
		}
#endif
#pragma unroll
		for(int idir=0;idir<4;idir++){
			ZHalo_swap_dir(z1,1,idir,DOWN);
			ZHalo_swap_dir(z2,1,idir,DOWN);
		}
#ifdef USE_MKL
		cblas_zcopy(kvol+halo, z1, 1, &u11t[0][mu], ndim);
		cblas_zcopy(kvol+halo, z2, 1, &u12t[0][mu], ndim);
#else
		for(int i=0; i<kvol;i++){
			u11t[i][mu]=z1[i];
			u12t[i][mu]=z2[i];
		}
#endif
	}
	complex Sigma11[kvol], Sigma12[kvol] __attribute__((aligned(AVX)));
	//	The fortran code used several consecutive loops to get the plaquette
	//	Instead we'll just make a11 and a12 values and do everything in one loop
	//	complex a11[kvol], a12[kvol]  __attribute__((aligned(AVX)));
	double hgs = 0; double hgt = 0;
	for(int mu=0;mu<ndim;mu++)
		for(int nu=0;nu<mu;nu++){
			//Don't merge into a single loop. Makes vectorisation easier?
			//Or merge into a single loop and dispense with the a arrays?
#pragma omp parallel for
			for(int i=0;i<kvol;i++){
				int uidm = iu[mu][i]; 

				Sigma11[i]=u11t[i][mu]*u11t[uidm][nu]-u12t[i][mu]*conj(u12t[uidm][nu]);
				Sigma12[i]=u11t[i][mu]*u12t[uidm][nu]+u12t[i][mu]*conj(u11t[uidm][nu]);
				//			}
				//			for(i=0;i<kvol;i++){
				int uidn = iu[nu][i]; 
				complex a11=Sigma11[i]*conj(u11t[uidn][mu])+Sigma12[i]*conj(u12t[uidn][mu]);
				complex a12=-Sigma12[i]*u12t[uidn][mu]+Sigma12[i]*u11t[uidn][mu];
				//			}
				//			for(i=0;i<kvol;i++){
				Sigma11[i]=a11*conj(u11t[i][nu])+a12*conj(u12t[i][nu]);
				//				Sigma12[i]=-a11[i]*u12t[i][nu]+a12*u11t[i][mu];
				//				Not needed in final result as it traces out?
		}
		//Space component
		if(mu<ndim-1)
			//Is there a BLAS routine for this
#pragma omp parallel for reduction(+:hgs)
			for(int i=0;i<kvol;i++)
				hgs-=creal(Sigma11[i]);
		//Time component
		else
#pragma omp parallel for reduction(+:hgt)
			for(int i=0;i<kvol;i++)
				hgt-=creal(Sigma11[i]);
		}
	Par_dsum(&hgs); Par_dsum(&hgt);
	*avplaqs=-hgs/(3*gvol); *avplaqt=-hgt/(gvol*3);
	*hg=(hgs+hgt)*beta;
#ifdef DEBUG
	if(!rank)
		printf("hgs=%f  hgt=%f  hg=%f\n", hgs, hgt, *hg);
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
	complex Sigma11[kvol3], Sigma12[kvol3] __attribute__((aligned(AVX)));
#ifdef USE_MKL
	cblas_zcopy(kvol3, &u11t[0][3], ndim, Sigma11, 1);
	cblas_zcopy(kvol3, &u12t[0][3], ndim, Sigma12, 1);
#else
	for(int i=0; i<kvol3; i++){
		Sigma11[i]=u11t[i][3];
		Sigma12[i]=u12t[i][3];
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
	int indexu=kvol3;
	//Buffers
	complex a11=0; complex  a12 = 0;
	for(int it=1;it<ksizet;it++){
		for(int i=0;i<kvol3;i++){
			//Seems a bit more efficient to increment indexu instead of reassigning
			//it every single loop
			a11=Sigma11[i]*u11t[indexu][3]-Sigma12[i]*conj(u12t[indexu][3]);
			a12=Sigma11[i]*u12t[indexu][3]+Sigma12[i]*conj(u11t[indexu][3]);
			Sigma11[i]=a11; Sigma12[i]=a12;
			indexu++;
		}
	}
	//Multiply this partial loop with the contributions of the other cores in the
	//timelike dimension
#ifdef DEBUG
	printf("Multiplying with MPI\n");
#endif
	//Par_tmul does nothing if there is only a single processor in the time direction. So we only compile
	//its call if it is required
#if (npt>1)
	Par_tmul(Sigma11, Sigma12);
#endif
	double poly = 0;
	//There has to be a vectorised method of doing this somewhere, or a reduction method
	//for large k
#pragma omp parallel for reduction(+:poly)
	for(int i=0;i<kvol3;i++)
		poly+=creal(Sigma11[i]);
	//Now all cores have the value for the complete Polyakov line at all spacial sites
	//We need to globally sum over spacial processores but not across time as these
	//are duplicates. So we zero the value for all but t=0

	//This is (according to the FORTRAN code) a bit of a hack
	if(pcoord[3][rank]) poly = 0;
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

#pragma omp parallel for collapse(2)
	for(int i=0; i<kvol; i++)
		for(int mu = 0; mu<ndim; mu++){
			//Declaring anorm inside the loop will hopefully let the compiler know it
			//is safe to vectorise aggessively
			double anorm=sqrt(conj(u11t[i][mu])*u11t[i][mu]+conj(u12t[i][mu])*u12t[i][mu]);
			if(anorm==0){
				fprintf(stderr, "Error %i in %s on rank %i: anorm = 0 for μ=%i and i=%i.\nExiting...\n\n",
						DIVZERO, funcname, rank, mu, i);
				MPI_Finalize();
				exit(DIVZERO);
			}
			u11t[i][mu]/=anorm;
			u12t[i][mu]/=anorm;
		}
	return 0;
}
inline int Z_gather(complex *x, complex *y, int n, int *table){
	for(int i=0; i<n; i++)
		x[i]=y[table[i]];
	return 0;
}
inline int Fill_Small_Phi(int na, complex smallPhi[][ndirac][nc]){
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
#pragma omp parallel for collapse(2)
	for(int i = 0; i<kvol;i++)
		for(int idirac = 0; idirac<ndirac; idirac++)
#pragma unroll
			for(int ic= 0; ic<nc; ic++){
				// The original code behaves in this manner, same loop order and same formula for PHI_index
				// We end up hiting the first 8 elements of the Phi array. But because i is multiplied by
				// 2*ndirac*nc we end up skipping the second, fourth, sixth etc. groups of 8 elements.
				// This is not yet clear to me why, but I'll update when it is clarified.
				//	  PHI_index=i*16+j*2+k;
				smallPhi[i][idirac][ic]=Phi[na][2*i][idirac][ic];
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
	for(int i=0; i<n; i++)
		norm+=z[i]*conj(z[i]);
	return norm;
}
