/**
 * @file random.c
 *
 * @brief Random number generator related routines
 */
#include "coord.h"
#ifdef	__GPU__
#include <hiprand.h>
#endif
#include "errorcodes.h"
#ifdef	__INTEL_MKL__
#include <mkl.h>
#include <mkl_vsl.h>
//Bad practice? Yes but it is convenient
#endif
#include "par_mpi.h"
#include "random.h"
#include "sizes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//Declaring external variables
#if (defined USE_RAN2||(!defined __INTEL_MKL__&&!defined __RANLUX__))
/// @brief RAN2 seed
long seed;
#elif defined __RANLUX__
/// @brief RANLUX instance
gsl_rng *ranlux_instd;
/// @brief RANLUX seed
unsigned long seed;
#elif defined __INTEL_MKL__
/// @brief Intel Mersene Twister seed
unsigned int seed;
/// @brief Intel Mersene Twister stream
VSLStreamStatePtr stream;
#endif
#ifndef M_PI
/// @brief	@f$\pi@f$ if not defined elsewhere	
#define M_PI  acos(-1)
#endif

#ifdef __RANLUX__
/*
 * @brief Seed the ranlux generator from GSL
 *
 * @param *seed pointer to seed
 *
 * @see gsl_rng_alloc(), gsl_rng_set()
 * 
 *	@return 0
 */
inline int ranset(unsigned long *seed)
#elif (defined __INTEL_MKL__&&!defined USE_RAN2)
/*
 * @brief Seed the Intel Mersenne twister generator
 *
 * @param *seed pointer to seed
 *
 * @see vslNewStream()
 * 
 *	@return 0
 */
inline int ranset(unsigned int *seed)
#else
/*
 * @brief Dummy seed the ran2 generator
 *
 * @param seed pointer to seed
 * 
 *	@return 0
 */
inline int ranset(long *seed)
#endif
{
#ifdef __RANLUX__
	ranlux_instd=gsl_rng_alloc(gsl_rng_ranlxd2);
	gsl_rng_set(ranlux_instd,*seed);
	return 0;
#elif (defined __INTEL_MKL__&& !defined USE_RAN2)
	vslNewStream( &stream, VSL_BRNG_MT19937, *seed );
	return 0;
#else
	return 0;
#endif
}
int Par_ranread(char *filename, double *ranval){
	/*
	 * @brief Reads ps from a file
	 * Since this function is very similar to Par_sread, I'm not really going to comment it
	 * check there if you are confused about things. 
	 *
	 * @param	*filename: The name of the file we're reading from
	 * @param	ps:	The destination for the file's contents
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	const char *funcname = "Par_psread";
	FILE *dest;
	if(!rank){
		if(!(dest = fopen(filename, "rb"))){
			fprintf(stderr, "Error %i in %s: Failed to open %s.\nExiting...\n\n", OPENERROR, funcname, filename);
#if(nproc>1)
			MPI_Abort(comm,OPENERROR); 
#else
			exit(OPENERROR);
#endif

		}
		fread(&ranval, sizeof(ranval), 1, dest);	
		fclose(dest);
	}
#if(nproc>1)
	Par_dcopy(ranval);
#endif
	return 0;
}
#if (defined USE_RAN2||(!defined __INTEL_MKL__&&!defined __RANLUX__))
	/*
	 * @brief Uses the rank to get a new seed.
	 * Copying from the FORTRAN description here 
	 * c     create new seeds in range seed to 9*seed
	 * c     having a range of 0*seed gave an unfortunate pattern
	 * c     in the underlying value of ds(1) (it was always 10 times bigger
	 * c     on the last processor). This does not appear to happen with 9.
	 *
	 * @param	*seed:	The seed from the rank in question.
	 * @param	iread:	Do we read from file or not. Don't remember why it's here as it's not used	
	 *
	 * @see ranset() (used to initialise the stream for MKL at the moment. Legacy from Fortran)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
int Par_ranset(long *seed,int iread)
#elif defined __RANLUX__
	/*
	 * @brief Uses the rank to get a new seed.
	 * Copying from the FORTRAN description here 
	 * c     create new seeds in range seed to 9*seed
	 * c     having a range of 0*seed gave an unfortunate pattern
	 * c     in the underlying value of ds(1) (it was always 10 times bigger
	 * c     on the last processor). This does not appear to happen with 9.
	 *
	 * @param	*seed:	The seed from the rank in question.
	 * @param	iread:	Do we read from file or not. Don't remember why it's here as it's not used	
	 *
	 * @see ranset() (used to initialise the stream for MKL at the moment. Legacy from Fortran)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
int Par_ranset(unsigned long *seed,int iread)
#elif (defined __INTEL_MKL__||defined __RANLUX__)
	/*
	 * @brief Uses the rank to get a new seed.
	 * Copying from the FORTRAN description here 
	 * c     create new seeds in range seed to 9*seed
	 * c     having a range of 0*seed gave an unfortunate pattern
	 * c     in the underlying value of ds(1) (it was always 10 times bigger
	 * c     on the last processor). This does not appear to happen with 9.
	 *
	 * @param	*seed:	The seed from the rank in question.
	 * @param	iread:	Do we read from file or not. Don't remember why it's here as it's not used	
	 *
	 * @see ranset() (used to initialise the stream for MKL at the moment. Legacy from Fortran)
	 *
	 * @return Zero on success, integer error code otherwise
	 */
int Par_ranset(unsigned int *seed,int iread)
#endif
{
	const char *funcname = "Par_ranset";
	//If we're not using the master thread, we need to change the seed
#ifdef _DEBUG
	printf("Master seed: %i\t",*seed);
#endif
	if(rank)
		*seed *= 1.0f+8.0f*(float)rank/(float)(size-1);
#ifdef _DEBUG
	printf("Rank:  %i\tSeed %i\n",rank, *seed);
#endif
	//Next we set the seed using ranset
	//This is one of the really weird FORTRAN 66-esque functions with ENTRY points, so good luck!
#if (defined __INTEL_MKL__||defined __RANLUX__)
	return ranset(seed);
#else
	return 0;
#endif
}
double Par_granf(){
	/*
	 * @brief Generates a random double which is then sent to the other ranks
	 *
	 * @see ran2(), par_dcopy(), gsl_rng_uniform(), vdRngUniform()
	 *
	 * @return the random number generated
	 */
	const char *funcname = "Par_granf";
	double ran_val=0;
	if(!rank){
#if (defined USE_RAN2||(!defined __INTEL_MKL__&&!defined __RANLUX__))
		ran_val = ran2(&seed);
#elif defined __RANLUX__
		ran_val = gsl_rng_uniform(ranlux_instd);
#else
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 1, &ran_val, 0,1);
#endif
	}
#if(nproc>1)
	Par_dcopy(&ran_val);
#endif
	return ran_val;
}
int Gauss_z(Complex *ps, unsigned int n, const Complex mu, const double sigma){
	/*
	 * @brief	Generates a vector of normally distributed random double precision complex numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 *
	 * @see ran2(), par_dcopy(), gsl_rng_uniform(), vdRngUniform()
	 * 
	 * @return Zero on success integer error code otherwise
	 */
	const char *funcname = "Gauss_z";
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
#pragma unroll
	for(int i=0;i<n;i++){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r + I*(mu+v*r);
		 */
#ifdef __RANLUX__
		double	r =sigma*sqrt(-2*log(gsl_rng_uniform(ranlux_instd)));
		double	theta=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		double	r =sigma*sqrt(-2*log(ran2(&seed)));
		double	theta=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=r*(cos(theta)+sin(theta)*I)+mu;
	}     
	return 0;
}
int Gauss_c(Complex_f *ps, unsigned int n, const Complex_f mu, const float sigma){
	/*
	 * @brief	Generates a vector of normally distributed random single precision complex numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 *
	 * @see ran2(), par_dcopy(), gsl_rng_uniform(), vdRngUniform()
	 * 
	 * @return Zero on success integer error code otherwise
	 */
	const char *funcname = "Gauss_z";
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
#pragma unroll
	for(int i=0;i<n;i++){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r + I*(mu+v*r);
		 */
#ifdef __RANLUX__
		float r =sigma*sqrt(-2*log(gsl_rng_uniform(ranlux_instd)));
		float theta=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		float r =sigma*sqrt(-2*log(ran2(&seed)));
		float theta=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=r*(cos(theta)+mu+sin(theta)*I)+mu;
	}     
	return 0;
}
int Gauss_d(double *ps, unsigned int n, const double mu, const double sigma){
	/*
	 * @brief	Generates a vector of normally distributed random double precision numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 *
	 * @see ran2(), par_dcopy(), gsl_rng_uniform(), vdRngUniform()
	 * 
	 * @return Zero on success integer error code otherwise
	 */
	const char *funcname = "Gauss_z";
	//The FORTRAN Code had two different Gauss Routines. gaussp having unit
	//mean and variance and gauss0 where the variance would appear to be 1/sqrt(2)
	//(Since we multiply by sqrt(-ln(r)) instead of sqrt(-2ln(r)) )
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
	int i;
	double r, u, v;
	//If n is odd we calculate the last index seperately and the rest in pairs
	if(n%2==1){
		n--;
#ifdef __RANLUX__
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
		ps[n]=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*cos(r);
#else
		r=2.0*M_PI*ran2(&seed);
		ps[n]=sqrt(-2*log(ran2(&seed)))*cos(r);
#endif
	}
	for(i=0;i<n;i+=2){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r; 
			ps[i+1]=mu+v*r;
		 */
#ifdef __RANLUX__
		u=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*sigma;
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		u=sqrt(-2*log(ran2(&seed)))*sigma;
		r=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=u*cos(r)+mu;
		ps[i+1]=u*sin(r)+mu;
	}     
	return 0;
}
int Gauss_f(float *ps, unsigned int n, const float mu, const float sigma){
	/*
	 * @brief	Generates a vector of normally distributed random single precision numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 *
	 * @see ran2(), par_dcopy(), gsl_rng_uniform(), vdRngUniform()
	 * 
	 * @return Zero on success integer error code otherwise
	 */
	const char *funcname = "Gauss_z";
	//The FORTRAN Code had two different Gauss Routines. gaussp having unit
	//mean and variance and gauss0 where the variance would appear to be 1/sqrt(2)
	//(Since we multiply by sqrt(-ln(r)) instead of sqrt(-2ln(r)) )
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
	int i;
	float r, u, v;
	//If n is odd we calculate the last index seperately and the rest in pairs
	if(n%2==1){
		n--;
#ifdef __RANLUX__
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
		ps[n]=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*cos(r);
#else
		r=2.0*M_PI*ran2(&seed);
		ps[n]=sqrt(-2*log(ran2(&seed)))*cos(r);
#endif
	}
#ifdef __RANLUX__
	r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
	ps[n]=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*cos(r);
#else
	r=2.0*M_PI*ran2(&seed);
	ps[n]=sqrt(-2*log(ran2(&seed)))*cos(r);
#endif
	for(i=0;i<n;i+=2){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r; 
			ps[i+1]=mu+v*r;
		 */
#ifdef __RANLUX__
		u=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*sigma;
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		u=sqrt(-2*log(ran2(&seed)))*sigma;
		r=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=u*cos(r)+mu;
		ps[i+1]=u*sin(r)+mu;
	}
	return 0;
}
#ifndef __RANLUX__
double ran2(long *idum) {
	/*
	 * @brief	Generates uniformly distributed random double between zero and one as
	 * 			described in numerical recipes. It's also thread-safe for different seeds.
	 *
	 * @param	idum: Pointer to the seed
	 *
	 * @return	The random double between zero and one
	 *
	 */
	long k;
	int j;
	static long idum2=123456789; 
	static long iy=0;
	static long iv[NTAB];
	////Combining this with a different seed for each thread should give a thread-safe and repeatable result
#pragma omp threadprivate(idum2, iy, iv)
	//No worries
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum=1; 
		else *idum = -(*idum);
		{
			idum2=(*idum);

			for(j=NTAB+7;j>=0;j--) {
				k=(*idum)/IQ1; 
				*idum=IA1*(*idum-k*IQ1)-k*IR1; 
				if (*idum < 0) *idum += IM1; 
				if (j < NTAB)
					iv[j] = *idum;
			}
			iy=iv[0];
		}
	}
	k=(*idum)/IQ1; 
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if (*idum < 0) *idum += IM1; 
	k=idum2/IQ2; 
	idum2=IA2*(idum2-k*IQ2)-k*IR2;

	if (idum2 < 0) idum2 += IM2; j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) 
		return RNMX; 

	else return temp;

}
#endif

/*
	int ran_test(){
	const char *funcname ="ran_test";
	const double mu = 0.3;
	const double sigma = 2;
	const float mu_f = 0.7;
	const float sigma_f = 1.6;
	long seed = 10;
	ranset(&seed);
	Complex *z_arr=(Complex *)malloc(1024*1024*sizeof(Complex));
	FILE *z_out=fopen("z_ran.out","w");
	Gauss_z(z_arr,1024*1024,mu,sigma);
	for(int i =0; i<1024*1024;i++)
	fprintf(z_out, "%f\t%f\n", creal(z_arr[i]), cimag(z_arr[i]));
	fclose(z_out);
	free(z_arr);
	Complex_f *c_arr=(Complex_f *)malloc(1024*1024*sizeof(Complex_f));
	Gauss_c(c_arr,1024*1024,mu_f,sigma_f);
	FILE *c_out=fopen("c_ran.out","w");
	free(c_arr);
	for(int i =0; i<1024*1024;i++)
	fprintf(c_out,"%f\t%f\n", creal(c_arr[i]), cimag(c_arr[i]));
	fclose(c_out);
	double *d_arr=(double *)malloc(1024*1024*sizeof(double));
	Gauss_d(d_arr,1024*1024,mu,sigma);
	FILE *d_out=fopen("d_ran.out","w");
	for(int i =0; i<1024*1024;i++)
	fprintf(d_out,"%f\n", d_arr[i]);
	fclose(d_out);
	free(d_arr);
	float *f_arr=(float *)malloc(1024*1024*sizeof(float));
	Gauss_f(f_arr,1024*1024,mu_f,sigma_f);
	FILE *f_out=fopen("f_ran.out","w");
	for(int i =0; i<1024*1024;i++)
	fprintf(f_out,"%f\n", f_arr[i]);
	fclose(f_out);
	free(f_arr);
	return 0;
	}
 */
