#include "coord.h"
#include <complex.h>
#ifdef	__NVCC__
#include <curand.h>
#endif
#include "errorcodes.h"
#ifdef	USE_MKL
#include <mkl.h>
#include <mkl_vsl.h>
//Bad practice? Yes but it is convenient
VSLStreamStatePtr stream;
#endif
#include <mpi.h>
#include "par_mpi.h"
#include "random.h"
#include "sizes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Declaring external variables
#ifdef	USE_RAN2
long seed = 967580161;
#elif defined(USE_MKL)
unsigned int seed = 967580161;
#else
int seed = 967580161;
#endif


#ifdef USE_MKL
inline int ranset(unsigned int *seed)
#else
inline int ranset(int *seed)
#endif
{
#ifdef USE_MKL
	vslNewStream( &stream, VSL_BRNG_MT19937, *seed );
#endif
	return 0;
}
int Par_ranread(char *filename, double *ranval){
	/* Reads ps from a file
	 * Since this function is very similar to Par_sread, I'm not really going to comment it
	 * check there if you are confused about things. 
	 *
	 * Parameters;
	 * ==========
	 * char 	*filename: The name of the file we're reading from
	 * double	ps:	The destination for the file's contents
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Par_psread";
	FILE *dest;
	if(!rank){
		if(!(dest = fopen(filename, "rb"))){
			fprintf(stderr, "Error %i in %s: Failed to open %s.\nExiting...\n\n", OPENERROR, funcname, filename);
			exit(OPENERROR); 
		}
		fread(&ranval, sizeof(ranval), 1, dest);	
		fclose(dest);
	}
	Par_dcopy(ranval);
	return 0;
}
#ifdef USE_RAN2
int Par_ranset(long *seed)
#elif defined(USE_MKL)
int Par_ranset(unsigned int *seed)
#else
int Par_ranset(unsigned int *seed)
#endif
{
	/* Uses the rank to get a new seed.
	 * Copying from the fortran description here 
	 * c     create new seeds in range seed to 9*seed
	 * c     having a range of 0*seed gave an unfortunate pattern
	 * c     in the underlying value of ds(1) (it was always 10 times bigger
	 * c     on the last processor). This does not appear to happen with 9.
	 *
	 * Parameters:
	 * ===========
	 * double *seed:  The seed from the rank in question.
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Par_ranset";
	//If we're not using the masterthread, we need to change the seed
	if(rank)
		*seed *= 1.0+8.0*(float)rank/(float)(size-1);
	//Next we set the seed using ranset
	//This is one of the really weird FORTRANN 66 esque functions with ENTRY points, so good luck!
#ifndef USE_RAN2
	return ranset(seed);
#else
	return 0;
#endif
}
double Par_granf(){
	/* Generates a random value which is then sent to the other ranks
	 *
	 * Parameters:
	 * ===========
	 * None!
	 *
	 * Returns:
	 * ========
	 * double: the random number generated
	 *
	 */
	char *funcname = "Par_granf";
	double ran_val;
	if(!rank)
#if defined(USE_RAN2)||!defined(USE_MKL)
	ran_val = ran2(&seed);
#else
	vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, 1, &ran_val, 0,1);
#endif
	Par_dcopy(&ran_val);
	return ran_val;
}
int Gauss_z(complex *ps, unsigned int n, const double mu, const double sigma){
	/* Generates a vector of normally distributed random complex numbers
	 * using the Box-Muller Method
	 * 
	 * Parameters:
	 * ==========
	 * complex *ps:   The array
	 * unsigned int n: The array length
	 * double mu:     mean
	 * double sigma:  variance
	 * 
	 * Returns:
	 * =======
	 * Zero on success integer error code otherwise
	 */
	const char *funcname = "Gauss_z";
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
		exit(ARRAYLEN);
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
#ifdef USE_RAN2
		double	r =sigma*sqrt(-2*log(ran2(&seed)));
		double	theta=2.0*M_PI*ran2(&seed);
		ps[i]=r*(cos(theta)+mu+(sin(theta)+mu)*I);
#endif
	}     
	return 0;
}
int Gauss_d(double *ps, unsigned int n, const double mu, const double sigma){
	/* Generates a vector of normally distributed random complex numbers
	 * using the Box-Muller Method
	 * 
	 * Parameters:
	 * ==========
	 * complex *ps:   The array
	 * unsigned int n: The array length
	 * double mu:     mean
	 * double sigma:  variance
	 * 
	 * Returns:
	 * =======
	 * Zero on success integer error code otherwise
	 */
	const char *funcname = "Gauss_z";
	//The FORTRAN Code had two different Gauss Routines. gaussp having unit
	//mean and variance and gauss0 where the variance would appear to be 1/sqrt(2)
	//(Since we multiply by sqrt(-ln(r)) instead of sqrt(-2ln(r)) )
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
		exit(ARRAYLEN);
	}
	int i;
	double r, u, v;
	//If n is odd we calculate the last index seperately and the rest in pairs
	if(n%2==1){
		n--;
#ifdef USE_RAN2
		r=2.0*M_PI*ran2(&seed);
		ps[n]=sqrt(-2*log(ran2(&seed)))*cos(r);
#else
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
#ifdef USE_RAN2
		u=sqrt(-2*log(ran2(&seed)))*sigma;
		r=2.0*M_PI*ran2(&seed);
#else
#endif
		ps[i]=u*cos(r)+mu;
		ps[i+1]=u*sin(r)+mu;
	}     
	return 0;
}
double ran2(long *idum) {
	long k;
	int j;
	static long idum2=123456789; 
	static long iy=0;
	static long iv[NTAB];
	//Hopefully not anymore. Combining with unique seeds
	//should do the trick (famous last words...)
	//Combining this with a different seed for each thread
	//should give a threadsafe and repeatable result
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
//Distributions
