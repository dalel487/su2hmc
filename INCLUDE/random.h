#ifndef	RANDOM
#define	RANDOM
//Need two cases here. MKL/CUDA or not for BLAS and CUDA or not for complex
#ifdef __NVCC__ 
#include <cublas_v2.h>
#endif
#ifdef __INTEL_MKL__
#include <mkl.h>
#include <mkl_vsl.h>
#define M_PI		3.14159265358979323846	/* pi */
#endif
#ifdef __RANLUX__
#include <gsl/gsl_rng.h>
#endif
#include <math.h>
#include <par_mpi.h>
#include <sizes.h>
//Configuration for existing generators if called
//===============================================
#if (defined USE_RAN2||(!defined __INTEL_MKL__&&!defined __RANLUX__))
extern long seed;
#ifdef __cplusplus
extern "C"
{
#endif
	int Par_ranset(long *seed, int iread);
	int ranset(long *seed);
	double ran2(long *idum); 
#ifdef __cplusplus
}
#endif
#elif defined __RANLUX__
extern gsl_rng *ranlux_instd;
//Need to get a float version that uses a different seed for performance reasons.
//Otherwise we get two generators (one float, one double) starting from the same seed. Not good
//For now, the float generator will be a cast of the double one.
//gsl_rng *ranlux_instf;
extern unsigned long seed;
#ifdef __cplusplus
extern "C"
{
#endif
	int Par_ranset(unsigned long *seed, int iread);
	int ranset(unsigned long *seed);
#ifdef __cplusplus
}
#endif
#elif defined __INTEL_MKL__
extern VSLStreamStatePtr stream;
extern unsigned int seed;
#ifdef __cplusplus
extern "C"
{
#endif
	int ranset(unsigned int *seed);
	int Par_ranset(unsigned int *seed, int iread);
#ifdef __cplusplus
}
#endif
#else
extern int seed;
#ifdef __cplusplus
extern "C"
{
#endif
	//Mersenne Twister
	int ranset(int *seed);
	int Par_ranset(int *seed);
#ifdef __cplusplus
}
#endif
#endif
#ifdef __cplusplus
extern "C"
{
#endif
	//Generators:
	//==========
	//Distributions
	//=============
	//Use Box-Müller to generate an array of complex numbers
	int Gauss_z(Complex *ps, unsigned int n, const Complex mu, const double sigma);
	int Gauss_d(double *ps, unsigned int n, const double mu, const double sigma);
	int Gauss_c(Complex_f *ps, unsigned int n, const Complex_f mu, const float sigma);
	int Gauss_f(float *ps, unsigned int n, const float mu, const float sigma);

	//MPI
	//===
	int Par_ranread(char *filename, double *ranval);
	double Par_granf();
	//Test Functions
	int	ran_test();

#ifdef __cplusplus
}
#endif
//RAN2 stuff treat it seperately so as to avoid any accidents
#ifndef RAN2
#define RAN2
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#endif
#endif
