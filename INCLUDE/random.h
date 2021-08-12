#ifndef	RANDOM
#define	RANDOM
//Need two cases here. MKL/CUDA or not for BLAS and CUDA or not for complex
#ifdef USE_MKL
#include <mkl.h>
#include <mkl_vsl.h>
#define M_PI		3.14159265358979323846	/* pi */
#elif (defined __NVCC__ || defined __CUDACC__)
#include <cublas.h>
#else
#include	<cblas.h>
#endif

#include <math.h>
#include <par_mpi.h>
//Configuration for existing generators if called
//===============================================
#if (defined USE_RAN2||!defined USE_MKL)
extern long seed;
int Par_ranset(long *seed);
#elif defined(USE_MKL)
extern unsigned int seed;
int ranset(unsigned int *seed);
int Par_ranset(unsigned int *seed);
extern VSLStreamStatePtr stream;
#else
extern int seed;
//Mersenne Twister
int ranset(int *seed);
int Par_ranset(int *seed);
#endif
//Luxury for the RANLUX generator, default to 3
//#define lux 3;
int Rand_init();
//PLACEHOLDERS TO TRICK COMPLIER
double ranget(double *seed);
//Generators:
//==========
//Distributions
//=============
//Use Box-Müller to generate an array of complex numbers
int Gauss_z(Complex *ps, unsigned int n, const double mu, const double sigma);
int Gauss_d(double *ps, unsigned int n, const double mu, const double sigma);

//MPI
//===
int Par_ranread(char *filename, double *ranval);
double Par_granf();

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

//Prototypes
double ran2(long *idum); 
int	ran_test();
#endif
