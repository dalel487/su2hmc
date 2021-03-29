#ifndef	RANDOM
#define	RANDOM
#include <complex.h>
#ifdef USE_MKL
      #include <mkl.h>
      #include <mkl_vsl.h>
#endif
#include <par_mpi.h>
#include <SFMT.h>
//Configuration for existing generators if called
//===============================================
extern int seed;
//Luxury for the RANLUX generator, default to 3
//#define lux 3;
#ifdef USE_MKL
	extern VSLStreamStatePtr stream;
#endif
//Mersenne Twister
extern sfmt_t sfmt;
int Rand_init();
//PLACEHOLDERS TO TRICK COMPLIER
double ranget(double *seed);
int ranset(int *seed);
//Generators:
//==========
//Distributions
//=============
//Use Box-Müller to generate an array of complex numbers
int Gauss_z(complex *ps, unsigned int n, const double mu, const double sigma);
int Gauss_d(double *ps, unsigned int n, const double mu, const double sigma);

//MPI
//===
int Par_ranread(char *filename, double *ranval);
int Par_ranset(int *seed);
double Par_granf();

#endif
