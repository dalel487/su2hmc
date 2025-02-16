/**
 * @file	random.h
 *
 * @brief	Header for random number configuration
 */
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
#if (defined__INTEL_COMPILER || __INTEL_LLVM_COMPILER)
#include <mathimf.h>
#endif
#include <par_mpi.h>
//Configuration for existing generators if called
//===============================================
#if (defined USE_RAN2||(!defined __INTEL_MKL__&&!defined __RANLUX__))
extern long seed;
#ifdef __cplusplus
extern "C"
{
#endif
/**
 * @brief Dummy seed the ran2 generator
 *
 * @param seed pointer to seed
 * 
 *	@return 0
 */
	int ranset(long *seed);
	/**
	 * @brief Uses the rank to get a new seed.
	 * Copying from the FORTRAN description here 
	 * c     create new seeds in range seed to 9*seed
	 * c     having a range of 0*seed gave an unfortunate pattern
	 * c     in the underlying value of ds(1) (it was always 10 times bigger
	 * c     on the last processor). This does not appear to happen with 9.
	 *
	 * @param	seed:	The seed from the rank in question.
	 * @param	iread:	Do we read from file or not. Don't remember why it's here as it's not used	
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Par_ranset(long *seed, int iread);
	/**
	 * @brief	Generates uniformly distributed random double between zero and one as
	 * 			described in numerical recipes. It's also thread-safe for different seeds.
	 *
	 * @param	idum: Pointer to the seed
	 *
	 * @return	The random double between zero and one
	 *
	 */
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
/**
 * @brief Seed the ranlux generator from GSL
 *
 * @param seed pointer to seed
 * 
 *	@return 0
 */
	int ranset(unsigned long *seed);
	/**
	 * @brief Uses the rank to get a new seed.
	 * Copying from the FORTRAN description here 
	 * c     create new seeds in range seed to 9*seed
	 * c     having a range of 0*seed gave an unfortunate pattern
	 * c     in the underlying value of ds(1) (it was always 10 times bigger
	 * c     on the last processor). This does not appear to happen with 9.
	 *
	 * @param	seed:	The seed from the rank in question.
	 * @param	iread:	Do we read from file or not. Don't remember why it's here as it's not used	
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Par_ranset(unsigned long *seed, int iread);
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
/**
 * @brief Seed the Intel Mersenne twister generator
 *
 * @param seed pointer to seed
 *
 *	@return 0
 */
	int ranset(unsigned int *seed);
	/**
	 * @brief Uses the rank to get a new seed.
	 * Copying from the FORTRAN description here 
	 * c     create new seeds in range seed to 9*seed
	 * c     having a range of 0*seed gave an unfortunate pattern
	 * c     in the underlying value of ds(1) (it was always 10 times bigger
	 * c     on the last processor). This does not appear to happen with 9.
	 *
	 * @param	seed:	The seed from the rank in question.
	 * @param	iread:	Do we read from file or not. Don't remember why it's here as it's not used	
	 *
	 * @return Zero on success, integer error code otherwise
	 */
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
	/**
	 * @brief	Generates a vector of normally distributed random double precision complex numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 * 
	 * @return Zero on success integer error code otherwise
	 */
	int Gauss_z(Complex *ps, unsigned int n, const Complex mu, const double sigma);
	/**
	 * @brief	Generates a vector of normally distributed random double precision numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 *
	 * @return Zero on success integer error code otherwise
	 */
	int Gauss_d(double *ps, unsigned int n, const double mu, const double sigma);
	/**
	 * @brief	Generates a vector of normally distributed random single precision complex numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 * 
	 * @return Zero on success integer error code otherwise
	 */
	int Gauss_c(Complex_f *ps, unsigned int n, const Complex_f mu, const float sigma);
	/**
	 * @brief	Generates a vector of normally distributed random single precision numbers using the Box-Muller Method
	 * 
	 * @param	ps:		The output array
	 * @param	n:			The array length
	 * @param	mu:		mean
	 * @param	sigma:	variance
	 *
	 * @return Zero on success integer error code otherwise
	 */
	int Gauss_f(float *ps, unsigned int n, const float mu, const float sigma);

	//MPI
	//===
	/**
	 * @brief Reads ps from a file
	 * Since this function is very similar to Par_sread, I'm not really going to comment it
	 * check there if you are confused about things. 
	 *
	 * @param	filename: The name of the file we're reading from
	 * @param	ranval:	The destination for the file's contents
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Par_ranread(char *filename, double *ranval);
	/**
	 * @brief Generates a random double which is then sent to the other ranks
	 *
	 * @return the random number generated
	 */
	double Par_granf();
	/// @brief Test Functions
	int	ran_test();

#ifdef __cplusplus
}
#endif
//RAN2 stuff treat it separately so as to avoid any accidents
#if !(defined RAN2) && !(defined __RANLUX__)
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
