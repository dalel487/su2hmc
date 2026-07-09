/**
 * @file thrust_complex.h
 *
 * @brief Complex Header for CUDA. Sets macros for C compatability
 *
 * We are also adding the macros for extracting the real, imaginary parts.
 * This way they match the C standard library calls
 */
#ifndef TCMPLX
#define TCMPLX
#include <thrust/complex.h>
using thrust::complex;
#undef conj
//#undef	complex
///@brief Single precision complex number 
#define	Complex_f	 complex<float>
///@brief Double precision complex number 
#define	Complex	 complex<double>


///@brief	Exponentiate using C standard notation
#define	cexp(z)	thrust::exp(z)
///@brief	Extract Imaginary Component using C standard notation
#define	cimag(z)	z.imag()
///@brief	Extract Real Component using C standard notation
#define	creal(z)	z.real()
///@brief 	Define I in double precision using C standard notation
#define	I	Complex(0.0,1.0)	
///@brief 	Define I in single precision
#define	I_f	Complex_f(0.0f,1.0f)	
/**
 * @brief	Complex Conjugation
 * @param	z Number to be conjugated
 */
template <typename T> __device__ __forceinline__ T conj(const T& z);
#endif
