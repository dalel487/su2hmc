/*
 * @file thrust_complex.h
 *
 * @brief Complex Header for CUDA. Sets macros for C compatability
 *
 * We are also adding dding the macros for extracting the real, imaginary parts.
 * This way they match the C standard library calles
 */
#ifndef TCMPLX
#define TCMPLX
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#define SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#include <complex>
using namespace sycl::ext::oneapi::experimental;
//#undef	complex
///@brief Single precision complex number 
#define	Complex_f	complex<float>
///@brief Double precision complex number 
#define	Complex		complex<double>

///@brief	Exponentiate
#define	cexp(z)	exp(z)
///@brief	Extract Imaginary Component
#define	cimag(z)	z.imag()
///@brief	Extract Real Component
#define	creal(z)	z.real()
///@brief	Complex Conjugation
#define	conj(z)	conj(z)
///@brief 	Define I
#define	I	Complex(0.0,1.0)	
#endif
