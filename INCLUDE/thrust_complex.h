#ifndef ACCCOMPLEX
#define ACCCOMPLEX
/*
 * @file thrust_complex.h
 *
 * @brief Complex Header for CUDA. Sets macros for C compatability
 *
 * We are also adding dding the macros for extracting the real, imaginary parts.
 * This way they match the C standard library calles
 */
#warning "Compiling with thrust"
#include <thrust/complex.h>
#include <math.h>
#define TCMPLX
using thrust::complex;
///@brief	Exponentiate
#define	cexp(z)	thrust::exp(z)
///@brief	Extract Imaginary Component
#define	cimag(z)	z.imag()
///@brief	Extract Real Component
#define	creal(z)	z.real()
///@brief	Complex Conjugation
#define	conj(z)	thrust::conj(z)
///@brief 	Define I
#define	I	Complex(0.0,1.0)	

///@brief Single precision complex number 
typedef	complex<float>		Complex_f;
///@brief Double precision complex number 
typedef	complex<double>	Complex;
#endif
/*
#ifdef __CUDACC__ || defined __HIP__
#elif defined __HIPCC__
#warning "Hip Device Compiling"
#include <complex>
using std::complex;
///@brief	Exponentiate
#define	cexp(z)	std::exp(z)
///@brief	Extract Imaginary Component
#define	cimag(z)	z.imag()
///@brief	Extract Real Component
#define	creal(z)	z.real()
///@brief	Complex Conjugation
#define	conj(z)	std::conj(z)
///@brief 	Define I
#define	I	Complex(0.0,1.0)	

///@brief Single precision complex number 
typedef	std::complex<float>		Complex_f;
///@brief Double precision complex number 
typedef	std::complex<double>	Complex;
#endif
#elif
#include <hip/hip_complex.h>
#warning "Compiling with hip_complex"
///@brief Single precision complex number 
typedef hipFloatComplex	Complex_f;
///@brief Double precision complex number 
typedef hipDoubleComplex Complex;

///@brief	Extract Real Component
#define creal(z)	z.x
///@brief	Extract Imaginary Component
#define cimag(z)	z.y
///@brief	Complex Conjugation
///@param	z
__HOST_DEVICE__ static __inline__ Complex conj(Complex z){
	return hipConj(z);
}
///@brief	Single precision Complex Conjugation
///@param	z
__HOST_DEVICE__ static __inline__ Complex_f conj(Complex_f z){
	return hipConjf(z);
}

#ifndef _CEXP
#define _CEXP
///@brief HIP does not contain a cexp routine so I've taken the one from GCC
///and tweaked it slightly. 
///@param	z
__HOST_DEVICE__ static __inline__ Complex cexp(Complex z)
{
	double a = creal(z); double b = cimag(z);
	Complex v;
	v.x=cos(b);v.y=sin(b);
	return exp (a) * v;
}
__HOST_DEVICE__ static __inline__ Complex_f cexp(Complex_f z)
{

	float a = creal(z); float b = cimag(z);
	Complex_f v;
	v.x=cos(b);v.y=sin(b);
	return exp (a) * v;
}
#endif
*/
