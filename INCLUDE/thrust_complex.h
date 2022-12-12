//Complex Header for CUDA. Sets macros for C compatability
#ifndef TCMPLX
#define TCMPLX
#include <thrust/complex.h>
using thrust::complex;
//#undef	complex
#define	Complex_f	 complex<float>
#define	Complex	 complex<double>
//Adding the macros for extracting the real, imaginary parts 
#define	cexp(z)	exp(z)
#define	cimag(z)	z.imag()
#define	creal(z)	z.real()
#define	I	Complex(0.0,1.0)	
#endif
