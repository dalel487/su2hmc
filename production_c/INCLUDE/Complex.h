/* $Id: Complex.h,v 0.3 2006/09/06 10:59:55 jonivar Exp $ */
//Cudified by Dale Lawlor May 2021
/*** definition of Complex type */ 
#ifndef _CMPLX_H_
#define _CMPLX_H_
#include <assert.h>
#include <math.h>

typedef struct{
	double re;
	double im;
}complex_cuda;
static const complex_cuda I   = {0.0,1.0};
//Routine complex_cuda operations
//#########################
inline __device__ double creal(complex_cuda z){
	return z.re;
}
inline __device__ double cimag(complex_cuda z){
	return z.im;
}
inline __device__ complex_cuda conj(complex_cuda z){
	return {z.re,-z.im};
}
inline __device__ double cnormsq(complex_cuda z){
	return z.re*z.re+z.im*z.im;
}
inline __device__ double cabs(complex_cuda z){
	return sqrt(cnormsq(z));
}

//Binary operators
//###############
//Equivalence
//===========
inline __device__ bool operator==(const complex_cuda &w, const complex_cuda &z){
	if(w.re==z.re && w.im == z.im) return true;
	else return false;
}
inline __device__ bool operator!=(const complex_cuda &w, const complex_cuda &z){
	if(w.re!=z.re || w.im != z.im) return true;
	else return false;
}
inline __device__ bool operator==(const double &x, const complex_cuda &z){
	if(x==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const complex_cuda &z, const double &x){
	if(x!=z.re || z.im != 0) return true;
	else return false;
}
inline __device__ bool operator==(const complex_cuda &z, const double &x){
	if(x==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const double &x, const complex_cuda &z){
	if(x!=z.re || z.im != 0) return true;
	else return false;
}
inline __device__ bool operator==(const complex_cuda &z, const int &n){
	if(n==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const complex_cuda &z, const int &n){
	if(n!=z.re || z.im != 0) return true;
	else return false;
}
inline __device__ bool operator==(const int &n, const complex_cuda &z){
	if(n==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const int &n, const complex_cuda &z){
	if(n!=z.re || z.im != 0) return true;
	else return false;
}
//Complex-Double
//==============
//I copied these from the double-complex_cuda case which is why the z and x appear backwards
inline __device__ complex_cuda operator+(const complex_cuda &z, const double &x){
	return {x+z.re,z.im};
}
inline __device__ complex_cuda operator-(const complex_cuda &z, const double &x){
	//z-x leaves imaginary part of z unchanged
	return {x-z.re,z.im};
}
inline __device__ complex_cuda operator*(const complex_cuda &z, const double &x){
	return {x*z.re,x*z.im};
}
inline __device__ complex_cuda operator/(const complex_cuda &z, const double &x){
	assert(x!=0);
	return {z.re/(x),z.im/(x)};
}

//Double-Complex
//==============
inline __device__ complex_cuda operator+(const double &x, const complex_cuda &z){
	return {x+z.re,z.im};
}
inline __device__ complex_cuda operator-(const double &x, const complex_cuda &z){
	//x-z flips the imaginary sign of z
	return {x-z.re,-z.im};
}
inline __device__ complex_cuda operator*(const double &x, const complex_cuda &z){
	return {x*z.re,x*z.im};
}
inline __device__ complex_cuda operator/(const double &x, const complex_cuda &z){
	//This short way of doing it will require defining complex_cuda/double first
	assert(z!=0);
	return (x*conj(z))/cnormsq(z);
}

//Complex-Int
//==============
//I copied these from the int-complex_cuda case which is why the z and n appear backwards
inline __device__ complex_cuda operator+(const complex_cuda &z, const int &n){
	return {n+z.re,z.im};
}
inline __device__ complex_cuda operator-(const complex_cuda &z, const int &n){
	//z-n leaves imaginary part of z unchanged
	return {n-z.re,z.im};
}
inline __device__ complex_cuda operator*(const complex_cuda &z, const int &n){
	return {n*z.re,n*z.im};
}
inline __device__ complex_cuda operator/(const complex_cuda &z, const int &n){
	assert(n!=0);
	return {z.re/(n),z.im/(n)};
}

//Int-Complex
//==============
inline __device__ complex_cuda operator+(const int &n, const complex_cuda &z){
	return {n+z.re,z.im};
}
inline __device__ complex_cuda operator-(const int &n, const complex_cuda &z){
	//n-z flips the imaginary sign of z
	return {n-z.re,-z.im};
}
inline __device__ complex_cuda operator*(const int &n, const complex_cuda &z){
	return {n*z.re,n*z.im};
}
inline __device__ complex_cuda operator/(const int &n, const complex_cuda &z){
	//This short way of doing it will require defining complex_cuda/int first
	assert(z!=0);
	return (n*conj(z))/cnormsq(z);
}

//Complex-Complex
//==============
inline __device__ complex_cuda operator+(const complex_cuda &w, const complex_cuda &z){
	return {w.re+z.re,w.im+z.im};
}
inline __device__ complex_cuda operator-(const complex_cuda &w, const complex_cuda &z){
	return {w.re-z.re,w.im-z.im};
}
inline __device__ complex_cuda operator*(const complex_cuda &w, const complex_cuda &z){
	return {w.re*z.re-w.im*z.im,w.re*z.im+w.im*z.re};
}
inline __device__ complex_cuda operator/(const complex_cuda &w, const complex_cuda &z){
	//This short way of doing it will require defining complex_cuda/double first
	assert(z!=0);
	return (w*conj(z))/cnormsq(z);
}
//Less routine complex_cuda functions
//#############################
//inline __device__ complex_cuda cexp(complex_cuda z){
//	return exp(z.re)*{cos(z.im),sin(z.im)};
//}
#endif
