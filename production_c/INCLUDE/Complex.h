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
}complex;
static const complex I   = {0.0,1.0};
//Routine complex operations
//#########################
inline __device__ double creal(complex z){
	return z.re;
}
inline __device__ double cimag(complex z){
	return z.im;
}
inline __device__ complex conj(complex z){
	return {z.re,-z.im};
}
inline __device__ double cnormsq(complex z){
	return z.re*z.re+z.im*z.im;
}
inline __device__ double cabs(complex z){
	return sqrt(cnormsq(z));
}

//Binary operators
//###############
//Equivalence
//===========
inline __device__ bool operator==(const complex &w, const complex &z){
	if(w.re==z.re && w.im == z.im) return true;
	else return false;
}
inline __device__ bool operator!=(const complex &w, const complex &z){
	if(w.re!=z.re || w.im != z.im) return true;
	else return false;
}
inline __device__ bool operator==(const double &x, const complex &z){
	if(x==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const complex &z, const double &x){
	if(x!=z.re || z.im != 0) return true;
	else return false;
}
inline __device__ bool operator==(const complex &z, const double &x){
	if(x==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const double &x, const complex &z){
	if(x!=z.re || z.im != 0) return true;
	else return false;
}
inline __device__ bool operator==(const complex &z, const int &n){
	if(n==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const complex &z, const int &n){
	if(n!=z.re || z.im != 0) return true;
	else return false;
}
inline __device__ bool operator==(const int &n, const complex &z){
	if(n==z.re && z.im == 0) return true;
	else return false;
}
inline __device__ bool operator!=(const int &n, const complex &z){
	if(n!=z.re || z.im != 0) return true;
	else return false;
}
//Complex-Double
//==============
//I copied these from the double-complex case which is why the z and x appear backwards
inline __device__ complex operator+(const complex &z, const double &x){
	return {x+z.re,z.im};
}
inline __device__ complex operator-(const complex &z, const double &x){
	//z-x leaves imaginary part of z unchanged
	return {x-z.re,z.im};
}
inline __device__ complex operator*(const complex &z, const double &x){
	return {x*z.re,x*z.im};
}
inline __device__ complex operator/(const complex &z, const double &x){
	assert(x!=0);
	return {z.re/(x),z.im/(x)};
}

//Double-Complex
//==============
inline __device__ complex operator+(const double &x, const complex &z){
	return {x+z.re,z.im};
}
inline __device__ complex operator-(const double &x, const complex &z){
	//x-z flips the imaginary sign of z
	return {x-z.re,-z.im};
}
inline __device__ complex operator*(const double &x, const complex &z){
	return {x*z.re,x*z.im};
}
inline __device__ complex operator/(const double &x, const complex &z){
	//This short way of doing it will require defining complex/double first
	assert(z!=0);
	return (x*conj(z))/cnormsq(z);
}

//Complex-Int
//==============
//I copied these from the int-complex case which is why the z and n appear backwards
inline __device__ complex operator+(const complex &z, const int &n){
	return {n+z.re,z.im};
}
inline __device__ complex operator-(const complex &z, const int &n){
	//z-n leaves imaginary part of z unchanged
	return {n-z.re,z.im};
}
inline __device__ complex operator*(const complex &z, const int &n){
	return {n*z.re,n*z.im};
}
inline __device__ complex operator/(const complex &z, const int &n){
	assert(n!=0);
	return {z.re/(n),z.im/(n)};
}

//Int-Complex
//==============
inline __device__ complex operator+(const int &n, const complex &z){
	return {n+z.re,z.im};
}
inline __device__ complex operator-(const int &n, const complex &z){
	//n-z flips the imaginary sign of z
	return {n-z.re,-z.im};
}
inline __device__ complex operator*(const int &n, const complex &z){
	return {n*z.re,n*z.im};
}
inline __device__ complex operator/(const int &n, const complex &z){
	//This short way of doing it will require defining complex/int first
	assert(z!=0);
	return (n*conj(z))/cnormsq(z);
}

//Complex-Complex
//==============
inline __device__ complex operator+(const complex &w, const complex &z){
	return {w.re+z.re,w.im+z.im};
}
inline __device__ complex operator-(const complex &w, const complex &z){
	return {w.re-z.re,w.im-z.im};
}
inline __device__ complex operator*(const complex &w, const complex &z){
	return {w.re*z.re-w.im*z.im,w.re*z.im+w.im*z.re};
}
inline __device__ complex operator/(const complex &w, const complex &z){
	//This short way of doing it will require defining complex/double first
	assert(z!=0);
	return (w*conj(z))/cnormsq(z);
}
//Less routine complex functions
//#############################
//inline __device__ complex cexp(complex z){
//	return exp(z.re)*{cos(z.im),sin(z.im)};
//}
#endif
