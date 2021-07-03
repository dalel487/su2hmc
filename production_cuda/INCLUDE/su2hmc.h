#ifndef SU2HEAD
#define SU2HEAD
#ifdef __NVCC__
	#include <cuda_complex.hpp>
#else
	#include <complex.h>
#endif
//MKL is powerful, but not guaranteed to be available (especially on AMD systems or future
//ARM Based machines.) BLAS routines should work with other libraries, so we can set a compiler
//flag to sort them out. But the PRNG routines etc. are MKL exclusive
#ifdef	USE_MKL
#include	<mkl.h>
#endif
#include	<sizes.h>

//Definitions:
//###########
//Variables:
//#########
int ibound; 
//Arrays:
//------
//Seems a bit redundant looking
extern const int gamin[4][4];
//We have the four γ Matrices, and in the final index (labelled 4 in C) is γ_5)
extern complex gamval[5][4];

//From common_pseud
complex *Phi, *R1, *X0, *X1, *xi;
//From common_mat
//double dk4m[kvol+halo], dk4p[kvol+halo] __attribute__((aligned(AVX)));
double *dk4m, *dk4p, *pp;
//From common_trial_u11u12
//complex *u11, *u12;
//double pp[kvol+halo][nadj][ndim] __attribute__((aligned(AVX)));

//Values:
//------
//The diquark
extern complex jqq;

//Average # of congrad iter guidance and acceptance
double ancg, ancgh;
extern double fmu, beta, akappa;

//Function Declarations:
//#####################
int Force(double *dSdpi, int iflag, double res1);
int Init(int istart);
int Gauge_force(double *dSdpi);
int Hamilton(double *h, double *s, double res2);
int Congradq(int na, double res, complex *smallPhi, int *itercg);
int Congradp(int na, double res, int *itercg);
int Measure(double *pbp, double *endenf, double *denf, complex *qq, complex *qbqb, double res, int *itercg);
int SU2plaq(double *hg, double *avplaqs, double *avplaqt);
double Polyakov();

inline int Reunitarise();
inline int Z_gather(complex *x, complex *y, int n, unsigned int *table, unsigned int mu);
inline int Fill_Small_Phi(int na, complex *smallPhi);
double Norm_squared(complex *z, int n);
#endif