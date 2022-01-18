//Header file containing field declarations
//Were previously in su2hmc.h and par_mpi.h but makes sense to keep them seperate
#ifndef FIELDS
#define FIELDS
#ifdef __NVCC__
#include	<thrust_complex.h>
#else
#include	<complex.h>
#define	Complex_f	float	complex
#define	Complex	complex
#endif
	//Gauges and trial fields 
#ifdef __NVCC__
	__managed__ extern 
#endif 
		Complex *u11, *u12, *u11t, *u12t;
#ifdef __NVCC__
	__managed__ extern 
#endif 
		Complex_f *u11t_f, *u12t_f;

	//From common_pseud
#ifdef __NVCC__
	__managed__ extern 
#endif 
		Complex *Phi, *R1, *X0, *X1;
	//From common_mat
#ifdef __NVCC__
	__managed__ extern 
#endif 
		double *dk4m, *dk4p, *pp;
#ifdef __NVCC__
	__managed__ extern 
#endif 
		float	*dk4m_f, *dk4p_f;
#endif
