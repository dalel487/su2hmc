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
#endif
