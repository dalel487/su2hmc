#include <complex.h>
#include "coord.h"
#include "errorcodes.h"
#include <math.h>
#ifdef 	USE_MKL
	#include <mkl.h>
#endif
//#include <mpi.h>
#ifdef __OPEN_MP
	#include <omp.h>
#endif
#include "random.h"
#include "sizes.h"
#include <stdio.h>
#include <stdlib.h>
#include "su2hmc.h"

int main(int argc, char *argv[]){
	/*
	 * This is supposed to be a testbed for the other functions in the programme
	 * I might change it to use #ifdefs as a switch for different tests, or
	 * use command line arguments. But for now I'll just comment stuff out
	 *
	 */
	 char *funcname = "Test_call.c main";

	/*First Test: The index to coordinate functions
	//They were already defined in the FORTRAN code so I'll just call them 
	//here.
	int cap = pow(2,16);
	#pragma omp parallel sections
	{
	#pragma omp section
	{
		Testlcoord(cap);
	}
	#pragma omp section
	{
		Testgcoord(cap);
	}
	}
	*/
	 /*Second test, is the Addrc function functioning?
	  *
	  */
//	Addrc();
	/* Test III: Gaussian distribution
	 *
	 */
	double *R = mkl_malloc(1024*1024*sizeof(double),AVX);
	Gauss_d(R, 1024*1024, 0,1.0);
	FILE *dub_out = fopen("double_out", "w");
	for(int i = 0; i<1024*1024;i++)
		fprintf(dub_out, "%f\n", R[i]);
	fclose(dub_out);
	free(R);
	complex *Rz = mkl_malloc(1024*1024*sizeof(complex),AVX);
	Gauss_z(Rz, 1024*1024, 0,1.0);
	FILE *com_out= fopen("complex_out", "w");
	for(int i = 0; i<1024*1024;i++)
		fprintf(com_out, "%f\t%f\n", creal(Rz[i]), cimag(Rz[i]));
	fclose(dub_out);
	free(Rz);
	return 0;
}
