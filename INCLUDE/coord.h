#ifndef COORD
#define COORD
#ifdef __NVCC__
#include <cuda.h>
#define USE_BLAS
#include <cublas_v2.h>
#endif
#if defined __INTEL_MKL__
#define USE_BLAS
#include <mkl.h>
#elif defined GSL_BLAS
#define USE_BLAS
#include <gsl/gsl_cblas.h>
#elif defined AMD_BLAS
#define USE_BLAS
#include <cblas.h>
#endif
#include <par_mpi.h>
#include <sizes.h>
//Global Variables
//unsigned int id[ndim][kvol], iu[ndim][kvol] __attribute__((aligned(AVX)));
//unsigned int hu[4][halo], hd[4][halo] __attribute__((aligned(AVX)));
#ifdef __NVCC__
__managed__
#endif
extern unsigned int *hu, *hd, *h1u, *h1d, *halosize;;
#if (defined __NVCC__ || defined __cplusplus)
extern "C"
{
#endif
	//Functions
	//========
	int Addrc(unsigned int *iu, unsigned int *id);
	/*
	 * Loads the addresses required during the update
	 */
	int ia(int x,int y,int z, int t);
	int Check_addr(unsigned int *table, int lns, int lnt, int imin, int imax);
	/*  Checks that the addresses are all correct before an update
	 *  Depends on nothing else
	 */
	int Index2lcoord(int index, int *coord);
	/* Converts the index of a point in memory to the equivalent point
	 * in the 4 dimensional array, where the time index is the last
	 * coordinate in the array
	 * Depends on nothing else.
	 */
	int Index2gcoord(int index, int *coord);
	/* Converts the index of a point in memory to the equivalent point
	 * in the 4 dimensional array, where the time index is the last
	 * coordinate in the array
	 * Depends on nothing else.
	 */
	int Coord2lindex(int ix, int iy, int iz, int it);
	/* Converts the coordinates of a point to its relative index in the 
	 * computer memory to the first point in the memory
	 * Depends on nothing else.
	 */
	int Coord2gindex(int ix, int iy, int iz, int it);
	/* Converts the coordinates of a point to its relative index in the 
	 * computer memory to the first point in the memory
	 * Depends on nothing else.
	 */
	int Testlcoord(int cap);
	/* Tests if the coordinate transformation functions are working
	 * Depends on Index2lcoord and Coord2lindex
	 */
	int Testgcoord(int cap);
	/* Tests if the coordinate transformation functions are working
	 * Depends on Index2gcoord and Coordglindex
	 */
#if (defined __NVCC__ || defined __cplusplus)
}
#endif
#endif
