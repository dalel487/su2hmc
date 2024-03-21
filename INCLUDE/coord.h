/**
 *	@file coord.h
 *	@brief Header for routines related to lattice sites
 */
#ifndef COORD
#define COORD
#ifdef __CUDACC__
#include <cuda.h>
#define USE_BLAS
#include <cublas_v2.h>
#endif
#include <math.h>
#if (defined__INTEL_COMPILER || __INTEL_LLVM_COMPILER)
#include <mathimf.h>
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
#include <sizes.h>
#ifdef __CUDACC__
__managed__
#endif
extern unsigned int *hu, *hd, *h1u, *h1d, *halosize;;
#ifdef __cplusplus
extern "C"
{
#endif
	//Functions
	//========
	/**
	 * @brief	Loads the addresses required during the update
	 * 
	 * @param	iu:	Upper halo indices
	 * @param	id:	Lower halo indices
	 *
	 * @see hu, hd, h1u, h1d, h2u, h2d, halosize
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Addrc(unsigned int *iu, unsigned int *id);
	/**
	 * @brief Described as a 21st Century address calculator, it gets the memory
	 * address of an array entry.
	 *
	 * @param x, y, z, t. The coordinates
	 *
	 * @return An integer corresponding to the position of the entry in a flattened
	 * row-major array.
	 *
	 * @todo	Future... Switch for Row and column major, and zero or one indexing
	 */
	int ia(int x,int y,int z, int t);
	/** Checks that the addresses are within bounds before an update
	 *
	 * @param	table:	Pointer to the table in question
	 * @param	lns:		Size of each spacial dimension
	 * @param	lnt:		Size of the time dimension
	 * @param	imin:		Lower bound for element of the table
	 * @param	imax:		Upper bound for an element of the table
	 *
	 * @return	Zero on success, integer error code otherwise.
	 */
	int Check_addr(unsigned int *table, int lns, int lnt, int imin, int imax);
	/**
	 *@brief Converts the index of a point in memory to the equivalent point
	 * in the 4 dimensional array, where the time index is the last
	 * coordinate in the array.
	 *
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 *
	 * @param	index:	The index of the point as stored linearly in computer memory
	 * @param	coord:	The 4-array for the coordinates. The first three spots are for the time index.
	 *
	 * @return Zero on success. Integer Error code otherwise
	 */ 
	int Index2lcoord(int index, int *coord);
	/**
	 * @brief Converts the index of a point in memory to the equivalent point
	 * in the 4 dimensional array, where the time index is the last
	 * coordinate in the array.
	 *
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 *
	 * @param	index:	The index of the point as stored linearly in computer memory
	 * @param 	coord:	The 4-array for the coordinates. The first three spots are for the time index.
	 *
	 * @return	Zero on success. Integer Error code otherwise
	 */ 
	int Index2gcoord(int index, int *coord);
	/**
	 * @brief Converts the coordinates of a local lattice point to its index in the 
	 * computer memory.
	 *
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 *
	 * @param index: The flattend index of the coordinate
	 * @param coord: The coordinate 
	 *
	 * Returns:
	 * ========
	 * int index: The position of the point
	 */
	int Coord2lindex(int ix, int iy, int iz, int it);
	/**
	 * @brief Converts the coordinates of a global lattice point to its index in the 
	 * computer memory.
	 *
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 *
	 * @param index: The flattend index of the coordinate
	 * @param coord: The coordinate 
	 *
	 * Returns:
	 * ========
	 * int index: The position of the point
	 */
	int Coord2gindex(int ix, int iy, int iz, int it);
	/**
	 * @brief Tests if the local coordinate transformation functions are working
	 * 
	 * Going to expand a little on the original here and do the following
	 * 1. Convert from int to lcoord (the original code)
	 * And the planned additional features
	 * 2. Convert from lcoord to int (new, function doesn't exist in the original
	 * If we get the same value we started with then we're probably doing
	 * something right.
	 *
	 * @param cap: The max value the index can take on. Should be the size of the array
	 *
	 * @return Zero on success, integer error code otherwise.
	 */
	int Testlcoord(int cap);
	/**
	 * @brief This is completely new and missing from the original code.
	 *
	 * We test the coordinate conversion functions by doing the following
	 * 1. Convert from int to gcoord (new)
	 * 2. Convert from gcoord to int (also new) and compare to input.
	 * If we get the same value we started with then we're probably doing
	 * something right
	 *
	 * The code is basically the same as the previous function with different
	 * magic numbers.
	 *
	 * @param cap: The max value the index can take on. Should be the size of our array
	 *
	 * @return Zero on success, integer error code otherwise 
	 */
	int Testgcoord(int cap);
#ifdef __cplusplus
}
#endif
#endif
