/**
 *	@file cucoord.h
 *
 *	@brief	Working with lattice coordinates on the GPU. Mainly neighbouring term access
 *	@author	D. Lawlor
 */
#pragma once
#include <par_mpi.h>
namespace Device{
	///@brief Halo sizes. Mu index is slowest moving. Then down spacing, then up spacing.
	extern
	#ifdef __HIP_PLATFORM_AMD__
	__device__
	#else
	__constant__ 
	#endif
	unsigned int hsize[ndim][nc];
	///@brief Lattice extents
	extern
	#ifdef __HIP_PLATFORM_AMD__
	__device__
	#else
	__constant__ 
	#endif
	unsigned short kind[ndim];
	///@brief Processor grid sizes
	extern
	#ifdef __HIP_PLATFORM_AMD__
	__device__
	#else
	__constant__ 
	#endif
	unsigned short np[ndim];

	//Don't use const here. index is passed by value
	/**
	 *	@brief	From a flattened array index, get the x,y,z,t coordinates
	 *
	 *	@param[in]	index:	Flattened index
	 *	@param[out]	coord:	x,y,z,t coordinates
	 *	@param[in]	tid:		Block thread index
	 *
	 *	@post	@p coord updated with x,y,z,t coordinates.
	 */
	__device__ inline void Index2lcoord(unsigned int index, short site[ndim][__BSIZE__],
			const unsigned short tid);
	/**
	 *	@brief	From  x,y,z,t coordinates, get thea flattened array index
	 *
	 *	@param[in]	coord:	x,y,z,t coordinates
	 *	@param[in]	tid:		Block thread index
	 *
	 * @return	Flattened array index
	 */
	__device__ unsigned int Coord2lindex(short site[ndim][__BSIZE__],const unsigned short tid);
	/**
	 *	@brief	Get the index of a neighbouring site on the lattice, including boundary condition
	 *
	 *	@param[in]	i:		Flattened index
	 *	@param[in]	mu:	Direction of neighbour (x,y,z,t)
	 *	@param[in]	dir:	Are we looking for the neighbour in the up or down direction?
	 *
	 *	@return	Flattened index of neighbouring lattice site
	 */
	__device__ unsigned int HaloIndex(const unsigned int i, const unsigned short mu, const unsigned short dir,
			const unsigned short tid);
}
namespace Kernels{
	extern
	#ifdef __HIP_PLATFORM_AMD__
	__device__ const
	#else
	__constant__ 
	#endif
	unsigned short gamin[16];
}
