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
		__constant__ unsigned int hsize[ndim][nc]={{kvol,kvol+halox},
			{kvol+2*halox,kvol+2*halox+haloy},
			{kvol+2*halox+2*haloy, kvol+2*halox+2*haloy+haloz},
			{kvol+2*halox+2*haloy+2*haloz, kvol+2*halox+2*haloy+2*haloz+halot}};
		///@brief Lattice extents
		__constant__ unsigned short kind[ndim]={ksizex,ksizey,ksizez,ksizet};
		///@brief Processor grid sizes
		__constant__ unsigned short np[ndim]={npx,npy,npz,npt};

	//Don't use const here. index is passed by value
	/**
	 *	@brief	From a flattened array index, get the x,y,z,t coordinates
	 *
	 *	@param[in]	index:	Flattened index
	 *	@param[out]	coord:	x,y,z,t coordinates
	 *
	 *	@post	@p coord updated with x,y,z,t coordinates.
	 */
	__device__ inline void Index2lcoord(unsigned int index, short site[ndim]){
		//A divide and conquer approach. Going from the deepest coordinate
		//to the least deep coordinate, we take the modulo of the index by
		//the length of that axis to get the coordinate, and then divide
		//the index by the length of that coordinate to set up for the
		//next coordinate. This works since int/int gives an int.
		site[0] = index%ksizex; index/=ksizex;
		site[1] = index%ksizey; index/=ksizey;
		site[2] = index%ksizez; index/=ksizez;
		site[3] = index; //No need to divide by nt since were done.
		return;
	}
	/**
	 *	@brief	From  x,y,z,t coordinates, get thea flattened array index
	 *
	 *	@param[in]	coord:	x,y,z,t coordinates
	 *
	 * @return	Flattened array index
	 */
	__device__ unsigned int Coord2lindex(short site[ndim]){
		//Ensure index is in range. Reverse module arithmetic basically
		while(site[0]<0) site[0]+=ksizex; while(site[0]>=ksizex) site[0]-= ksizex;
		while(site[1]<0) site[1]+=ksizey; while(site[1]>=ksizey) site[1]-= ksizey;
		while(site[2]<0) site[2]+=ksizez; while(site[2]>=ksizez) site[2]-= ksizez;
		while(site[3]<0) site[3]+=ksizet; while(site[3]>=ksizet) site[3]-= ksizet;

		return site[0]+ksizex*(site[1]+ksizey*(site[2]+ksizez*site[3]));
	}
	/**
	 *	@brief	Get the index of a neighbouring site on the lattice, including boundary condition
	 *
	 *	@param[in]	i:		Flattened index
	 *	@param[in]	mu:	Direction of neighbour (x,y,z,t)
	 *	@param[in]	dir:	Are we looking for the neighbour in the up or down direction?
	 *
	 *	@return	Flattened index of neighbouring lattice site
	 */
	__device__ unsigned int HaloIndex(const unsigned int i, const unsigned short mu, const unsigned short dir){
		short site[ndim];
		Index2lcoord(i,site);
		switch(dir){
			case(UP):
				//If we're not at the edge, just increment. Same if only one rank in that direction
				if(np[mu]==1 || site[mu]<kind[mu] ){
					site[mu]++;
					return Coord2lindex(site);
				}
				//Otherwise, have some fun with indexing
				//TODO: Check if we've split the lattice in that direction
				else
					switch(dir){
						case(0):
							return hsize[mu][dir]+(site[1]+kind[1]*(site[2]+kind[2]*site[3]));
							break;
						case(1):
							return hsize[mu][dir]+(site[0]+kind[0]*(site[2]+kind[2]*site[3]));
							break;
						case(2):
							return hsize[mu][dir]+(site[0]+kind[0]*(site[1]+kind[1]*site[3]));
							break;
						case(3):
							return hsize[mu][dir]+(site[0]+kind[0]*(site[1]+kind[1]*site[2]));
							break;
					}
				break;
			case(DOWN):
				//If we're not at the edge, just increment. Same if only one rank in that direction
				if(np[mu]==1 || site[mu]>0 ){
					site[mu]--;
					return Coord2lindex(site);
				}
				//Otherwise, have some fun with indexing
				else
					switch(dir){
						case(0):
							return hsize[mu][dir]+(site[1]+kind[1]*(site[2]+kind[2]*site[3]));
							break;
						case(1):
							return hsize[mu][dir]+(site[0]+kind[0]*(site[2]+kind[2]*site[3]));
							break;
						case(2):
							return hsize[mu][dir]+(site[0]+kind[0]*(site[1]+kind[1]*site[3]));
							break;
						case(3):
							return hsize[mu][dir]+(site[0]+kind[0]*(site[1]+kind[1]*site[2]));
							break;
					}
				break;
		}
	}
}
