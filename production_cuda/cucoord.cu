/**
 *	@file cucoord.cu
 *
 *	@brief	Working with lattice coordinates on the GPU. Mainly neighbouring term access
 *	@author	D. Lawlor
 */
#include <cucoord.h>
namespace Device{
	///@brief Halo sizes. Mu index is slowest moving. Then down spacing, then up spacing.
	#ifdef __HIP_PLATFORM_AMD__
	__device__
	#else
	__constant__ 
	#endif
	unsigned int hsize[ndim][nc]={{kvol,kvol+halox},
		{kvol+2*halox,kvol+2*halox+haloy},
		{kvol+2*halox+2*haloy, kvol+2*halox+2*haloy+haloz},
		{kvol+2*halox+2*haloy+2*haloz, kvol+2*halox+2*haloy+2*haloz+halot}};
	///@brief Lattice extents
	#ifdef __HIP_PLATFORM_AMD__
	__device__
	#else
	__constant__ 
	#endif
	unsigned short kind[ndim]={ksizex,ksizey,ksizez,ksizet};
	///@brief Processor grid sizes
	#ifdef __HIP_PLATFORM_AMD__
	__device__
	#else
	__constant__ 
	#endif
	unsigned short np[ndim]={npx,npy,npz,npt};

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
			const unsigned short tid){
		//A divide and conquer approach. Going from the deepest coordinate
		//to the least deep coordinate, we take the modulo of the index by
		//the length of that axis to get the coordinate, and then divide
		//the index by the length of that coordinate to set up for the
		//next coordinate. This works since int/int gives an int.
		site[0][tid] = index%ksizex; index/=ksizex;
		site[1][tid] = index%ksizey; index/=ksizey;
		site[2][tid] = index%ksizez; index/=ksizez;
		site[3][tid] = index; //No need to divide by nt since were done.
		return;
	}
	/**
	 *	@brief	From  x,y,z,t coordinates, get thea flattened array index
	 *
	 *	@param[in]	coord:	x,y,z,t coordinates
	 *	@param[in]	tid:		Block thread index
	 *
	 * @return	Flattened array index
	 */
	__device__ unsigned int Coord2lindex(short site[ndim][__BSIZE__],const unsigned short tid){
		//Ensure index is in range. Reverse module arithmetic basically
		while(site[0][tid]<0) site[0][tid]+=ksizex; while(site[0][tid]>=ksizex) site[0][tid]-= ksizex;
		while(site[1][tid]<0) site[1][tid]+=ksizey; while(site[1][tid]>=ksizey) site[1][tid]-= ksizey;
		while(site[2][tid]<0) site[2][tid]+=ksizez; while(site[2][tid]>=ksizez) site[2][tid]-= ksizez;
		while(site[3][tid]<0) site[3][tid]+=ksizet; while(site[3][tid]>=ksizet) site[3][tid]-= ksizet;

		return site[0][tid]+ksizex*(site[1][tid]+ksizey*(site[2][tid]+ksizez*site[3][tid]));
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
	__device__ unsigned int HaloIndex(const unsigned int i, const unsigned short mu, const unsigned short dir,
			const unsigned short tid){
			//Buffer to get x,y,z,t index. Shared to prevent it from being stored in RAM
		__shared__ short site[ndim][__BSIZE__];
		Index2lcoord(i,site,tid);
		if(dir==UP){
			//If we're not at the edge, just increment. Same if only one rank in that direction
			if(np[mu]==1 || site[mu][tid]<kind[mu] ){
				site[mu][tid]++;
				return Coord2lindex(site,tid);
			}
			//Otherwise, have some fun with indexing
			//TODO: This causes occupancy issues. Disable for nproc==1
#if(nproc>1)
			else{
				if(mu==0)
					return hsize[mu][dir]+(site[1][tid]+ksizey*(site[2][tid]+ksizez*site[3][tid]));
				else if(mu==1)
					return hsize[mu][dir]+(site[0][tid]+ksizex*(site[2][tid]+ksizez*site[3][tid]));
				else if(mu==2)
					return hsize[mu][dir]+(site[0][tid]+ksizex*(site[1][tid]+ksizey*site[3][tid]));
				else if(mu==3)
					return hsize[mu][dir]+(site[0][tid]+ksizex*(site[1][tid]+ksizey*site[2][tid]));
			}
#endif
		}
		else if(dir==DOWN){
			//If we're not at the edge, just increment. Same if only one rank in that direction
			if(np[mu]==1 || site[mu][tid]>0){
				site[mu][tid]--;
				return Coord2lindex(site,tid);
			}
			//Otherwise, have some fun with indexing
			//TODO: This causes occupancy issues. Disable for nproc==1
#if(nproc>1)
			else{
				if(mu==0)
					return hsize[mu][dir]+(site[1][tid]+ksizey*(site[2][tid]+ksizez*site[3][tid]));
				else if(mu==1)
					return hsize[mu][dir]+(site[0][tid]+ksizex*(site[2][tid]+ksizez*site[3][tid]));
				else if(mu==2)
					return hsize[mu][dir]+(site[0][tid]+ksizex*(site[1][tid]+ksizey*site[3][tid]));
				else if(mu==3)
					return hsize[mu][dir]+(site[0][tid]+ksizex*(site[1][tid]+ksizey*site[2][tid]));
			}
#endif
		}
	}
}
