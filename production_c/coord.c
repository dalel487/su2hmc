/**
 *	@file coord.c
 *	@brief functions for dealing with lattice and MPI coordinates
 */
#include <coord.h>
#include <errorcodes.h>
#ifdef  __OPENMP
#include <omp.h>
#endif
#include <par_mpi.h>
#include <sizes.h>
#include <stdlib.h>
#include <stdio.h>

unsigned int *hu, *hd, *h1u, *h1d, *halosize;
int Addrc(unsigned int *iu, unsigned int *id){
	/*
	 * Loads the addresses required during the update
	 * 
	 * Globals (Only referenced by the CPU):
	 * ======
	 * hu, hd, h1u, h1d, h2u, h2d, halosize
	 * 
	 * Parameters (Used for CPU and GPU):
	 * =========
	 * unsigned int *iu:	Upper halo indices
	 * unsigned int *id:	Lower halo indices
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Addrc";
		//Rather than having 8 ih variables I'm going to use a 2x4 array
		//down is 0, up is 1
		int ih[2][4] = {{-1,-1,-1,-1},{-1,-1,-1,-1}};
		hd = (unsigned int*)aligned_alloc(AVX,ndim*halo*sizeof(int));
		hu = (unsigned int*)aligned_alloc(AVX,ndim*halo*sizeof(int));
		h1u = (unsigned int*)aligned_alloc(AVX,ndim*sizeof(int));
		h1d = (unsigned int*)aligned_alloc(AVX,ndim*sizeof(int));
		halosize= (unsigned int*)aligned_alloc(AVX,ndim*sizeof(int));

		//Do the lookups appropriate for over indexing into halos
		//order is down, up for each x y z t
		// 
		// Since the h2? terms are related to the h1? terms I've dropped them in the C Version
		// saving about 4 billionths of a second in the process
		//
		//Need to watch these +/- 1 at the end. Is that a FORTRAN thing or a program thing?
		//The only time I see h1d called that +1 term gets cancelled by a -1 so I'm going
		//to omit it here at my own peril. (Turned out I was right)
		h1d[0]=kvol; h1u[0]=h1d[0]+halox;
		halosize[0]=halox;

		h1d[1]=h1u[0]+halox; h1u[1]=h1d[1]+haloy;
		halosize[1]=haloy;

		h1d[2]=h1u[1]+haloy; h1u[2]=h1d[2]+haloz;
		halosize[2]=haloz;

		h1d[3]=h1u[2]+haloz; h1u[3]=h1d[3]+halot;
		halosize[3]=halot;

		//Time for the nitty-gritty
		/*
		 * Variables are:
		 *
		 * h1d(mu) = starting  point in tail of down halo in direction mu
		 * h2d(mu) = finishing point in tail of down halo in direction mu
		 *
		 * h1u(mu) = starting  point in tail of up   halo in direction mu
		 * h2u(mu) = finishing point in tail of up   halo in direction mu
		 *
		 * hd(i,mu) = index in core of point that should be packed into the
		 *            ith location of the down halo in direction mu
		 *
		 * hu(i,mu) = index in core of point that should be packed into the
		 *            ith location of the up   halo in direction mu
		 *
		 * Note that hd and hu should be used for PACKING before SENDING
		 *
		 * Unpacking would be done with a loop over ALL the core sites with
		 * reference to normal dn/up lookups, ie we DO NOT have a list of
		 * where in the halo the core point i should go
		 *
		 * Halo points are ordered "as they come" in the linear loop over
		 * core sites
		 */	
		int iaddr, ic;
		//if using ic++ inside the loop instead
		//	ic=-1;
#ifdef _DEBUG
		printf("ksizex = %i, ksizet=%i\n", ksizex, ksizet);
#endif
		//The loop order here matters as it affects the value of ic corresponding to each entry
		for(int jt=0;jt<ksizet;jt++)
			for(int jz=0;jz<ksizez;jz++)
				for(int jy=0;jy<ksizey;jy++)
					for(int jx=0;jx<ksizex;jx++){
						//First value of ic is zero as planned.
						//ic++;
						ic=((jt*ksizez+jz)*ksizey+jy)*ksizex+jx;
						//jx!=0 is logically equivalent to if(jx)
						//If we're inside the sublattice, take the down nearest neightbour from inside the sublattice
						if(jx)
							iaddr = ia(jx-1,jy,jz,jt);
						//Else if we're at the "down" edge, the down nearest neighbour is in the halo
						else{
							ih[0][0]++;
#if npx>1
							if(ih[0][0]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."\
										"\nExiting...\n\n", HALOLIM, funcname, 0, 0, ih[0][0], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hd[0+ndim*ih[0][0]]=ic;
							iaddr=h1d[0]+ih[0][0];
#elif npx==1
							iaddr = ia(jx-1,jy,jz,jt);
#endif
						}
						id[0+ndim*ic]=iaddr;

						if(jx<ksize-1)
							iaddr = ia(jx+1,jy,jz,jt);
						else{
							ih[1][0]++;
#if npx>1
							if(ih[1][0]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."
										"\nExiting...\n\n", HALOLIM, funcname, 1, 0, ih[1][0], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hu[0+ndim*ih[1][0]]=ic;
							iaddr=ih[1][0]+h1u[0];	
#elif npx==1
							iaddr = ia(jx+1,jy,jz,jt);
#endif
						}
						iu[0+ndim*ic]=iaddr;

						if(jy)
							iaddr = ia(jx,jy-1,jz,jt);
						else{
							ih[0][1]++;
#if npy>1
							if(ih[0][1]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."\
										"\nExiting...\n\n", HALOLIM, funcname, 0, 1, ih[0][1], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hd[1+ndim*ih[0][1]]=ic;
							iaddr=h1d[1]+ih[0][1];
#elif npy==1
							iaddr = ia(jx,jy-1,jz,jt);
#endif
						}
						id[1+ndim*ic]=iaddr;

						if(jy<ksize-1)
							iaddr = ia(jx,jy+1,jz,jt);
						else{
							ih[1][1]++;
#if npy>1
							if(ih[1][1]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."
										"\nExiting...\n\n", HALOLIM, funcname, 1, 1, ih[1][1], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hu[1+ndim*ih[1][1]]=ic;
							iaddr=ih[1][1]+h1u[1];	
#elif npy==1
							iaddr = ia(jx,jy+1,jz,jt);
#endif
						}
						iu[1+ndim*ic]=iaddr;

						if(jz)
							iaddr = ia(jx,jy,jz-1,jt);
						else{
							ih[0][2]++;
#if npz>1
							if(ih[0][2]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."\
										"\nExiting...\n\n", HALOLIM, funcname, 0, 2, ih[0][2], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hd[2+ndim*ih[0][2]]=ic;
							iaddr=h1d[2]+ih[0][2];
#elif npz==1
							iaddr = ia(jx,jy,jz-1,jt);
#endif
						}
						id[2+ndim*ic]=iaddr;

						if(jz<ksize-1)
							iaddr = ia(jx,jy,jz+1,jt);
						else{
							ih[1][2]++;
#if npz>1
							if(ih[1][2]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."
										"\nExiting...\n\n", HALOLIM, funcname, 1, 2, ih[1][2], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hu[2+ndim*ih[1][2]]=ic;
							iaddr=ih[1][2]+h1u[2];	
#elif npz==1
							iaddr = ia(jx,jy,jz+1,jt);
#endif
						}
						iu[2+ndim*ic]=iaddr;

						if(jt)
							iaddr = ia(jx,jy,jz,jt-1);
						else{
							ih[0][3]++;
#if npt>1
							if(ih[0][3]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."\
										"\nExiting...\n\n", HALOLIM, funcname, 0, 3, ih[0][3], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hd[3+ndim*ih[0][3]]=ic;
							iaddr=h1d[3]+ih[0][3];
#elif npt==1
							iaddr = ia(jx,jy,jz,jt-1);
#endif
						}
						id[3+ndim*ic]=iaddr;

						if(jt<ksizet-1)
							iaddr = ia(jx,jy,jz,jt+1);
						else{
							ih[1][3]++;
#if npt>1
							if(ih[1][3]>= halo){
								fprintf(stderr, "Error %i in %s: Index ih[%i][%i]=%i is larger than the halo size %i."
										"\nExiting...\n\n", HALOLIM, funcname, 1, 3, ih[1][3], halo);
#if(nproc>1)
								MPI_Abort(comm,HALOLIM);
#else
								exit(HALOLIM);
#endif
							}
							hu[3+ndim*ih[1][3]]=ic;
							iaddr=ih[1][3]+h1u[3];	
#elif npt==1
							iaddr = ia(jx,jy,jz,jt+1);
#endif
						}
						iu[3+ndim*ic]=iaddr;
					}
		//Print iu and id for diagnostics
#ifdef _DEBUG
#pragma omp parallel sections
		{
#pragma omp section
			{
				FILE *id_out = fopen("id_out", "w");
				for(int i=0;i<kvol;i++)
					fprintf(id_out,"%i\t%i\t%i\t%i\n",id[i*ndim],id[i*ndim+1],id[i*ndim+2],id[i*ndim+3]);
				fclose(id_out);
			}
#pragma omp section
			{
				FILE *iu_out = fopen("iu_out", "w");
				for(int i=0;i<kvol;i++)
					fprintf(iu_out,"%i\t%i\t%i\t%i\n",iu[i*ndim],iu[i*ndim+1],iu[i*ndim+2],iu[i*ndim+3]);
				fclose(iu_out);

			}

		}
#endif
		return 0;
}
//No point making this parallel because Addrc is serial and the only thing that calls ia
inline int ia(int x, int y, int z, int t){
	/*
	 * Described as a 21st Century address calculator, it gets the memory
	 * address of an array entry.
	 *
	 * Parameters:
	 * ==========
	 * int x, y, z, t. The coordinates
	 *
	 * Returns:
	 * =======
	 * An integer corresponding to the position of the entry in a flattened
	 * row-major array
	 *
	 * Future... Switch for Row and column major, and zero or one indexing
	 */
	const char *funcname = "ia";
	//We need to ensure that the indices aren't out of bounds using while loops
	while(x<0) x+=ksizex; while(x>=ksizex) x-= ksizex;
	while(y<0) y+=ksizey; while(y>=ksizey) y-= ksizey;
	while(z<0) z+=ksizez; while(z>=ksizez) z-= ksizez;
	while(t<0) t+=ksizet; while(t>=ksizet) t-= ksizet;

	//And flattening.
	//return t+ksizet*(z+ksizez*(y+ksizey*x));
	return ((t*ksizez+z)*ksizey+y)*ksizex+x;
}
int Check_addr(unsigned int *table, int lns, int lnt, int imin, int imax){
	/* Checks that the addresses are within bounds before an update
	 *
	 * Parameters:
	 * ==========
	 * int *table:	Pointer to the table in question
	 * int lns:	Size of each spacial dimension
	 * int lnt:	Size of the time dimension
	 * int imin:	Lower bound for element of the table
	 * int imax:	Upper bound for an element of the table
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise.
	 */
	const char *funcname = "Check_addr";
	//Get the total number of elements in each dimension of the table
	int ntable = lns*lns*lns*lnt;
	int iaddr;
	//Collapsing two for loops together
	for(int j=0; j<ntable*ndim; j++){
		iaddr = table[j];
		if((iaddr<imin) || (iaddr>= imax)){
			fprintf(stderr, "Error %i in %s: %i is out of the bounds of (%i,%i)\n"\
					"for a table of size %i^3 *%i.\nExiting...\n\n",\
					BOUNDERROR,funcname,iaddr,imin,imax,lns,lnt);
#if(nproc>1)
			MPI_Abort(comm,BOUNDERROR);
#else
			exit(BOUNDERROR);
#endif
		}
	}
	return 0;
}
inline int Index2lcoord(int index, int *coord){
	/* Converts the index of a point in memory to the equivalent point
	 * in the 4 dimensional array, where the time index is the last
	 * coordinate in the array
	 *
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 *
	 * Parameters:
	 * ==========
	 * int index:	The index of the point as stored linearly in computer
	 *			memory
	 * int *coord:	The 4-array for the coordinates. The first three spots
	 *			are for the time index.
	 *
	 * Returns:
	 * ========
	 * Zero on success. Integer Error code otherwise
	 */ 

	const char *funcname = "Index2lcoord";
	//A divide and conquer approach. Going from the deepest coordinate
	//to the least deep coordinate, we take the modulo of the index by
	//the length of that axis to get the coordinate, and then divide
	//the index by the length of that coordinate to set up for the
	//next coordinate. This works since int/int gives an int.
	coord[3] = index%ksizet; index/=ksizet;
	coord[2] = index%ksizez; index/=ksizez;
	coord[1] = index%ksizey; index/=ksizey;
	coord[0] = index; //No need to divide by nt since were done.

	return 0;
}
inline int Index2gcoord(int index, int *coord){
	/* Converts the index of a point in memory to the equivalent point
	 * in the 4 dimensional array, where the time index is the last
	 * coordinate in the array
	 *
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 *
	 * Parameters:
	 * ==========
	 * int index:	The index of the point as stored linearly in computer
	 *			memory
	 * int *coord:	The 4-array for the coordinates. The first three spots
	 *			are for the time index.
	 *
	 * Returns:
	 * ========
	 * Zero on success. Integer Error code otherwise
	 */ 

	const char *funcname = "Index2gcoord";
	//A divide and conquer approach. Going from the deepest coordinate
	//to the least deep coordinate, we take the modulo of the index by
	//the length of that axis to get the coordinate, and then divide
	//the index by the length of that coordinate to set up for the
	//next coordinate. This works since int/int gives an int.
	coord[3] = index%nt; index/=nt;
	coord[2] = index%nz; index/=nz;
	coord[1] = index%ny; index/=ny;
	coord[0] = index; //No need to divide by nt since were done.

	return 0;
}
inline int Coord2lindex(int ix, int iy, int iz, int it){
	/* Converts the coordinates of a local lattice point to its index in the 
	 * computer memory
	 *
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 * Parameters:
	 * ==========
	 * int i?: The coordinate being converted 
	 *
	 * Returns:
	 * ========
	 * int index: The position of the point
	 */
	const char *funcname = "Coord2gindex";

	//I've factorised this function compared to its original 
	//implementation to reduce the number of multiplications
	//and hopefully improve performance
	//int index = coord[3]+ksizez*(coord[2]+ksizey*(coord[1]+ksizex*coord[0]));
	return it+ksizet*(iz+ksizez*(iy+ksizey*ix));
}
inline int Coord2gindex(int ix, int iy, int iz, int it){
	/* Converts the coordinates of a point in the global gauge field 
	 * to its flattened index in the computer memory
	 * 
	 * This is a rather nuanced function, as C and Fortran are rather
	 * different in how they store arrays. C starts with index 0 and
	 * Fortran (by default) starts with index 1
	 *
	 * Also C and Fortran store data in the opposite memory order so
	 * be careful when calling this function!
	 * Parameters:
	 * ==========
	 * int *coord: The pointer to the 4-vector being considered
	 *
	 * Returns:
	 * ========
	 * int index: The position of the point
	 */
	const char *funcname = "Coord2gindex";

	//I've factorised this function compared to its original 
	//implementation to reduce the number of multiplications
	//and hopefully improve performance
	//	return it+nt*(iz+nz*(iy+ny*ix));
	return ix+nx*(iy+ny*(iz+nz*it));
}
int Testlcoord(int cap){
	/* Tests if the coordinate transformation functions are working
	 * Going to expand a little on the original here and do the following
	 * 1. Convert from int to lcoord (the original code)
	 * And the planned additional features
	 * 2. Convert from lcoord to int (new, function doesn't exist in the original
	 * If we get the same value we started with then we're probably doing
	 * something right.
	 *
	 * Parameters:
	 * ===========
	 * int cap: The max value the index can take on. Should be the size of the array
	 *
	 * Calls:
	 * =====
	 * Index2lcoord, Coord2lindex
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise.
	 */
	const char *funcname = "Testlcoord";
	//The storage array for the coordinates, and the index and its test value.
	int coord[4], index, index2;
	for(index =0; index<cap; index++){
		Index2lcoord(index, coord);
		printf("Coordinates for %i are (x,y,z,t):[%i,%i,%i,%i].\n", index,\
				coord[0], coord[1], coord[2], coord[3]);
		//index2 = Coord2lindex(coord);
		if(!(index==index2)){
			fprintf(stderr, "Error %i in %s: Converted index %i does not match "
					"original index %i.\nExiting...\n\n",\
					INDTOCOORD, funcname, index2, index);
#if(nproc>1)
			MPI_Abort(comm,INDTOCOORD);
#else
			exit(INDTOCOORD);
#endif
		}
	}
	return 0;
}
int Testgcoord(int cap){
	/* This is completely new and missing from the original code.
	 * We test the coordinate conversion functions by doing the following
	 * 1. Convert from int to gcoord (new)
	 * 2. Convert from gcoord to int (also new) and compare to input.
	 * If we get the same value we started with then we're probably doing
	 * something right
	 *
	 * The code is basically the same as the previous function with different
	 * magic numbers.
	 *
	 * Parameters:
	 * ===========
	 * int cap: The max value the index can take on. Should be the size of our array
	 *
	 * Calls:
	 * ======
	 * Index2gcoord, Coord2gindex
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise 
	 */
	const char *funcname = "Testgcoord";
	int coord[4], index, index2;
#pragma omp parallel for private(coord, index, index2)
	for(index=0; index<cap; index++){
		Index2gcoord(index, coord);
#pragma omp critical
		printf("Coordinates for %i are (x,y,z,t):[%i,%i,%i,%i].\n", index,\
				coord[0], coord[1], coord[2], coord[3]);
		//index2 = Coord2gindex(coord);
		if(!(index==index2)){
			fprintf(stderr, "Error %i in %s: Converted index %i does not match "\
					"original index %i.\nExiting...\n\n",\
					INDTOCOORD, funcname, index2, index);
#if(nproc>1)
			MPI_Abort(comm,INDTOCOORD);
#else
			exit(INDTOCOORD);
#endif
		}
	}
	return 0;
}
