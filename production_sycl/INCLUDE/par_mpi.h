/**
 * @file par_mpi.h
 *
 * @brief MPI headers
 */
#ifndef	PAR_MPI
#define	PAR_MPI
#include <coord.h>
#include	<errorcodes.h>
#if (nproc >1)
#include	<mpi.h> 
#endif
#ifdef _OPENMP
#include	<omp.h>
#endif
//#include	<random.h>
#include	<sizes.h>
#ifdef __cplusplus
#include	<cstdio>
#include	<cstdlib>
#include	<cstring>
#else
#include	<stdbool.h>
#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#endif

/// @brief Avoid any accidents with US/UK spelling
#define MPI_Finalise() MPI_Finalize()

//Definitions
//==========
/// @brief Flag for send down
#define	DOWN	0
/// @brief Flag for send up
#define	UP		1

/// @brief The main rank. Used for serial tasks
#define masterproc 0

///@brief default MPI tag
#define tag   0
//#define _STAT_SIZE_  sizeof(MPI_Status)
//Variables
//=========
//Up/Down arrays
/// @brief Processors in the up direction
extern int __attribute__((aligned(AVX))) pu[ndim];
/// @brief Processors in the down direction
extern int __attribute__((aligned(AVX))) pd[ndim];

//MPI Stuff
#if (nproc >1)
/// @brief MPI communicator
extern MPI_Comm comm ;
/// @brief MPI request. Required for send/receive
extern MPI_Request request;
#endif

/// @brief The processor grid
extern int *pcoord;
/// @brief The initial lattice site on each sublattice in a given direction
extern int  __attribute__((aligned(AVX))) pstart[ndim][nproc];
/// @brief The final lattice site on each sublattice in a given direction
extern int  __attribute__((aligned(AVX))) pstop[ndim][nproc];
///	@brief The MPI rank
extern int rank;
///	@brief The number of MPI ranks in total
extern int size;
//The common keyword from fortran is largely redundant here as everything
//is already global scope.

/*common /par/ pu, pd, procid, comm,
  1             gsize, lsize, pcoord, pstart, pstop,
  1             ismaster, masterproc
 */	

#ifdef __cplusplus
extern "C"
{
#endif
	//Function Declarations
	//=====================
	/**
	 * @brief Initialises the MPI configuration
	 *
	 * @param	argc		Number of arguments given to the programme
	 * @param	argv		Array of arguments
	 *
	 * @return Zero on success, integer error code otherwise.
	 */
	int Par_begin(int argc, char *argv[]);
	/**
	 * @brief Reads and assigns the gauges from file
	 *	
	 *	@param	iread:		Configuration to read in
	 *	@param	beta:			Inverse gauge coupling
	 *	@param   fmu:			Chemical potential
	 *	@param	akappa:		Hopping parameter
	 *	@param	ajq:			Diquark source
	 *	@param	u11,u12:		Gauge fields
	 *	@param	u11t,u12t:	Trial fields
	 * 
	 * @return	Zero on success, integer error code otherwise
	 */
	int Par_sread(const int iread, const float beta, const float fmu, const float akappa, const Complex_f ajq,
			Complex *u11, Complex *u12, Complex *u11t, Complex *u12t);
	/**
	 * @brief	Copies u11 and u12 into arrays without halos which then get written to output
	 *
	 * Modified from an original version of swrite in FORTRAN
	 *	
	 *	@param	itraj:		Trajectory to write
	 *	@param	icheck:		Not currently used but haven't gotten around to removing it
	 *	@param	beta:			Inverse gauge coupling
	 *	@param   fmu:			Chemical potential
	 *	@param	akappa:		Hopping parameter
	 *	@param	ajq:			Diquark source
	 *	@param	u11,u12:		Gauge fields
	 * 
	 * @return	Zero on success, integer error code otherwise
	 */
	int Par_swrite(const int itraj, const int icheck, const float beta, const float fmu, const float akappa,
			const Complex_f ajq,	Complex *u11, Complex *u12);
	//Shortcuts for reductions and broadcasts. These should be inlined
	/**
	 * @brief	Performs a reduction on an integer ival to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param ival: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 *
	 */
	int Par_isum(int *ival);
	/**
	 * @brief	Performs a reduction on a double dval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param dval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 *
	 */
	int Par_dsum(double *dval);
	/**
	 * @brief	Performs a reduction on a float dval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param dval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 *
	 */
	int Par_fsum(float *dval);
	/**
	 * @brief	Performs a reduction on a complex float cval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param cval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 *
	 */
	int Par_csum(Complex_f *cval);
	/**
	 * @brief	Performs a reduction on a complex double zval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param zval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 *
	 */
	int Par_zsum(Complex *zval);
	/**
	 * @brief Broadcasts an integer to the other processes
	 *
	 * @param	ival: Integer being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int Par_icopy(int *ival);
	/**
	 * @brief Broadcasts a double to the other processes
	 *
	 * @param	dval: double being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int Par_dcopy(double *dval);
	/**
	 * @brief Broadcasts a float to the other processes
	 *
	 * @param	fval: float being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int Par_fcopy(float *fval);
	/**
	 * @brief Broadcasts a complex float to the other processes
	 *
	 * @param	cval: Complex float being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int Par_ccopy(Complex *cval);
	/**
	 * @brief Broadcasts a complex double to the other processes
	 *
	 * @param	zval: Complex double being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int Par_zcopy(Complex *zval);
	//Halo Manipulation
	/**
	 * @brief Calls the functions to send data to both the up and down halos
	 *
	 * @param	z:		The data being sent
	 * @param	ncpt:	Number of components being sent
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int ZHalo_swap_all(Complex *z, int ncpt);
	/**
	 * @brief	Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 *  @param	z:			The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 *  @param	ncpt: 	Number of components being sent
	 *  @param	idir:		The axis being moved along in C Indexing
	 *  @param	layer:	Either DOWN (0) or UP (1)
	 *
	 *  @return Zero on success, Integer Error code otherwise
	 */
	int ZHalo_swap_dir(Complex *z, int ncpt, int idir, int layer);
	/**
	 * @brief Calls the functions to send data to both the up and down halos
	 *
	 * @param	c:		The data being sent
	 * @param	ncpt:	Number of components being sent
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int CHalo_swap_all(Complex_f *c, int ncpt);
	/**
	 * @brief	Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 *  @param	c:			The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 *  @param	ncpt: 	Number of components being sent
	 *  @param	idir:		The axis being moved along in C Indexing
	 *  @param	layer:	Either DOWN (0) or UP (1)
	 *
	 *  @return Zero on success, Integer Error code otherwise
	 */
	int CHalo_swap_dir(Complex_f *c, int ncpt, int idir, int layer);
	/**
	 * @brief Calls the functions to send data to both the up and down halos
	 *
	 * @param	d:		The data being sent
	 * @param	ncpt:	Number of components being sent
	 *
	 * @return	Zero on success, integer error code otherwise
	 */
	int DHalo_swap_all(double *d, int ncpt);
	/**
	 * @brief	Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 *  @param	d:			The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 *  @param	ncpt: 	Number of components being sent
	 *  @param	idir:		The axis being moved along in C Indexing
	 *  @param	layer:	Either DOWN (0) or UP (1)
	 *
	 *  @return Zero on success, Integer Error code otherwise
	 */
	int DHalo_swap_dir(double *d, int ncpt, int idir, int layer);
	/**
	 *	@brief Exchanges the trial fields.
	 *
	 *	I noticed that this halo exchange was happening
	 *	even though the trial fields hadn't been updated. To get around this
	 *	I'm making a function that does the halo exchange and only calling it after
	 *	the trial fields get updated.
	 *
	 *	@param u11t,u12t			Double precision trial fields
	 *	@param u11t_f,u12t_f		Single precision trial fields
	 *
	 *  @return Zero on success, Integer Error code otherwise
	 */
	int Trial_Exchange(Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f);
	//If we have more than two processors on the time axis, there's an extra step in the Polyakov loop calculation
#if(npt>1)
	/**
	 * @brief	Multiplication along the time extent for the polyakov loop
	 *
	 * @param	z11,z12	The inputs and the products
	 *
	 * @return Zero on success, integer error code otherwise.
	 */
	int Par_tmul(Complex_f *z11, Complex_f *z12);
#endif
#ifdef __cplusplus
}
#endif
#endif
