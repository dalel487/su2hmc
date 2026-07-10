/**
 * @file par_mpi.h
 *
 * @brief MPI headers
 */
#ifndef	PAR_MPI
#define	PAR_MPI
#include	<coord.h>
#include	<errorcodes.h>
#if (nproc >1)
#include	<mpi.h> 
#endif
#ifdef _OPENMP
#include	<omp.h>
#endif
//#include	<random.h>
#ifdef __cplusplus
#include	<cstdio>
#include	<cstdlib>
#include	<cstring>
#else
#include <stdalign.h>
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
	 * @param[in]	argc		Number of arguments given to the programme
	 * @param[in]	argv		Array of arguments
	 *
	 * @return Zero on success, integer error code otherwise.
	 * @post MPI communicator configured, processor and sublattice topology determined
	 */
	int Par_begin(int argc, char *argv[]);
	/**
	 * @brief Reads and assigns the gauges from file
	 *	
	 *	@param[in]	iread:		Configuration to read in
	 *	@param[in]	beta:			Inverse gauge coupling
	 *	@param[in]   fmu:			Chemical potential
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	ajq:			Diquark source
	 *	@param[in]	c_sw:			Clover coefficient
	 *	@param[out]	u11,u12:		Gauge fields
	 *	@param[out]	u11t,u12t:	Trial fields
	 * 
	 * @return	Zero on success, integer error code otherwise
	 * @post	Contents of gauge fields replaced with read in values
	 */
	int Par_sread(const int iread, const float beta, const float fmu, const float akappa, const Complex_f ajq,\
			const float c_sw, Complex *u11, Complex *u12, Complex *u11t, Complex *u12t);
	/**
	 * @brief	Copies u11 and u12 into arrays without halos which then get written to output
	 *
	 * Modified from an original version of swrite in FORTRAN
	 *	
	 *	@param[in]	itraj:		Trajectory to write
	 *	@param[in]	icheck:		Not currently used but haven't gotten around to removing it
	 *	@param[in]	beta:			Inverse gauge coupling
	 *	@param[in]   fmu:			Chemical potential
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	ajq:			Diquark source
	 *	@param[in]	c_sw:			Clover coefficient
	 *	@param[in]	u11,u12:		Gauge fields
	 * 
	 * @return	Zero on success, integer error code otherwise
	 * @post	Gauge fields saved to file
	 */
	int Par_swrite(const int itraj, const int icheck, const float beta, const float fmu, const float akappa,\
			const float c_sw, const Complex_f ajq,	Complex *u11, Complex *u12);
	//Shortcuts for reductions and broadcasts. These should be inlined
	/**
	 * @brief	Performs a reduction on an integer ival to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param[in,out] ival: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 * @post	Reduced sum stored in @p ival
	 */
	int Par_isum(int *ival);
	/**
	 * @brief	Performs a reduction on a double dval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param[in,out] dval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 * @post	Reduced sum stored in @p dval
	 *
	 */
	int Par_dsum(double *dval);
	/**
	 * @brief	Performs a reduction on a float dval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param[in,out] dval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 * @post	Reduced sum stored in @p dval
	 *
	 */
	int Par_fsum(float *dval);
	/**
	 * @brief	Performs a reduction on a complex float cval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param[in,out] cval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 * @post	Reduced sum stored in @p cval
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 *
	 */
	int Par_csum(Complex_f *cval);
	/**
	 * @brief	Performs a reduction on a complex double zval to get a sum which is
	 * 			then distributed to all ranks.
	 *
	 * @param[in,out] zval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * @return	Zero on success. Integer error code otherwise.
	 * @post	Reduced sum stored in @p zval
	 *
	 */
	int Par_zsum(Complex *zval);
	/**
	 * @brief Broadcasts an integer to the other processes
	 *
	 * @param[in,out]	ival: Integer being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	All ranks not broadcasting have their value of @p ival overwritten
	 */
	int Par_icopy(int *ival);
	/**
	 * @brief Broadcasts a double to the other processes
	 *
	 * @param[in,out]	dval: double being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	All ranks not broadcasting have their value of @p dval overwritten
	 */
	int Par_dcopy(double *dval);
	/**
	 * @brief Broadcasts a float to the other processes
	 *
	 * @param[in,out]	fval: float being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	All ranks not broadcasting have their value of @p fval overwritten
	 */
	int Par_fcopy(float *fval);
	/**
	 * @brief Broadcasts a complex float to the other processes
	 *
	 * @param[in,out]	cval: Complex float being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	All ranks not broadcasting have their value of @p cval overwritten
	 */
	int Par_ccopy(Complex *cval);
	/**
	 * @brief Broadcasts a complex double to the other processes
	 *
	 * @param[in,out]	zval: Complex double being broadcast
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	All ranks not broadcasting have their value of @p zval overwritten
	 */
	int Par_zcopy(Complex *zval);
	//Halo Manipulation
	/**
	 * @brief Calls the functions to send data to both the up and down halos
	 *
	 * @param[in,out]	z:		The data being sent
	 * @param[in]	ncpt:	Number of components being sent
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	Halo terms of @p z updated
	 */
	int ZHalo_swap_all(Complex *z, int ncpt);
	/**
	 * @brief	Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * @param[in,out]	z:			The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 * @param[in]	ncpt: 	Number of components being sent
	 * @param[in]	idir:		The axis being moved along in C Indexing
	 * @param[in]	layer:	Either DOWN (0) or UP (1)
	 *
	 * @return Zero on success, Integer Error code otherwise
	 * @post	Halo terms of @p z updated in direction @p idir and layer @p layer
	 */
	int ZHalo_swap_dir(Complex *z, int ncpt, int idir, int layer);
	/**
	 * @brief Calls the functions to send data to both the up and down halos
	 *
	 * @param[in,out]	c:		The data being sent
	 * @param[in]	ncpt:	Number of components being sent
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	Halo terms of @p c updated
	 */
	int CHalo_swap_all(Complex_f *c, int ncpt);
	/**
	 * @brief	Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * @param[in,out]	c:			The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 * @param[in]	ncpt: 	Number of components being sent
	 * @param[in]	idir:		The axis being moved along in C Indexing
	 * @param[in]	layer:	Either DOWN (0) or UP (1)
	 *
	 * @return Zero on success, Integer Error code otherwise
	 * @post	Halo terms of @p c updated in direction @p idir and layer @p layer
	 */
	int CHalo_swap_dir(Complex_f *c, int ncpt, int idir, int layer);
	/**
	 * @brief Calls the functions to send data to both the up and down halos
	 *
	 * @param[in,out]	d:		The data being sent
	 * @param[in]	ncpt:	Number of components being sent
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	Halo terms of @p d updated
	 */
	int DHalo_swap_all(double *d, int ncpt);
	/**
	 * @brief	Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * @param[in,out]	d:			The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 * @param[in]	ncpt: 	Number of components being sent
	 * @param[in]	idir:		The axis being moved along in C Indexing
	 * @param[in]	layer:	Either DOWN (0) or UP (1)
	 *
	 * @return Zero on success, Integer Error code otherwise
	 * @post	Halo terms of @p d updated in direction @p idir and layer @p layer
	 */
	int DHalo_swap_dir(double *d, int ncpt, int idir, int layer);
	/**
	 * @brief Calls the functions to send data to both the up and down halos
	 *
	 * @param[in,out]	d:		The data being sent
	 * @param[in]	ncpt:	Number of components being sent
	 *
	 * @return	Zero on success, integer error code otherwise
	 * @post	Halo terms of @p d updated
	 */
	int SHalo_swap_all(float *d, int ncpt);
	/**
	 * @brief	Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * @param[in,out]	d:			The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 * @param[in]	ncpt: 	Number of components being sent
	 * @param[in]	idir:		The axis being moved along in C Indexing
	 * @param[in]	layer:	Either DOWN (0) or UP (1)
	 *
	 * @return Zero on success, Integer Error code otherwise
	 * @post	Halo terms of @p d updated in direction @p idir and layer @p layer
	 */
	int SHalo_swap_dir(float *d, int ncpt, int idir, int layer);
	/**
	 *	@brief Exchanges the trial fields.
	 *
	 *	I noticed that this halo exchange was happening
	 *	even though the trial fields hadn't been updated. To get around this
	 *	I'm making a function that does the halo exchange and only calling it after
	 *	the trial fields get updated.
	 *
	 *	@param[in,out] ut		Double precision trial fields
	 *	@param[out] ut_f:	Single precision trial fields
	 *
	 * @return Zero on success, Integer Error code otherwise
	 * @post	Halos of @p ut updated. @p ut_f overwritten with single precision values of @p ut
	 */
	int Trial_Exchange(Complex *ut[2], Complex_f *ut_f[2]);
	//If we have more than two processors on the time axis, there's an extra step in the Polyakov loop calculation
#if(npt>1)
	/**
	 * @brief	Multiplication along the time extent for the polyakov loop
	 *
	 * @param[in,out]	z11,z12	The inputs and the products
	 *
	 * @return Zero on success, integer error code otherwise.
	 * @post	Products stored in @p z11 and @p z12
	 */
	int Par_tmul(Complex_f *z11, Complex_f *z12);
#endif
#ifdef __cplusplus
}
#endif
#endif
