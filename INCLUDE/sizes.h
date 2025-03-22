/**
 * 	@file sizes.h
 *
 *    @brief Defines the constants of the code and other parameters 
 *    for loop dimensions.
 *    Each subroutine includes these definitions using:
 *    @verbatim INCLUDE sizes.h	@endverbatim
 *
 *    Or at least they used to. But this is the C version, so
 *    no need for that. Include once in each source file and
 *    it should work just fine.
 *    
 *    The only potential snag is that including the header makes
 *    its content global in scope. Where this could cause an issue
 *    we shall instead declare the variable inside the function and
 *    its scope shall override that of the global variable.
 *
 *    I've tried keeping the comments as close to the original
 *    as possible. Of course some magic numbers will be redundant
 *    and some will need to be added or redefined. Anything that
 *    was commented out has been left commented out with a second
 *    C commented out version beneath it.
 *
 *    @author D. Lawlor September 2020
 *
 ******************************************************************/
#ifndef	SIZES
#define	SIZES
#ifdef	__INTEL_MKL__
#define	USE_BLAS
#include	<mkl.h>
#elif defined GSL_BLAS
#define	USE_BLAS
#include <gsl/gsl_cblas.h>
#elif defined AMD_BLAS
#define	USE_BLAS
#include	<cblas.h>
#endif

#if defined __HIPCC__ || defined __NVCC__
#define __GPU__
#endif

#ifdef	__NVCC__
#include	<cuda.h>
#include	<cuda_runtime_api.h>
#include	<cublas_v2.h>
extern cublasHandle_t cublas_handle;
extern cublasStatus_t cublas_status;
extern cudaMemPool_t mempool;
///@brief	Get rid of that bastardised yankee English
#define cudaDeviceSynchronise() cudaDeviceSynchronize()
#elif defined	__HIPCC__
#include	<hip/hip_runtime.h>
#include	<hip/hip_runtime_api.h>
#include	<hipblas.h>
extern hipblasHandle_t cublas_handle;
extern hipblasStatus_t cublas_status;
extern hipMemPool_t mempool;
///@brief	Get rid of that bastardised yankee English
#define cudaDeviceSynchronise() hipDeviceSynchronize()
#endif
#if defined __CUDACC__ || defined __HIP__
#include	<thrust_complex.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#else
#include	<complex.h>
///@brief Single precision complex number 
#define	Complex_f	float	complex
///@brief Double precision complex number 
#define	Complex		double complex
#endif

/// @brief	Default file name length
#define	FILELEN	64
// Common block definition for parallel variables

///	@brief Lattice x extent
#define	nx 24
#if(nx<1)
#error "nx is expected it to be greater than or equal to 1"
#endif

// Keep original restriction of single spatial extent

///	@brief Lattice y extent. We normally use cubic lattices so this is the same as nx
#define	ny    nx
#if(ny<1)
#error "ny is expected it to be greater than or equal to 1"
#endif

///	@brief Lattice z extent. We normally use cubic lattices so this is the same as nx
#define	nz    nx
#if(nz<1)
#error "nz is expected it to be greater than or equal to 1"
#endif

///	@brief	Lattice temporal extent. This also corresponds to the inverse temperature
#define	nt	32
#if(nt<1)
#error "nt is expected it to be greater than or equal to 1"
#endif

///	@brief	Lattice volume
#define	gvol    (nx*ny*nz*nt)
///	@brief	Lattice spatial volume
#define	gvol3   (nx*ny*nz)

///	@brief Processor grid x extent. This must be a divisor of nx
#define	npx	1
#if(npx<1)
#error "npx is expected it to be greater than or equal to 1"
#elif(nx%npx!=0)
#error "npx should be a divisor of nx"
#endif

// Initially restrict to npz = npy = npx
// This allows us to have a single ksize variable

///	@brief Processor grid y extent
#define	npy	npx
#if(npy<1)
#error "npy is expected it to be greater than or equal to 1"
#elif(ny%npy!=0)
#error "npy should be a divisor of ny"
#endif

///	@brief Processor grid z extent
#define	npz	npx
#if(npz<1)
#error "npz is expected it to be greater than or equal to 1"
#elif(nz%npz!=0)
#error "npz should be a divisor of nz"
#endif

///	@brief Processor grid t extent
#define	npt	1
#if(npt<1)
#error "npt is expected it to be greater than or equal to 1"
#elif(nt%npt!=0)
#error "npt should be a divisor of nt"
#endif

///	@brief	Number of processors for MPI
#define	nproc	(npx*npy*npz*npt)

///	@brief Number of threads for OpenMP, which can be overwritten at runtime
#define	nthreads	32

//    Existing parameter definitions.
///	@brief Sublattice x extent
#define	ksizex	(nx/npx)
///	@brief Sublattice y extent
#define	ksizey	(ny/npy)
///	@brief Sublattice z extent
#define	ksizez	(nz/npz)

///	@brief Sublattice spatial extent for a cubic lattice
#define	ksize	ksizex

///	@brief Sublattice t extent
#define	ksizet	(nt/npt)
///	@brief Fermion flavours (double it)
#define	nf	1

///	@brief	Sublattice volume
#define	kvol	(ksizet*ksizez*ksizey*ksizex)
///	@brief	Sublattice spatial volume
#define	kvol3	(ksizez*ksizey*ksizex)

//     integer, parameter :: niterc=2*gvol  
//      #define niterc 2*gvol
//    jis: hard limit to avoid runaway trajectories
#if	(nx*ny*nz*nt<=16384)
///	@brief	Hard limit for runaway trajectories in Conjugate gradient
#define	niterc	gvol
#elif (nx>=(3*nt)/2)
///	@brief	Hard limit for runaway trajectories in Conjugate gradient
#define	niterc	gvol3
#else
///	@brief	Hard limit for runaway trajectories in Conjugate gradient
#define	niterc	(gvol/4)
#endif
//    Constants for dimensions.
///	@brief	Colours
#define	nc	2
///	@brief	adjacent spatial indices
#define	nadj	3
///	@brief	Dirac indices
#define	ndirac	4
///	@brief	Dimensions
#define	ndim	4
///	@brief	Gor'kov indices
#define	ngorkov	8

///		@brief	sublattice momentum sites
#define	kmom	(ndim*nadj*kvol)
///		@brief	sublattice size including Gor'kov indices
#define	kferm	(nc*ngorkov*kvol)
///		@brief	sublattice size including Dirac indices
#define	kferm2	(nc*ndirac*kvol)
/**
 * 	@subsection halo MPI Halos 
 *
 *    For those who may not have used MPI Before, halos are just a bit 
 *    of padding we put outside of the sublattices we're using in MPI
 *    so we can look at terms outside the sublattice we're actively working
 *    on with that process.
 */
///	@brief	x Halo size
#if(npx>1)
#define	halox	(ksizey*ksizez*ksizet)
#else
#define	halox	0
#endif
///	@brief	y Halo size
#if(npy>1)
#define	haloy	(ksizex*ksizez*ksizet)
#else
#define	haloy	0
#endif
///	@brief	z Halo size
#if(npz>1)
#define	haloz	(ksizex*ksizey*ksizet)
#else
#define	haloz	0
#endif
///	@brief	t Halo size
#if(npt>1)
#define	halot	(ksizex*ksizey*ksizez)
#else
#define	halot	0
#endif
///	@brief	Total Halo size
#define	halo	(2*(halox+haloy+haloz+halot))

///	@brief	Gor'kov lattice and halo
#define	kfermHalo	(nc*ngorkov*(kvol+halo))
///	@brief	Dirac lattice and halo
#define	kferm2Halo	(nc*ndirac*(kvol+halo))
///	@brief	Momentum lattice and halo
#define	kmomHalo	(ndim*nadj*(kvol+halo))

///	@brief Conjugate gradient residue for @f$\langle\bar{\Psi}\Psi\rangle@f$
//		These all used to be multipled by kferm or kferm2 at the start of Congradq or Congradp
//		On 20240516 in Room 2.19 of the Lloyd building of Trinity we copped that doing so means that the residue is larger
//		if running on a smaller number of cores. In the extreme GPU case the subvolume is the entire volume so the residue
//		can be several orders of magnitude larger than in the smallest sublattice case.
//		Instead, we rescale all the default residues here by sqrt(kferm) or sqrt(kferm2). No matter what size sublattice
//		we use now, the residue will match that of a 2^3X4 sublattice used in the earlier FORTRAN runs
#define	respbp	3.2E-5
///	@brief Conjugate gradient residue for update
#define	rescgg	2.26E-5 
///	@brief Conjugate gradient residue for acceptance
#define	rescga	2.26E-8 


#ifdef	__AVX512F__
#ifdef __unix__
#warning	AVX512 detected
#elif (defined WIN32||_WIN32)
#pragma message("AVX512 detected")
#endif
/// @brief Alignment of arrays. 64 for AVX-512, 32 for AVX/AVX2. 16 for SSE. Since AVX is standard on modern x86 machines I've called it that
#define	AVX	64
#elif defined	__AVX__
#ifdef __unix__
#warning	AVX or AVX2 detected
#elif (defined WIN32||_WIN32)
#pragma message("AVX or AVX2 detected")
#endif
/// @brief Alignment of arrays. 64 for AVX-512, 32 for AVX/AVX2. 16 for SSE. Since AVX is standard on modern x86 machines I've called it that
#define	AVX	32
#else
#ifdef __unix__
#warning	No AVX detected, assuming SSE is present
#elif (defined WIN32||_WIN32)
#pragma message("No AVX detected, assuming SSE is present")
#endif
/// @brief Alignment of arrays. 64 for AVX-512, 32 for AVX/AVX2. 16 for SSE. Since AVX is standard on modern x86 machines I've called it that
#define	AVX	16
#endif

#ifdef __GPU__
/*
 * @section gridblock Grids and Blocks
 *
 * Threads are grouped together to form warps of 32 threads
 * best to keep the block dimension (ksizex*ksizey) multiples of 32,
 * usually between 128 and 256
 * Note that from Volta/Turing  each SM (group of processors)
 * is smaller than on previous generations of GPUs
 */
extern dim3	dimBlock;//	=dim3(nx,ny,nz);
extern dim3	dimGrid;//	=dim3(nt,1,1);
						  //For copying over gamval
extern dim3	dimBlockOne;//	=dim3(nx,ny,nz);
extern dim3	dimGridOne;//	=dim3(nt,1,1);
#define	USE_BLAS
#endif
#endif
