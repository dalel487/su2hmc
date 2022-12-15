/*******************************************************************
 *
 *    Defines the constants of the code and other parameters 
 *    for loop dimensions.
 *    Each subroutine includes these definitions using:
 *    INCLUDE sizes.h
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
 *    C Version translated by:
 *    D. Lawlor September 2020
 *
 ******************************************************************/
#ifndef	SIZES
#define	SIZES
#ifdef	__INTEL_MKL__
#include	<mkl.h>
#endif
#ifdef	__NVCC__
#include	<cuda.h>
#include	<cuda_runtime_api.h>
#endif
#ifdef __CUDACC__
#include	<thrust_complex.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#else
#include	<complex.h>
#define	Complex_f	float	complex
#define	Complex		double complex
#endif

#define	FILELEN	64
// Common block definition for parallel variables

#define	nx 12
#if(nx<1)
#error "nx is expected it to be greater than or equal to 1"
#endif

// Keep original restriction of single spatial extent

#define	ny    nx
#if(ny<1)
#error "ny is expected it to be greater than or equal to 1"
#endif

#define	nz    nx
#if(nz<1)
#error "nz is expected it to be greater than or equal to 1"
#endif

#define	nt	32
#if(nt<1)
#error "nt is expected it to be greater than or equal to 1"
#endif

#define	gvol    (nx*ny*nz*nt)
#define	gvol3   (nx*ny*nz)

#define	npx	1
#if(npx<1)
#error "npx is expected it to be greater than or equal to 1"
#elif(nx%npx!=0)
#error "npx should be a divisor of nx"
#endif

// Initially restrict to npz = npy = npx
// This allows us to have a single ksize variable

#define	npy	npx
#if(npy<1)
#error "npy is expected it to be greater than or equal to 1"
#elif(ny%npy!=0)
#error "npy should be a divisor of ny"
#endif

#define	npz	npx
#if(npz<1)
#error "npz is expected it to be greater than or equal to 1"
#elif(nz%npz!=0)
#error "npz should be a divisor of nz"
#endif

#define	npt	1
#if(npt<1)
#error "npt is expected it to be greater than or equal to 1"
#elif(nt%npt!=0)
#error "npt should be a divisor of nt"
#endif

#define	nproc	(npx*npy*npz*npt)

//Number of threads for OpenMP, which can be overwritten at runtime
#define	nthreads	16

//    Existing parameter definitions.
#define	ksizex	(nx/npx)
#define	ksizey	(ny/npy)
#define	ksizez	(nz/npz)

#define	ksize	ksizex

#define	ksizet	(nt/npt)
#define	nf	1

#define	kvol	(ksizet*ksizez*ksizey*ksizex)
#define	kvol3	(ksizez*ksizey*ksizex)

#define	stepmax	1000
//     integer, parameter :: niterc=2*gvol  
//      #define niterc 2*gvol
//    jis: hard limit to avoid runaway trajectories
#define	niterc	gvol	
//    Constants for dimensions.
#define	nc	2
#define	nadj	3
#define	ndirac	4
#define	ndim	4
#define	ngorkov	8

#define	kmom	(ndim*nadj*kvol)
#define	kferm	(nc*ngorkov*kvol)
#define	kferm2	(nc*ndirac*kvol)
//    For those who may not have used MPI Before, halos are just a bit 
//    of padding we put on the outside of the sub-arrays we're using in MPI
//    so we can look at terms outside the sub-array we're actively working
//    on with that process.
//TODO: Sort out coord.h so we can run without halos
#if(npx>1)
#define	halox	(ksizey*ksizez*ksizet)
#else
#define	halox	0
#endif
#if(npy>1)
#define	haloy	(ksizex*ksizez*ksizet)
#else
#define	haloy	0
#endif
#if(npz>1)
#define	haloz	(ksizex*ksizey*ksizet)
#else
#define	haloz	0
#endif
#if(npt>1)
#define	halot	(ksizex*ksizey*ksizez)
#else
#define	halot	0
#endif
#define	halo	(2*(halox+haloy+haloz+halot))

#define	kfermHalo	(nc*ngorkov*(kvol+halo))
#define	kferm2Halo	(nc*ndirac*(kvol+halo))
#define	kmomHalo	(ndim*nadj*(kvol+halo))

#define	respbp	1E-6
#define	rescgg	1E-6 
#define	rescga	1E-9 


//Alignment of arrays. 64 for AVX-512, 32 for AVX/AVX2. 16 for SSE. Since AVX is standard
//on modern x86 machines I've called it that
#ifdef	__AVX512F__
#ifdef __unix__
#warning	AVX512 detected
#elif (defined WIN32||_WIN32)
#pragma message("AVX512 detected")
#endif
#define	AVX	64
#elif defined	__AVX__
#ifdef __unix__
#warning	AVX or AVX2 detected
#elif (defined WIN32||_WIN32)
#pragma message("AVX or AVX2 detected")
#endif
#define	AVX	32
#else
#ifdef __unix__
#warning	No AVX detected, assuming SSE is present
#elif (defined WIN32||_WIN32)
#pragma message("No AVX detected, assuming SSE is present")
#endif
#define	AVX	16
#endif

#ifdef	__NVCC__
//Threads are grouped together to form warps of 32 threads
//best to keep the block dimension (ksizex*ksizey) multiples of 32,
//usually between 128 and 256
//Note that from Volta/Turing  each SM (group of processors)
//is smaller than on previous generations of GPUs
extern dim3	dimBlock;//	=dim3(nx,ny,nz);
extern dim3	dimGrid;//	=dim3(nt,1,1);
#endif
#endif
