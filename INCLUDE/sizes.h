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
#endif
#ifdef __CUDACC__
#include	<thrust_complex.h>
#else
#include	<complex.h>
#define	Complex_f	float	complex
#define	Complex		double complex
#endif

#define	FILELEN	64
// Common block definition for parallel variables

#define	nx 24
#define	nt	3
// Keep original restriction of single spatial extent

#define	ny    nx
#define	nz    nx
#define	gvol    (nx*ny*nz*nt)
#define	gvol3   (nx*ny*nz)

#define	npx	2
#define	npt	1
//Number of threads for OpenMP
#define	nthreads	1

// Initially restrict to npz = npy = npx
// This allows us to have a single ksize variable

#define	npy	npx
#define	npz	npx

#define	nproc	(npx*npy*npz*npt)

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
#define	halox	(ksizey*ksizez*ksizet)
#define	haloy	(ksizex*ksizez*ksizet)
#define	haloz	(ksizex*ksizey*ksizet)
#define	halot	(ksizex*ksizey*ksizez)
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
extern dim3	dimBlock;
extern dim3	dimGrid;
#endif
#endif
