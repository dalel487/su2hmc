#ifndef PAR_MPI
#define	PAR_MPI
#include	<complex.h>
#include	<coord.h>
#ifdef __NVCC__
#include	<cuComplex.h>
#endif
#include	<errorcodes.h>
#include	<math.h>
#ifdef	USE_MKL
//If using mkl and BLAS, it is good practice to use mkl_malloc to align the arrays better
//for the AVX-512 FMA Units
#include	<mkl.h>
#endif
#include	<mpi.h> 
#ifdef _OPENMP
#include	<omp.h>
#endif
#include	<sizes.h>
#include	<stdio.h>
#include	<stdlib.h>
#include    <string.h>

//Avoid any accidents with US/UK spelling
#define MPI_Finalise() MPI_Finalize()

//Definitons
//==========
#define	DOWN	0
#define     UP	1

#define masterproc 0

//	Used the actual name instead of the alias when converting, but it's still good to have here
//	I may change to the alias so that we can update it quicker in the future.
#define MPI_CMPLXKIND   MPI_C_DOUBLE_COMPLEX
#define MPI_REALKIND    MPI_DOUBLE

#define tag   0
//#define _STAT_SIZE_  sizeof(MPI_Status)
//Variables
//=========
//Up/Down arrays
int pu[ndim] __attribute__((aligned(AVX)));
int pd[ndim] __attribute__((aligned(AVX))); 

//MPI Stuff
int procid;
int ierr;


//In C and Fortran 2008, MPI_STATUS_SIZE is replaced with 
//MPI_Status
//      int status[_STAT_SIZE_];
//      int statarray[_STAT_SIZE_][2*ndim];
MPI_Status status;
extern MPI_Comm comm ;
int request;

int gsize[ndim];
int lsize[ndim];

int *pcoord;
int pstart[ndim][nproc] __attribute__((aligned(AVX)));
int pstop [ndim][nproc] __attribute__((aligned(AVX)));
int rank, size;
//C doesn't have logicals so we'll instead use an integer
//although using rank 0 could also work?
//      logical ismaster
int ismaster;

      //The common keyword is largely redundant here as everything
      //is already global scope.
      
      /*common /par/ pu, pd, procid, comm,
     1             gsize, lsize, pcoord, pstart, pstop,
     1             ismaster, masterproc
*/	

//A couple of other components usually defined in common_*.h files in fortran. But since C has global scope
//may as well put them in here instead.
//Gauges
#ifdef __NVCC__
cuDoubleComplex *u11, *u12;
#else
complex *u11, *u12;
#endif
// Trial matrices
#ifdef __NVCC__
cuDoubleComplex *u11t, *u12t;
#else
complex *u11t, *u12t;
#endif
//Halos indices
//-------------
static unsigned int *iu, *id;
unsigned int h1u[4], h1d[4];
unsigned int halosize[4];

//Function Declarations
//=====================
int Par_begin(int argc, char *argv[]);
int Par_sread();
int Par_psread(char *filename, double *ps);
int Par_swrite(const int itraj, const int icheck, const double beta, const double fmu, const double akappa, const double ajq);
int Par_end();
//Shortcuts for reductions and broadcasts
int Par_isum(int *ival);
int Par_dsum(double *dval);
int Par_zsum(complex *zval);
int Par_icopy(int *ival);
int Par_dcopy(double *dval);
int Par_zcopy(complex *zval);
//Halo Manipulation
int ZHalo_swap_all(complex *z, int ncpt);
int ZHalo_swap_dir(complex *z, int ncpt, int idir, int layer);
int DHalo_swap_dir(double *d, int ncpt, int idir, int layer);
int Trial_Exchange();

int Par_tmul(complex *z11, complex *z12);
#endif
