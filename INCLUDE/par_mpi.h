#ifndef	PAR_MPI
#define	PAR_MPI
#include	<coord.h>
#include	<errorcodes.h>
#include	<fields.h>
#include	<math.h>
#include	<mpi.h> 
#ifdef _OPENMP
#include	<omp.h>
#endif
#include	<random.h>
#include	<sizes.h>
#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>

//Avoid any accidents with US/UK spelling
#define MPI_Finalise() MPI_Finalize()

//Definitons
//==========
#define	DOWN	0
#define	UP		1

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
//halos indices
//-------------
//static unsigned int *iu, *id;

//Function Declarations
//=====================
#ifdef __cplusplus
extern "C"
{
#endif
	int Par_begin(int argc, char *argv[]);
	int Par_sread(const int iread, const double beta, const double fmu, const double akappa, const Complex ajq,\
			Complex *u11, Complex *u12, Complex *u11t, Complex *u12t);
	//	int Par_psread(char *filename, double *ps);
	int Par_swrite(const int itraj, const int icheck, const double beta, const double fmu, const double akappa, const Complex ajq,\
			Complex *u11, Complex *u12);
	int Par_end();
	//Shortcuts for reductions and broadcasts. These should be inlined
	int Par_isum(int *ival);
	int Par_dsum(double *dval);
	int Par_fsum(float *dval);
	int Par_zsum(Complex *zval);
	int Par_icopy(int *ival);
	int Par_dcopy(double *dval);
	int Par_fcopy(float *fval);
	int Par_zcopy(Complex *zval);
	//Halo Manipulation
	int ZHalo_swap_all(Complex *z, int ncpt);
	int ZHalo_swap_dir(Complex *z, int ncpt, int idir, int layer);
	int CHalo_swap_all(Complex_f *c, int ncpt);
	int CHalo_swap_dir(Complex_f *c, int ncpt, int idir, int layer);
	int DHalo_swap_dir(double *d, int ncpt, int idir, int layer);
	int Trial_Exchange(Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f);
	//If we have more than two processors on the time axis, there's an extra step in the Polyakov loop calculation
	int Par_tmul(Complex *z11, Complex *z12);
#ifdef __cplusplus
}
#endif
#endif
