# su2hmc: Two Colour Hybrid Monte Carlo with Wilson Fermions
## Introduction
Hybrid Monte Carlo algorithm for Two Color QCD with Wilson-Gor'kov fermions
based on the algorithm of Duane et al. Phys. Lett. B195 (1987) 216. 

There is "up/down partitioning": each update requires
one operation of congradq on complex*16 vectors to determine
(Mdagger M)**-1  Phi where Phi has dimension 4*kvol*nc*Nf - 
The matrix M is the Wilson matrix for a single flavor
there is no extra species doubling as a result

matrix multiplies done using routines hdslash and hdslashd

Hence, the number of lattice flavors Nf is related to the
number of continuum flavors N_f by
              N_f = 2 * Nf

Fermion expectation values are measured using a noisy estimator.
on the Wilson-Gor'kov matrix, which has dimension 8*kvol*nc*Nf
inversions done using congradp, and matrix multiplies with dslash,
dslashd

trajectory length is random with mean dt*stepl
The code runs for a fixed number ntraj of trajectories.

Phi: pseudofermion field 
bmass: bare fermion mass 
fmu: chemical potential 
actiona: running average of total action

Fermion expectation values are measured using a noisy estimator.
outputs:
Bosonic_Observables:		spatial plaquette, temporal plaquette, Polyakov line
PBP-Density:				psibarpsi, energy density, baryon density
Diquark:					   real<qq>

                                           SJH March 2005

 Hybrid code, P.Giudice, May 2013
 Converted from Fortran to C by D. Lawlor March 2021
### Conversion notes
This two colour implementation was originally written in FORTRAN for:
[S. Hands, S. Kim and J.-I. Skullerud, Deconfinement in
dense 2-color QCD, Eur. Phys. J. C48, 193 (2006), hep-
lat/0604004](https://arxiv.org/abs/hep-lat/0604004)

It has since been rewritten in C and is in the process of being adapted for CUDA.

Some adaptions from the original are:
-	Mixed precision conjugate gradient
-	Implementation of BLAS routines for vector operations
-	Removal of excess halo exchanges
-	'''omp simd''' instructions
-	Makefiles for Intel, GCC and AMD compilers with flags set for latest machines
-	GSL ranlux support

Other works in progress include:
-	CUDA implementation. Compiles on my machine but refuses to link.
-	OpenACC.

##Getting started
This code is written for MPI, thus has a few caveats to get up and running
1.	In sizes.h, set the lattice size. By default we assume the spatial components
	to be equal
2.	Also in sizes.h set the processor grid size by setting the values of
	'''
	npx npy npz npt
	'''
	These **MUST**  be divisors of 
	'''
	nx ny nz nt
	'''
	set in step one.
3.	Compile the code using the desired Makefile. Please note that the paths given in the Makefiles for
	BLAS libraries etc. are based on my own system. You may need to adjust these manually.
4.	Run the code. This may differ from system to system, especially if a task scheduler like SLURM is being used.
	On my desktop it can be run locally using the following command
	'''sh
	mpirun -n<nproc> ./su2hmc <input_file>
	'''
-	'''nproc''' is the number of processors, given by the product of '''npx npy npz npt'''
-	If no input file is given, the programme defaults to  midout

