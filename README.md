# su2hmc: Two Colour Hybrid Monte Carlo with Wilson Fermions
## Introduction
Hybrid Monte Carlo algorithm for Two Color QCD with Wilson-Gor'kov fermions
based on the algorithm of Duane et al. Phys. Lett. B195 (1987) 216. 

There is "up/down partitioning": each update requires
one operation of congradq on complex vectors to determine
$$
\left(M\dagger M\right)^{-1}\Phi
$$ 
where $\Phi$ has dimension 4 * kvol * nc * Nf -
The matrix M is the Wilson matrix for a single flavor
there is no extra species doubling as a result

matrix multiplies done using routines hdslash and hdslashd

Hence, the number of lattice flavors Nf is related to the
number of continuum flavors N_f by
              N_f = 2 * Nf

Fermion expectation values are measured using a noisy estimator.
on the Wilson-Gor'kov matrix, which has dimension 8 * kvol * nc * Nf
inversions done using congradp, and matrix multiplies with dslash,
dslashd

trajectory length is random with mean dt * stepl
The code runs for a fixed number ntraj of trajectories.

Phi: pseudofermion field 
bmass: bare fermion mass 
fmu: chemical potential 
actiona: running average of total action

Fermion expectation values are measured using a noisy estimator.
The code productes the following outputs:
|File Name| Data type|
|---------|:---------|
|config.bβββkκκκmuμμμμjJJJsNXtNT.XXXXXX| Lattice configuration for given parameters. Last digits are the configuration number|
|Output|	Number of conjugate gradient steps for each trajectory. Also contains general simulation details upon completion|
|Bosonic_Observables|		spatial plaquette, temporal plaquette, Polyakov line|
|PBP-Density|				psibarpsi, energy density, baryon density|
|Diquark|					real<qq>|

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
-	`#pragma omp simd` instructions
-	Makefiles for Intel, GCC and AMD compilers with flags set for latest machines
-	GSL ranlux support

Other works in progress include:
-	CUDA implementation. Compiles on my machine but refuses to link.
-	OpenACC.

## Getting started
This code is written for MPI on Linux, thus has a few caveats to get up and running
1.	In sizes.h, set the lattice size. By default we assume the spatial components
	to be equal
2.	Also in sizes.h set the processor grid size by setting the values of
	``` c
	npx npy npz npt
	```
	These **MUST**  be divisors of 
	``` c
	nx ny nz nt
	```
	set in step one.
3.	Compile the code using the desired Makefile. Please note that the paths given in the Makefiles for
	BLAS libraries etc. are based on my own system. You may need to adjust these manually.
4.	Run the code. This may differ from system to system, especially if a task scheduler like SLURM is being used.
	On my desktop it can be run locally using the following command

	``` sh
	mpirun -n<nproc> ./su2hmc <input_file>
	```

-	`nproc` is the number of processors, given by the product of `npx npy npz npt`
-	If no input file is given, the programme defaults to midout. I need to ask why that name was chosen. Maybe even change it...

### Input parameters
A sample input file looks like
```
0.00238	1.7	0.1780	0.00	0.000	0.0	0.0	420	20	1	1	420
dt	beta	akappa	jqq	thetaq	fmu	aNf	stepl	ntraj	istart	icheck	iread
```
where
- `dt` is the step size for the update
- `beta` is β, given up to three significant figures
- `akappa` is hopping parameter, given up to four significant figures
- `jqq` is the diquark source, given up to three significant figures
- `thetaq` is the diquark mixing angle
- `fmu` is the chemical potential
- `aNf` is ignored and left as a legacy of the FORTRAN input, where it was also ignored
- `stepl` is the average number of steps per trajectory. For a single trajectory it times dt should equal 1
- `ntraj` is the umber of trajectories
- `istart` signals a hot start (>=1) or cold start (<=0)
- `icheck` is how often to print out a configuration. We typically use 5 for an 80% acceptance rate
- `iread` is the starting configuration for continuation runs. If zero, start without reading

The bottom line of the input is ignored by the programme and is just there to make your life easier
Blank space does not matter, so long as there is some gap between the input parameters in the file and they are all
on a single line.
