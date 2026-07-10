/**
 * @file random.c
 *
 * @brief Random number generator related routines
 */
#ifdef	__NVCC__
#include <curand.h>
#endif
#ifdef	__USE_MKL__
#include <mkl.h>
//Bad practice? Yes but it is convenient
#endif
#include "random.h"
#include <time.h>

//Declaring external variables
#ifdef __RANLUX__
/// @brief RANLUX instance
gsl_rng *ranlux_instd;
/// @brief RANLUX seed
unsigned long seed;
#else
/// @brief RAN2 seed
long seed;
#endif
#ifndef M_PI
/// @brief	@f$\pi@f$ if not defined elsewhere	
#define M_PI  acos(-1)
#endif

#ifdef __RANLUX__
inline int ranset(unsigned long *seed)
#else
inline int ranset(long *seed)
#endif
{
#ifdef __RANLUX__
	ranlux_instd=gsl_rng_alloc(gsl_rng_ranlxd2);
	gsl_rng_set(ranlux_instd,*seed);
	return 0;
#else
	return 0;
#endif
}
int Par_ranread(char *filename, double *ranval){
	const char funcname[] = "Par_psread";
	FILE *dest;
	if(!rank){
		if(!(dest = fopen(filename, "rb"))){
			fprintf(stderr, "Error %i in %s: Failed to open %s.\nExiting...\n\n", OPENERROR, funcname, filename);
#if(nproc>1)
			MPI_Abort(comm,OPENERROR); 
#else
			exit(OPENERROR);
#endif

		}
		fread(&ranval, sizeof(ranval), 1, dest);	
		fclose(dest);
	}
#if(nproc>1)
	Par_dcopy(ranval);
#endif
	return 0;
}
#ifdef __RANLUX__
int Par_ranset(unsigned long *seed,int iread)
#else
int Par_ranset(long *seed,int iread)
#endif
{
	const char funcname[] = "Par_ranset";
	//If we're not using the master thread, we need to change the seed
#ifdef _DEBUG
	printf("Master seed: %lu\t",*seed);
#endif
	if(rank)
		*seed *= 1.0f+8.0f*(float)rank/(float)(size-1);
#ifdef _DEBUG
	printf("Rank:  %i\tSeed %lu\n",rank, *seed);
#endif
	//Next we set the seed using ranset
	//This is one of the really weird FORTRAN 66-esque functions with ENTRY points, so good luck!
#ifdef __RANLUX__
	return ranset(seed);
#else
	return 0;
#endif
}
double Par_granf(){
	const char funcname[] = "Par_granf";
	double ran_val=0;
	if(!rank){
#ifdef __RANLUX__
		ran_val = gsl_rng_uniform(ranlux_instd);
		#else
		ran_val = ran2(&seed);
#endif
	}
#if(nproc>1)
	Par_dcopy(&ran_val);
#endif
	return ran_val;
}
int Gauss_z(Complex *ps, unsigned int n, const Complex mu, const double sigma){
	const char funcname[] = "Gauss_z";
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
#pragma unroll
	for(int i=0;i<n;i++){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r + I*(mu+v*r);
		 */
#ifdef __RANLUX__
		double	r =sigma*sqrt(-2*log(gsl_rng_uniform(ranlux_instd)));
		double	theta=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		double	r =sigma*sqrt(-2*log(ran2(&seed)));
		double	theta=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=r*(cos(theta)+sin(theta)*I)+mu;
	}     
	return 0;
}
int Gauss_c(Complex_f *ps, unsigned int n, const Complex_f mu, const float sigma){
	const char funcname[] = "Gauss_z";
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
#pragma unroll
	for(int i=0;i<n;i++){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r + I*(mu+v*r);
		 */
#ifdef __RANLUX__
		float r =sigma*sqrt(-2*log(gsl_rng_uniform(ranlux_instd)));
		float theta=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		float r =sigma*sqrt(-2*log(ran2(&seed)));
		float theta=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=r*(cos(theta)+mu+sin(theta)*I)+mu;
	}     
	return 0;
}
int Gauss_d(double *ps, unsigned int n, const double mu, const double sigma){
	const char funcname[] = "Gauss_z";
	//The FORTRAN Code had two different Gauss Routines. gaussp having unit
	//mean and variance and gauss0 where the variance would appear to be 1/sqrt(2)
	//(Since we multiply by sqrt(-ln(r)) instead of sqrt(-2ln(r)) )
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
	int i;
	double r, u, v;
	//If n is odd we calculate the last index seperately and the rest in pairs
	if(n%2==1){
		n--;
#ifdef __RANLUX__
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
		ps[n]=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*cos(r);
#else
		r=2.0*M_PI*ran2(&seed);
		ps[n]=sqrt(-2*log(ran2(&seed)))*cos(r);
#endif
	}
	for(i=0;i<n;i+=2){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r; 
			ps[i+1]=mu+v*r;
		 */
#ifdef __RANLUX__
		u=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*sigma;
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		u=sqrt(-2*log(ran2(&seed)))*sigma;
		r=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=u*cos(r)+mu;
		ps[i+1]=u*sin(r)+mu;
	}     
	return 0;
}
int Gauss_f(float *ps, unsigned int n, const float mu, const float sigma){
	const char funcname[] = "Gauss_z";
	//The FORTRAN Code had two different Gauss Routines. gaussp having unit
	//mean and variance and gauss0 where the variance would appear to be 1/sqrt(2)
	//(Since we multiply by sqrt(-ln(r)) instead of sqrt(-2ln(r)) )
	if(n<=0){
		fprintf(stderr, "Error %i in %s: Array cannot have length %i.\nExiting...\n\n",
				ARRAYLEN, funcname, n);
#if(nproc>1)
		MPI_Abort(comm,ARRAYLEN);
#else
		exit(ARRAYLEN);
#endif
	}
	int i;
	float r, u, v;
	//If n is odd we calculate the last index seperately and the rest in pairs
	if(n%2==1){
		n--;
#ifdef __RANLUX__
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
		ps[n]=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*cos(r);
#else
		r=2.0*M_PI*ran2(&seed);
		ps[n]=sqrt(-2*log(ran2(&seed)))*cos(r);
#endif
	}
#ifdef __RANLUX__
	r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
	ps[n]=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*cos(r);
#else
	r=2.0*M_PI*ran2(&seed);
	ps[n]=sqrt(-2*log(ran2(&seed)))*cos(r);
#endif
	for(i=0;i<n;i+=2){
		/* Marsaglia Method for fun
			do{
			u=sfmt_genrand_real1(sfmt);
			v=sfmt_genrand_real1(sfmt);
			r=u*u+v*v;
			}while(0<r & r<1);
			r=sqrt(r);
			r=sqrt(-2.0*log(r)/r)*sigma;
			ps[i] = mu+u*r; 
			ps[i+1]=mu+v*r;
		 */
#ifdef __RANLUX__
		u=sqrt(-2*log(gsl_rng_uniform(ranlux_instd)))*sigma;
		r=2.0*M_PI*gsl_rng_uniform(ranlux_instd);
#else
		u=sqrt(-2*log(ran2(&seed)))*sigma;
		r=2.0*M_PI*ran2(&seed);
#endif
		ps[i]=u*cos(r)+mu;
		ps[i+1]=u*sin(r)+mu;
	}
	return 0;
}
#ifndef __RANLUX__
double ran2(long *idum) {
	long k;
	int j;
	static long idum2=123456789; 
	static long iy=0;
	static long iv[NTAB];
	////Combining this with a different seed for each thread should give a thread-safe and repeatable result
#pragma omp threadprivate(idum2, iy, iv)
	//No worries
	double temp;

	if (*idum <= 0) {
		if (-(*idum) < 1) *idum=1; 
		else *idum = -(*idum);
		{
			idum2=(*idum);

			for(j=NTAB+7;j>=0;j--) {
				k=(*idum)/IQ1; 
				*idum=IA1*(*idum-k*IQ1)-k*IR1; 
				if (*idum < 0) *idum += IM1; 
				if (j < NTAB)
					iv[j] = *idum;
			}
			iy=iv[0];
		}
	}
	k=(*idum)/IQ1; 
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if (*idum < 0) *idum += IM1; 
	k=idum2/IQ2; 
	idum2=IA2*(idum2-k*IQ2)-k*IR2;

	if (idum2 < 0) idum2 += IM2; j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp=AM*iy) > RNMX) 
		return RNMX; 

	else return temp;

}
#endif

