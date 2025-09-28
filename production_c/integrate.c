/**
 *	@file 	integrate.c
 *
 *	@brief	Molecular dynamics integrators and support routines
 *
 *	@author	D. Lawlor
 */
#include <su2hmc.h>

int Gauge_Update(const double d, double *pp, Complex *ut[2],Complex_f *ut_f[2]){
	/*
	 * @brief Generates new trial fields
	 *
	 * @see cuGauge_Update (CUDA Wrapper)
	 * 
	 * @param	d:		Half lattice spacing
	 * @param	pp:		Momentum field
	 * @param	ut[0]:		First colour field
	 * @param	ut[1]:		Second colour field
	 *
	 * @returns	Zero on success, integer error code otherwise
	 */
	char funcname[] = "Gauge_Update"; 
#ifdef __NVCC__
	cuGauge_Update(d,pp,ut[0],ut[1],dimGrid,dimBlock);
#else
#pragma omp parallel for simd collapse(2) aligned(pp:AVX) 
	for(int i=0;i<kvol;i++)
		for(int mu = 0; mu<ndim; mu++){
			/*
			 * Sticking to what was in the FORTRAN for variable names.
			 * CCC for cosine SSS for sine AAA for...
			 * Re-exponentiating the force field. Can be done analytically in SU(2)
			 * using sine and cosine which is nice
			 */

			double AAA = d*sqrt(pp[i*nadj*ndim+mu]*pp[i*nadj*ndim+mu]\
					+pp[(i*nadj+1)*ndim+mu]*pp[(i*nadj+1)*ndim+mu]\
					+pp[(i*nadj+2)*ndim+mu]*pp[(i*nadj+2)*ndim+mu]);
			double CCC = cos(AAA);
			double SSS = d*sin(AAA)/AAA;
			Complex a11 = CCC+I*SSS*pp[(i*nadj+2)*ndim+mu];
			Complex a12 = pp[(i*nadj+1)*ndim+mu]*SSS + I*SSS*pp[i*nadj*ndim+mu];
			//b11 and b12 are ut[0] and ut[1] terms, so we'll use ut[1] directly
			//but use b11 for ut[0] to prevent RAW dependency
			Complex b11 = ut[0][i*ndim+mu];
			ut[0][i*ndim+mu] = a11*b11-a12*conj(ut[1][i*ndim+mu]);
			ut[1][i*ndim+mu] = a11*ut[1][i*ndim+mu]+a12*conj(b11);
		}
#endif
	Reunitarise(ut);
	//Get trial fields from accelerator for halo exchange
	Trial_Exchange(ut,ut_f);
	return 0;
}
inline int Momentum_Update(const double d, const double *dSdpi, double *pp)
{
	const char funcname[] = "Momentum_Update";
#ifdef __NVCC__
	cublasDaxpy(cublas_handle,kmom, &d, dSdpi, 1, pp, 1);
	cudaDeviceSynchronise();
#elif defined USE_BLAS
	cblas_daxpy(kmom, d, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd
	for(int i=0;i<kmom;i++)
		pp[i]+=d*dSdpi[i];
#endif
	return 0;
}
int Leapfrog(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
		double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex *gamval, Complex_f *gamval_f, int *gamin,
		Complex *sigval, Complex_f *sigval_f, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
		const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby)
{
	const char funcname[] = "Leapfrog";
	//This was originally in the half-step of the FORTRAN code, but it makes more sense to declare
	//it outside the loop. Since it's always being subtracted we'll define it as negative
	const	double d =-dt*0.5;
	//Half step forward for p
	//=======================
#ifdef _DEBUG
	printf("Evaluating force on rank %i\n", rank);
#endif
	Force(dSdpi, 1, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
	Momentum_Update(d,dSdpi,pp);
	//Main loop for classical time evolution
	//======================================
	bool end_traj=false; int step =1;
	//	for(int step = 1; step<=stepmax; step++){
	do{
#ifdef _DEBUG
		if(!rank)
			printf("Step: %d\n",  step);
#endif
		//The FORTRAN redefines d=dt here, which makes sense if you have a limited line length.
		//I'll stick to using dt though.
		//step (i) st(t+dt)=st(t)+p(t+dt/2)*dt;
		//Note that we are moving from kernel to kernel within the default streams so don't need a Device_Sync here
		Gauge_Update(dt,pp,ut,ut_f);

		//p(t+3et/2)=p(t+dt/2)-dSds(t+dt)*dt
		//	Force(dSdpi, 0, rescgg);
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);

		//	if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
		if(step==stepl){
			//Final trajectory has a half momentum step
			Momentum_Update(d,dSdpi,pp);
			*itot+=step;
			*ancg/=step;
			end_traj=true;
			break;
		}
		else{
			//Otherwise, there's a half step at the end and start of each trajectory, so we combine them into one full step.
			Momentum_Update(-dt,dSdpi,pp);
			step++;
		}
	}while(!end_traj);

	return 0;
}
int OMF2(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
		double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex *gamval, Complex_f *gamval_f, int *gamin,
		Complex *sigval, Complex_f *sigval_f, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
		const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby)
{
	const char funcname[] = "OMF2";
	const double lambda=0.5-(pow(2.0*sqrt(326.0)+36.0,1.0/3.0)/12.0)+1.0/(6*pow(2.0*sqrt(326.0) + 36.0,1.0/3.0));
	//const double lambda=1.0/6.0;
	//	const double lambda=0.5;

	//Gauge update by half dt
	const	double dU = dt*0.5;

	//Momentum updates by lambda, 2lambda and (1-2lambda) in the middle
	const double dp= -lambda*dt;
	const double dp2= 2.0*dp;
	const double dpm= -(1.0-2.0*lambda)*dt;
	//Initial step forward for p
	//=======================
#ifdef _DEBUG
	printf("Evaluating force on rank %i, dSdpi[0] %e\n", rank,dSdpi[0]);
#endif
	Force(dSdpi, 1, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
#ifdef _DEBUG
	if(!rank)
		printf("Initial force on rank %i, dSdpi[0] %e\n", rank,dSdpi[0]);
#endif
	//Initial momentum update
	Momentum_Update(dp,dSdpi,pp);
#ifdef _DEBUG
	if(!rank)
		printf("Initial momentum on rank %i, pp[0] %e\n", rank,pp[0]);
#endif

	//Main loop for classical time evolution
	//======================================
	bool end_traj=false; int step =1;
	//	for(int step = 1; step<=stepmax; step++){
	do{
#ifdef _DEBUG
		if(!rank)
			printf("Step: %d\n", step);
#endif
		//First gauge update
		Gauge_Update(dU,pp,ut,ut_f);

		//Calculate force for middle momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
#ifdef _DEBUG
		if(!rank)
			printf("First force update on step %i, dSdpi[0] %e\n", step,dSdpi[0]);
#endif
		//Now do the middle momentum update
		Momentum_Update(dpm,dSdpi,pp);
#ifdef _DEBUG
		if(!rank)
			printf("Middle momentum on step %i, pp[0] %e\n", step,pp[0]);
#endif

		//Second gauge update
		Gauge_Update(dU,pp,ut,ut_f);

		//Calculate force for second momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
#ifdef _DEBUG
		if(!rank)
			printf("Second force update on step %i, dSdpi[0] %e\n", step,dSdpi[0]);
#endif

		//if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
		if(step==stepl){
			//Final momentum step
			Momentum_Update(dp,dSdpi,pp);
#ifdef _DEBUG
			if(!rank)
				printf("Final momentum on step %i, pp[0] %e\n", step,pp[0]);
#endif
			*itot+=step;
			//Two force terms, so an extra factor of two in the average?
			//Or leave it as it was, to get the average CG iterations per trajectory rather than force
			*ancg/=step;
			end_traj=true;
			break;
		}
		else{
			//Since we apply the momentum at the start and end of a step we instead apply a double step here
			Momentum_Update(dp2,dSdpi,pp);
#ifdef _DEBUG
			if(!rank)
				printf("Intermediate momentum on step %i, pp[0] %e\n", step,pp[0]);
#endif
			step++;
		}
	}while(!end_traj);
	return 0;
}
int OMF4(Complex *ut[2],Complex_f *ut_f[2],Complex *X0,Complex *X1, Complex *Phi,double *dk[2],float *dk_f[2],
		double *dSdpi,double *pp, unsigned int *iu,unsigned int *id, Complex *gamval, Complex_f *gamval_f, int *gamin,
		Complex *sigval, Complex_f *sigval_f, unsigned short *sigin, const Complex jqq, const float beta, const float akappa, 
		const float c_sw, const int stepl, const float dt, double *ancg, int *itot, const float proby)
{
	const char funcname[] = "OMF4";
	//These values were lifted from openqcd-fastsum, and should probably be tuned for QC2D. They also probably never
	//will be...
	const double r1 = 0.08398315262876693;
	const double r2 = 0.2539785108410595;
	const double r3 = 0.6822365335719091;
	const double r4 = -0.03230286765269967;
	///@brief Momentum updates
	///@brief Outer updates depend on r1. We have two of these, doubled for between full steps
	const double dpO= -r1*dt;
	const double dpO2= 2*dpO;
	///@brief Middle updates depend on r3
	const double dpM= -r3*dt;
	///@brief Inner updates depend on r1 and r3
	const double dpI= -(0.5-r1-r3)*dt;

	///@brief Gauge updates. These depend on r2 and r4
	///@brief Outer gauge update depends on r2
	const	double duO = dt*r2;
	///@brief Middle gauge update depends on r4
	const	double duM = dt*r4;
	///@brief Inner gauge update depends on r2 and r4
	const	double duI = dt*(1-2*(r2+r4));

	//Initial step forward for p
	//=======================
#ifdef _DEBUG
	printf("Evaluating force on rank %i\n", rank);
#endif
	Force(dSdpi, 1, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
	Momentum_Update(dpO,dSdpi,pp);

	//Main loop for classical time evolution
	//======================================
	bool end_traj=false; int step =1;
	//	for(int step = 1; step<=stepmax; step++){
	do{
#ifdef _DEBUG
		if(!rank)
			printf("Step: %d\n", step);
#endif
		//First outer gauge update
		Gauge_Update(duO,pp,ut,ut_f);

		//Calculate force for first middle momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
		//Now do the first middle momentum update
		Momentum_Update(dpM,dSdpi,pp);

		//First middle gauge update
		Gauge_Update(duM,pp,ut,ut_f);

		//Calculate force for first inner momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
		//Now do the first inner momentum update
		Momentum_Update(dpI,dSdpi,pp);

		//Inner gauge update
		Gauge_Update(duI,pp,ut,ut_f);

		//Calculate force for second inner momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
		//Now do the second inner momentum update
		Momentum_Update(dpI,dSdpi,pp);

		//Second middle gauge update
		Gauge_Update(duM,pp,ut,ut_f);

		//Calculate force for second middle momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);
		//Now do the second middle momentum update
		Momentum_Update(dpM,dSdpi,pp);

		//Second outer gauge update
		Gauge_Update(duO,pp,ut,ut_f);

		//Calculate force for outer momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,ut,ut_f,iu,id,gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,akappa,beta,c_sw,ancg);

		//Outer momentum update depends on if we've finished the trajectory
		//if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
		if(step==stepl){
			//Final momentum step
			Momentum_Update(dpO,dSdpi,pp);
			*itot+=step;

			//Four force terms, so an extra factor of four in the average?
			//Or leave it as it was, to get the average CG iterations per trajectory rather than force
			*ancg/=step;
			end_traj=true;
			break;
		}
		else{
			//Since we apply the momentum at the start and end of a step we instead apply a double step here
			Momentum_Update(dpO2,dSdpi,pp);
			step++;
		}
	}while(!end_traj);
	return 0;
}
