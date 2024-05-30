#include <su2hmc.h>

int Leapfrog(Complex *u11t,Complex *u12t,Complex_f *u11t_f,Complex_f *u12t_f,Complex *X0,Complex *X1,
		Complex *Phi,double *dk4m,double *dk4p,float *dk4m_f,float *dk4p_f,double *dSdpi,double *pp,
		int *iu,int *id, Complex *gamval, Complex_f *gamval_f, int *gamin, Complex jqq,
		float beta, float akappa, int stepl, float dt, double *ancg, int *itot, float proby)
{
	//This was originally in the half-step of the FORTRAN code, but it makes more sense to declare
	//it outside the loop. Since it's always being subtracted we'll define it as negative
	const	double d = -dt*0.5;
	//Half step forward for p
	//=======================
#ifdef _DEBUG
	printf("Evaluating force on rank %i\n", rank);
#endif
	Force(dSdpi, 1, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
			dk4m_f,dk4p_f,jqq,akappa,beta,ancg);
#ifdef __NVCC__
	cublasDaxpy(cublas_handle,kmom, &d, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
	cblas_daxpy(kmom, d, dSdpi, 1, pp, 1);
#else
	for(int i=0;i<kmom;i++)
		//d negated above
		pp[i]+=d*dSdpi[i];
#endif
	//Main loop for classical time evolution
	//======================================
	bool end_traj=false; int step =1;
	//	for(int step = 1; step<=stepmax; step++){
	do{
#ifdef _DEBUG
		if(!rank)
			printf("Traj: %d\tStep: %d\n", itraj, step);
#endif
		//The FORTRAN redefines d=dt here, which makes sense if you have a limited line length.
		//I'll stick to using dt though.
		//step (i) st(t+dt)=st(t)+p(t+dt/2)*dt;
		//Note that we are moving from kernel to kernel within the default streams so don't need a Device_Sync here
		New_trial(dt,pp,u11t,u12t);
		Reunitarise(u11t,u12t);
#ifdef __NVCC__
		cudaDeviceSynchronise();
#endif
		//Get trial fields from accelerator for halo exchange
		Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
		//Mark trial fields as primarily read only here? Can re-enable writing at the end of each trajectory

		//p(t+3et/2)=p(t+dt/2)-dSds(t+dt)*dt
		//	Force(dSdpi, 0, rescgg);
		Force(dSdpi, 0, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
				dk4m_f,dk4p_f,jqq,akappa,beta,ancg);
		//The same for loop is given in both the if and else
		//statement but only the value of d changes. This is due to the break in the if part
		if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
#ifdef __NVCC__
			cublasDaxpy(cublas_handle,kmom, &d, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
			cblas_daxpy(kmom, d, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd aligned(pp,dSdpi:AVX)
			for(int i = 0; i<kmom; i++)
				//d negated above
				pp[i]+=d*dSdpi[i];
#endif
			*itot+=step;
			*ancg/=step;
			end_traj=true;
			break;
		}
		else{
#ifdef __NVCC__
			//dt is needed for the trial fields so has to be negated every time.
			double dt_d=-1*dt;
			cublasDaxpy(cublas_handle,kmom, &dt_d, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
			cblas_daxpy(kmom, -dt, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd aligned(pp,dSdpi:AVX)
			for(int i = 0; i<kmom; i++)
				pp[i]-=dt*dSdpi[i];
#endif
			step++;
		}
	}while(!end_traj);
	return 0;
}
int OMF2(Complex *u11t,Complex *u12t,Complex_f *u11t_f,Complex_f *u12t_f,Complex *X0,Complex *X1,
		Complex *Phi,double *dk4m,double *dk4p,float *dk4m_f,float *dk4p_f,double *dSdpi,double *pp,
		int *iu,int *id, Complex *gamval, Complex_f *gamval_f, int *gamin, Complex jqq,
		float beta, float akappa, int stepl, float dt, double *ancg, int *itot, float proby, float alpha)
{
	//Gauge update by half dt
	const	double dh = dt*0.5;
	//Momentum updates by alpha, 2alpha and (1-2alpha) in the middle
	const double da= -alpha*dt;
	const double da2= -2*alpha*dt;
	const double dam= -(1-2*alpha)*dt;
	//Initial step forward for p
	//=======================
#ifdef _DEBUG
	printf("Evaluating force on rank %i\n", rank);
#endif
	Force(dSdpi, 1, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
			dk4m_f,dk4p_f,jqq,akappa,beta,ancg);
#ifdef __NVCC__
	cublasDaxpy(cublas_handle,kmom, &da, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
	cblas_daxpy(kmom, da, dSdpi, 1, pp, 1);
#else
	for(int i=0;i<kmom;i++)
		//da negated above
		pp[i]+=da*dSdpi[i];
#endif
	//Main loop for classical time evolution
	//======================================
	bool end_traj=false; int step =1;
	//	for(int step = 1; step<=stepmax; step++){
	do{
#ifdef _DEBUG
		if(!rank)
			printf("Traj: %d\tStep: %d\n", itraj, step);
#endif
		//First gauge update
		New_trial(dh,pp,u11t,u12t);
		Reunitarise(u11t,u12t);
#ifdef __NVCC__
		cudaDeviceSynchronise();
#endif
		//Get trial fields from accelerator for halo exchange
		Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
		//Mark trial fields as primarily read only here? Can re-enable writing at the end of each trajectory

		//Calculate force for middle momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
				dk4m_f,dk4p_f,jqq,akappa,beta,ancg);
		//Now do the middle momentum update
#ifdef __NVCC__
		cublasDaxpy(cublas_handle,kmom, &dam, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
		cblas_daxpy(kmom, dam, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd aligned(pp,dSdpi:AVX)
		for(int i = 0; i<kmom; i++)
			pp[i]+=dam*dSdpi[i];
#endif

		//Another gauge update
		New_trial(dh,pp,u11t,u12t);
		Reunitarise(u11t,u12t);
#ifdef __NVCC__
		cudaDeviceSynchronise();
#endif
		//Get trial fields from accelerator for halo exchange
		Trial_Exchange(u11t,u12t,u11t_f,u12t_f);
		//Mark trial fields as primarily read only here? Can re-enable writing at the end of each trajectory

		//Calculate force for final momentum update
		Force(dSdpi, 0, rescgg,X0,X1,Phi,u11t,u12t,u11t_f,u12t_f,iu,id,gamval,gamval_f,gamin,dk4m,dk4p,\
				dk4m_f,dk4p_f,jqq,akappa,beta,ancg);
		//The same for loop is given in both the if and else
		//statement but only the value of d changes. This is due to the break in the if part
		if(step>=stepl*4.0/5.0 && (step>=stepl*(6.0/5.0) || Par_granf()<proby)){
			//Final momentum step
#ifdef __NVCC__
			cublasDaxpy(cublas_handle,kmom, &da, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
			cblas_daxpy(kmom, da, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd aligned(pp,dSdpi:AVX)
			for(int i = 0; i<kmom; i++)
				//d negated above
				pp[i]+=da*dSdpi[i];
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
#ifdef __NVCC__
			cublasDaxpy(cublas_handle,kmom, &da2, dSdpi, 1, pp, 1);
#elif defined USE_BLAS
			cblas_daxpy(kmom, da2, dSdpi, 1, pp, 1);
#else
#pragma omp parallel for simd aligned(pp,dSdpi:AVX)
			for(int i = 0; i<kmom; i++)
				pp[i]+=da2*dSdpi[i];
#endif
			step++;
		}
	}while(!end_traj);
	return 0;
}
