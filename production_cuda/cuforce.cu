/**
 * @file
 * @brief Code for force calculations.
 * 		Requires multiply.cu to work
 * @author	D. Lawlor
 */
#include	<matrices.h>
#include	<su2hmc.h>
//CUDA Kernels
/**
 * @brief Calculates the staple in the positive @f$\mu@f$ direction
 *
 * @param mu:						@f$\mu@f$ direction
 * @param nu:						@f$\nu@f$ direction
 * @param iu:						Upper indices
 * @param Sigma11,Sigma12:		Staple output
 * @param u11t,u12t:				Gauge fields
 *
 */
__global__ void Plus_staple(const int mu, const int nu,unsigned int *iu, Complex_f *Sigma11, Complex_f *Sigma12, Complex_f *u11t, Complex_f *u12t){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(unsigned int i=threadId;i<kvol;i+=gsize*bsize){
		const unsigned int uidm = iu[mu*kvol+i];
		unsigned int indn=uidm+kvolHalo*nu;
		const unsigned int uidn = iu[nu*kvol+i];
		unsigned int indm=uidn+kvolHalo*mu;
		Complex_f	a11=u11t[indn]*conj(u11t[indm])+\
							 u12t[indn]*conj(u12t[indm]);
		Complex_f	a12=-u11t[indn]*u12t[indm]+\
							 u12t[indn]*u11t[indm];
		indn=i+kvolHalo*nu;
		Sigma11[i]+=a11*conj(u11t[indn])+a12*conj(u12t[indn]);
		Sigma12[i]+=-a11*u12t[indn]+a12*u11t[indn];
	}
}
/**
 * @brief Calculates the staple in the positive @f$\mu@f$ direction
 *
 * @param mu:						@f$\mu@f$ direction
 * @param nu:						@f$\nu@f$ direction
 * @param iu:						Upper indices
 * @param Sigma11,Sigma12:		Staple output
 * @param u11sh,u12sh:			Gauge fields in @f$\mu@f$ direction only 
 * @param u11t,u12t:				Gauge fields
 *
 */
__global__ void Minus_staple(const int mu,const int nu,unsigned int *iu,unsigned int *id, Complex_f *Sigma11, Complex_f *Sigma12,\
		Complex_f *u11sh, Complex_f *u12sh, Complex_f *u11t, Complex_f *u12t){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(unsigned int i=threadId;i<kvol;i+=gsize*bsize){
		const unsigned int uidm = iu[mu*kvol+i];
		const unsigned int didn = id[nu*kvol+i];
		//uidm is correct here
		unsigned int ind=didn+kvolHalo*mu;
		Complex_f u11s=u11t[ind]; Complex_f u12s=u12t[ind];
		Complex_f a11=conj(u11sh[uidm])*conj(u11s)-\
						  u12sh[uidm]*conj(u12s);
		Complex_f a12=-conj(u11sh[uidm])*u12s-\
						  u12sh[uidm]*u11s;
		ind=didn+kvolHalo*nu;
		u11s=u11t[ind]; u12s=u12t[ind];
		Sigma11[i]+=a11*u11s-a12*conj(u12s);
		Sigma12[i]+=a11*u12s+a12*conj(u11s);
	}
}
__global__ void cuGaugeForce(int mu, Complex_f *Sigma11, Complex_f *Sigma12,double* dSdpi,Complex_f *u11t, Complex_f *u12t, float beta){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(unsigned int i=threadId;i<kvol;i+=gsize*bsize){
		const unsigned int ind = i+kvolHalo*mu;
		Complex_f a11 = u11t[ind]*Sigma12[i]+u12t[ind]*conj(Sigma11[i]);
		Complex_f a12 = u11t[ind]*Sigma11[i]+conj(u12t[ind])*Sigma12[i];
		//Not worth splitting into different streams, before we get ideas...
		dSdpi[i+kvol*mu]=beta*a11.imag();
		dSdpi[i+kvol*(1*ndim+mu)]=beta*a11.real();
		dSdpi[i+kvol*(2*ndim+mu)]=beta*a12.imag();
	}
}
/**
 * @brief	Extracts all the single precision gauge links in the @f$\mu@f$ direction only
 *
 * @param	x:			The output 
 * @param	y:			The gauge field for a particular colour
 * @param	n:			Number of sites in the gauge field. This is typically kvol
 * @param	table:	Table containing information on nearest neighbours. Usually id or iu
 * @param	mu:		Direciton we're interested in extractng	
 *
 */
	template <typename T>
__global__ void Gather(T *x, T *y, const unsigned int n, unsigned int *table, const unsigned short mu)
{
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;
	const unsigned int kvbmu=kvolHalo*mu;
	for(unsigned int i = gthreadId; i<kvol;i+=gsize*bsize)
		x[i]=y[table[i+kvbmu]+kvbmu];
}

__global__ void cuForce_s(double *dSdpi, Complex_f *u11t, Complex_f *u12t, Complex_f *X1, Complex_f *X2, Complex_f gamval[20],\
		unsigned int *iu, const unsigned short gamin[16],float akappa, const unsigned short mu){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;

	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		const unsigned int ind=i+kvolHalo*mu;
		const Complex_f u11s=u11t[ind]; const Complex_f u12s=u12t[ind];
		const unsigned int uid = iu[i+kvol*mu];
		//Similarly to Hdslash we always see idirac*nc so we do that here too.
		for(unsigned short idirac=0;idirac<nc*ndirac;idirac+=nc){
			Complex_f X1s[nc];	 Complex_f X1su[nc];
			Complex_f X2s[nc];	 Complex_f X2su[nc];

			X1s[0]=X1[i+kvolHalo*(idirac)]; X1s[1]=X1[i+kvolHalo*(1+idirac)];
			X1su[0]=X1[uid+kvolHalo*(idirac)]; X1su[1]=X1[uid+kvolHalo*(1+idirac)];
			X2s[0]=X2[i+kvolHalo*(idirac)]; X2s[1]=X2[i+kvolHalo*(1+idirac)];
			X2su[0]=X2[uid+kvolHalo*(idirac)]; X2su[1]=X2[uid+kvolHalo*(1+idirac)];

		//			Need to be double to avoid accumulation errors
			double dSdpis[3];
			//Careful!! cant use ind here as dSdpi has no halo!
			dSdpis[0]=dSdpi[i+kvol*mu];
			//Multiplying by i and taking the real component is the same as taking the negative imaginary component
			//The positions of u11 and u12 might look a bit funky here. That's just because we've multiplied by the
			//generators by hand
			dSdpis[0]+=-akappa*(
					conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					+conj(X1s[1])*(u11s*X2su[0]+u12s*X2su[1])
					+conj(X1su[0])*(u12s*X2s[0]-conj(u11s)*X2s[1])
					+conj(X1su[1])*(-u11s*X2s[0]-conj(u12s)*X2s[1])).imag();

			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=akappa*(
					(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					 +conj(X1s[1])*(-u11s*X2su[0]-u12s*X2su[1])
					 +conj(X1su[0])*(-u12s*X2s[0]-conj(u11s)*X2s[1])
					 +conj(X1su[1])*(u11s*X2s[0]-conj(u12s)*X2s[1]))).real();

			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-akappa*(
					conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					+conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					+conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
					+conj(X1su[1])*(-conj(u12s)*X2s[0]+u11s *X2s[1])).imag();

			const unsigned short gindex=mu*ndirac+(idirac>>1);
			const Complex_f gamval_c=gamval[gindex];
			//Rescaling gind by nc
			const unsigned short gind = gamin[gindex]<<1;	
			X2s[0]=X2[i+kvolHalo*(gind)]; X2s[1]=X2[i+kvolHalo*(1+gind)];
			X2su[0]=X2[uid+kvolHalo*(gind)]; X2su[1]=X2[uid+kvolHalo*(1+gind)];

			//If you are asked to rederive the force from Montvay and Munster you'll notice that it should be kappa*gamma
			//but below is only gamma. We rescaled gamma by kappa already when we defined it so that's where it has gone
			dSdpis[0]+=-(gamval_c*
					(conj(X1s[0])* (-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					 +conj(X1s[1])* (u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1su[0])* (-u12s *X2s[0] +conj(u11s)*X2s[1])
					 +conj(X1su[1])*(u11s *X2s[0] +conj(u12s)*X2s[1]))).imag();
			dSdpi[i+kvol*mu]=dSdpis[0];

			dSdpis[1]+=(gamval_c*
					(conj(X1s[0])* (-conj(u12s)*X2su[0] +conj(u11s)*X2su[1])
					 +conj(X1s[1])*(-u11s *X2su[0]-u12s *X2su[1])
					 +conj(X1su[0])* (u12s *X2s[0]+conj(u11s)*X2s[1])
					 +conj(X1su[1])* (-u11s *X2s[0]+conj(u12s)*X2s[1]))).real();
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-(gamval_c*
					(conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					 +conj(X1su[0])*(conj(u11s)*X2s[0]+u12s *X2s[1])
					 +conj(X1su[1])*(conj(u12s)*X2s[0]-u11s *X2s[1]))).imag();
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
}
__global__ void cuForce_t(double *dSdpi, Complex_f *u11t, Complex_f *u12t,Complex_f *X1, Complex_f *X2, Complex_f gamval[20],\
		float *dk4m, float *dk4p, unsigned int *iu, const unsigned short gamin[16],float akappa){
	const unsigned int gsize = gridDim.x*gridDim.y*gridDim.z;
	const unsigned int bsize = blockDim.x*blockDim.y*blockDim.z;
	const unsigned int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const unsigned int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const unsigned int gthreadId= blockId * bsize+bthreadId;

	const unsigned short mu=3;
	for(unsigned int i=gthreadId;i<kvol;i+=gsize*bsize){
		const unsigned int ind=i+kvolHalo*mu;
		const Complex_f u11s=u11t[ind];	const Complex_f u12s=u12t[ind];
		const float dk4ms=dk4m[i];	const float dk4ps=dk4p[i];
		//Up indices
		const unsigned int uid = iu[i+kvol*mu];
		//Similarly to Hdslash we always see idirac*nc so we do that here too.
		for(unsigned short idirac=0;idirac<ndirac*nc;idirac+=nc){
			Complex_f X1s[nc];	 Complex_f X1su[nc];
			Complex_f X2s[nc];	 Complex_f X2su[nc];

			X1s[0]=X1[i+kvolHalo*(idirac)]; X1s[1]=X1[i+kvolHalo*(1+idirac)];
			X1su[0]=X1[uid+kvolHalo*(idirac)]; X1su[1]=X1[uid+kvolHalo*(1+idirac)];
			X2s[0]=X2[i+kvolHalo*(idirac)]; X2s[1]=X2[i+kvolHalo*(1+idirac)];
			X2su[0]=X2[uid+kvolHalo*(idirac)]; X2su[1]=X2[uid+kvolHalo*(1+idirac)];

		//			Need to be double to avoid accumulation errors
			double dSdpis[3];
			dSdpis[0]=dSdpi[i+kvol*mu];
			//Multiplying by i and taking the real component is the same as taking the negative imaginary component
			//The positions of u11 and u12 might look a bit funky here. That's just because we've multiplied by the
			//generators by hand
			dSdpis[0]+=-(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(u11s *X2su[0]+u12s *X2su[1]))
					+dk4ps*(conj(X1su[0])*(+u12s*X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(-u11s*X2s[0]-conj(u12s)*X2s[1]))).imag();

			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(-u11s *X2su[0]-u12s *X2su[1]))
					+dk4ps*(conj(X1su[0])*(-u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*( u11s *X2s[0]-conj(u12s)*X2s[1]))).real();

			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-(dk4ms* (conj(X1s[0])* (u11s *X2su[0]+u12s *X2su[1])
						+conj(X1s[1])* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					+dk4ps*(conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
						+conj(X1su[1])* (-conj(u12s)*X2s[0]+u11s *X2s[1]))).imag();

			const unsigned short gindex=mu*ndirac+(idirac>>1);
			//Rescaling gind by nc
			const unsigned short gind = gamin[gindex]<<1;	
			X2s[0]=X2[i+kvolHalo*(gind)]; X2s[1]=X2[i+kvolHalo*(1+gind)];
			X2su[0]=X2[uid+kvolHalo*(gind)]; X2su[1]=X2[uid+kvolHalo*(1+gind)];

			dSdpis[0]+=-(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(u11s *X2su[0]+u12s *X2su[1]))
					-dk4ps*(conj(X1su[0])* (u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(-u11s *X2s[0]-conj(u12s)*X2s[1]))).imag();
			dSdpi[i+kvol*mu]=dSdpis[0];

			dSdpis[1]+=(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(-u11s*X2su[0]-u12s *X2su[1]))
					-dk4ps*(conj(X1su[0])*(-u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(u11s*X2s[0]-conj(u12s)*X2s[1]))).real();
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-(dk4ms*(conj(X1s[0])*(u11s*X2su[0] +u12s *X2su[1])
						+conj(X1s[1])* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					-dk4ps*(conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
						+conj(X1su[1])*(-conj(u12s)*X2s[0]+u11s *X2s[1]))).imag();
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
}

//Calling functions
void cuGauge_force(Complex_f *ut[2],double *dSdpi,float beta,unsigned int *iu,unsigned int *id,dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Gauge_force";
	int device=-1;
	cudaGetDevice(&device);
	Complex_f *Sigma[ndim][2], *ush[ndim][2];
	for(unsigned short i=0;i<ndim;i++){
#ifdef _DEBUG
		cudaMallocManaged((void **)&Sigma[i][0],kvol*sizeof(Complex_f),cudaMemAttachGlobal);
		cudaMallocManaged((void **)&Sigma[i][1],kvol*sizeof(Complex_f),cudaMemAttachGlobal);
		cudaMallocManaged((void **)&ush[i][0],kvolHalo*sizeof(Complex_f),cudaMemAttachGlobal);
		cudaMallocManaged((void **)&ush[i][1],kvolHalo*sizeof(Complex_f),cudaMemAttachGlobal);
#else
		cudaMallocAsync((void **)&Sigma[i][0],kvol*sizeof(Complex_f),streams[i]);
		cudaMallocAsync((void **)&Sigma[i][1],kvol*sizeof(Complex_f),streams[i]);
		cudaMallocAsync((void **)&ush[i][0],kvolHalo*sizeof(Complex_f),streams[i]);
		cudaMallocAsync((void **)&ush[i][1],kvolHalo*sizeof(Complex_f),streams[i]);
#endif
	}
	for(unsigned short mu=0; mu<ndim; mu++){
		cudaMemsetAsync(Sigma[mu][0],0, kvol*sizeof(Complex_f),streams[mu]);
		cudaMemsetAsync(Sigma[mu][1],0, kvol*sizeof(Complex_f),streams[mu]);
		for(unsigned short nu=0; nu<ndim; nu++)
			if(nu!=mu){
				//The @f$-\nu@f$ Staple
				Plus_staple<<<dimGrid,dimBlock,0,streams[mu]>>>(mu, nu, iu, Sigma[mu][0], Sigma[mu][1],ut[0],ut[1]);
				Gather<<<dimGrid,dimBlock,0,streams[mu]>>>(ush[mu][0], ut[0], kvol, id, nu);
				Gather<<<dimGrid,dimBlock,0,streams[mu]>>>(ush[mu][1], ut[1], kvol, id, nu);

#if(nproc>1)
				//Prefetch to the CPU for until we get NCCL working
				//cudaMemPrefetchAsync(ush[0], kvolHalo*sizeof(Complex_f),cudaCpuDeviceId,streams[0]);
				//cudaMemPrefetchAsync(ush[1], kvolHalo*sizeof(Complex_f),cudaCpuDeviceId,streams[1]);
				CHalo_swap_dir(ush[mu][0], 1, mu, DOWN); CHalo_swap_dir(ush[mu][1], 1, mu, DOWN);
				//cudaMemPrefetchAsync(ush[0]+kvol, halo*sizeof(Complex_f),device,streams[0]);
				//cudaMemPrefetchAsync(ush[1]+kvol, halo*sizeof(Complex_f),device,streams[1]);
#endif
				//Next up, the @f$-\nu@f$ staple
				Minus_staple<<<dimGrid,dimBlock,0,streams[mu]>>>(mu, nu, iu, id,Sigma[mu][0],Sigma[mu][1],\
						ush[mu][0],ush[mu][1],ut[0],ut[1]);
			}
		//Now get the gauge force acting in the @f$\mu@f$ direction
		cuGaugeForce<<<dimGrid,dimBlock,0,streams[mu]>>>(mu,Sigma[mu][0],Sigma[mu][1],dSdpi,ut[0],ut[1],beta);
	}
	for(unsigned short i=0;i<ndim;i++){
#ifdef _DEBUG
		cudaFree(Sigma[i][0]); cudaFree(Sigma[i][1]);
		cudaFree(ush[i][0]); cudaFree(ush[i][1]);
#else
		cudaFreeAsync(Sigma[i][0],streams[i]); cudaFreeAsync(Sigma[i][1],streams[i]);
		cudaFreeAsync(ush[i][0],streams[i]); cudaFreeAsync(ush[i][1],streams[i]);
#endif
	}
	cudaDeviceSynchronise();
}
void cuForce(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, \
		Complex_f gamval[20],float *dk[2],unsigned int *iu,const unsigned short gamin[16],\
		float akappa, dim3 dimGrid, dim3 dimBlock){
	const char *funcname = "Force";
	//X1=(M†M)^{1} Phi
	//	Transpose_z(X1,ndirac*nc,kvol); Transpose_z(X2,ndirac*nc,kvol);
	cudaDeviceSynchronise();
#pragma unroll
	for(unsigned short mu=0;mu<3;mu++){
		cuForce_s<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,ut[0],ut[1],X1,X2,gamval,iu,gamin,akappa,mu);
	}
	//Set stream for time direction
	unsigned short mu=3;
	cuForce_t<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,ut[0],ut[1],X1,X2,gamval,dk[0],dk[1],iu,gamin,akappa);
	cudaDeviceSynchronise();
	//	Transpose_z(X1,kvol,ndirac*nc); Transpose_z(X2,kvol,ndirac*nc);
	cudaDeviceSynchronise();
}
