/*
 * Code for force calculations.
 * Requires multiply.cu to work
 */
#include	<matrices.h>
#include	<par_mpi.h>
#include	<su2hmc.h>
//Calling functions
void cuGauge_force(Complex_f *ut[2],double *dSdpi,float beta,unsigned int *iu,unsigned int *id,dim3 dimGrid, dim3 dimBlock){
	const char funcname[] = "Gauge_force";
	int device=-1;
	cudaGetDevice(&device);
	Complex_f *Sigma[2], *ush[2];
#ifdef _DEBUG
	cudaMallocManaged((void **)&Sigma[0],kvol*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&Sigma[1],kvol*sizeof(Complex_f),cudaMemAttachGlobal);
#else
	cudaMallocAsync((void **)&Sigma[0],kvol*sizeof(Complex_f),streams[0]);
	cudaMallocAsync((void **)&Sigma[1],kvol*sizeof(Complex_f),streams[1]);
#endif
	cudaMallocManaged((void **)&ush[0],(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&ush[1],(kvol+halo)*sizeof(Complex_f),cudaMemAttachGlobal);
	for(int mu=0; mu<ndim; mu++){
		cudaMemset(Sigma[0],0, kvol*sizeof(Complex_f));
		cudaMemset(Sigma[1],0, kvol*sizeof(Complex_f));
		for(int nu=0; nu<ndim; nu++)
			if(nu!=mu){
				//The @f$-\nu@f$ Staple
				cuPlus_staple(mu,nu,iu,Sigma[0],Sigma[1],ut[0],ut[1],dimGrid,dimBlock);
				C_gather(ush[0], ut[0], kvol, id, nu);
				C_gather(ush[1], ut[1], kvol, id, nu);

				//Prefetch to the CPU for until we get NCCL working
				cudaMemPrefetchAsync(ush[0], kvol*sizeof(Complex_f),cudaCpuDeviceId,streams[0]);
				cudaMemPrefetchAsync(ush[1], kvol*sizeof(Complex_f),cudaCpuDeviceId,streams[1]);
#if(nproc>1)
				CHalo_swap_dir(ush[0], 1, mu, DOWN); CHalo_swap_dir(ush[1], 1, mu, DOWN);
				cudaMemPrefetchAsync(ush[0]+kvol, halo*sizeof(Complex_f),device,streams[0]);
				cudaMemPrefetchAsync(ush[1]+kvol, halo*sizeof(Complex_f),device,streams[1]);
#endif
				//Next up, the @f$-\nu@f$ staple
				cuMinus_staple(mu,nu,iu,id,Sigma[0],Sigma[1],ush[0],ush[1],ut[0],ut[1],dimGrid,dimBlock);
			}
		//Now get the gauge force acting in the @f$\mu@f$ direction
		cuGaugeForce<<<dimGrid,dimBlock>>>(mu,Sigma[0],Sigma[1],dSdpi,ut[0],ut[1],beta);
		cudaDeviceSynchronise();
	}
#ifdef _DEBUG
	cudaFree(Sigma[0]); cudaFree(Sigma[1]);
#else
	cudaFreeAsync(Sigma[0],streams[0]); cudaFreeAsync(Sigma[1],streams[1]);
#endif
	cudaFree(ush[0]); cudaFree(ush[1]);
}
void cuPlus_staple(int mu, int nu, unsigned int *iu, Complex_f *Sigma11, Complex_f *Sigma12, Complex_f *u11t, Complex_f *u12t,\
		dim3 dimGrid, dim3 dimBlock){
	const char *funcname="Plus_staple";
	Plus_staple<<<dimGrid,dimBlock>>>(mu, nu, iu, Sigma11, Sigma12,u11t,u12t);
}
void cuMinus_staple(int mu, int nu, unsigned int *iu, unsigned int *id, Complex_f *Sigma11, Complex_f *Sigma12,\
		Complex_f *u11sh, Complex_f *u12sh,Complex_f *u11t, Complex_f *u12t,dim3 dimGrid, dim3 dimBlock){
	const char *funcname="Minus_staple";
	Minus_staple<<<dimGrid,dimBlock>>>(mu, nu, iu, id,Sigma11,Sigma12,u11sh,u12sh,u11t,u12t);
}
void cuForce(double *dSdpi, Complex_f *ut[2], Complex_f *X1, Complex_f *X2, \
		Complex_f *gamval,float *dk[2],unsigned int *iu,int *gamin,\
		float akappa, dim3 dimGrid, dim3 dimBlock){
	const char *funcname = "Force";
	//X1=(M†M)^{1} Phi
	//	Transpose_z(X1,ndirac*nc,kvol); Transpose_z(X2,ndirac*nc,kvol);
	cudaDeviceSynchronise();
#pragma unroll
	for(int mu=0;mu<3;mu++){
		cuForce_s<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,ut[0],ut[1],X1,X2,gamval,iu,gamin,akappa,mu);
		//			cuForce_s1<<<dimGrid,dimBlock,0,streams[mu*nadj+1]>>>(dSdpi,ut[0],ut[1],X1,X2,gamval,dk[1],dk[1],iu,gamin,akappa,idirac,mu);
		//			cuForce_s2<<<dimGrid,dimBlock,0,streams[mu*nadj+2]>>>(dSdpi,ut[0],ut[1],X1,X2,gamval,dk[1],dk[1],iu,gamin,akappa,idirac,mu);
	}
	//Set stream for time direction
	int mu=3;
	cuForce_t<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,ut[0],ut[1],X1,X2,gamval,dk[0],dk[1],iu,gamin,akappa);
	cudaDeviceSynchronise();
	//	Transpose_z(X1,kvol,ndirac*nc); Transpose_z(X2,kvol,ndirac*nc);
	cudaDeviceSynchronise();
}

//CUDA Kernels
//TODO: Split cuForce into seperateable streams. Twelve in total I Believe?
//A stream for each nadj index,dirac index and each μ (ndim) value
//3*4*4=36 streams total... Pass dirac and μ spatial indices as arguments
__global__ void Plus_staple(int mu, int nu,unsigned int *iu, Complex_f *Sigma11, Complex_f *Sigma12, Complex_f *u11t, Complex_f *u12t){
	const char *funcname = "Plus_staple";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		int uidm = iu[mu*kvol+i];
		int uidn = iu[nu*kvol+i];
		Complex_f	a11=u11t[uidm+kvol*nu]*conj(u11t[uidn+kvol*mu])+\
							 u12t[uidm+kvol*nu]*conj(u12t[uidn+kvol*mu]);
		Complex_f	a12=-u11t[uidm+kvol*nu]*u12t[uidn+kvol*mu]+\
							 u12t[uidm+kvol*nu]*u11t[uidn+kvol*mu];
		Sigma11[i]+=a11*conj(u11t[i+kvol*nu])+a12*conj(u12t[i+kvol*nu]);
		Sigma12[i]+=-a11*u12t[i+kvol*nu]+a12*u11t[i+kvol*nu];
	}
}
__global__ void Minus_staple(int mu,int nu,unsigned int *iu,unsigned int *id, Complex_f *Sigma11, Complex_f *Sigma12,\
		Complex_f *u11sh, Complex_f *u12sh, Complex_f *u11t, Complex_f *u12t){
	const char *funcname = "Minus_staple";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		int uidm = iu[mu*kvol+i];
		int didn = id[nu*kvol+i];
		//uidm is correct here
		Complex_f a11=conj(u11sh[uidm])*conj(u11t[didn+kvol*mu])-\
						  u12sh[uidm]*conj(u12t[didn+kvol*mu]);
		Complex_f a12=-conj(u11sh[uidm])*u12t[didn+kvol*mu]-\
						  u12sh[uidm]*u11t[didn+kvol*mu];
		Sigma11[i]+=a11*u11t[didn+kvol*nu]-a12*conj(u12t[didn+kvol*nu]);
		Sigma12[i]+=a11*u12t[didn+kvol*nu]+a12*conj(u11t[didn+kvol*nu]);
	}
}
__global__ void cuGaugeForce(int mu, Complex_f *Sigma11, Complex_f *Sigma12,double* dSdpi,Complex_f *u11t, Complex_f *u12t, float beta){
	const char *funcname = "cuGaugeForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		Complex_f a11 = u11t[i+kvol*mu]*Sigma12[i]+u12t[i+kvol*mu]*conj(Sigma11[i]);
		Complex_f a12 = u11t[i+kvol*mu]*Sigma11[i]+conj(u12t[i+kvol*mu])*Sigma12[i];
		//Not worth splitting into different streams, before we get ideas...
		dSdpi[i+kvol*(mu)]=beta*a11.imag();
		dSdpi[i+kvol*(1*ndim+mu)]=beta*a11.real();
		dSdpi[i+kvol*(2*ndim+mu)]=beta*a12.imag();
	}
}

__global__ void cuForce_s(double *dSdpi, Complex_f *u11t, Complex_f *u12t, Complex_f *X1, Complex_f *X2, Complex_f *gamval,\
		unsigned int *iu, int *gamin,float akappa, int mu){
	const char *funcname = "cuForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Complex_f u11s=u11t[i*ndim+mu];	Complex_f u12s=u12t[i*ndim+mu];
		const Complex_f u11s=u11t[i+kvol*mu];
		const Complex_f u12s=u12t[i+kvol*mu];
		//const int uid = iu[mu+ndim*i];
		const int uid = iu[mu*kvol+i];
		for(int idirac=0;idirac<ndirac;idirac++){
			Complex_f X1s[nc];	 Complex_f X1su[nc];
			Complex_f X2s[nc];	 Complex_f X2su[nc];

			X1s[0]=X1[i+kvol*(nc*idirac)]; X1s[1]=X1[i+kvol*(1+nc*idirac)];
			X1su[0]=X1[uid+kvol*(nc*idirac)]; X1su[1]=X1[uid+kvol*(1+nc*idirac)];
			X2s[0]=X2[i+kvol*(nc*idirac)]; X2s[1]=X2[i+kvol*(1+nc*idirac)];
			X2su[0]=X2[uid+kvol*(nc*idirac)]; X2su[1]=X2[uid+kvol*(1+nc*idirac)];

			float dSdpis[3];
			//dSdpis[0]=dSdpi[(i*nadj)*ndim+mu];
			dSdpis[0]=dSdpi[i+kvol*(mu)];
			//Multiplying by i and taking the real component is the same as taking the negative imaginary component
			dSdpis[0]+=-akappa*(
					conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					+conj(X1s[1])*(u11s*X2su[0]+u12s*X2su[1])
					+conj(X1su[0])*(u12s*X2s[0]-conj(u11s)*X2s[1])
					+conj(X1su[1])*(-u11s*X2s[0]-conj(u12s)*X2s[1])).imag();

			//dSdpis[1]=dSdpi[(i*nadj+1)*ndim+mu];
			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=akappa*(
					(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					 +conj(X1s[1])*(-u11s*X2su[0]-u12s*X2su[1])
					 +conj(X1su[0])*(-u12s*X2s[0]-conj(u11s)*X2s[1])
					 +conj(X1su[1])*(u11s*X2s[0]-conj(u12s)*X2s[1]))).real();

			//dSdpis[2]=dSdpi[(i*nadj+2)*ndim+mu];
			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-akappa*(
					conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					+conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					+conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
					+conj(X1su[1])*(-conj(u12s)*X2s[0]+u11s *X2s[1])).imag();

			const int igork1 = gamin[mu*ndirac+idirac];	
			//X2s[0]=X2[(i*ndirac+igork1)*nc];	X2s[1]=X2[(i*ndirac+igork1)*nc+1];
			//X2su[0]=X2[(uid*ndirac+igork1)*nc];	X2su[1]=X2[(uid*ndirac+igork1)*nc+1];
			X2s[0]=X2[i+kvol*(nc*igork1)]; X2s[1]=X2[i+kvol*(1+nc*igork1)];
			X2su[0]=X2[uid+kvol*(nc*igork1)]; X2su[1]=X2[uid+kvol*(1+nc*igork1)];

			//If you are asked to rederive the force from Montvay and Munster you'll notice that it should be kappa*gamma
			//but below is only gamma. We rescaled gamma by kappa already when we defined it so that's where it has gone
			dSdpis[0]+=-(gamval[mu*ndirac+idirac]*
					(conj(X1s[0])* (-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					 +conj(X1s[1])* (u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1su[0])* (-u12s *X2s[0] +conj(u11s)*X2s[1])
					 +conj(X1su[1])*(u11s *X2s[0] +conj(u12s)*X2s[1]))).imag();
			//dSdpi[(i*nadj)*ndim+mu]=dSdpis[0];
			dSdpi[i+kvol*(mu)]=dSdpis[0];

			dSdpis[1]+=(gamval[mu*ndirac+idirac]*
					(conj(X1s[0])* (-conj(u12s)*X2su[0] +conj(u11s)*X2su[1])
					 +conj(X1s[1])*(-u11s *X2su[0]-u12s *X2su[1])
					 +conj(X1su[0])* (u12s *X2s[0]+conj(u11s)*X2s[1])
					 +conj(X1su[1])* (-u11s *X2s[0]+conj(u12s)*X2s[1]))).real();
			//dSdpi[(i*nadj+1)*ndim+mu]=dSdpis[1];
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-(gamval[mu*ndirac+idirac]*
					(conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					 +conj(X1su[0])*(conj(u11s)*X2s[0]+u12s *X2s[1])
					 +conj(X1su[1])*(conj(u12s)*X2s[0]-u11s *X2s[1]))).imag();
			//dSdpi[(i*nadj+2)*ndim+mu]=dSdpis[2];
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
}
__global__ void cuForce_t(double *dSdpi, Complex_f *u11t, Complex_f *u12t,Complex_f *X1, Complex_f *X2, Complex_f *gamval,\
		float *dk4m, float *dk4p, unsigned int *iu, int *gamin,float akappa){
	const char *funcname = "cuForce";
	//Up indices
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	const int mu=3;
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Up indices
		//const int uid = iu[mu+ndim*i];
		const int uid = iu[mu*kvol+i];
		//	Complex_f u11s=u11t[i*ndim+mu];	Complex_f u12s=u12t[i*ndim+mu];
		const Complex_f u11s=u11t[i+kvol*mu];	const Complex_f u12s=u12t[i+kvol*mu];
		//TODO: The only diffrence with these is that the sign flips for the temporal components
		//			Can we figure out a way of doing this without having to read in a large array. 
		//			Will result in a conditional inside a CUDA loop. If i>kvol3
		const float dk4ms=dk4m[i];	const float dk4ps=dk4p[i];

		for(int idirac=0;idirac<ndirac;idirac++){
			Complex X1s[nc];	 Complex X1su[nc];
			Complex_f X2s[nc];	 Complex_f X2su[nc];
			//X1s[0]=X1[(i*ndirac+idirac)*nc];	X1s[1]=X1[(i*ndirac+idirac)*nc+1];
			//X1su[0]=X1[(uid*ndirac+idirac)*nc];	X1su[1]=X1[(uid*ndirac+idirac)*nc+1];
			//X2s[0]=X2[(i*ndirac+idirac)*nc];	X2s[1]=X2[(i*ndirac+idirac)*nc+1];
			//X2su[0]=X2[(uid*ndirac+idirac)*nc];	X2su[1]=X2[(uid*ndirac+idirac)*nc+1];
			X1s[0]=X1[i+kvol*(nc*idirac)]; X1s[1]=X1[i+kvol*(1+nc*idirac)];
			X1su[0]=X1[uid+kvol*(nc*idirac)]; X1su[1]=X1[uid+kvol*(1+nc*idirac)];
			X2s[0]=X2[i+kvol*(nc*idirac)]; X2s[1]=X2[i+kvol*(1+nc*idirac)];
			X2su[0]=X2[uid+kvol*(nc*idirac)]; X2su[1]=X2[uid+kvol*(1+nc*idirac)];

			float dSdpis[3];
			//	dSdpis[0]=dSdpi[(i*nadj)*ndim+mu];
			dSdpis[0]=dSdpi[i+kvol*(mu)];
			dSdpis[0]+=-(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(u11s *X2su[0]+u12s *X2su[1]))
					+dk4ps*(conj(X1su[0])*(+u12s*X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(-u11s*X2s[0]-conj(u12s)*X2s[1]))).imag();

			//	dSdpis[1]=dSdpi[(i*nadj+1)*ndim+mu];
			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(-u11s *X2su[0]-u12s *X2su[1]))
					+dk4ps*(conj(X1su[0])*(-u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*( u11s *X2s[0]-conj(u12s)*X2s[1]))).real();

			//dSdpis[2]=dSdpi[(i*nadj+2)*ndim+mu];
			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-(dk4ms* (conj(X1s[0])* (u11s *X2su[0]+u12s *X2su[1])
						+conj(X1s[1])* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					+dk4ps*(conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
						+conj(X1su[1])* (-conj(u12s)*X2s[0]+u11s *X2s[1]))).imag();

			const int igork1 = gamin[mu*ndirac+idirac];	
			//X2s[0]=X2[(i*ndirac+igork1)*nc];	X2s[1]=X2[(i*ndirac+igork1)*nc+1];
			//X2su[0]=X2[(uid*ndirac+igork1)*nc];	X2su[1]=X2[(uid*ndirac+igork1)*nc+1];
			X2s[0]=X2[i+kvol*(nc*igork1)]; X2s[1]=X2[i+kvol*(1+nc*igork1)];
			X2su[0]=X2[uid+kvol*(nc*igork1)]; X2su[1]=X2[uid+kvol*(1+nc*igork1)];

			dSdpis[0]+=-(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(u11s *X2su[0]+u12s *X2su[1]))
					-dk4ps*(conj(X1su[0])* (u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(-u11s *X2s[0]-conj(u12s)*X2s[1]))).imag();
			//dSdpi[(i*nadj)*ndim+mu]=dSdpis[0];
			dSdpi[i+kvol*(mu)]=dSdpis[0];

			dSdpis[1]+=(dk4ms*(conj(X1s[0])*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
						+conj(X1s[1])*(-u11s*X2su[0]-u12s *X2su[1]))
					-dk4ps*(conj(X1su[0])*(-u12s *X2s[0]-conj(u11s)*X2s[1])
						+conj(X1su[1])*(u11s*X2s[0]-conj(u12s)*X2s[1]))).real();
			//dSdpi[(i*nadj+1)*ndim+mu]=dSdpis[1];
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-(dk4ms*(conj(X1s[0])*(u11s*X2su[0] +u12s *X2su[1])
						+conj(X1s[1])* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					-dk4ps*(conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
						+conj(X1su[1])*(-conj(u12s)*X2s[0]+u11s *X2s[1]))).imag();
			//dSdpi[(i*nadj+2)*ndim+mu]=dSdpis[2];
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
}
