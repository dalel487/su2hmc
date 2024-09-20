/*
 * Code for force calculations.
 * Requires multiply.cu to work
 */
#include	<matrices.h>
#include	<par_mpi.h>
#include	<su2hmc.h>
//Calling functions
void cuGauge_force(int mu, Complex_f *Sigma11, Complex_f *Sigma12, Complex_f *u11t,Complex_f *u12t,double *dSdpi,float beta,\
		dim3 dimGrid, dim3 dimBlock){
	const char *funcname = "Gauge_force";
	cuGaugeForce<<<dimGrid,dimBlock>>>(mu,Sigma11,Sigma12,dSdpi,u11t,u12t,beta);
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
void cuForce(double *dSdpi, Complex_f *u11t, Complex_f *u12t, Complex_f *X1, Complex_f *X2, \
		Complex_f *gamval,float *dk4m, float *dk4p,unsigned int *iu,int *gamin,\
		float akappa, dim3 dimGrid, dim3 dimBlock){
	const char *funcname = "Force";
	//X1=(M†M)^{1} Phi
	Transpose_U(iu,ndim,kvol,dimGrid,dimBlock);
	Transpose_d(dSdpi,nadj*ndim,kvol,dimGrid,dimBlock);
	Transpose_c(u11t,ndim,kvol,dimGrid,dimBlock); Transpose_c(u12t,ndim,kvol,dimGrid,dimBlock);
//	Transpose_z(X1,ndirac*nc,kvol,dimGrid,dimBlock); Transpose_z(X2,ndirac*nc,kvol,dimGrid,dimBlock);
	cudaDeviceSynchronise();
	for(int mu=0;mu<3;mu++){
		cuForce_s<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,u11t,u12t,X1,X2,gamval,iu,gamin,akappa,mu);
		//			cuForce_s1<<<dimGrid,dimBlock,0,streams[mu*nadj+1]>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa,idirac,mu);
		//			cuForce_s2<<<dimGrid,dimBlock,0,streams[mu*nadj+2]>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa,idirac,mu);
	}
	//Set stream for time direction
	int mu=3;
	cuForce_t<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa);
	cudaDeviceSynchronise();
//	Transpose_z(X1,kvol,ndirac*nc,dimGrid,dimBlock); Transpose_z(X2,kvol,ndirac*nc,dimGrid,dimBlock);
	Transpose_c(u11t,kvol,ndim,dimGrid,dimBlock); Transpose_c(u12t,kvol,ndim,dimGrid,dimBlock);
	Transpose_U(iu,kvol,ndim,dimGrid,dimBlock); 	
	Transpose_d(dSdpi,kvol,nadj*ndim,dimGrid,dimBlock);
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
		int uidm = iu[mu+ndim*i];
		int uidn = iu[nu+ndim*i];
		Complex_f	a11=u11t[uidm*ndim+nu]*conj(u11t[uidn*ndim+mu])+\
							 u12t[uidm*ndim+nu]*conj(u12t[uidn*ndim+mu]);
		Complex_f	a12=-u11t[uidm*ndim+nu]*u12t[uidn*ndim+mu]+\
							 u12t[uidm*ndim+nu]*u11t[uidn*ndim+mu];
		Sigma11[i]+=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
		Sigma12[i]+=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+nu];
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
		int uidm = iu[mu+ndim*i];
		int didn = id[nu+ndim*i];
		//uidm is correct here
		Complex_f a11=conj(u11sh[uidm])*conj(u11t[didn*ndim+mu])-\
						  u12sh[uidm]*conj(u12t[didn*ndim+mu]);
		Complex_f a12=-conj(u11sh[uidm])*u12t[didn*ndim+mu]-\
						  u12sh[uidm]*u11t[didn*ndim+mu];
		Sigma11[i]+=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
		Sigma12[i]+=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+nu]);
	}
}
__global__ void cuGaugeForce(int mu, Complex_f *Sigma11, Complex_f *Sigma12,double* dSdpi,Complex_f *u11t, Complex_f *u12t, float beta){
	const char *funcname = "cuGaugeForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		Complex_f a11 = u11t[i*ndim+mu]*Sigma12[i]+u12t[i*ndim+mu]*conj(Sigma11[i]);
		Complex_f a12 = u11t[i*ndim+mu]*Sigma11[i]+conj(u12t[i*ndim+mu])*Sigma12[i];
		//Not worth splitting into different streams, before we get ideas...
		dSdpi[(i*nadj)*ndim+mu]=beta*a11.imag();
		dSdpi[(i*nadj+1)*ndim+mu]=beta*a11.real();
		dSdpi[(i*nadj+2)*ndim+mu]=beta*a12.imag();
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
		const Complex_f u11s=u11t[i+kvol*mu];	const Complex_f u12s=u12t[i+kvol*mu];
		//const int uid = iu[mu+ndim*i];
		const int uid = iu[mu*kvol+i];
		for(int idirac=0;idirac<ndirac;idirac++){
			Complex_f X1s[nc];	 Complex_f X1su[nc];
			Complex_f X2s[nc];	 Complex_f X2su[nc];
			//	phi[(i*ndirac+idirac)*nc]+=
			//phi_s[idirac*nc+c]=phi[i+kvol*(c+nc*idirac)];
			//X1s[0]=X1[(i*ndirac+idirac)*nc];	X1s[1]=X1[(i*ndirac+idirac)*nc+1];
			//X1su[0]=X1[(uid*ndirac+idirac)*nc];	X1su[1]=X1[(uid*ndirac+idirac)*nc+1];
			//X2s[0]=X2[(i*ndirac+idirac)*nc];	X2s[1]=X2[(i*ndirac+idirac)*nc+1];
			//X2su[0]=X2[(uid*ndirac+idirac)*nc];	X2su[1]=X2[(uid*ndirac+idirac)*nc+1];
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
					+conj(X1su[0])*(u12s*X2s[0]-conj(u11s)*X2s[1])
					+conj(X1s[1])*(u11s*X2su[0]+u12s*X2su[1])
					+conj(X1su[1])*(-u11s*X2s[0]-conj(u12s)*X2s[1])).imag();

			//dSdpis[1]=dSdpi[(i*nadj+1)*ndim+mu];
			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=akappa*(
					(conj(X1s[0])* (-conj(u12s)*X2su[0] +conj(u11s)*X2su[1])
					 +conj(X1su[0])* (-u12s *X2s[0] -conj(u11s)*X2s[1])
					 +conj(X1s[1])* (-u11s *X2su[0]-u12s *X2su[1])
					 +conj(X1su[1])* (u11s *X2s[0]-conj(u12s)*X2s[1]))).real();

			//dSdpis[2]=dSdpi[(i*nadj+2)*ndim+mu];
			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-akappa*(
					conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					+conj(X1su[0])*(-conj(u11s)*X2s[0]-u12s *X2s[1])
					+conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					+conj(X1su[1])*(-conj(u12s)*X2s[0]+u11s *X2s[1])).imag();

			const int igork1 = gamin[mu*ndirac+idirac];	
			//X2s[0]=X2[(i*ndirac+igork1)*nc];	X2s[1]=X2[(i*ndirac+igork1)*nc+1];
			//X2su[0]=X2[(uid*ndirac+igork1)*nc];	X2su[1]=X2[(uid*ndirac+igork1)*nc+1];
			X2s[0]=X2[i+kvol*(nc*igork1)]; X2s[1]=X2[i+kvol*(1+nc*igork1)];
			X2su[0]=X2[uid+kvol*(nc*igork1)]; X2su[1]=X2[uid+kvol*(1+nc*igork1)];

			dSdpis[0]+=-(gamval[mu*ndirac+idirac]*
					(conj(X1s[0])* (-conj(u12s)*X2su[0]+conj(u11s)*X2su[1])
					 +conj(X1su[0])* (-u12s *X2s[0] +conj(u11s)*X2s[1])
					 +conj(X1s[1])* (u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1su[1])*(u11s *X2s[0] +conj(u12s)*X2s[1]))).imag();
			//dSdpi[(i*nadj)*ndim+mu]=dSdpis[0];
			dSdpi[i+kvol*(mu)]=dSdpis[0];

			dSdpis[1]+=(gamval[mu*ndirac+idirac]*
					(conj(X1s[0])* (-conj(u12s)*X2su[0] +conj(u11s)*X2su[1])
					 +conj(X1su[0])* (u12s *X2s[0]+conj(u11s)*X2s[1])
					 +conj(X1s[1])*(-u11s *X2su[0]-u12s *X2su[1])
					 +conj(X1su[1])* (-u11s *X2s[0]+conj(u12s)*X2s[1]))).real();
			//dSdpi[(i*nadj+1)*ndim+mu]=dSdpis[1];
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-(gamval[mu*ndirac+idirac]*
					(conj(X1s[0])*(u11s *X2su[0]+u12s *X2su[1])
					 +conj(X1su[0])*(conj(u11s)*X2s[0]+u12s *X2s[1])
					 +conj(X1s[1])*(conj(u12s)*X2su[0]-conj(u11s)*X2su[1])
					 +conj(X1su[1])*(conj(u12s)*X2s[0]-u11s *X2s[1]))).imag();
			//dSdpi[(i*nadj+2)*ndim+mu]=dSdpis[2];
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
}
__global__ void cuForce_t(double *dSdpi, Complex_f *u11t, Complex_f *u12t, Complex_f *X1, Complex_f *X2, Complex_f *gamval,\
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
			dSdpis[0]+=-(conj(X1s[0])* (dk4ms*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1]))
					 +conj(X1su[0])* (dk4ps*(+u12s *X2s[0]-conj(u11s)*X2s[1]))
					 +conj(X1s[1])*(dk4ms*(u11s *X2su[0]+u12s *X2su[1]))
					 +conj(X1su[1])*(dk4ps*(-u11s *X2s[0]-conj(u12s)*X2s[1]))).imag();
		//	dSdpis[1]=dSdpi[(i*nadj+1)*ndim+mu];
			dSdpis[1]=dSdpi[i+kvol*(ndim+mu)];
			dSdpis[1]+=(conj(X1s[0])*(dk4ms*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1]))
					+conj(X1su[0])*(dk4ps*(-u12s *X2s[0]-conj(u11s)*X2s[1]))
					+conj(X1s[1])*(dk4ms*(-u11s *X2su[0]-u12s *X2su[1]))
					+conj(X1su[1])*(dk4ps*( u11s *X2s[0]-conj(u12s)*X2s[1]))).real();

			//dSdpis[2]=dSdpi[(i*nadj+2)*ndim+mu];
			dSdpis[2]=dSdpi[i+kvol*(2*ndim+mu)];
			dSdpis[2]+=-(conj(X1s[0])* (dk4ms* (u11s *X2su[0]+u12s *X2su[1]))
					 +conj(X1su[0])*(dk4ps*(-conj(u11s)*X2s[0]-u12s *X2s[1]))
					 +conj(X1s[1])* (dk4ms* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					 +conj(X1su[1])* (dk4ps*(-conj(u12s)*X2s[0]+u11s *X2s[1]))).imag();

			const int igork1 = gamin[mu*ndirac+idirac];	
			//X2s[0]=X2[(i*ndirac+igork1)*nc];	X2s[1]=X2[(i*ndirac+igork1)*nc+1];
			//X2su[0]=X2[(uid*ndirac+igork1)*nc];	X2su[1]=X2[(uid*ndirac+igork1)*nc+1];
			X2s[0]=X2[i+kvol*(nc*igork1)]; X2s[1]=X2[i+kvol*(1+nc*igork1)];
			X2su[0]=X2[uid+kvol*(nc*igork1)]; X2su[1]=X2[uid+kvol*(1+nc*igork1)];

			dSdpis[0]+=-(conj(X1s[0])*(dk4ms*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1]))
					 +conj(X1su[0])* (-dk4ps*(u12s *X2s[0]-conj(u11s)*X2s[1]))
					 +conj(X1s[1])*(dk4ms*(u11s *X2su[0]+u12s *X2su[1]))
					 +conj(X1su[1])*(-dk4ps*(-u11s *X2s[0]-conj(u12s)*X2s[1]))).imag();
			//dSdpi[(i*nadj)*ndim+mu]=dSdpis[0];
			dSdpi[i+kvol*(mu)]=dSdpis[0];

			dSdpis[1]+=(
					(conj(X1s[0])*(dk4ms*(-conj(u12s)*X2su[0]+conj(u11s)*X2su[1]))
					 +conj(X1su[0])*(-dk4ps*(-u12s *X2s[0]-conj(u11s)*X2s[1]))
					 +conj(X1s[1])*(dk4ms*(-u11s*X2su[0]-u12s *X2su[1]))
					 +conj(X1su[1])*(-dk4ps*(u11s*X2s[0]-conj(u12s)*X2s[1])))).real();
			//dSdpi[(i*nadj+1)*ndim+mu]=dSdpis[1];
			dSdpi[i+kvol*(ndim+mu)]=dSdpis[1];

			dSdpis[2]+=-(conj(X1s[0])*(dk4ms*(u11s*X2su[0] +u12s *X2su[1]))
					 +conj(X1su[0])*(-dk4ps*(-conj(u11s)*X2s[0]-u12s *X2s[1]))
					 +conj(X1s[1])*(dk4ms* (conj(u12s)*X2su[0]-conj(u11s)*X2su[1]))
					 +conj(X1su[1])*(-dk4ps*(-conj(u12s)*X2s[0]+u11s *X2s[1]))).imag();
			//dSdpi[(i*nadj+2)*ndim+mu]=dSdpis[2];
			dSdpi[i+kvol*(2*ndim+mu)]=dSdpis[2];
		}
	}
}
