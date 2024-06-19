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
void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, \
		Complex *gamval,double *dk4m, double *dk4p,unsigned int *iu,int *gamin,\
		float akappa, dim3 dimGrid, dim3 dimBlock){
	const char *funcname = "Force";
	//X1=(M†M)^{1} Phi
	//cuForce<<<dimGrid,dimBlock>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa);
	for(int mu=0;mu<3;mu++){
		cuForce_s<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa,mu);
		//			cuForce_s1<<<dimGrid,dimBlock,0,streams[mu*nadj+1]>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa,idirac,mu);
		//			cuForce_s2<<<dimGrid,dimBlock,0,streams[mu*nadj+2]>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa,idirac,mu);
	}
	//Set stream for time direction
	int mu=3;
	cuForce_t<<<dimGrid,dimBlock,0,streams[mu]>>>(dSdpi,u11t,u12t,X1,X2,gamval,dk4m,dk4p,iu,gamin,akappa);
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

__global__ void cuForce_s(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa, int mu){
	const char *funcname = "cuForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	__shared__ Complex_f u11s[128];	__shared__ Complex_f u12s[128];
	__shared__ Complex X1s[128*nc];	__shared__ Complex X1su[128*nc];
	__shared__ Complex_f X2s[128*nc];	__shared__ Complex_f X2su[128*nc];
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Up indices
		int uid = iu[mu+ndim*i];
		u11s[bthreadId]=u11t[i*ndim+mu];	u12s[bthreadId]=u12t[i*ndim+mu];

		for(int idirac=0;idirac<ndirac;idirac++){
			X1s[bthreadId]=X1[(i*ndirac+idirac)*nc];	X1s[bthreadId+1]=X1[(i*ndirac+idirac)*nc+1];
			X1su[bthreadId]=X1[(uid*ndirac+idirac)*nc];	X1su[bthreadId+1]=X1[(uid*ndirac+idirac)*nc+1];

			X2s[bthreadId]=X2[(i*ndirac+idirac)*nc];	X2s[bthreadId+1]=X2[(i*ndirac+idirac)*nc+1];
			X2su[bthreadId]=X2[(uid*ndirac+idirac)*nc];	X2su[bthreadId+1]=X2[(uid*ndirac+idirac)*nc+1];

			dSdpi[(i*nadj)*ndim+mu]+=akappa*(I*
					(conj(X1s[bthreadId])*
					 (-conj(u12s[bthreadId])*X2su[bthreadId]
					  +conj(u11s[bthreadId])*X2su[bthreadId+1])
					 +conj(X1su[bthreadId])*
					 ( u12s[bthreadId] *X2s[bthreadId]
						-conj(u11s[bthreadId])*X2s[bthreadId+1])
					 +conj(X1s[bthreadId+1])*
					 (u11s[bthreadId] *X2su[bthreadId]
					  +u12s[bthreadId] *X2su[bthreadId+1])
					 +conj(X1su[bthreadId+1])*
					 (-u11s[bthreadId] *X2s[bthreadId]
					  -conj(u12s[bthreadId])*X2s[bthreadId+1]))).real();

			dSdpi[(i*nadj+1)*ndim+mu]+=akappa*(
					(conj(X1s[bthreadId])*
					 (-conj(u12s[bthreadId])*X2su[bthreadId]
					  +conj(u11s[bthreadId])*X2su[bthreadId+1])
					 +conj(X1su[bthreadId])*
					 (-u12s[bthreadId] *X2s[bthreadId]
					  -conj(u11s[bthreadId])*X2s[bthreadId+1])
					 +conj(X1s[bthreadId+1])*
					 (-u11s[bthreadId] *X2su[bthreadId]
					  -u12s[bthreadId] *X2su[bthreadId+1])
					 +conj(X1su[bthreadId+1])*
					 (u11s[bthreadId] *X2s[bthreadId]
					  -conj(u12s[bthreadId])*X2s[bthreadId+1]))).real();

			dSdpi[(i*nadj+2)*ndim+mu]+=akappa*(I*
					(conj(X1s[bthreadId])*
					 (u11s[bthreadId] *X2su[bthreadId]
					  +u12s[bthreadId] *X2su[bthreadId+1])
					 +conj(X1su[bthreadId])*
					 (-conj(u11s[bthreadId])*X2s[bthreadId]
					  -u12s[bthreadId] *X2s[bthreadId+1])
					 +conj(X1s[bthreadId+1])*
					 (conj(u12s[bthreadId])*X2su[bthreadId]
					  -conj(u11s[bthreadId])*X2su[bthreadId+1])
					 +conj(X1su[bthreadId+1])*
					 (-conj(u12s[bthreadId])*X2s[bthreadId]
					  +u11s[bthreadId] *X2s[bthreadId+1]))).real();

			const int igork1 = gamin[mu*ndirac+idirac];	
			X2s[bthreadId]=X2[(i*ndirac+igork1)*nc];	X2s[bthreadId+1]=X2[(i*ndirac+igork1)*nc+1];
			X2su[bthreadId]=X2[(uid*ndirac+igork1)*nc];	X2su[bthreadId+1]=X2[(uid*ndirac+igork1)*nc+1];

			dSdpi[(i*nadj)*ndim+mu]+=(I*gamval[mu*ndirac+idirac]*
					(conj(X1s[bthreadId])*
					 (-conj(u12s[bthreadId])*X2su[bthreadId]
					  +conj(u11s[bthreadId])*X2su[bthreadId+1])
					 +conj(X1su[bthreadId])*
					 (-u12s[bthreadId] *X2s[bthreadId]
					  +conj(u11s[bthreadId])*X2s[bthreadId+1])
					 +conj(X1s[bthreadId+1])*
					 (u11s[bthreadId] *X2su[bthreadId]
					  +u12s[bthreadId] *X2su[bthreadId+1])
					 +conj(X1su[bthreadId+1])*
					 (u11s[bthreadId] *X2s[bthreadId]
					  +conj(u12s[bthreadId])*X2s[bthreadId+1]))).real();

			dSdpi[(i*nadj+1)*ndim+mu]+=(gamval[mu*ndirac+idirac]*
					(conj(X1s[bthreadId])*
					 (-conj(u12s[bthreadId])*X2su[bthreadId]
					  +conj(u11s[bthreadId])*X2su[bthreadId+1])
					 +conj(X1su[bthreadId])*
					 (u12s[bthreadId] *X2s[bthreadId]
					  +conj(u11s[bthreadId])*X2s[bthreadId+1])
					 +conj(X1s[bthreadId+1])*
					 (-u11s[bthreadId] *X2su[bthreadId]
					  -u12s[bthreadId] *X2su[bthreadId+1])
					 +conj(X1su[bthreadId+1])*
					 (-u11s[bthreadId] *X2s[bthreadId]
					  +conj(u12s[bthreadId])*X2s[bthreadId+1]))).real();

			dSdpi[(i*nadj+2)*ndim+mu]+=(I*gamval[mu*ndirac+idirac]*
					(conj(X1s[bthreadId])*
					 (u11s[bthreadId] *X2su[bthreadId]
					  +u12s[bthreadId] *X2su[bthreadId+1])
					 +conj(X1su[bthreadId])*
					 (conj(u11s[bthreadId])*X2s[bthreadId]
					  +u12s[bthreadId] *X2s[bthreadId+1])
					 +conj(X1s[bthreadId+1])*
					 (conj(u12s[bthreadId])*X2su[bthreadId]
					  -conj(u11s[bthreadId])*X2su[bthreadId+1])
					 +conj(X1su[bthreadId+1])*
					 (conj(u12s[bthreadId])*X2s[bthreadId]
					  -u11s[bthreadId] *X2s[bthreadId+1]))).real();
		}
	}
}
__global__ void cuForce_t(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa){
	const char *funcname = "cuForce";
	//Up indices
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;
	__shared__ Complex_f u11s[128];	__shared__ Complex_f u12s[128];
	__shared__ Complex X1s[128*nc];	__shared__ Complex X1su[128*nc];
	__shared__ Complex_f X2s[128*nc];	__shared__ Complex_f X2su[128*nc];
	__shared__ float dk4ms[128]; __shared__ float dk4ps[128];

	const int mu=3;
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		//Up indices
		int uid = iu[mu+ndim*i];
		u11s[bthreadId]=u11t[i*ndim+mu];	u12s[bthreadId]=u12t[i*ndim+mu];
		dk4ms[bthreadId]=dk4m[i];	dk4ps[bthreadId]=dk4p[i];

		for(int idirac=0;idirac<ndirac;idirac++){
			X1s[bthreadId]=X1[(i*ndirac+idirac)*nc];	X1s[bthreadId+1]=X1[(i*ndirac+idirac)*nc+1];
			X1su[bthreadId]=X1[(uid*ndirac+idirac)*nc];	X1su[bthreadId+1]=X1[(uid*ndirac+idirac)*nc+1];

			X2s[bthreadId]=X2[(i*ndirac+idirac)*nc];	X2s[bthreadId+1]=X2[(i*ndirac+idirac)*nc+1];
			X2su[bthreadId]=X2[(uid*ndirac+idirac)*nc];	X2su[bthreadId+1]=X2[(uid*ndirac+idirac)*nc+1];

			dSdpi[(i*nadj)*ndim+mu]+=(I*
					(conj(X1s[bthreadId])*
					 (dk4ms[bthreadId]*(-conj(u12s[bthreadId])*X2su[bthreadId]
								  +conj(u11s[bthreadId])*X2su[bthreadId+1]))
					 +conj(X1su[bthreadId])*
					 (dk4ps[bthreadId]*      (+u12s[bthreadId] *X2s[bthreadId]
										  -conj(u11s[bthreadId])*X2s[bthreadId+1]))
					 +conj(X1s[bthreadId+1])*
					 (dk4ms[bthreadId]*       (u11s[bthreadId] *X2su[bthreadId]
											+u12s[bthreadId] *X2su[bthreadId+1]))
					 +conj(X1su[bthreadId+1])*
					 (dk4ps[bthreadId]*      (-u11s[bthreadId] *X2s[bthreadId]
										  -conj(u12s[bthreadId])*X2s[bthreadId+1])))).real();
			dSdpi[(i*nadj+1)*ndim+mu]+=(
					conj(X1s[bthreadId])*
					(dk4ms[bthreadId]*(-conj(u12s[bthreadId])*X2su[bthreadId]
								 +conj(u11s[bthreadId])*X2su[bthreadId+1]))
					+conj(X1su[bthreadId])*
					(dk4ps[bthreadId]*      (-u12s[bthreadId] *X2s[bthreadId]
										 -conj(u11s[bthreadId])*X2s[bthreadId+1]))
					+conj(X1s[bthreadId+1])*
					(dk4ms[bthreadId]*      (-u11s[bthreadId] *X2su[bthreadId]
										 -u12s[bthreadId] *X2su[bthreadId+1]))
					+conj(X1su[bthreadId+1])*
					(dk4ps[bthreadId]*      ( u11s[bthreadId] *X2s[bthreadId]
										  -conj(u12s[bthreadId])*X2s[bthreadId+1]))).real();

			dSdpi[(i*nadj+2)*ndim+mu]+=(I*
					(conj(X1s[bthreadId])*
					 (dk4ms[bthreadId]*       (u11s[bthreadId] *X2su[bthreadId]
											+u12s[bthreadId] *X2su[bthreadId+1]))
					 +conj(X1su[bthreadId])*
					 (dk4ps[bthreadId]*(-conj(u11s[bthreadId])*X2s[bthreadId]
								  -u12s[bthreadId] *X2s[bthreadId+1]))
					 +conj(X1s[bthreadId+1])*
					 (dk4ms[bthreadId]* (conj(u12s[bthreadId])*X2su[bthreadId]
									-conj(u11s[bthreadId])*X2su[bthreadId+1]))
					 +conj(X1su[bthreadId+1])*
					 (dk4ps[bthreadId]*(-conj(u12s[bthreadId])*X2s[bthreadId]
								  +u11s[bthreadId] *X2s[bthreadId+1])))).real();

			const int igork1 = gamin[mu*ndirac+idirac];	
			X2s[bthreadId]=X2[(i*ndirac+igork1)*nc];	X2s[bthreadId+1]=X2[(i*ndirac+igork1)*nc+1];
			X2su[bthreadId]=X2[(uid*ndirac+igork1)*nc];	X2su[bthreadId+1]=X2[(uid*ndirac+igork1)*nc+1];

			dSdpi[(i*nadj)*ndim+mu]+=(I*
					(conj(X1s[bthreadId])*
					 (dk4ms[bthreadId]*(-conj(u12s[bthreadId])*X2su[bthreadId]
								  +conj(u11s[bthreadId])*X2su[bthreadId+1]))
					 +conj(X1su[bthreadId])*
					 (-dk4ps[bthreadId]*       (u12s[bthreadId] *X2s[bthreadId]
											 -conj(u11s[bthreadId])*X2s[bthreadId+1]))
					 +conj(X1s[bthreadId+1])*
					 (dk4ms[bthreadId]*       (u11s[bthreadId] *X2su[bthreadId]
											+u12s[bthreadId] *X2su[bthreadId+1]))
					 +conj(X1su[bthreadId+1])*
					 (-dk4ps[bthreadId]*      (-u11s[bthreadId] *X2s[bthreadId]
											-conj(u12s[bthreadId])*X2s[bthreadId+1])))).real();

			dSdpi[(i*nadj+1)*ndim+mu]+=(
					(conj(X1s[bthreadId])*
					 (dk4ms[bthreadId]*(-conj(u12s[bthreadId])*X2su[bthreadId]
								  +conj(u11s[bthreadId])*X2su[bthreadId+1]))
					 +conj(X1su[bthreadId])*
					 (-dk4ps[bthreadId]*      (-u12s[bthreadId] *X2s[bthreadId]
											-conj(u11s[bthreadId])*X2s[bthreadId+1]))
					 +conj(X1s[bthreadId+1])*
					 (dk4ms[bthreadId]*      (-u11s[bthreadId] *X2su[bthreadId]
										  -u12s[bthreadId] *X2su[bthreadId+1]))
					 +conj(X1su[bthreadId+1])*
					 (-dk4ps[bthreadId]*       (u11s[bthreadId] *X2s[bthreadId]
											 -conj(u12s[bthreadId])*X2s[bthreadId+1])))).real();

			dSdpi[(i*nadj+2)*ndim+mu]+=(I*
					(conj(X1s[bthreadId])*
					 (dk4ms[bthreadId]*       (u11s[bthreadId] *X2su[bthreadId]
											+u12s[bthreadId] *X2su[bthreadId+1]))
					 +conj(X1su[bthreadId])*
					 (-dk4ps[bthreadId]*(-conj(u11s[bthreadId])*X2s[bthreadId]
									-u12s[bthreadId] *X2s[bthreadId+1]))
					 +conj(X1s[bthreadId+1])*
					 (dk4ms[bthreadId]* (conj(u12s[bthreadId])*X2su[bthreadId]
									-conj(u11s[bthreadId])*X2su[bthreadId+1]))
					 +conj(X1su[bthreadId+1])*
					 (-dk4ps[bthreadId]*(-conj(u12s[bthreadId])*X2s[bthreadId]
									+u11s[bthreadId] *X2s[bthreadId+1])))).real();
		}
	}
}
