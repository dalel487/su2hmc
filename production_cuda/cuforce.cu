/*
 * Code for force calculations.
 * Requires multiply.cu to work
 */
#include	<matrices.h>
#include	<par_mpi.h>
#include	<su2hmc.h>
//Calling functions
void cuGauge_force(int mu, Complex *Sigma11, Complex *Sigma12, Complex *u11t,Complex *u12t,double *dSdpi,float beta,\
		dim3 dimGrid, dim3 dimBlock){
	const char *funcname = "Gauge_force";
	cuGaugeForce<<<dimGrid,dimBlock>>>(mu,Sigma11,Sigma12,dSdpi,u11t,u12t,beta);
}
void cuPlus_staple(int mu, int nu, unsigned int *iu, Complex *Sigma11, Complex *Sigma12, Complex *u11t, Complex *u12t,\
		dim3 dimGrid, dim3 dimBlock){
	const char *funcname="Plus_staple";
	Plus_staple<<<dimGrid,dimBlock>>>(mu, nu, iu, Sigma11, Sigma12,u11t,u12t);
}
void cuMinus_staple(int mu, int nu, unsigned int *iu, unsigned int *id, Complex *Sigma11, Complex *Sigma12,\
		Complex *u11sh, Complex *u12sh,Complex *u11t, Complex*u12t,dim3 dimGrid, dim3 dimBlock){
	const char *funcname="Minus_staple";
	Minus_staple<<<dimGrid,dimBlock>>>(mu, nu, iu, id,Sigma11,Sigma12,u11sh,u12sh,u11t,u12t);
}
void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, \
		Complex gamval[5][4],double *dk4m, double *dk4p,unsigned int *iu,int gamin[4][4],\
		float akappa, dim3 dimGrid, dim3 dimBlock){
	const char *funcname = "Force";
	//X1=(M†M)^{1} Phi
	cuForce<<<dimGrid,dimBlock>>>(dSdpi,u11t,u12t,X1,X2,&gamval[0][0],dk4m,dk4p,iu,&gamin[0][0],akappa);
}
//CUDA Kernels
//TODO: Split cuForce into seperateable streams
__global__ void cuForce(double *dSdpi, Complex *u11t, Complex *u12t, Complex *X1, Complex *X2, Complex *gamval,\
		double *dk4m, double *dk4p, unsigned int *iu, int *gamin,float akappa){
	char *funcname = "cuForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize)
		for(int idirac=0;idirac<ndirac;idirac++){
			int mu, uid, igork1;
#ifndef NO_SPACE
			for(mu=0; mu<3; mu++){
				//Long term ambition. I used the diff command on the different
				//spacial components of dSdpi and saw a lot of the values required
				//for them are duplicates (u11(i,mu)*X2(1,idirac,i) is used again with
				//a minus in front for example. Why not evaluate them first /and then plug 
				//them into the equation? Reduce the number of evaluations needed and look
				//a bit neater (although harder to follow as a consequence).

				//Up indices
				uid = iu[mu+ndim*i];
				igork1 = gamin[mu*ndirac+idirac];	
				dSdpi[(i*nadj)*ndim+mu]+=akappa*(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
						  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 ( u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
							-conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
						  +u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (-u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
						  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))).real();
				dSdpi[(i*nadj)*ndim+mu]+=(I*gamval[mu*ndirac+idirac]*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
						  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (-u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
						  +conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
						  +u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
						  +conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))).real();

				dSdpi[(i*nadj+1)*ndim+mu]+=akappa*(
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
						  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (-u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
						  -conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (-u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
						  -u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
						  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))).real();
				dSdpi[(i*nadj+1)*ndim+mu]+=(gamval[mu*ndirac+idirac]*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
						  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
						  +conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (-u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
						  -u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (-u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
						  +conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))).real();

				dSdpi[(i*nadj+2)*ndim+mu]+=akappa*(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
						  +u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (-conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
						  -u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
						  -conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (-conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
						  +u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1]))).real();
				dSdpi[(i*nadj+2)*ndim+mu]+=(I*gamval[mu*ndirac+idirac]*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
						  +u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
						  +u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1])
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
						  -conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1])
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
						  -u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1]))).real();

			}
#endif
			//We're not done tripping yet!! Time like term is different. dk4? shows up
			//For consistency we'll leave mu in instead of hard coding.
			mu=3;
			uid = iu[mu+ndim*i];
			//We are mutiplying terms by dk4?[i] Also there is no akappa or gamval factor in the time direction	
			//for the "gamval" terms the sign of d4kp flips
#ifndef NO_TIME
			dSdpi[(i*nadj)*ndim+mu]+=(I*
					(conj(X1[(i*ndirac+idirac)*nc])*
					 (dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
								  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
					 +conj(X1[(uid*ndirac+idirac)*nc])*
					 (dk4p[i]*      (+u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
										  -conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))
					 +conj(X1[(i*ndirac+idirac)*nc+1])*
					 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
											+u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
					 +conj(X1[(uid*ndirac+idirac)*nc+1])*
					 (dk4p[i]*      (-u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
										  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1])))).real()
				+(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
									  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (-dk4p[i]*       (u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
												 -conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
												+u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (-dk4p[i]*      (-u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
												-conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])))).real();

			dSdpi[(i*nadj+1)*ndim+mu]+=(
					conj(X1[(i*ndirac+idirac)*nc])*
					(dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
								 +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
					+conj(X1[(uid*ndirac+idirac)*nc])*
					(dk4p[i]*      (-u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
										 -conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))
					+conj(X1[(i*ndirac+idirac)*nc+1])*
					(dk4m[i]*      (-u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
										 -u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
					+conj(X1[(uid*ndirac+idirac)*nc+1])*
					(dk4p[i]*      ( u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc]
										  -conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc+1]))).real()
				+(
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (dk4m[i]*(-conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
									  +conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (-dk4p[i]*      (-u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
												-conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1]))
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (dk4m[i]*      (-u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
											  -u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (-dk4p[i]*       (u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc]
												 -conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc+1])))).real();

			dSdpi[(i*nadj+2)*ndim+mu]+=(I*
					(conj(X1[(i*ndirac+idirac)*nc])*
					 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc]
											+u12t[i*ndim+mu] *X2[(uid*ndirac+idirac)*nc+1]))
					 +conj(X1[(uid*ndirac+idirac)*nc])*
					 (dk4p[i]*(-conj(u11t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
								  -u12t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1]))
					 +conj(X1[(i*ndirac+idirac)*nc+1])*
					 (dk4m[i]* (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc]
									-conj(u11t[i*ndim+mu])*X2[(uid*ndirac+idirac)*nc+1]))
					 +conj(X1[(uid*ndirac+idirac)*nc+1])*
					 (dk4p[i]*(-conj(u12t[i*ndim+mu])*X2[(i*ndirac+idirac)*nc]
								  +u11t[i*ndim+mu] *X2[(i*ndirac+idirac)*nc+1])))).real()
				+(I*
						(conj(X1[(i*ndirac+idirac)*nc])*
						 (dk4m[i]*       (u11t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc]
												+u12t[i*ndim+mu] *X2[(uid*ndirac+igork1)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc])*
						 (-dk4p[i]*(-conj(u11t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
										-u12t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1]))
						 +conj(X1[(i*ndirac+idirac)*nc+1])*
						 (dk4m[i]* (conj(u12t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc]
										-conj(u11t[i*ndim+mu])*X2[(uid*ndirac+igork1)*nc+1]))
						 +conj(X1[(uid*ndirac+idirac)*nc+1])*
						 (-dk4p[i]*(-conj(u12t[i*ndim+mu])*X2[(i*ndirac+igork1)*nc]
										+u11t[i*ndim+mu] *X2[(i*ndirac+igork1)*nc+1])))).real();

#endif
		}
}
__global__ void Plus_staple(int mu, int nu,unsigned int *iu, Complex *Sigma11, Complex *Sigma12, Complex *u11t, Complex *u12t){
	char *funcname = "Plus_staple";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		int uidm = iu[mu+ndim*i];
		int uidn = iu[nu+ndim*i];
		Complex	a11=u11t[uidm*ndim+nu]*conj(u11t[uidn*ndim+mu])+\
						 u12t[uidm*ndim+nu]*conj(u12t[uidn*ndim+mu]);
		Complex	a12=-u11t[uidm*ndim+nu]*u12t[uidn*ndim+mu]+\
						 u12t[uidm*ndim+nu]*u11t[uidn*ndim+mu];
		Sigma11[i]+=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
		Sigma12[i]+=-a11*u12t[i*ndim+nu]+a12*u11t[i*ndim+nu];
	}
}
__global__ void Minus_staple(int mu,int nu,unsigned int *iu,unsigned int *id, Complex *Sigma11, Complex *Sigma12,\
		Complex *u11sh, Complex *u12sh, Complex *u11t, Complex *u12t){
	char *funcname = "Minus_staple";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		int uidm = iu[mu+ndim*i];
		int didn = id[nu+ndim*i];
		//uidm is correct here
		Complex a11=conj(u11sh[uidm])*conj(u11t[didn*ndim+mu])-\
						u12sh[uidm]*conj(u12t[didn*ndim+mu]);
		Complex a12=-conj(u11sh[uidm])*u12t[didn*ndim+mu]-\
						u12sh[uidm]*u11t[didn*ndim+mu];
		Sigma11[i]+=a11*u11t[didn*ndim+nu]-a12*conj(u12t[didn*ndim+nu]);
		Sigma12[i]+=a11*u12t[didn*ndim+nu]+a12*conj(u11t[didn*ndim+nu]);
	}
}
__global__ void cuGaugeForce(int mu, Complex *Sigma11, Complex *Sigma12,double*dSdpi,Complex *u11t, Complex *u12t, float beta){
	char *funcname = "cuGaugeForce";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		Complex a11 = u11t[i*ndim+mu]*Sigma12[i]+u12t[i*ndim+mu]*conj(Sigma11[i]);
		Complex a12 = u11t[i*ndim+mu]*Sigma11[i]+conj(u12t[i*ndim+mu])*Sigma12[i];

		dSdpi[(i*nadj)*ndim+mu]=beta*a11.imag();
		dSdpi[(i*nadj+1)*ndim+mu]=beta*a11.real();
		dSdpi[(i*nadj+2)*ndim+mu]=beta*a12.imag();
	}
}
