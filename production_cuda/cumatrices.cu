#include <assert.h>
#include <matrices.h>
#include <string.h>
#include	<thrust_complex.h>
__global__ void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, Complex_f jqq, float akappa){
	const char *funcname = "cuDslash";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			a_1=conj(jqq)*gamval_d[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval_d[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc+0];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+0]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin_d[mu*ndirac+idirac] : gamin_d[mu*ndirac+idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]-\
						u12t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
													  //Dirac term
													  gamval_d[mu*ndirac+idirac]*(u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]+\
															  u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]-\
															  conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]+\
															  u12t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]+\
						u11t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
														 //Dirac term
														 gamval_d[mu*ndirac+idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]+\
																 conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]-\
																 conj(u12t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]-\
																 u11t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int igorkov=0; igorkov<4; igorkov++){
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1 = gamin_d[3*ndirac+igorkov];	int igork1PP = igork1+4;

			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4p[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));

			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])+\
					u12t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))-\
													 dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])-\
															 u12t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=-dk4m[i]*(conj(-u12t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])+\
					conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))-\
														dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])+\
																u11t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));
		}
#endif
	}
}
__global__ void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, Complex_f jqq, float akappa){
	const char *funcname = "cuDslashd";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval_d[4*ndirac+idirac];
			a_2=jqq*gamval_d[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc];
			phi[(i*ngorkov+igork)*nc]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin_d[mu*ndirac+idirac] : gamin_d[mu*ndirac+idirac]+4;
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=
					-akappa*(      u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]
							+u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							-u12t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval_d[mu*ndirac+idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]
								  +u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]
								  -conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
								  +u12t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							+u11t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval_d[mu*ndirac+idirac]*
					(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]
					 +conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]
					 -conj(u12t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
					 -u11t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int igorkov=0; igorkov<4; igorkov++){
			//the FORTRAN code did it.
			int igork1 = gamin_d[3*ndirac+igorkov];	
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4m[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));


			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])+\
					u12t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))-\
													 dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])-\
															 u12t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

			phi[(i*ngorkov+igorkovPP)*nc+1]+=dk4p[i]*(conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])-\
					conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))-\
														dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])+
																u11t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));

		}
#endif
	}
}
__global__ void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, float akappa){
	const char *funcname = "cuHdslash";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin_d[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]-\
						u12t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
													//Dirac term
													gamval_d[mu*ndirac+idirac]*(u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]+\
															u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]-\
															conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]+\
															u12t[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]+\
						u11t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
													  //Dirac term
													  gamval_d[mu*ndirac+idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]+\
															  conj(u11t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]-\
															  conj(u12t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]-\
															  u11t[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ndirac+idirac)*nc]+=
				-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4p[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
}
__global__ void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, float akappa){
	const char *funcname = "cuHdslashd";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize*bsize){
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin_d[mu*ndirac+idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				phi[(i*ndirac+idirac)*nc]+=
					-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]
							+u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							-u12t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval_d[mu*ndirac+idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]
								  +u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]
								  -conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
								  +u12t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							+u11t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval_d[mu*ndirac+idirac]*
					(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]
					 +conj(u11t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]
					 -conj(u12t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
					 -u11t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
			phi[(i*ndirac+idirac)*nc]+=
				-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4m[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
}

//Float editions
//TODO: On newer GPUs (Ada and later) the L2 cache is big enough that the u11t or u12t fields can fit on it. In that
//case it may be worth investigating shared memory access. Preload on the u11t and u12t before the kernel call even
//https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory

__global__ void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		__shared__ Complex_f *gamval_d,	int *gamin_d,	float *dk4m, float *dk4p, Complex_f jqq, float akappa){
	const char *funcname = "cuDslash_f";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		Complex_f ru[nc]; Complex_f rd[nc];
		Complex_f rgu[nc]; Complex_f rgd[nc];
		Complex_f phi_s[ngorkov*nc];
		for(int idirac=0;idirac<ndirac;idirac++){
			int igork = idirac+4;
			Complex_f a_1=conj(jqq)*gamval_d[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			Complex_f a_2=-jqq*gamval_d[4*ndirac+idirac];
			phi_s[idirac*nc]=phi[i+kvol*(idirac*nc)]+a_1*r[i+kvol*(igork*nc)];
			phi_s[igork*nc]=phi[i+kvol*(igork*nc)]+a_2*r[i+kvol*(idirac*nc)];
			phi_s[idirac*nc+1]=phi[i+kvol*(idirac*nc+1)]+a_1*r[i+kvol*(igork*nc+1)];
			phi_s[igork*nc+1]=phi[i+kvol*(igork*nc+1)]+a_2*r[i+kvol*(idirac*nc+1)];
		}
		Complex_f u11s;	Complex_f u12s;
		Complex_f u11sd; Complex_f u12sd;
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			u11s=u11t[i+kvol*mu]; u12s=u12t[i+kvol*mu];
			u11sd=u11t[did+kvol*mu]; u12sd=u12t[did+kvol*mu];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				int idirac=igorkov%4;		
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int igork1 = (igorkov<4) ? gamin_d[mu*ndirac+idirac] : gamin_d[mu*ndirac+idirac]+4;
				for(int c=0;c<nc;c++){
					ru[c]=r[uid+kvol*(igorkov*nc+c)];
					rd[c]=r[did+kvol*(igorkov*nc+c)];
					rgu[c]=r[uid+kvol*(igork1*nc+c)];
					rgd[c]=r[did+kvol*(igork1*nc+c)];
				}
				//Can manually vectorise with a pragma?
//				phi_s[bthreadId]=phi[i+kvol*(igorkov*nc)];
				phi_s[igorkov*nc]+=-akappa*(u11s*ru[0]+\
						u12s*ru[1]+\
						conj(u11sd)*rd[0]-\
						u12sd*rd[1]);
				//Dirac term
				phi_s[igorkov*nc]+=gamval_d[mu*ndirac+idirac]*(u11s*rgu[0]+\
						u12s*rgu[1]-\
						conj(u11sd)*rgd[0]+\
						u12sd*rgd[1]);
//				phi[i+kvol*(igorkov*nc)]=phi_s[bthreadId];

//				phi_s[bthreadId]=phi[i+kvol*(igorkov*nc+1)];
				phi_s[igorkov*nc+1]+=-akappa*(-conj(u12s)*ru[0]+\
						conj(u11s)*ru[1]+\
						conj(u12sd)*rd[0]+\
						u11sd*rd[1]);
				//Dirac term
				phi_s[igorkov*nc+1]+=gamval_d[mu*ndirac+idirac]*(-conj(u12s)*rgu[0]+\
						conj(u11s)*rgu[1]-\
						conj(u12sd)*rgd[0]-\
						u11sd*rgd[1]);
//				phi[i+kvol*(igorkov*nc+1)]=phi_s[bthreadId];
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
#ifndef NO_TIME
		u11s=u11t[i+kvol*3]; u12s=u12t[i+kvol*3];
		float dk4ms=dk4m[i];	float dk4ps=dk4p[i];
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		u11sd=u11t[did+kvol*3]; u12sd=u12t[did+kvol*3];
		float dk4msd=dk4m[did];	float dk4psd=dk4p[did];
		for(int igorkov=0;igorkov<ndirac;igorkov++){
			int igork1 = gamin_d[3*ndirac+igorkov];
			for(int c=0;c<nc;c++){
				ru[c]=r[uid+kvol*(igorkov*nc+c)];
				rd[c]=r[did+kvol*(igorkov*nc+c)];
				rgu[c]=r[uid+kvol*(igork1*nc+c)];
				rgd[c]=r[did+kvol*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
//			phi_s[igorkov*nc]=phi[i+kvol*(igorkov*nc)];
			phi_s[igorkov*nc]+=
				-dk4ps*(u11s*(ru[0]-rgu[0])
						+u12s*(ru[1]-rgu[1]))
				-dk4msd*(conj(u11sd)*(rd[0]+rgd[0])
						-u12sd *(rd[1]+rgd[1]));
			phi[i+kvol*(igorkov*nc)]=phi_s[igorkov*nc];

//			phi_s[igorkov*nc]=phi[i+kvol*(igorkov*nc+1)];
			phi_s[igorkov*nc+1]+=
				-dk4ps*(-conj(u12s)*(ru[0]-rgu[0])
						+conj(u11s)*(ru[1]-rgu[1]))
				-dk4msd*(conj(u12sd)*(rd[0]+rgd[0])
						+u11sd *(rd[1]+rgd[1]));
			phi[i+kvol*(igorkov*nc+1)]=phi_s[igorkov*nc+1];
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1PP = igork1+4;
			//And the gorkov terms. Note that dk4p and dk4m swap positions compared to the above				
			for(int c=0;c<nc;c++){
				ru[c]=r[uid+kvol*(igorkovPP*nc+c)];
				rd[c]=r[did+kvol*(igorkovPP*nc+c)];
				rgu[c]=r[uid+kvol*(igork1PP*nc+c)];
				rgd[c]=r[did+kvol*(igork1PP*nc+c)];
			}
//			phi_s[igorkovPP*nc]=phi[i+kvol*(igorkovPP*nc)];
			phi_s[igorkovPP*nc]+=-dk4ms*(u11s*(ru[0]-rgu[0])+
					u12s*(ru[1]-rgu[1]))-
				dk4psd*(conj(u11sd)*(rd[0]+rgd[0])-
						u12sd*(rd[1]+rgd[1]));
			phi[i+kvol*(igorkovPP*nc)]=phi_s[igorkovPP*nc];

//			phi_s[bthreadId]=phi[i+kvol*(igorkovPP*nc+1)];
			phi_s[igorkovPP*nc+1]+=-dk4ms*(conj(-u12s)*(ru[0]-rgu[0])
					+conj(u11s)*(ru[1]-rgu[1]))
				-dk4psd*(conj(u12sd)*(rd[0]+rgd[0])
						+u11sd*(rd[1]+rgd[1]));
			phi[i+kvol*(igorkovPP*nc+1)]=phi_s[igorkovPP*nc+1];
		}
#endif
	}
}
__global__ void cuDslashd_f(Complex_f *phi, const Complex_f *r, const Complex_f *u11t, const Complex_f *u12t,const unsigned int *iu, const unsigned int *id,\
		__shared__ Complex_f *gamval_d,	int *gamin_d,	const float *dk4m, const float *dk4p, const Complex_f jqq, const float akappa){
	const char *funcname = "cuDslashd_f";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	Complex_f u11s;	 Complex_f u12s;
	Complex_f u11sd;	 Complex_f u12sd;
	Complex_f ru[nc];  Complex_f rd[nc];
	Complex_f rgu[nc];  Complex_f rgd[nc];
	Complex_f phi_s[ngorkov*nc];

	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		volatile int did=0; volatile int uid = 0;
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
#pragma unroll
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval_d[4*ndirac+idirac];
			a_2=jqq*gamval_d[4*ndirac+idirac];
			phi_s[idirac*nc]=phi[i+kvol*(idirac*nc)]+a_1*r[i+kvol*(igork*nc)];
			phi_s[igork*nc]=phi[i+kvol*(igork*nc)]+a_2*r[i+kvol*(idirac*nc)];
			phi_s[idirac*nc+1]=phi[i+kvol*(idirac*nc+1)]+a_1*r[i+kvol*(igork*nc+1)];
			phi_s[igork*nc+1]=phi[i+kvol*(igork*nc+1)]+a_2*r[i+kvol*(idirac*nc+1)];
		}
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
#pragma unroll
		for(int mu = 0; mu <3; mu++){
			did=id[mu+ndim*i]; uid = iu[mu+ndim*i];
			u11s=u11t[i+kvol*mu]; u12s=u12t[i+kvol*mu];
			u11sd=u11t[did+kvol*mu]; u12sd=u12t[did+kvol*mu];
#pragma unroll
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin_d[mu*ndirac+idirac] : gamin_d[mu*ndirac+idirac]+4;
#pragma unroll
				for(int c=0;c<nc;c++){
					ru[c]=r[uid+kvol*(igorkov*nc+c)];
					rd[c]=r[did+kvol*(igorkov*nc+c)];
					rgu[c]=r[uid+kvol*(igork1*nc+c)];
					rgd[c]=r[did+kvol*(igork1*nc+c)];
				}
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//	phi_s[bthreadId]=phi[i+kvol*(igorkov*nc)];
				phi_s[igorkov*nc]-=
					akappa*(u11s*ru[0]
							+u12s*ru[1]
							+conj(u11sd)*rd[0]
							-u12sd *rd[1]);

				//Dirac term
				phi_s[igorkov*nc]-=gamval_d[mu*ndirac+idirac]*
					(u11s*rgu[0]
					 +u12s*rgu[1]
					 -conj(u11sd)*rgd[0]
					 +u12sd *rgd[1]);
				//				phi[i+kvol*(igorkov*nc)]=phi_s[bthreadId];

				//				phi_s[bthreadId]=phi[i+kvol*(igorkov*nc+1)];
				phi_s[igorkov*nc+1]-=
					akappa*(-conj(u12s)*ru[0]
							+conj(u11s)*ru[1]
							+conj(u12sd)*rd[0]
							+u11sd *rd[1]);
				//Dirac term
				phi_s[igorkov*nc+1]-=gamval_d[mu*ndirac+idirac]*
					(-conj(u12s)*rgu[0]
					 +conj(u11s)*rgu[1]
					 -conj(u12sd)*rgd[0]
					 -u11sd *rgd[1]);
				//				phi[i+kvol*(igorkov*nc+1)]=phi_s[bthreadId];

			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
#ifndef NO_TIME
		did=id[3+ndim*i]; uid = iu[3+ndim*i];
		u11s=u11t[i+kvol*3]; u12s=u12t[i+kvol*3];
		u11sd=u11t[did+kvol*3]; u12sd=u12t[did+kvol*3];
		Complex_f dk4msd=dk4m[did];	Complex_f dk4psd=dk4p[did];
		Complex_f dk4ms=dk4m[i];	Complex_f dk4ps=dk4p[i];
#pragma unroll
		for(int igorkov=0; igorkov<ndirac; igorkov++){
			int igork1 = gamin_d[3*ndirac+igorkov];	
#pragma unroll
			for(int c=0;c<nc;c++){
				ru[c]=r[uid+kvol*(igorkov*nc+c)];
				rd[c]=r[did+kvol*(igorkov*nc+c)];
				rgu[c]=r[uid+kvol*(igork1*nc+c)];
				rgd[c]=r[did+kvol*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//			phi_s[bthreadId]=phi[i+kvol*(igorkov*nc)];
			phi_s[igorkov*nc]+=
				-dk4ms*(u11s*(ru[0]+rgu[0])
						+u12s*(ru[1]+rgu[1]))
				-dk4psd*(conj(u11sd)*(rd[0]-rgd[0])
						-u12sd *(rd[1]-rgd[1]));
			phi[i+kvol*(igorkov*nc)]=phi_s[igorkov*nc];

			//			phi_s[bthreadId]=phi[i+kvol*(igorkov*nc+1)];
			phi_s[igorkov*nc+1]+=
				-dk4ms*(-conj(u12s)*(ru[0]+rgu[0])
						+conj(u11s)*(ru[1]+rgu[1]))
				-dk4psd*(conj(u12sd)*(rd[0]-rgd[0])
						+u11sd *(rd[1]-rgd[1]));
			phi[i+kvol*(igorkov*nc+1)]=phi_s[igorkov*nc+1];
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1PP = igork1+4;
			for(int c=0;c<nc;c++){
				ru[c]=r[uid+kvol*(igorkovPP*nc+c)];
				rd[c]=r[did+kvol*(igorkovPP*nc+c)];
				rgu[c]=r[uid+kvol*(igork1PP*nc+c)];
				rgd[c]=r[did+kvol*(igork1PP*nc+c)];
			}
			//And the Gor'kov terms. Note that dk4p and dk4m swap positions compared to the above				
			//			phi_s[bthreadId]=phi[i+kvol*(igorkovPP*nc)];
			phi_s[igorkovPP*nc]+=-dk4ps*(u11s*(ru[0]+rgu[0])
					+u12s*(ru[1]+rgu[1]))
				-dk4msd*(conj(u11sd)*(rd[0]-rgd[0])
						-u12sd*(rd[1]-rgd[1]));
			phi[i+kvol*(igorkovPP*nc)]=phi_s[igorkovPP*nc];

			//			phi_s[bthreadId]=phi[i+kvol*(igorkovPP*nc+1)];
			phi_s[igorkovPP*nc+1]+=dk4ps*(conj(u12s)*(ru[0]+rgu[0])
					-conj(u11s)*(ru[1]+rgu[1]))
				-dk4msd*(conj(u12sd)*(rd[0]-rgd[0])
						+u11sd*(rd[1]-rgd[1]));
			phi[i+kvol*(igorkovPP*nc+1)]=phi_s[igorkovPP*nc+1];
		}
#endif
	}
}

__global__ void cuHdslash_f(Complex_f *phi, const Complex_f *r, const Complex_f *u11t, const Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		__constant__ Complex_f gamval[20],	__constant__ int gamin_d[16],	const float *dk4m, const float *dk4p, const float akappa){
	/*
	 * Half Dslash float precision
	 */
	const volatile char *funcname = "cuHdslash0_f";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	//Right. Time to prefetch
	Complex_f u11s;	 Complex_f u12s;
	Complex_f u11sd;	 Complex_f u12sd;
	Complex_f ru[2];  Complex_f rd[2];
	Complex_f rgu[2];  Complex_f rgd[2];
	Complex_f phi_s[ndirac*nc];
	for(int i=gthreadId;i<kvol;i+=bsize*gsize){
		//Do we need to sync threads if each thread only accesses the value it put in shared memory?
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++)
#pragma unroll
			for(int c=0; c<nc; c++)
				phi_s[idirac*nc+c]=phi[i+kvol*(c+nc*idirac)];
#ifndef NO_SPACE
#pragma unroll
		for(int mu = 0; mu <3; mu++){
			u11s=u11t[i+kvol*mu];	u12s=u12t[i+kvol*mu];
			int did=id[mu*kvol+i];
			u11sd=u11t[did+kvol*mu];	u12sd=u12t[did+kvol*mu];
			int uid = iu[mu*kvol+i];
#pragma unroll
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin_d[mu*ndirac+idirac];
#pragma unroll
				for(int c=0;c<nc;c++){
					ru[c]=r[uid+kvol*(idirac*nc+c)];
					rd[c]=r[did+kvol*(idirac*nc+c)];
					rgu[c]=r[uid+kvol*(igork1*nc+c)];
					rgd[c]=r[did+kvol*(igork1*nc+c)];
				}
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi_s[idirac*nc]+=-akappa*(u11s*ru[0]+\
						u12s*ru[1]+\
						conj(u11sd)*rd[0]-\
						u12sd*rd[1]);
				//Dirac term
				phi_s[idirac*nc]+=gamval[mu*ndirac+idirac]*(u11s*rgu[0]+\
						u12s*rgu[1]-\
						conj(u11sd)*rgd[0]+\
						u12sd*rgd[1]);

				phi_s[idirac*nc+1]+=-akappa*(-conj(u12s)*ru[0]+\
						conj(u11s)*ru[1]+\
						conj(u12sd)*rd[0]+\
						u11sd*rd[1]);
				//Dirac term
				phi_s[idirac*nc+1]+=gamval[mu*ndirac+idirac]*(-conj(u12s)*rgu[0]+\
						conj(u11s)*rgu[1]-\
						conj(u12sd)*rgd[0]-\
						u11sd*rgd[1]);
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		u11s=u11t[i+kvol*3];	u12s=u12t[i+kvol*3];
		int did=id[3*kvol+i]; 
		u11sd=u11t[did+kvol*3];	u12sd=u12t[did+kvol*3];
		float dk4ms=dk4m[did];   float dk4ps=dk4p[i];
		int uid = iu[3*kvol+i];
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
#pragma unroll
			for(int c=0;c<nc;c++){
				ru[c]=r[uid+kvol*(idirac*nc+c)];
				rd[c]=r[did+kvol*(idirac*nc+c)];
				rgu[c]=r[uid+kvol*(igork1*nc+c)];
				rgd[c]=r[did+kvol*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)

			phi_s[idirac*nc+0]-=
				dk4ps*(u11s*(ru[0]-rgu[0])
						+u12s*(ru[1]-rgu[1]));
			phi_s[idirac*nc+0]-=
				dk4ms*(conj(u11sd)*(rd[0]+rgd[0])
						-u12sd *(rd[1]+rgd[1]));
			phi[i+kvol*(0+nc*idirac)]=phi_s[idirac*nc+0];

			phi_s[idirac*nc+1]-=
				dk4ps*(-conj(u12s)*(ru[0]-rgu[0])
						+conj(u11s)*(ru[1]-rgu[1]));
			phi_s[idirac*nc+1]-=
				dk4ms*(conj(u12sd)*(rd[0]+rgd[0])
						+u11sd *(rd[1]+rgd[1]));
			phi[i+kvol*(1+nc*idirac)]=phi_s[idirac*nc+1];
#endif
		}
	}
}
__global__ void cuHdslashd_f(Complex_f *phi, const Complex_f* r, const Complex_f* u11t, const Complex_f* u12t,unsigned int* iu, unsigned int* id,\
		__constant__ Complex_f gamval[20],	__constant__ int gamin_d[16],	const float* dk4m, const float* dk4p, const float akappa){
	/*
	 * Half Dslash Dagger float precision 
	 */
	const volatile char *funcname = "cuHdslashd0_f";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const volatile int gthreadId= blockId * bsize+bthreadId;

	//Right. Time to prefetch
	Complex_f u11s;	 Complex_f u12s;
	Complex_f u11sd;	 Complex_f u12sd;
	Complex_f ru[2];  Complex_f rd[2];
	Complex_f rgu[2];  Complex_f rgd[2];
	Complex_f phi_s[ndirac*nc];
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++)
#pragma unroll
			for(int c=0; c<nc; c++)
				phi_s[idirac*nc+c]=phi[i+kvol*(c+nc*idirac)];
#ifndef NO_SPACE
#pragma unroll
		for(int mu = 0; mu <ndim-1; mu++){
			//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
			u11s=u11t[i+kvol*mu];	u12s=u12t[i+kvol*mu];
			int did=id[i+kvol*mu];
			u11sd=u11t[did+kvol*mu];	u12sd=u12t[did+kvol*mu];
			int uid = iu[i+kvol*mu];
#pragma unroll
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin_d[mu*ndirac+idirac];
#pragma unroll
				for(int c=0;c<nc;c++){
					ru[c]=r[uid+kvol*(idirac*nc+c)];
					rd[c]=r[did+kvol*(idirac*nc+c)];
					rgu[c]=r[uid+kvol*(igork1*nc+c)];
					rgd[c]=r[did+kvol*(igork1*nc+c)];
				}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi_s[idirac*nc]-=akappa*(u11s*ru[0]
						+u12s*ru[1]
						+conj(u11sd)*rd[0]
						-u12sd *rd[1]);
				//Dirac term
				phi_s[idirac*nc]-=gamval[mu*ndirac+idirac]*
					(u11s*rgu[0]
					 +u12s*rgu[1]
					 -conj(u11sd)*rgd[0]
					 +u12sd *rgd[1]);

				phi_s[idirac*nc+1]-=akappa*(-conj(u12s)*ru[0]
						+conj(u11s)*ru[1]
						+conj(u12sd)*rd[0]
						+u11sd *rd[1]);
				//Dirac term
				phi_s[idirac*nc+1]-=gamval[mu*ndirac+idirac]*(-conj(u12s)*rgu[0]
						+conj(u11s)*rgu[1]
						-conj(u12sd)*rgd[0]
						-u11sd *rgd[1]);
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		u11s=u11t[i+kvol*3];	u12s=u12t[i+kvol*3];
		int did=id[i+kvol*3];
		u11sd=u11t[did+kvol*3];	u12sd=u12t[did+kvol*3];
		float  dk4ms=dk4m[i];  float dk4ps=dk4p[did];
		int uid = iu[i+kvol*3];
#pragma unroll
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
#pragma unroll
			for(int c=0;c<nc;c++){
				ru[c]=r[uid+kvol*(idirac*nc+c)];
				rd[c]=r[did+kvol*(idirac*nc+c)];
				rgu[c]=r[uid+kvol*(igork1*nc+c)];
				rgd[c]=r[did+kvol*(igork1*nc+c)];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
			phi_s[idirac*nc]+=
				-dk4ms*(u11s*(ru[0]+rgu[0])
						+u12s*(ru[1]+rgu[1]));
			phi_s[idirac*nc]+=
				-dk4ps*(conj(u11sd)*(rd[0]-rgd[0])
						-u12sd *(rd[1]-rgd[1]));
			phi[i+kvol*(0+nc*idirac)]=phi_s[idirac*nc+0];

			phi_s[idirac*nc+1]-=
				dk4ms*(-conj(u12s)*(ru[0]+rgu[0])
						+conj(u11s)*(ru[1]+rgu[1]));
			phi_s[idirac*nc+1]-=
				+dk4ps*(conj(u12sd)*(rd[0]-rgd[0])
						+u11sd *(rd[1]-rgd[1]));
			phi[i+kvol*(1+nc*idirac)]=phi_s[idirac*nc+1];
#endif
		}
	}
}

/**
 * @brief Swaps the order of the gauge field so that it is now SoA instead of AoS and it is nice and coalesced in memory
 * 
 * @param out:	The flipped array
 * @param in:	The original array
 * @param nx:	The size of the slowest moving dimension. This is the lattice site when read in from disk
 * @param ny:	The size of the fastest moving dimension, This is the direction index when read in from disk.
 * 
 */
template <typename T>
__global__ void Transpose(T *out, const T *in, const int fast_in, const int fast_out){
	const volatile char *funcname="Transpose_f";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	//The if/else here is only to ensure we maximise GPU bandwidth
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=gthreadId;x<fast_out;x+=gsize*bsize)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=gthreadId;y<fast_in;y+=gsize*bsize)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
}
/*
__global__ void Transpose_f(Complex_f *out, Complex_f *in, const int fast_in, const int fast_out){
	const volatile char *funcname="Transpose_f";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	//The if/else here is only to ensure we maximise GPU bandwidth
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=gthreadId;x<fast_out;x+=gsize*bsize)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=gthreadId;y<fast_in;y+=gsize*bsize)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
}
__global__ void Transpose_I(int *out, int *in, const int fast_in, const int fast_out){
	const volatile char *funcname="Transpose_I";
	const volatile int gsize = gridDim.x*gridDim.y*gridDim.z;
	const volatile int bsize = blockDim.x*blockDim.y*blockDim.z;
	const volatile int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const volatile int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	//The if/else here is only to ensure we maximise GPU bandwidth
	//Typically this is used to write back to the AoS/Coalseced format
	if(fast_out>fast_in){
		for(int x=gthreadId;x<fast_out;x+=gsize*bsize)
			for(int y=0; y<fast_in;y++)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
	//Typically this is used to write back to the SoA/saved config format
	else{
		for(int x=0; x<fast_out;x++)
			for(int y=gthreadId;y<fast_in;y+=gsize*bsize)
				out[y*fast_out+x]=in[x*fast_in+y];
	}
}
*/

//Calling Functions
//================
void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
		Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
		dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * Complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslash";
	cudaMemcpy(phi, r, kferm*sizeof(Complex),cudaMemcpyDeviceToDevice);
	cuDslash<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
}
void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
		Complex *gamval, int *gamin,	double *dk4m, double *dk4p, Complex_f jqq, float akappa,\ 
		dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 *
	 * Complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslashd";
	cudaMemcpy(phi, r, kferm*sizeof(Complex),cudaMemcpyDeviceToDevice);
	cuDslashd<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
}
void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
		Complex *gamval, int *gamin,	double *dk4m, double *dk4p, float akappa,\ 
		dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * Complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslash";
	cudaMemcpy(phi, r, kferm2*sizeof(Complex),cudaMemcpyDeviceToDevice);
	cuHdslash<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
}
void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
		Complex *gamval, int *gamin,double *dk4m, double *dk4p, float akappa,\ 
		dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * Complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslashd";
	//Spacelike term
	cudaMemcpy(phi, r, kferm2*sizeof(Complex),cudaMemcpyDeviceToDevice);
	cuHdslashd<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
}

//Float editions
void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval,int *gamin,	float *dk4m, float *dk4p, Complex_f jqq, float akappa,\ 
		dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * Complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslash_f";
	int cuCpyStat=0;
	if((cuCpyStat=cudaMemcpy(phi, r, kferm*sizeof(Complex_f),cudaMemcpyDefault))){
		fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
				CPYERROR,funcname,cuCpyStat);
		exit(cuCpyStat);
	}
	cuDslash_f<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
}
void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval,int *gamin,	float *dk4m, float *dk4p, Complex_f jqq, float akappa,\ 
		dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parameter:
	 * ==========
	 *
	 * Complex *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Dslashd_f";
	int cuCpyStat=0;
	if((cuCpyStat=cudaMemcpy(phi, r, kferm*sizeof(Complex_f),cudaMemcpyDefault))){
		fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
				CPYERROR,funcname,cuCpyStat);
		exit(cuCpyStat);
	}
	cuDslashd_f<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,jqq,akappa);
}
void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu,unsigned int *id, Complex_f *gamval,
					int *gamin,	float *dk[2], float akappa, dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * Complex_f *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex_f r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslash_f";
	int cuCpyStat=0;
	if((cuCpyStat=cudaMemcpy(phi, r, kferm2*sizeof(Complex_f),cudaMemcpyDefault))){
		fprintf(stderr,"Error %d in %s: Cuda failed to copy r into device Phi with code %d.\nExiting,,,\n\n",\
				CPYERROR,funcname,cuCpyStat);
		exit(cuCpyStat);
	}
	const int bsize=dimGrid.x*dimGrid.y*dimGrid.z;
	const int shareSize= ndim*bsize*nc*sizeof(Complex_f);
	cuHdslash_f<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
}
void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu,unsigned int *id,
						Complex_f*gamval,int *gamin,float *dk[2], float akappa,dim3 dimGrid, dim3 dimBlock){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, (*akappa), jqq 
	 *
	 * Calls:
	 * ======
	 * zhaloswapdir, chaloswapdir, zhaloswapall (Non-mpi version could do without these)
	 *
	 * Parametrer:
	 * ==========
	 *
	 * Complex_f *phi:	The result container. This is NOT THE SAME AS THE GLOBAL Phi. But
	 * 			for consistency with the fortran code I'll keep the name here
	 * Complex_f r:		The array being acted on by M
	 *
	 * Returns:
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Hdslashd_f";
	int cuCpyStat=0;
	//__shared__ int gamin_s[16]; __shared__ Complex_f gamval_s[20];
	//intShare(gamin_s,gamin,16); floatShare(gamval_s,gamval,2*20);
	if((cuCpyStat=cudaMemcpy(phi, r, kferm2*sizeof(Complex_f),cudaMemcpyDefault))){
		fprintf(stderr,"Error %d in %s: Cuda failed to copy managed r into device Phi with code %d.\nExiting,,,\n\n",\
				CPYERROR,funcname,cuCpyStat);
		exit(cuCpyStat);
	}
	cuHdslashd_f<<<dimGrid,dimBlock>>>(phi,r,ut[0],ut[1],iu,id,gamval,gamin,dk[0],dk[1],akappa);
}

void cuTranspose_z(Complex *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	Complex *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(Complex));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(Complex),cudaMemcpyDefault);
	Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	//cublasCgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,fast_in,fast_out,(cuComplex *)&alpha,\
	//(cuComplex *)out,fast_out,NULL,(cuComplex *)&beta,fast_out,(cuComplex *)holder,fast_out);
	//cudaMemcpy(out,holder,fast_in*fast_out*sizeof(Complex_f),cudaMemcpyDefault);
	cudaFree(holder);
}
void cuTranspose_c(Complex_f *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	Complex_f *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(Complex_f));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(Complex_f),cudaMemcpyDefault);
	Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	cudaDeviceSynchronise();
	//cublasCgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,fast_in,fast_out,(cuComplex *)&alpha,\
	//(cuComplex *)out,fast_out,NULL,(cuComplex *)&beta,fast_out,(cuComplex *)holder,fast_out);
	//cudaMemcpy(out,holder,fast_in*fast_out*sizeof(Complex_f),cudaMemcpyDefault);
	cudaFree(holder);
}
void cuTranspose_d(double *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	double *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(double));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(double),cudaMemcpyDefault);
	Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	//cublasCgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,fast_in,fast_out,(cuComplex *)&alpha,\
	//(cuComplex *)out,fast_out,NULL,(cuComplex *)&beta,fast_out,(cuComplex *)holder,fast_out);
	//cudaMemcpy(out,holder,fast_in*fast_out*sizeof(double),cudaMemcpyDefault);
	cudaFree(holder);
}
void cuTranspose_f(float *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	float *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(float));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(float),cudaMemcpyDefault);
	Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	//cublasCgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,fast_in,fast_out,(cuComplex *)&alpha,\
	//(cuComplex *)out,fast_out,NULL,(cuComplex *)&beta,fast_out,(cuComplex *)holder,fast_out);
	//cudaMemcpy(out,holder,fast_in*fast_out*sizeof(float),cudaMemcpyDefault);
	cudaFree(holder);
}
void cuTranspose_I(int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	int *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(int));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(int),cudaMemcpyDefault);
	Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	//cublasCgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,fast_in,fast_out,(cuComplex *)&alpha,\
	//(cuComplex *)out,fast_out,NULL,(cuComplex *)&beta,fast_out,(cuComplex *)holder,fast_out);
	//cudaMemcpy(out,holder,fast_in*fast_out*sizeof(int),cudaMemcpyDefault);
	cudaFree(holder);
}
void cuTranspose_U(unsigned int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	unsigned int *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(unsigned int));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(unsigned int),cudaMemcpyDefault);
	Transpose<<<dimGrid,dimBlock>>>(out,holder,fast_in,fast_out);
	//cublasCgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,fast_in,fast_out,(cuComplex *)&alpha,\
	//(cuComplex *)out,fast_out,NULL,(cuComplex *)&beta,fast_out,(cuComplex *)holder,fast_out);
	//cudaMemcpy(out,holder,fast_in*fast_out*sizeof(int),cudaMemcpyDefault);
	cudaFree(holder);
}

template __global__ void Transpose<float>(float *, const float*, const int, const int);
template __global__ void Transpose<double>(double *, const double*, const int, const int);
template __global__ void Transpose<int>(int *, const int*, const int, const int);
template __global__ void Transpose<unsigned int>(unsigned int *, const unsigned int*, const int, const int);
template __global__ void Transpose<Complex_f>(Complex_f *, const Complex_f*, const int, const int);
template __global__ void Transpose<Complex>(Complex *, const Complex*, const int, const int);
