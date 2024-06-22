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
	const char *funcname = "cuDslash0_f";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	__shared__ Complex_f u11s[128];	__shared__ Complex_f u12s[128];
	__shared__ Complex_f u11sd[128];	__shared__ Complex_f u12sd[128];
	__shared__ Complex_f ru[128*nc]; __shared__ Complex_f rd[128*nc];
	__shared__ Complex_f rgu[128*nc]; __shared__ Complex_f rgd[128*nc];
	__shared__ float  dk4ms[128]; __shared__ float dk4ps[128];
	__shared__ float  dk4msd[128]; __shared__ float dk4psd[128];
	__shared__ Complex_f phi_s[128];

	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		dk4ms[bthreadId]=dk4m[i];	dk4ps[bthreadId]=dk4p[i];
		for(int idirac=0;idirac<ndirac;idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			a_1=conj(jqq)*gamval_d[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval_d[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc];
			phi[(i*ngorkov+igork)*nc]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			u11s[bthreadId]=u11t[i*ndim+mu]; u12s[bthreadId]=u12t[i*ndim+mu];
			u11sd[bthreadId]=u11t[did*ndim+mu]; u12sd[bthreadId]=u12t[did*ndim+mu];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				int idirac=igorkov%4;		
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int igork1 = (igorkov<4) ? gamin_d[mu*ndirac+idirac] : gamin_d[mu*ndirac+idirac]+4;
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[(uid*ngorkov+igorkov)*nc+c];
					rd[bthreadId+128*c]=r[(did*ngorkov+igorkov)*nc+c];
					rgu[bthreadId+128*c]=r[(uid*ngorkov+igork1)*nc+c];
					rgd[bthreadId+128*c]=r[(did*ngorkov+igork1)*nc+c];
				}
				//Can manually vectorise with a pragma?
				phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc];
				phi_s[bthreadId]+=-akappa*(u11s[bthreadId]*ru[bthreadId]+\
						u12s[bthreadId]*ru[bthreadId+128]+\
						conj(u11sd[bthreadId])*rd[bthreadId]-\
						u12sd[bthreadId]*rd[bthreadId+128]);
				//Dirac term
				phi_s[bthreadId]+=gamval_d[mu*ndirac+idirac]*(u11s[bthreadId]*rgu[bthreadId]+\
						u12s[bthreadId]*rgu[bthreadId+128]-\
						conj(u11sd[bthreadId])*rgd[bthreadId]+\
						u12sd[bthreadId]*rgd[bthreadId+128]);
				phi[(i*ngorkov+igorkov)*nc]=phi_s[bthreadId];

				phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc+1];
				phi_s[bthreadId]+=-akappa*(-conj(u12s[bthreadId])*ru[bthreadId]+\
						conj(u11s[bthreadId])*ru[bthreadId+128]+\
						conj(u12sd[bthreadId])*rd[bthreadId]+\
						u11sd[bthreadId]*rd[bthreadId+128]);
				//Dirac term
				phi_s[bthreadId]+=gamval_d[mu*ndirac+idirac]*(-conj(u12s[bthreadId])*rgu[bthreadId]+\
						conj(u11s[bthreadId])*rgu[bthreadId+128]-\
						conj(u12sd[bthreadId])*rgd[bthreadId]-\
						u11sd[bthreadId]*rgd[bthreadId+128]);
				phi[(i*ngorkov+igorkov)*nc+1]=phi_s[bthreadId];
			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
#ifndef NO_TIME
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		u11s[bthreadId]=u11t[i*ndim+3]; u12s[bthreadId]=u12t[i*ndim+3];
		u11sd[bthreadId]=u11t[did*ndim+3]; u12sd[bthreadId]=u12t[did*ndim+3];
		dk4msd[bthreadId]=dk4m[did];	dk4psd[bthreadId]=dk4p[did];
		for(int igorkov=0;igorkov<ndirac;igorkov++){
			int igork1 = gamin_d[3*ndirac+igorkov];
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[(uid*ngorkov+igorkov)*nc+c];
					rd[bthreadId+128*c]=r[(did*ngorkov+igorkov)*nc+c];
					rgu[bthreadId+128*c]=r[(uid*ngorkov+igork1)*nc+c];
					rgd[bthreadId+128*c]=r[(did*ngorkov+igork1)*nc+c];
				}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc];
			phi_s[bthreadId]+=
				-dk4ps[bthreadId]*(u11s[bthreadId]*(ru[bthreadId]-rgu[bthreadId])
						+u12s[bthreadId]*(ru[bthreadId+128]-rgu[bthreadId+128]))
				-dk4msd[bthreadId]*(conj(u11sd[bthreadId])*(rd[bthreadId]+rgd[bthreadId])
						-u12sd[bthreadId] *(rd[bthreadId+128]+rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkov)*nc]=phi_s[bthreadId];

			phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc+1];
			phi_s[bthreadId]+=
				-dk4ps[bthreadId]*(-conj(u12s[bthreadId])*(ru[bthreadId]-rgu[bthreadId])
						+conj(u11s[bthreadId])*(ru[bthreadId+128]-rgu[bthreadId+128]))
				-dk4msd[bthreadId]*(conj(u12sd[bthreadId])*(rd[bthreadId]+rgd[bthreadId])
						+u11sd[bthreadId] *(rd[bthreadId+128]+rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkov)*nc+1]=phi_s[bthreadId];
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1PP = igork1+4;
			//And the gorkov terms. Note that dk4p and dk4m swap positions compared to the above				
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[(uid*ngorkov+igorkovPP)*nc+c];
					rd[bthreadId+128*c]=r[(did*ngorkov+igorkovPP)*nc+c];
					rgu[bthreadId+128*c]=r[(uid*ngorkov+igork1PP)*nc+c];
					rgd[bthreadId+128*c]=r[(did*ngorkov+igork1PP)*nc+c];
				}
			phi_s[bthreadId]=phi[(i*ngorkov+igorkovPP)*nc];
			phi_s[bthreadId]+=-dk4ms[bthreadId]*(u11s[bthreadId]*(ru[bthreadId]-rgu[bthreadId])+
					u12s[bthreadId]*(ru[bthreadId+128]-rgu[bthreadId+128]))-
				dk4psd[bthreadId]*(conj(u11sd[bthreadId])*(rd[bthreadId]+rgd[bthreadId])-
						u12sd[bthreadId]*(rd[bthreadId+128]+rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkovPP)*nc]=phi_s[bthreadId];

			phi_s[bthreadId]=phi[(i*ngorkov+igorkovPP)*nc+1];
			phi_s[bthreadId]+=-dk4ms[bthreadId]*(conj(-u12s[bthreadId])*(ru[bthreadId]-rgu[bthreadId])
					+conj(u11s[bthreadId])*(ru[bthreadId+128]-rgu[bthreadId+128]))
				-dk4psd[bthreadId]*(conj(u12sd[bthreadId])*(rd[bthreadId]+rgd[bthreadId])
						+u11sd[bthreadId]*(rd[bthreadId+128]+rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkovPP)*nc+1]=phi_s[bthreadId];
		}
#endif
	}
}
__global__ void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		__shared__ Complex_f *gamval_d,	int *gamin_d,	float *dk4m, float *dk4p, Complex_f jqq, float akappa){
	const char *funcname = "cuDslashd_f";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	__shared__ Complex_f u11s[128];	__shared__ Complex_f u12s[128];
	__shared__ Complex_f u11sd[128];	__shared__ Complex_f u12sd[128];
	__shared__ Complex_f ru[128*nc]; __shared__ Complex_f rd[128*nc];
	__shared__ Complex_f rgu[128*nc]; __shared__ Complex_f rgd[128*nc];
	__shared__ float  dk4ms[128]; __shared__ float dk4ps[128];
	__shared__ float  dk4msd[128]; __shared__ float dk4psd[128];
	__shared__ Complex_f phi_s[128];

	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		dk4ms[bthreadId]=dk4m[i];	dk4ps[bthreadId]=dk4p[i];
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
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
			u11s[bthreadId]=u11t[i*ndim+mu]; u12s[bthreadId]=u12t[i*ndim+mu];
			u11sd[bthreadId]=u11t[did*ndim+mu]; u12sd[bthreadId]=u12t[did*ndim+mu];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin_d[mu*ndirac+idirac] : gamin_d[mu*ndirac+idirac]+4;
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[(uid*ngorkov+igorkov)*nc+c];
					rd[bthreadId+128*c]=r[(did*ngorkov+igorkov)*nc+c];
					rgu[bthreadId+128*c]=r[(uid*ngorkov+igork1)*nc+c];
					rgd[bthreadId+128*c]=r[(did*ngorkov+igork1)*nc+c];
				}
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc];
				phi_s[bthreadId]-=
					akappa*(u11s[bthreadId]*ru[bthreadId]
							+u12s[bthreadId]*ru[bthreadId+128]
							+conj(u11sd[bthreadId])*rd[bthreadId]
							-u12sd[bthreadId] *rd[bthreadId+128]);

				//Dirac term
				phi_s[bthreadId]-=gamval_d[mu*ndirac+idirac]*
					(u11s[bthreadId]*rgu[bthreadId]
					 +u12s[bthreadId]*rgu[bthreadId+128]
					 -conj(u11sd[bthreadId])*rgd[bthreadId]
					 +u12sd[bthreadId] *rgd[bthreadId+128]);
				phi[(i*ngorkov+igorkov)*nc]=phi_s[bthreadId];

				phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc+1];
				phi_s[bthreadId]-=
					akappa*(-conj(u12s[bthreadId])*ru[bthreadId]
							+conj(u11s[bthreadId])*ru[bthreadId+128]
							+conj(u12sd[bthreadId])*rd[bthreadId]
							+u11sd[bthreadId] *rd[bthreadId+128]);
				//Dirac term
				phi_s[bthreadId]-=gamval_d[mu*ndirac+idirac]*
					(-conj(u12s[bthreadId])*rgu[bthreadId]
					 +conj(u11s[bthreadId])*rgu[bthreadId+128]
					 -conj(u12sd[bthreadId])*rgd[bthreadId]
					 -u11sd[bthreadId] *rgd[bthreadId+128]);
				phi[(i*ngorkov+igorkov)*nc+1]=phi_s[bthreadId];

			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
#ifndef NO_TIME
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		u11s[bthreadId]=u11t[i*ndim+3]; u12s[bthreadId]=u12t[i*ndim+3];
		u11sd[bthreadId]=u11t[did*ndim+3]; u12sd[bthreadId]=u12t[did*ndim+3];
		dk4msd[bthreadId]=dk4m[did];	dk4psd[bthreadId]=dk4p[did];
		for(int igorkov=0; igorkov<ndirac; igorkov++){
			int igork1 = gamin_d[3*ndirac+igorkov];	
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[(uid*ngorkov+igorkov)*nc+c];
					rd[bthreadId+128*c]=r[(did*ngorkov+igorkov)*nc+c];
					rgu[bthreadId+128*c]=r[(uid*ngorkov+igork1)*nc+c];
					rgd[bthreadId+128*c]=r[(did*ngorkov+igork1)*nc+c];
				}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc];
			phi_s[bthreadId]+=
				-dk4ms[bthreadId]*(u11s[bthreadId]*(ru[bthreadId]+rgu[bthreadId])
						+u12s[bthreadId]*(ru[bthreadId+128]+rgu[bthreadId+128]))
				-dk4psd[bthreadId]*(conj(u11sd[bthreadId])*(rd[bthreadId]-rgd[bthreadId])
						-u12sd[bthreadId] *(rd[bthreadId+128]-rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkov)*nc]=phi_s[bthreadId];

			phi_s[bthreadId]=phi[(i*ngorkov+igorkov)*nc+1];
			phi_s[bthreadId]+=
				-dk4ms[bthreadId]*(-conj(u12s[bthreadId])*(ru[bthreadId]+rgu[bthreadId])
						+conj(u11s[bthreadId])*(ru[bthreadId+128]+rgu[bthreadId+128]))
				-dk4psd[bthreadId]*(conj(u12sd[bthreadId])*(rd[bthreadId]-rgd[bthreadId])
						+u11sd[bthreadId] *(rd[bthreadId+128]-rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkov)*nc+1]=phi_s[bthreadId];
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1PP = igork1+4;
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[(uid*ngorkov+igorkovPP)*nc+c];
					rd[bthreadId+128*c]=r[(did*ngorkov+igorkovPP)*nc+c];
					rgu[bthreadId+128*c]=r[(uid*ngorkov+igork1PP)*nc+c];
					rgd[bthreadId+128*c]=r[(did*ngorkov+igork1PP)*nc+c];
				}
			//And the Gor'kov terms. Note that dk4p and dk4m swap positions compared to the above				
			phi_s[bthreadId]=phi[(i*ngorkov+igorkovPP)*nc];
			phi_s[bthreadId]+=-dk4ps[bthreadId]*(u11s[bthreadId]*(ru[bthreadId]+rgu[bthreadId])
					+u12s[bthreadId]*(ru[bthreadId+128]+rgu[bthreadId+128]))
				-dk4msd[bthreadId]*(conj(u11sd[bthreadId])*(rd[bthreadId]-rgd[bthreadId])
						-u12sd[bthreadId]*(rd[bthreadId+128]-rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkovPP)*nc]=phi_s[bthreadId];

			phi_s[bthreadId]=phi[(i*ngorkov+igorkovPP)*nc+1];
			phi_s[bthreadId]+=dk4ps[bthreadId]*(conj(u12s[bthreadId])*(ru[bthreadId]+rgu[bthreadId])
					-conj(u11s[bthreadId])*(ru[bthreadId+128]+rgu[bthreadId+128]))
				-dk4msd[bthreadId]*(conj(u12sd[bthreadId])*(rd[bthreadId]-rgd[bthreadId])
						+u11sd[bthreadId]*(rd[bthreadId+128]-rgd[bthreadId+128]));
			phi[(i*ngorkov+igorkovPP)*nc+1]=phi_s[bthreadId];
		}
#endif
	}
}

__global__ void cuHdslash_f(Complex_f *phi, const Complex_f *r, const Complex_f *u11t, const Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		const Complex_f gamval[20],	int *gamin_d,	const float *dk4m, const float *dk4p, const float akappa){
	/*
	 * Half Dslash float precision acting on colour index zero
	 */
	const char *funcname = "cuHdslash0_f";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;


	//Right. Time to prefetch into shared memory
	__shared__ Complex_f ru[128*nc]; __shared__ Complex_f rd[128*nc];
	__shared__ Complex_f rgu[128*nc]; __shared__ Complex_f rgd[128*nc];
	__shared__ Complex_f u11s[128];	__shared__ Complex_f u12s[128];
	__shared__ Complex_f u11sd[128];	__shared__ Complex_f u12sd[128];
	__shared__ float  dk4ms[128]; __shared__  float dk4ps[128];
	__shared__ Complex_f phi_s[128];
	for(int i=gthreadId;i<kvol;i+=bsize*gsize){
		dk4ps[bthreadId]=dk4p[i];
		//Do we need to sync threads if each thread only accesses the value it put in shared memory?
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			u11s[bthreadId]=u11t[i+kvol*mu];	u12s[bthreadId]=u12t[i+kvol*mu];
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			u11sd[bthreadId]=u11t[did+kvol*mu];	u12sd[bthreadId]=u12t[did+kvol*mu];
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin_d[mu*ndirac+idirac];
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[uid+kvol*(idirac*nc+c)];
					rd[bthreadId+128*c]=r[did+kvol*(idirac*nc+c)];
					rgu[bthreadId+128*c]=r[uid+kvol*(igork1*nc+c)];
					rgd[bthreadId+128*c]=r[did+kvol*(igork1*nc+c)];
				}
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi_s[bthreadId]=phi[i+kvol*(0+nc*idirac)];
				phi_s[bthreadId]+=-akappa*(u11s[bthreadId]*ru[bthreadId]+\
						u12s[bthreadId]*ru[bthreadId+128]+\
						conj(u11sd[bthreadId])*rd[bthreadId]-\
						u12sd[bthreadId]*rd[bthreadId+128]);
				//Dirac term
				phi_s[bthreadId]+=gamval[mu*ndirac+idirac]*(u11s[bthreadId]*rgu[bthreadId]+\
						u12s[bthreadId]*rgu[bthreadId+128]-\
						conj(u11sd[bthreadId])*rgd[bthreadId]+\
						u12sd[bthreadId]*rgd[bthreadId+128]);
				phi[i+kvol*(0+nc*idirac)]=phi_s[bthreadId];

				phi_s[bthreadId]=phi[i+kvol*(1+nc*idirac)];
				phi_s[bthreadId]+=-akappa*(-conj(u12s[bthreadId])*ru[bthreadId]+\
						conj(u11s[bthreadId])*ru[bthreadId+128]+\
						conj(u12sd[bthreadId])*rd[bthreadId]+\
						u11sd[bthreadId]*rd[bthreadId+128]);
				//Dirac term
				phi_s[bthreadId]+=gamval[mu*ndirac+idirac]*(-conj(u12s[bthreadId])*rgu[bthreadId]+\
						conj(u11s[bthreadId])*rgu[bthreadId+128]-\
						conj(u12sd[bthreadId])*rgd[bthreadId]-\
						u11sd[bthreadId]*rgd[bthreadId+128]);
				phi[i+kvol*(1+nc*idirac)]=phi_s[bthreadId];
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		dk4ms[bthreadId]=dk4m[did];
		u11s[bthreadId]=u11t[i+kvol*3];	u12s[bthreadId]=u12t[i+kvol*3];
		u11sd[bthreadId]=u11t[did+kvol*3];	u12sd[bthreadId]=u12t[did+kvol*3];
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
			for(int c=0;c<nc;c++){
				ru[bthreadId+128*c]=r[uid+kvol*(idirac*nc+c)];
				rd[bthreadId+128*c]=r[did+kvol*(idirac*nc+c)];
				rgu[bthreadId+128*c]=r[uid+kvol*(igork1*nc+c)];
				rgd[bthreadId+128*c]=r[did+kvol*(igork1*nc+c)];
			}
			phi_s[bthreadId]=phi[i+kvol*(0+nc*idirac)];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi_s[bthreadId]-=
				dk4ps[bthreadId]*(u11s[bthreadId]*(ru[bthreadId]-rgu[bthreadId])
						+u12s[bthreadId]*(ru[bthreadId+128]-rgu[bthreadId+128]))
				+dk4ms[bthreadId]*(conj(u11sd[bthreadId])*(rd[bthreadId]+rgd[bthreadId])
						-u12sd[bthreadId] *(rd[bthreadId+128]+rgd[bthreadId+128]));
			phi[i+kvol*(0+nc*idirac)]=phi_s[bthreadId];

			phi_s[bthreadId]=phi[i+kvol*(1+nc*idirac)];
			phi_s[bthreadId]-=
				dk4ps[bthreadId]*(-conj(u12s[bthreadId])*(ru[bthreadId]-rgu[bthreadId])
						+conj(u11s[bthreadId])*(ru[bthreadId+128]-rgu[bthreadId+128]))
				+dk4ms[bthreadId]*(conj(u12sd[bthreadId])*(rd[bthreadId]+rgd[bthreadId])
						+u11sd[bthreadId] *(rd[bthreadId+128]+rgd[bthreadId+128]));
			phi[i+kvol*(1+nc*idirac)]=phi_s[bthreadId];
#endif
		}
	}
}
__global__ void cuHdslashd_f(Complex_f *phi, const Complex_f *r, const Complex_f *u11t, const Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		const Complex_f gamval[20],	const int *gamin_d,	const float *dk4m, const float *dk4p, const float akappa){
	const char *funcname = "cuHdslashd0_f";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	const int gthreadId= blockId * bsize+bthreadId;

	__shared__ Complex_f u11s[128];	__shared__ Complex_f u12s[128];
	__shared__ Complex_f u11sd[128];	__shared__ Complex_f u12sd[128];
	__shared__ Complex_f ru[128*nc]; __shared__ Complex_f rd[128*nc];
	__shared__ Complex_f rgu[128*nc]; __shared__ Complex_f rgd[128*nc];
	__shared__ float  dk4ms[128]; __shared__ float dk4ps[128];
	__shared__ Complex_f phi_s[128];
	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
		dk4ms[bthreadId]=dk4m[i];
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
			u11s[bthreadId]=u11t[i+kvol*mu];	u12s[bthreadId]=u12t[i+kvol*mu];
			u11sd[bthreadId]=u11t[did+kvol*mu];	u12sd[bthreadId]=u12t[did+kvol*mu];
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin_d[mu*ndirac+idirac];
				for(int c=0;c<nc;c++){
					ru[bthreadId+128*c]=r[uid+kvol*(idirac*nc+c)];
					rd[bthreadId+128*c]=r[did+kvol*(idirac*nc+c)];
					rgu[bthreadId+128*c]=r[uid+kvol*(igork1*nc+c)];
					rgd[bthreadId+128*c]=r[did+kvol*(igork1*nc+c)];
				}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi_s[bthreadId]=phi[i+kvol*(0+nc*idirac)];
				phi_s[bthreadId]-=akappa*(u11s[bthreadId]*ru[bthreadId]
						+u12s[bthreadId]*ru[bthreadId+128]
						+conj(u11sd[bthreadId])*rd[bthreadId]
						-u12sd[bthreadId] *rd[bthreadId+128]);
				//Dirac term
				phi_s[bthreadId]-=gamval[mu*ndirac+idirac]*
					(u11s[bthreadId]*rgu[bthreadId]
					 +u12s[bthreadId]*rgu[bthreadId+128]
					 -conj(u11sd[bthreadId])*rgd[bthreadId]
					 +u12sd[bthreadId] *rgd[bthreadId+128]);
				phi[i+kvol*(0+nc*idirac)]=phi_s[bthreadId];

				phi_s[bthreadId]=phi[i+kvol*(1+nc*idirac)];
				phi_s[bthreadId]-=akappa*(-conj(u12s[bthreadId])*ru[bthreadId]
						+conj(u11s[bthreadId])*ru[bthreadId+128]
						+conj(u12sd[bthreadId])*rd[bthreadId]
						+u11sd[bthreadId] *rd[bthreadId+128]);
				//Dirac term
				phi_s[bthreadId]-=gamval[mu*ndirac+idirac]*
					(-conj(u12s[bthreadId])*rgu[bthreadId]
					 +conj(u11s[bthreadId])*rgu[bthreadId+128]
					 -conj(u12sd[bthreadId])*rgd[bthreadId]
					 -u11sd[bthreadId] *rgd[bthreadId+128]);
				phi[i+kvol*(1+nc*idirac)]=phi_s[bthreadId];
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		u11s[bthreadId]=u11t[i+kvol*3];	u12s[bthreadId]=u12t[i+kvol*3];
		u11sd[bthreadId]=u11t[did+kvol*3];	u12sd[bthreadId]=u12t[did+kvol*3];
		dk4ps[bthreadId]=dk4p[did];
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
			for(int c=0;c<nc;c++){
				ru[bthreadId+128*c]=r[uid+kvol*(idirac*nc+c)];
				rd[bthreadId+128*c]=r[did+kvol*(idirac*nc+c)];
				rgu[bthreadId+128*c]=r[uid+kvol*(igork1*nc+c)];
				rgd[bthreadId+128*c]=r[did+kvol*(igork1*nc+c)];
			}
			phi_s[bthreadId]=phi[i+kvol*(0+nc*idirac)];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
			phi_s[bthreadId]+=
				-dk4ms[bthreadId]*(u11s[bthreadId]*(ru[bthreadId]+rgu[bthreadId])
						+u12s[bthreadId]*(ru[bthreadId+128]+rgu[bthreadId+128]))
				-dk4ps[bthreadId]*(conj(u11sd[bthreadId])*(rd[bthreadId]-rgd[bthreadId])
						-u12sd[bthreadId] *(rd[bthreadId+128]-rgd[bthreadId+128]));
			phi[i+kvol*(0+nc*idirac)]=phi_s[bthreadId];

			phi_s[bthreadId]=phi[i+kvol*(1+nc*idirac)];
			phi_s[bthreadId]-=
				dk4ms[bthreadId]*(-conj(u12s[bthreadId])*(ru[bthreadId]+rgu[bthreadId])
						+conj(u11s[bthreadId])*(ru[bthreadId+128]+rgu[bthreadId+128]))
				+dk4ps[bthreadId]*(conj(u12sd[bthreadId])*(rd[bthreadId]-rgd[bthreadId])
						+u11sd[bthreadId] *(rd[bthreadId+128]-rgd[bthreadId+128]));
			phi[i+kvol*(1+nc*idirac)]=phi_s[bthreadId];
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
__global__ void Transpose_f(Complex_f *out, Complex_f *in, const int fast_in, const int fast_out){
	const char *funcname="Transpose_f";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int bthreadId= (threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
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
void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval,int *gamin,	float *dk4m, float *dk4p, float akappa,\ 
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
	cuHdslash_f<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
}
void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
		Complex_f *gamval,int *gamin,	float *dk4m, float *dk4p, float akappa,\
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
	cuHdslashd_f<<<dimGrid,dimBlock>>>(phi,r,u11t,u12t,iu,id,gamval,gamin,dk4m,dk4p,akappa);
}

void Transpose_f(Complex_f *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock){
	Complex_f *holder;
	cudaMalloc((void **)&holder,fast_in*fast_out*sizeof(Complex_f));
	cudaMemcpy(holder,out,fast_in*fast_out*sizeof(Complex_f),cudaMemcpyDefault);
	Transpose_f<<<dimBlock,dimGrid>>>(out,holder,fast_in,fast_out);
	Complex_f alpha=1; Complex_f beta=0;
	//cublasCgeam(cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,fast_in,fast_out,(cuComplex *)&alpha,\
	//(cuComplex *)out,fast_out,NULL,(cuComplex *)&beta,fast_out,(cuComplex *)holder,fast_out);
	//cudaMemcpy(out,holder,fast_in*fast_out*sizeof(Complex_f),cudaMemcpyDefault);
	cudaFree(holder);
}
