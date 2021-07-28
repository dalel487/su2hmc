extern "C"{
#include <assert.h>
#include <matrices.h>
#include <string.h>
}
#include <cuComplex.h>
__global__ void cuDslash(Complex *phi, Complex *r){
	char *funcname = "cuDslash";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			a_1=conj(jqq)*gamval[4][idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval[4][idirac];
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
				int igork1 = (igorkov<4) ? gamin[mu][idirac] : gamin[mu][idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]-\
						u12t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
								     //Dirac term
								     gamval[mu][idirac]*(u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]+\
										     u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]-\
										     conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]+\
										     u12t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]+\
						u11t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1])+\
									 //Dirac term
									 gamval[mu][idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]+\
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
			int igork1 = gamin[3][igorkov];	int igork1PP = igork1+4;

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
__global__ void cuDslashd(Complex *phi, Complex *r){
	char *funcname = "cuDslashd";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval[4][idirac];
			a_2=jqq*gamval[4][idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
			phi[(i*ngorkov+igork)*nc]+=a_2*r[(i*ngorkov+idirac)*nc];
			phi[(i*ngorkov+igork)*nc+1]+=a_2*r[(i*ngorkov+idirac)*nc+1];
		}

		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing.
				int idirac=igorkov%4;		
				int igork1 = (igorkov<4) ? gamin[mu][idirac] : gamin[mu][idirac]+4;
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ngorkov+igorkov)*nc]+=
					-akappa*(      u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]
							+u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							-u12t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval[mu][idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]
						     +u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]
						     -conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
						     +u12t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);

				phi[(i*ngorkov+igorkov)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							+u11t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1])
					-gamval[mu][idirac]*
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
			int igork1 = gamin[3][igorkov];	
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
__global__ void cuHdslash(Complex *phi, Complex *r){
	char *funcname = "cuHdslash";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]-\
						u12t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
								   //Dirac term
								   gamval[mu][idirac]*(u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]+\
										   u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]-\
										   conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]+\
										   u12t[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]+\
						u11t[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
								     //Dirac term
								     gamval[mu][idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]+\
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
			int igork1 = gamin[3][idirac];
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
__global__ void cuHdslashd(Complex *phi, Complex *r){
	char *funcname = "cuHdslashd";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				phi[(i*ndirac+idirac)*nc]+=
					-akappa*(u11t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]
							+u12t[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							-u12t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu][idirac]*
					(          u11t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]
						     +u12t[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]
						     -conj(u11t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
						     +u12t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=
					-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							+u11t[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval[mu][idirac]*
					(-conj(u12t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]
					 +conj(u11t[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]
					 -conj(u12t[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
					 -u11t[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3][idirac];
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
__global__ void cuHdslash(Complex_f *phi, Complex_f *r){
	char *funcname = "cuHdslash";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa_f*(u11t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]+\
						u12t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u11t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]-\
						u12t_f[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
								   //Dirac term
								   gamval_f[mu][idirac]*(u11t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]+\
										   u12t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]-\
										   conj(u11t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]+\
										   u12t_f[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa_f*(-conj(u12t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]+\
						conj(u11t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]+\
						conj(u12t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]+\
						u11t_f[did*ndim+mu]*r[(did*ndirac+idirac)*nc+1])+\
								     //Dirac term
								     gamval_f[mu][idirac]*(-conj(u12t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]+\
										     conj(u11t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]-\
										     conj(u12t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]-\
										     u11t_f[did*ndim+mu]*r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3][idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ndirac+idirac)*nc]+=
				-dk4p_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m_f[did]*(conj(u11t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4p_f[i]*(-conj(u12t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]-r[(uid*ndirac+igork1)*nc])
						+conj(u11t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]-r[(uid*ndirac+igork1)*nc+1]))
				-dk4m_f[did]*(conj(u12t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]+r[(did*ndirac+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]+r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
}
__global__ void cuHdslashd(Complex_f *phi, Complex_f *r){
	char *funcname = "cuHdslashd";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int idirac=0; idirac<ndirac; idirac++){
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				int igork1 = gamin[mu][idirac];
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				phi[(i*ndirac+idirac)*nc]+=
					-akappa_f*(u11t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc]
							+u12t_f[i*ndim+mu]*r[(uid*ndirac+idirac)*nc+1]
							+conj(u11t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							-u12t_f[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval_f[mu][idirac]*
					(          u11t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc]
						     +u12t_f[i*ndim+mu]*r[(uid*ndirac+igork1)*nc+1]
						     -conj(u11t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
						     +u12t_f[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=
					-akappa_f*(-conj(u12t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc]
							+conj(u11t_f[i*ndim+mu])*r[(uid*ndirac+idirac)*nc+1]
							+conj(u12t_f[did*ndim+mu])*r[(did*ndirac+idirac)*nc]
							+u11t_f[did*ndim+mu] *r[(did*ndirac+idirac)*nc+1])
					-gamval_f[mu][idirac]*
					(-conj(u12t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc]
					 +conj(u11t_f[i*ndim+mu])*r[(uid*ndirac+igork1)*nc+1]
					 -conj(u12t_f[did*ndim+mu])*r[(did*ndirac+igork1)*nc]
					 -u11t_f[did*ndim+mu] *r[(did*ndirac+igork1)*nc+1]);
			}
		}
#endif
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin[3][idirac];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m_f and dk4p_f swap under dagger
			phi[(i*ndirac+idirac)*nc]+=
				-dk4m_f[i]*(u11t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+u12t_f[i*ndim+3]*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p_f[did]*(conj(u11t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						-u12t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]+=
				-dk4m_f[i]*(-conj(u12t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc]+r[(uid*ndirac+igork1)*nc])
						+conj(u11t_f[i*ndim+3])*(r[(uid*ndirac+idirac)*nc+1]+r[(uid*ndirac+igork1)*nc+1]))
				-dk4p_f[did]*(conj(u12t_f[did*ndim+3])*(r[(did*ndirac+idirac)*nc]-r[(did*ndirac+igork1)*nc])
						+u11t_f[did*ndim+3] *(r[(did*ndirac+idirac)*nc+1]-r[(did*ndirac+igork1)*nc+1]));
		}
#endif
	}
}

__global__ void New_trial(double dt){
	char *funcname = "New_trial";
	const	int gsize = gridDim.x*gridDim.y*gridDim.z;
	const	int bsize = blockDim.x*blockDim.y*blockDim.z;
	const	int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const	int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId;i<kvol;i+=gsize){
		for(int mu = 0; mu<ndim; mu++){
			//Sticking to what was in the FORTRAN for variable names.
			//CCC for cosine SSS for sine AAA for...
			//Re-exponentiating the force field. Can be done analytically in SU(2)
			//using sine and cosine which is nice
			double AAA = dt*sqrt(pp[i*nadj*ndim+mu]*pp[i*nadj*ndim+mu]\
					+pp[(i*nadj+1)*ndim+mu]*pp[(i*nadj+1)*ndim+mu]\
					+pp[(i*nadj+2)*ndim+mu]*pp[(i*nadj+2)*ndim+mu]);
			double CCC = cos(AAA);
			double SSS = dt*sin(AAA)/AAA;
			Complex a11 = CCC+I*SSS*pp[(i*nadj+2)*ndim+mu];
			Complex a12 = pp[(i*nadj+1)*ndim+mu]*SSS + I*SSS*pp[i*nadj*ndim+mu];
			//b11 and b12 are u11t and u12t terms, so we'll use u12t directly
			//but use b11 for u11t to prevent RAW dependency
			Complex b11 = u11t[i*ndim+mu];
			u11t[i*ndim+mu] = a11*b11-a12*conj(u12t[i*ndim+mu]);
			u12t[i*ndim+mu] = a11*u12t[i*ndim+mu]+a12*conj(b11);
		}
	}
}
__global__ inline void cuReunitarise(){
	/*
	 * Reunitarises u11t and u12t as in conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]=1
	 *
	 * If you're looking at the FORTRAN code be careful. There are two header files
	 * for the /trial/ header. One with u11 u12 (which was included here originally)
	 * and the other with u11t and u12t.
	 *
	 * Globals:
	 * =======
	 * u11t, u12t
	 *
	 * Returns:
	 * ========
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "Reunitarise";
	const int gsize = gridDim.x*gridDim.y*gridDim.z;
	const int bsize = blockDim.x*blockDim.y*blockDim.z;
	const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId= blockId * bsize+(threadIdx.z * blockDim.y+ threadIdx.y)* blockDim.x+ threadIdx.x;
	for(int i=threadId; i<kvol*ndim; i+=gsize){
		//Declaring anorm inside the loop will hopefully let the compiler know it
		//is safe to vectorise aggessively
		double anorm=sqrt(creal(conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]));
		//		Exception handling code. May be faster to leave out as the exit prevents vectorisation.
		//		if(anorm==0){
		//			fprintf(stderr, "Error %i in %s on rank %i: anorm = 0 for μ=%i and i=%i.\nExiting...\n\n",
		//					DIVZERO, funcname, rank, mu, i);
		//			MPI_Finalise();
		//			exit(DIVZERO);
		//		}
		u11t[i]/=anorm;
		u12t[i]/=anorm;
	}
}

__host__ int Dslash(Complex *phi, Complex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
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
	char *funcname = "Dslash";
	//Get the halos in order
	ZHalo_swap_all(r, 16);

	//Mass term
	memcpy(phi, r, kferm*sizeof(Complex));
	//	cudaMemPrefetchAsync(u11t,kvol+halo,0
	cuDslash<<<dimGrid,dimBlock>>>(phi,r);
	return 0;
}
__host__ int Dslashd(Complex *phi, Complex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
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
	char *funcname = "Dslashd";
	//Get the halos in order
	ZHalo_swap_all(r, 16);

	//Mass term
	memcpy(phi, r, kferm*sizeof(Complex));
	//	cudaMemPrefetchAsync(u11t,kvol+halo,0
	cuDslashd<<<dimGrid,dimBlock>>>(phi,r);
	return 0;
}
__host__ int Hdslash(Complex *phi, Complex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
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
	char *funcname = "Hdslash";
	//Get the halos in order
	ZHalo_swap_all(r, 8);

	//Mass term
	memcpy(phi, r, kferm2*sizeof(Complex));
	//Spacelike term
	//#pragma offload target(mic)\
	in(r: length(kferm2Halo))\
		in(dk4m, dk4p: length(kvol+halo))\
		in(id, iu: length(ndim*kvol))\
		in(u11t, u12t: length(ndim*(kvol+halo)))\
		inout(phi: length(kferm2Halo))
		//	cudaMemPrefetchAsync(u11t,kvol+halo,0
		cuHdslash<<<dimGrid,dimBlock>>>(phi,r);
	return 0;
}
__host__ int Hdslashd(Complex *phi, Complex *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
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
	char *funcname = "Hdslashd";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
	ZHalo_swap_all(r, 8);

	//Looks like flipping the array ordering for C has meant a lot
	//of for loops. Sense we're jumping around quite a bit the cache is probably getting refreshed
	//anyways so memory access patterns mightn't be as big of an limiting factor here anyway

	//Mass term
	memcpy(phi, r, kferm2*sizeof(Complex));
	//Spacelike term
	//#pragma offload target(mic)\
	in(r: length(kferm2Halo))\
		in(dk4m, dk4p: length(kvol+halo))\
		in(id, iu: length(ndim*kvol))\
		in(u11t, u12t: length(ndim*(kvol+halo)))\
		inout(phi: length(kferm2Halo))
		//	cudaMemPrefetchAsync(u11t,kvol+halo,0
		cuHdslashd<<<dimGrid,dimBlock>>>(phi,r);
	return 0;
}

//Float editions
__host__ int Hdslash_f(Complex_f *phi, Complex_f *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
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
	char *funcname = "Hdslash";
	//Get the halos in order
	CHalo_swap_all(r, 8);

	//Mass term
	memcpy(phi, r, kferm2*sizeof(Complex_f));
	//Spacelike term
	//#pragma offload target(mic)\
	in(r: length(kferm2Halo))\
		in(dk4m, dk4p: length(kvol+halo))\
		in(id, iu: length(ndim*kvol))\
		in(u11t, u12t: length(ndim*(kvol+halo)))\
		inout(phi: length(kferm2Halo))
		//	cudaMemPrefetchAsync(u11t,kvol+halo,0
		cuHdslash_f<<<dimGrid,dimBlock>>>(phi,r);
	return 0;
}
__host__ int Hdslashd_f(Complex_f *phi, Complex_f *r){
	/*
	 * Evaluates phi= M*r
	 *
	 * Globals
	 * =======
	 * u11t, u12t, dk4p, dk4m, akappa, jqq 
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
	char *funcname = "Hdslashd";
	//Get the halos in order. Because C is row major, we need to extract the correct
	//terms for each halo first. Changing the indices was considered but that caused
	//issues with the BLAS routines.
	CHalo_swap_all(r, 8);

	//Looks like flipping the array ordering for C has meant a lot
	//of for loops. Sense we're jumping around quite a bit the cache is probably getting refreshed
	//anyways so memory access patterns mightn't be as big of an limiting factor here anyway

	//Mass term
	memcpy(phi, r, kferm2*sizeof(Complex_f));
	//Spacelike term
	//#pragma offload target(mic)\
	in(r: length(kferm2Halo))\
		in(dk4m, dk4p: length(kvol+halo))\
		in(id, iu: length(ndim*kvol))\
		in(u11t, u12t: length(ndim*(kvol+halo)))\
		inout(phi: length(kferm2Halo))
		//	cudaMemPrefetchAsync(u11t,kvol+halo,0
		cuHdslashd_f<<<dimGrid,dimBlock>>>(phi,r);
	return 0;
}

__host__ inline int Reunitarise(){
	cuReunitarise<<<dimGrid,dimBlock>>>();
	return 0;
}

#ifdef DIAGNOSTIC
int Diagnostics(int istart){
	/*
	 * Routine to check if the multiplication routines are working or not
	 * How I hope this will work is that
	 * 1)	Initialise the system
	 * 2) Just after the initialisation of the system but before anything
	 * 	else call this routine using the C Preprocessor.
	 * 3) Give dummy values for the fields and then do work with them
	 * Caveats? Well this could get messy if we call something we didn't
	 * realise was being called and hadn't initialised it properly (Congradq
	 * springs to mind straight away)
	 */
	char *funcname = "Diagnostics";

	//Initialise the arrays being used. Just going to assume MKL is being
	//used here will also assert the number of flavours for now to avoid issues
	//later
	assert(nf==1);

	R1= mkl_malloc(kfermHalo*sizeof(Complex),AVX);
	xi= mkl_malloc(kfermHalo*sizeof(Complex),AVX);
	Phi= mkl_malloc(nf*kfermHalo*sizeof(Complex),AVX); 
	X0= mkl_malloc(nf*kferm2Halo*sizeof(Complex),AVX); 
	X1= mkl_malloc(kferm2Halo*sizeof(Complex),AVX); 
	double *dSdpi = mkl_malloc(kmomHalo*sizeof(double), AVX);
	//pp is the momentum field
	pp = mkl_malloc(kmomHalo*sizeof(double), AVX);

	//Trial fields don't get modified so I'll set them up outside
	switch(istart){
		case(2):
#pragma omp parallel for simd aligned(u11t:AVX,u12t:AVX)
			for(int i =0; i<ndim*kvol; i+=8){
				u11t[i+0]=1+I; u12t[i]=1+I;
				u11t[i+1]=1-I; u12t[i+1]=1+I;
				u11t[i+2]=1+I; u12t[i+2]=1-I;
				u11t[i+3]=1-I; u12t[i+3]=1-I;
				u11t[i+4]=-1+I; u12t[i+4]=1+I;
				u11t[i+5]=1+I; u12t[i+5]=-1+I;
				u11t[i+6]=-1+I; u12t[i+6]=-1+I;
				u11t[i+7]=-1-I; u12t[i+7]=-1-I;
			}
			break;
		case(1):
			Trial_Exchange();
#pragma omp parallel sections num_threads(2)
			{
#pragma omp section
				{
					FILE *trial_out = fopen("u11t", "w");
					for(int i=0;i<ndim*(kvol+halo);i+=4)
						fprintf(trial_out,"%f+%fI\t%f+%fI\t%f+%fI\t%f+%fI\n",
								creal(u11t[i]),cimag(u11t[i]),creal(u11t[i+1]),cimag(u11t[i+1]),
								creal(u11t[2+i]),cimag(u11t[2+i]),creal(u11t[i+3]),cimag(u11t[i+3]));

					fclose(trial_out);
				}
#pragma omp section
				{
					FILE *trial_out = fopen("u12t", "w");
					for(int i=0;i<ndim*(kvol+halo);i+=4)
						fprintf(trial_out,"%f+%fI\t%f+%fI\t%f+%fI\t%f+%fI\n",
								creal(u12t[i]),cimag(u12t[i]),creal(u12t[i+1]),cimag(u12t[i+1]),
								creal(u12t[2+i]),cimag(u12t[2+i]),creal(u12t[i+3]),cimag(u12t[i+3]));
					fclose(trial_out);
				}
			}
			break;
		default:
			//Cold start as a default
			memcpy(u11t,u11,kvol*ndim*sizeof(Complex));
			memcpy(u12t,u12,kvol*ndim*sizeof(Complex));
			break;
	}
#pragma omp parallel for simd aligned(u11t:AVX,u12t:AVX) 
	for(int i=0; i<kvol*ndim; i++){
		//Declaring anorm inside the loop will hopefully let the compiler know it
		//is safe to vectorise aggessively
		double anorm=sqrt(conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]);
		assert(anorm!=0);
		u11t[i]/=anorm;
		u12t[i]/=anorm;
	}

	Trial_Exchange();
	for(int test = 0; test<=6; test++){
		//Reset between tests
#pragma omp parallel for simd
		for(int i=0; i<kferm; i++){
			R1[i]=0.5; Phi[i]=0.5;xi[i]=0.5;
		}
#pragma omp parallel for simd
		for(int i=0; i<kferm2; i++){
			X0[i]=0.5;
			X1[i]=0.5;
		}
#pragma omp parallel for simd
		for(int i=0; i<kmomHalo; i++)
			dSdpi[i] = 0;
		FILE *output_old, *output;
		switch(test){
			case(0):
				output_old = fopen("dslash_old", "w");
				for(int i = 0; i< kferm; i+=8)
					fprintf(output_old, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
				fclose(output_old);
				Dslash(xi, R1);
				output = fopen("dslash", "w");
				for(int i = 0; i< kferm; i+=8)
					fprintf(output, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
				fclose(output);
				break;
			case(1):
				output_old = fopen("dslashd_old", "w");
				for(int i = 0; i< kferm; i+=8)
					fprintf(output_old, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
				fclose(output_old);
				Dslashd(xi, R1);
				output = fopen("dslashd", "w");
				for(int i = 0; i< kferm; i+=8)
					fprintf(output, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
				fclose(output);
				break;
			case(2):	
				output_old = fopen("hdslash_old", "w");
				for(int i = 0; i< kferm2; i+=8)
					fprintf(output_old, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				fclose(output_old);
				Hdslash(X1, X0);
				output = fopen("hdslash", "w");
				for(int i = 0; i< kferm2; i+=8)
					fprintf(output, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				fclose(output);
				break;
			case(3):	
				output_old = fopen("hdslashd_old", "w");
				for(int i = 0; i< kferm2; i+=8)
					fprintf(output_old, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				fclose(output_old);
				Hdslashd(X1, X0);
				output = fopen("hdslashd", "w");
				for(int i = 0; i< kferm2; i+=8)
					fprintf(output, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(X1[i]),cimag(X1[i]),creal(X1[i+1]),cimag(X1[i+1]),
							creal(X1[i+2]),cimag(X1[i+2]),creal(X1[i+3]),cimag(X1[i+3]),
							creal(X1[i+4]),cimag(X1[i+4]),creal(X1[i+5]),cimag(X1[i+5]),
							creal(X1[i+6]),cimag(X1[i+6]),creal(X1[i+7]),cimag(X1[i+7]));
				fclose(output);
				break;
				//Two force cases because of the flag
			case(4):	
				output_old = fopen("force_0_old", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output_old, "%f\t%f\t%f\t%f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output_old);
				Force(dSdpi, 0, rescgg);	
				output = fopen("force_0", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%f\t%f\t%f\t%f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
			case(5):	
				output_old = fopen("force_1_old", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output_old, "%f\t%f\t%f\t%f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output_old);
				Force(dSdpi, 1, rescgg);	
				output = fopen("force_1", "w");
				for(int i = 0; i< kmom; i+=4)
					fprintf(output, "%f\t%f\t%f\t%f\n", dSdpi[i], dSdpi[i+1], dSdpi[i+2], dSdpi[i+3]);
				fclose(output);
				break;
			case(6):
				output = fopen("Measure", "w");
				int itercg=0;
				double pbp, endenf, denf; Complex qq, qbqb;
				Measure(&pbp, &endenf, &denf, &qq, &qbqb, respbp, &itercg);
				fprintf(output,"pbp=%f\tendenf=%f\tdenf=%f\nqq=%f+(%f)i\tqbqb=%f+(%f)i\titercg=%i\n\n",
						pbp,endenf,denf,creal(qq),cimag(qq),creal(qbqb),cimag(qbqb),itercg);
				//				Congradp(0,respbp,&itercg);
				for(int i = 0; i< kferm; i+=8)
					fprintf(output, "%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n%f+%fI\t%f+%fI\n\n",
							creal(xi[i]),cimag(xi[i]),creal(xi[i+1]),cimag(xi[i+1]),
							creal(xi[i+2]),cimag(xi[i+2]),creal(xi[i+3]),cimag(xi[i+3]),
							creal(xi[i+4]),cimag(xi[i+4]),creal(xi[i+5]),cimag(xi[i+5]),
							creal(xi[i+6]),cimag(xi[i+6]),creal(xi[i+7]),cimag(xi[i+7])	);
				fclose(output);
				break;
		}
	}

	//George Michael's favourite bit of the code
	mkl_free(dk4m); mkl_free(dk4p); mkl_free(R1); mkl_free(dSdpi); mkl_free(pp);
	mkl_free(Phi); mkl_free(u11t); mkl_free(u12t); mkl_free(xi);
	mkl_free(X0); mkl_free(X1); mkl_free(u11); mkl_free(u12);
	mkl_free(id); mkl_free(iu); mkl_free(hd); mkl_free(hu);
	mkl_free(pcoord);

	MPI_Finalise();
	exit(0);
}
#endif
