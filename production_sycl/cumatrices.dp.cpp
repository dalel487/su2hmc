#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <matrices.h>
#include <string.h>
#include	<thrust_complex.h>
void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, Complex_f jqq, float akappa,
		const sycl::nd_item<3> &item_ct1){
	char *funcname = "cuDslash";
	const int gsize = item_ct1.get_group_range(2) *
		item_ct1.get_group_range(1) *
		item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2) *
		item_ct1.get_local_range(1) *
		item_ct1.get_local_range(0);
	const int blockId =
		item_ct1.get_group(2) +
		item_ct1.get_group(1) * item_ct1.get_group_range(2) +
		item_ct1.get_group_range(2) * item_ct1.get_group_range(1) *
		item_ct1.get_group(0);
	const int threadId =
		blockId * bsize +
		(item_ct1.get_local_id(0) * item_ct1.get_local_range(1) +
		 item_ct1.get_local_id(1)) *
		item_ct1.get_local_range(2) +
		item_ct1.get_local_id(2);
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			a_1=conj((Complex)jqq)*gamval_d[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			a_2=-(Complex)jqq*gamval_d[4*ndirac+idirac];
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
void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, Complex_f jqq, float akappa,
		const sycl::nd_item<3> &item_ct1){
	char *funcname = "cuDslashd";
	const int gsize = item_ct1.get_group_range(2) *
		item_ct1.get_group_range(1) *
		item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2) *
		item_ct1.get_local_range(1) *
		item_ct1.get_local_range(0);
	const int blockId =
		item_ct1.get_group(2) +
		item_ct1.get_group(1) * item_ct1.get_group_range(2) +
		item_ct1.get_group_range(2) * item_ct1.get_group_range(1) *
		item_ct1.get_group(0);
	const int threadId =
		blockId * bsize +
		(item_ct1.get_local_id(0) * item_ct1.get_local_range(1) +
		 item_ct1.get_local_id(1)) *
		item_ct1.get_local_range(2) +
		item_ct1.get_local_id(2);
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj((Complex)jqq)*gamval_d[4*ndirac+idirac];
			a_2=(Complex)jqq*gamval_d[4*ndirac+idirac];
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
void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, float akappa,
		const sycl::nd_item<3> &item_ct1){
	char *funcname = "cuHdslash";
	const int gsize = item_ct1.get_group_range(2) *
		item_ct1.get_group_range(1) *
		item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2) *
		item_ct1.get_local_range(1) *
		item_ct1.get_local_range(0);
	const int blockId =
		item_ct1.get_group(2) +
		item_ct1.get_group(1) * item_ct1.get_group_range(2) +
		item_ct1.get_group_range(2) * item_ct1.get_group_range(1) *
		item_ct1.get_group(0);
	const int threadId =
		blockId * bsize +
		(item_ct1.get_local_id(0) * item_ct1.get_local_range(1) +
		 item_ct1.get_local_id(1)) *
		item_ct1.get_local_range(2) +
		item_ct1.get_local_id(2);
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
void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval_d, int *gamin_d,	double *dk4m, double *dk4p, float akappa,
		const sycl::nd_item<3> &item_ct1){
	char *funcname = "cuHdslashd";
	const int gsize = item_ct1.get_group_range(2) *
		item_ct1.get_group_range(1) *
		item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2) *
		item_ct1.get_local_range(1) *
		item_ct1.get_local_range(0);
	const int blockId =
		item_ct1.get_group(2) +
		item_ct1.get_group(1) * item_ct1.get_group_range(2) +
		item_ct1.get_group_range(2) * item_ct1.get_group_range(1) *
		item_ct1.get_group(0);
	const int threadId =
		blockId * bsize +
		(item_ct1.get_local_id(0) * item_ct1.get_local_range(1) +
		 item_ct1.get_local_id(1)) *
		item_ct1.get_local_range(2) +
		item_ct1.get_local_id(2);
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

//Dslash_f Index 0
void cuDslash0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_d, int *gamin_d, float *dk4m, float *dk4p, Complex_f jqq, float akappa, const sycl::nd_item<3> &item_ct1){

	const int gsize = item_ct1.get_group_range(2)*item_ct1.get_group_range(1)*item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2)*item_ct1.get_local_range(1)*item_ct1.get_local_range(0);
	const int blockId = item_ct1.get_group(2)+ item_ct1.get_group(1) * item_ct1.get_group_range(2)+ item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
	const int threadId= blockId * bsize+(item_ct1.get_local_id(0) * item_ct1.get_local_range(1)+ item_ct1.get_local_id(1))* item_ct1.get_local_range(2)+ item_ct1.get_local_id(2);
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		for(int idirac=0;idirac<ndirac;idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			a_1=conj(jqq)*gamval_d[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval_d[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc];
			phi[(i*ngorkov+igork)*nc]+=a_2*r[(i*ngorkov+idirac)*nc];
		}
		//Spacelike terms. Here's hoping I haven't put time as the zeroth component somewhere!
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			for(int igorkov=0; igorkov<ngorkov; igorkov++){
				int idirac=igorkov%4;		
				//FORTRAN had mod((igorkov-1),4)+1 to prevent issues with non-zero indexing in the dirac term.
				int igork1 = (igorkov<4) ? gamin_d[mu*ndirac+idirac] : gamin_d[mu*ndirac+idirac]+4;
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				//				phi[(i*ngorkov+igorkov)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]+
				//						u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]+
				//						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]-
				//						u12t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1]);
				//				//Dirac term
				//				phi[(i*ngorkov+igorkov)*nc]+=gamval_d[mu*ndirac+idirac]*(u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]+
				//						u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]-
				//						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]+
				//						u12t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);
				phi[(i*ngorkov+igorkov)*nc]+=-akappa*(u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]-\
						u12t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1]);
				//Dirac term
				phi[(i*ngorkov+igorkov)*nc]+=gamval_d[mu*ndirac+idirac]*(u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]+\
						u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]-\
						conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]+\
						u12t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);

			}
		}
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
#endif
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int igorkov=0;igorkov<ndirac;igorkov++){
			int igork1 = gamin_d[3*ndirac+igorkov];
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])+
					u12t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))-
				dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])-
						u12t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));
		}
#endif
	}
}
void cuDslashd0_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_d, int *gamin_d, float *dk4m, float *dk4p, Complex_f jqq, float akappa, const sycl::nd_item<3> &item_ct1){

	const	int gsize = item_ct1.get_group_range(2)*item_ct1.get_group_range(1)*item_ct1.get_group_range(0);
	const	int bsize = item_ct1.get_local_range(2)*item_ct1.get_local_range(1)*item_ct1.get_local_range(0);
	const	int blockId = item_ct1.get_group(2)+ item_ct1.get_group(1) * item_ct1.get_group_range(2)+ item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
	const	int threadId= blockId * bsize+(item_ct1.get_local_id(0) * item_ct1.get_local_range(1)+ item_ct1.get_local_id(1))* item_ct1.get_local_range(2)+ item_ct1.get_local_id(2);
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval_d[4*ndirac+idirac];
			a_2=jqq*gamval_d[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc]+=a_1*r[(i*ngorkov+igork)*nc];
			phi[(i*ngorkov+igork)*nc]+=a_2*r[(i*ngorkov+idirac)*nc];
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
				phi[(i*ngorkov+igorkov)*nc]-=
					akappa*(u11t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc]
							+u12t[i*ndim+mu]*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u11t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							-u12t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1]);
				phi[(i*ngorkov+igorkov)*nc]-=gamval_d[mu*ndirac+idirac]*
					(u11t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc]
					 +u12t[i*ndim+mu]*r[(uid*ngorkov+igork1)*nc+1]
					 -conj(u11t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]
					 +u12t[did*ndim+mu] *r[(did*ngorkov+igork1)*nc+1]);

			}
		}
#endif
		//Timelike terms next. These run from igorkov=0..3 and 4..7 with slightly different rules for each
		//We can fit it into a single loop by declaring igorkovPP=igorkov+4 instead of looping igorkov=4..7  separately
		//Note that for the igorkov 4..7 loop idirac=igorkov-4, so we don't need to declare idiracPP separately
		//Under dagger, dk4p and dk4m get swapped and the dirac component flips sign.
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
#ifndef NO_TIME
		for(int igorkov=0; igorkov<ndirac; igorkov++){
			int igork1 = gamin_d[3*ndirac+igorkov];	
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ngorkov+igorkov)*nc]+=
				-dk4m[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+u12t[i*ndim+3]*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						-u12t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));
			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
												//the FORTRAN code did it.
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc]+=-dk4p[i]*(u11t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])
					+u12t[i*ndim+3]*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk4m[did]*(conj(u11t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])
						-u12t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));
		}
#endif
	}
}
//Dslash_f Index 1
void cuDslash1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_d, int *gamin_d, float *dk4m, float *dk4p, Complex_f jqq, float akappa, const sycl::nd_item<3> &item_ct1){

	const int gsize = item_ct1.get_group_range(2)*item_ct1.get_group_range(1)*item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2)*item_ct1.get_local_range(1)*item_ct1.get_local_range(0);
	const int blockId = item_ct1.get_group(2)+ item_ct1.get_group(1) * item_ct1.get_group_range(2)+ item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
	const int threadId= blockId * bsize+(item_ct1.get_local_id(0) * item_ct1.get_local_range(1)+ item_ct1.get_local_id(1))* item_ct1.get_local_range(2)+ item_ct1.get_local_id(2);
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			a_1=conj(jqq)*gamval_d[4*ndirac+idirac];
			//We subtract a_2, hence the minus
			a_2=-jqq*gamval_d[4*ndirac+idirac];
			phi[(i*ngorkov+idirac)*nc+1]+=a_1*r[(i*ngorkov+igork)*nc+1];
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
				//Wilson + Dirac term in that order. 
				//		phi[(i*ngorkov+igorkov)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]+\
				//				conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]+\
				//				conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]+\
				//				u11t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1]);
				//		//Dirac term
				//		phi[(i*ngorkov+igorkov)*nc+1]+=gamval_d[mu*ndirac+idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]+\
				//				conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc+1]-\
				//				conj(u12t[did*ndim+mu])*r[(did*ngorkov+igork1)*nc]-\
				//				u11t[did*ndim+mu]*r[(did*ngorkov+igork1)*nc+1]);
				phi[(i*ngorkov+igorkov)*nc+1]+=-akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]+\
						conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]+\
						conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]+\
						u11t[did*ndim+mu]*r[(did*ngorkov+igorkov)*nc+1]);
				//Dirac term
				phi[(i*ngorkov+igorkov)*nc+1]+=gamval_d[mu*ndirac+idirac]*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igork1)*nc]+\
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

			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4p[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]-r[(uid*ngorkov+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]-r[(uid*ngorkov+igork1)*nc+1]))
				-dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]+r[(did*ngorkov+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]+r[(did*ngorkov+igork1)*nc+1]));
			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc+1]+=-dk4m[i]*(conj(-u12t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]-r[(uid*ngorkov+igork1PP)*nc])
					+conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]-r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]+r[(did*ngorkov+igork1PP)*nc])
						+u11t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]+r[(did*ngorkov+igork1PP)*nc+1]));
		}
#endif
	}
}
void cuDslashd1_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval_d, int *gamin_d, float *dk4m, float *dk4p, Complex_f jqq, float akappa, const sycl::nd_item<3> &item_ct1){

	const	int gsize = item_ct1.get_group_range(2)*item_ct1.get_group_range(1)*item_ct1.get_group_range(0);
	const	int bsize = item_ct1.get_local_range(2)*item_ct1.get_local_range(1)*item_ct1.get_local_range(0);
	const	int blockId = item_ct1.get_group(2)+ item_ct1.get_group(1) * item_ct1.get_group_range(2)+ item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
	const	int threadId= blockId * bsize+(item_ct1.get_local_id(0) * item_ct1.get_local_range(1)+ item_ct1.get_local_id(1))* item_ct1.get_local_range(2)+ item_ct1.get_local_id(2);
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		//Diquark Term (antihermitian) The signs of a_1 and a_2 below flip under dagger
		for(int idirac = 0; idirac<ndirac; idirac++){
			int igork = idirac+4;
			Complex_f a_1, a_2;
			//We subtract a_1, hence the minus
			a_1=-conj(jqq)*gamval_d[4*ndirac+idirac];
			a_2=jqq*gamval_d[4*ndirac+idirac];
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
				phi[(i*ngorkov+igorkov)*nc+1]-=
					akappa*(-conj(u12t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc]
							+conj(u11t[i*ndim+mu])*r[(uid*ngorkov+igorkov)*nc+1]
							+conj(u12t[did*ndim+mu])*r[(did*ngorkov+igorkov)*nc]
							+u11t[did*ndim+mu] *r[(did*ngorkov+igorkov)*nc+1]);
				phi[(i*ngorkov+igorkov)*nc+1]-=gamval_d[mu*ndirac+idirac]*
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
			phi[(i*ngorkov+igorkov)*nc+1]+=
				-dk4m[i]*(-conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc]+r[(uid*ngorkov+igork1)*nc])
						+conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkov)*nc+1]+r[(uid*ngorkov+igork1)*nc+1]))
				-dk4p[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkov)*nc]-r[(did*ngorkov+igork1)*nc])
						+u11t[did*ndim+3] *(r[(did*ngorkov+igorkov)*nc+1]-r[(did*ngorkov+igork1)*nc+1]));

			int igorkovPP=igorkov+4; 	//idirac = igorkov; It is a bit redundant but I'll mention it as that's how
			int igork1PP = igork1+4;
			//And the +4 terms. Note that dk4p and dk4m swap positions compared to the above				
			phi[(i*ngorkov+igorkovPP)*nc+1]+=dk4p[i]*(conj(u12t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc]+r[(uid*ngorkov+igork1PP)*nc])
					-conj(u11t[i*ndim+3])*(r[(uid*ngorkov+igorkovPP)*nc+1]+r[(uid*ngorkov+igork1PP)*nc+1]))
				-dk4m[did]*(conj(u12t[did*ndim+3])*(r[(did*ngorkov+igorkovPP)*nc]-r[(did*ngorkov+igork1PP)*nc])
						+u11t[did*ndim+3]*(r[(did*ngorkov+igorkovPP)*nc+1]-r[(did*ngorkov+igork1PP)*nc+1]));
		}
#endif
	}
}
//IDEA: Split the Dslash routines into different streams for each colour index so the can run concurrently
//There are no race contitions.
//HDslash_f Index 0
void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin_d, float *dk4m, float *dk4p, float akappa, const sycl::nd_item<3> &item_ct1){

	const int gsize = item_ct1.get_group_range(2)*item_ct1.get_group_range(1)*item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2)*item_ct1.get_local_range(1)*item_ct1.get_local_range(0);
	const int blockId = item_ct1.get_group(2)+ item_ct1.get_group(1) * item_ct1.get_group_range(2)+ item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
	const int bthreadId= (item_ct1.get_local_id(0) * item_ct1.get_local_range(1)+ item_ct1.get_local_id(1))* item_ct1.get_local_range(2)+ item_ct1.get_local_id(2);
	const int gthreadId= blockId * bsize+bthreadId;


	//Right. Time to prefetch into shared memory
	//TODO:	Gorkov index terms
	//__shared__ Complex_f ru[ndim*bsize*nc], rd[ndim*bsize*nc];
	//extern __shared__ Complex_f s[];
	//Complex_f *ru = s;
	Complex_f ru[128*nc];  Complex_f rd[128*nc];
	Complex_f rgu[128*nc];  Complex_f rgd[128*nc];
	Complex_f u11s[128];	 Complex_f u12s[128];
	Complex_f u11sd[128];	 Complex_f u12sd[128];

	for(int i=gthreadId;i<kvol;i+=bsize*gsize){
		//Do we need to sync threads if each thread only accesses the value it put in shared memory?
#ifndef NO_SPACE
		for(int mu = 0; mu <3; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			u11s[bthreadId]=u11t[i*ndim+mu];	u12s[bthreadId]=u12t[i*ndim+mu];
			u11sd[bthreadId]=u11t[did*ndim+mu];	u12sd[bthreadId]=u12t[did*ndim+mu];
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin_d[mu*ndirac+idirac];
				for(int c=0;c<nc;c++){
					ru[bthreadId*nc+c]=r[(uid*ndirac+idirac)*nc+c];
					rd[bthreadId*nc+c]=r[(did*ndirac+idirac)*nc+c];
					rgu[bthreadId*nc+c]=r[(uid*ndirac+igork1)*nc+c];
					rgd[bthreadId*nc+c]=r[(did*ndirac+igork1)*nc+c];
				}
				//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way
				phi[(i*ndirac+idirac)*nc]+=-akappa*(u11s[bthreadId]*ru[bthreadId*nc]+\
						u12s[bthreadId]*ru[bthreadId*nc+1]+\
						conj(u11sd[bthreadId])*rd[bthreadId*nc]-\
						u12sd[bthreadId]*rd[bthreadId*nc+1]);
				//Dirac term
				phi[(i*ndirac+idirac)*nc]+=gamval[mu*ndirac+idirac]*(u11s[bthreadId]*rgu[bthreadId*nc]+\
						u12s[bthreadId]*rgu[bthreadId*nc+1]-\
						conj(u11sd[bthreadId])*rgd[bthreadId*nc]+\
						u12sd[bthreadId]*rgd[bthreadId*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]+=-akappa*(-conj(u12s[bthreadId])*ru[bthreadId*nc]+\
						conj(u11s[bthreadId])*ru[bthreadId*nc+1]+\
						conj(u12sd[bthreadId])*rd[bthreadId*nc]+\
						u11sd[bthreadId]*rd[bthreadId*nc+1]);
				//Dirac term
				phi[(i*ndirac+idirac)*nc+1]+=gamval[mu*ndirac+idirac]*(-conj(u12s[bthreadId])*rgu[bthreadId*nc]+\
						conj(u11s[bthreadId])*rgu[bthreadId*nc+1]-\
						conj(u12sd[bthreadId])*rgd[bthreadId*nc]-\
						u11sd[bthreadId]*rgd[bthreadId*nc+1]);
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		u11s[bthreadId]=u11t[i*ndim+3];	u12s[bthreadId]=u12t[i*ndim+3];
		u11sd[bthreadId]=u11t[did*ndim+3];	u12sd[bthreadId]=u12t[did*ndim+3];
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
			for(int c=0;c<nc;c++){
				ru[bthreadId*nc+c]=r[(uid*ndirac+idirac)*nc+c];
				rd[bthreadId*nc+c]=r[(did*ndirac+idirac)*nc+c];
				rgu[bthreadId*nc+c]=r[(uid*ndirac+igork1)*nc+c];
				rgd[bthreadId*nc+c]=r[(did*ndirac+igork1)*nc+c];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			phi[(i*ndirac+idirac)*nc]-=
				dk4p[i]*(u11s[bthreadId]*(ru[bthreadId*nc]-rgu[bthreadId*nc])
						+u12s[bthreadId]*(ru[bthreadId*nc+1]-rgu[bthreadId*nc+1]));
			phi[(i*ndirac+idirac)*nc]-=
				dk4m[did]*(conj(u11sd[bthreadId])*(rd[bthreadId*nc]+rgd[bthreadId*nc])
						-u12sd[bthreadId] *(rd[bthreadId*nc+1]+rgd[bthreadId*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]-=
				dk4p[i]*(-conj(u12s[bthreadId])*(ru[bthreadId*nc]-rgu[bthreadId*nc])
						+conj(u11s[bthreadId])*(ru[bthreadId*nc+1]-rgu[bthreadId*nc+1]))
				+dk4m[did]*(conj(u12sd[bthreadId])*(rd[bthreadId*nc]+rgd[bthreadId*nc])
						+u11sd[bthreadId] *(rd[bthreadId*nc+1]+rgd[bthreadId*nc+1]));
#endif
		}
	}
}
void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		Complex_f *gamval, int *gamin_d, float *dk4m, float *dk4p, float akappa, const sycl::nd_item<3> &item_ct1){

	const int gsize = item_ct1.get_group_range(2)*item_ct1.get_group_range(1)*item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2)*item_ct1.get_local_range(1)*item_ct1.get_local_range(0);
	const int blockId = item_ct1.get_group(2)+ item_ct1.get_group(1) * item_ct1.get_group_range(2)+ item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
	const int bthreadId= (item_ct1.get_local_id(0) * item_ct1.get_local_range(1)+ item_ct1.get_local_id(1))* item_ct1.get_local_range(2)+ item_ct1.get_local_id(2);
	const int gthreadId= blockId * bsize+bthreadId;

	Complex_f ru[128*nc];  Complex_f rd[128*nc];
	Complex_f rgu[128*nc];  Complex_f rgd[128*nc];
	Complex_f u11s[128];	 Complex_f u12s[128];
	Complex_f u11sd[128];	 Complex_f u12sd[128];

	for(int i=gthreadId;i<kvol;i+=gsize*bsize){
#ifndef NO_SPACE
		for(int mu = 0; mu <ndim-1; mu++){
			int did=id[mu+ndim*i]; int uid = iu[mu+ndim*i];
			//FORTRAN had mod((idirac-1),4)+1 to prevent issues with non-zero indexing.
			u11s[bthreadId]=u11t[i*ndim+mu];	u12s[bthreadId]=u12t[i*ndim+mu];
			u11sd[bthreadId]=u11t[did*ndim+mu];	u12sd[bthreadId]=u12t[did*ndim+mu];
			for(int idirac=0; idirac<ndirac; idirac++){
				int igork1 = gamin_d[mu*ndirac+idirac];
				for(int c=0;c<nc;c++){
					ru[bthreadId*nc+c]=r[(uid*ndirac+idirac)*nc+c];
					rd[bthreadId*nc+c]=r[(did*ndirac+idirac)*nc+c];
					rgu[bthreadId*nc+c]=r[(uid*ndirac+igork1)*nc+c];
					rgd[bthreadId*nc+c]=r[(did*ndirac+igork1)*nc+c];
				}
				//Can manually vectorise with a pragma?
				//Wilson + Dirac term in that order. Definitely easier
				//to read when split into different loops, but should be faster this way

				phi[(i*ndirac+idirac)*nc]-=akappa*(u11s[bthreadId]*ru[bthreadId*nc]
						+u12s[bthreadId]*ru[bthreadId*nc+1]
						+conj(u11sd[bthreadId])*rd[bthreadId*nc]
						-u12sd[bthreadId] *rd[bthreadId*nc+1]);
				//Dirac term
				phi[(i*ndirac+idirac)*nc]-=gamval[mu*ndirac+idirac]*
					(u11s[bthreadId]*rgu[bthreadId*nc]
					 +u12s[bthreadId]*rgu[bthreadId*nc+1]
					 -conj(u11sd[bthreadId])*rgd[bthreadId*nc]
					 +u12sd[bthreadId] *rgd[bthreadId*nc+1]);

				phi[(i*ndirac+idirac)*nc+1]-=akappa*(-conj(u12s[bthreadId])*ru[bthreadId*nc]
						+conj(u11s[bthreadId])*ru[bthreadId*nc+1]
						+conj(u12sd[bthreadId])*rd[bthreadId*nc]
						+u11sd[bthreadId] *rd[bthreadId*nc+1]);
				//Dirac term
				phi[(i*ndirac+idirac)*nc+1]-=gamval[mu*ndirac+idirac]*
					(-conj(u12s[bthreadId])*rgu[bthreadId*nc]
					 +conj(u11s[bthreadId])*rgu[bthreadId*nc+1]
					 -conj(u12sd[bthreadId])*rgd[bthreadId*nc]
					 -u11sd[bthreadId] *rgd[bthreadId*nc+1]);
			}
		}
#endif
#ifndef NO_TIME
		//Timelike terms
		int did=id[3+ndim*i]; int uid = iu[3+ndim*i];
		u11s[bthreadId]=u11t[i*ndim+3];	u12s[bthreadId]=u12t[i*ndim+3];
		u11sd[bthreadId]=u11t[did*ndim+3];	u12sd[bthreadId]=u12t[did*ndim+3];
		for(int idirac=0; idirac<ndirac; idirac++){
			int igork1 = gamin_d[3*ndirac+idirac];
			for(int c=0;c<nc;c++){
				ru[bthreadId*nc+c]=r[(uid*ndirac+idirac)*nc+c];
				rd[bthreadId*nc+c]=r[(did*ndirac+idirac)*nc+c];
				rgu[bthreadId*nc+c]=r[(uid*ndirac+igork1)*nc+c];
				rgd[bthreadId*nc+c]=r[(did*ndirac+igork1)*nc+c];
			}
			//Factorising for performance, we get dk4?*u1?*(+/-r_wilson -/+ r_dirac)
			//dk4m and dk4p swap under dagger
			phi[(i*ndirac+idirac)*nc]+=
				-dk4m[i]*(u11s[bthreadId]*(ru[bthreadId*nc]+rgu[bthreadId*nc])
						+u12s[bthreadId]*(ru[bthreadId*nc+1]+rgu[bthreadId*nc+1]))
				-dk4p[did]*(conj(u11sd[bthreadId])*(rd[bthreadId*nc]-rgd[bthreadId*nc])
						-u12sd[bthreadId] *(rd[bthreadId*nc+1]-rgd[bthreadId*nc+1]));

			phi[(i*ndirac+idirac)*nc+1]-=
				dk4m[i]*(-conj(u12s[bthreadId])*(ru[bthreadId*nc]+rgu[bthreadId*nc])
						+conj(u11s[bthreadId])*(ru[bthreadId*nc+1]+rgu[bthreadId*nc+1]))
				+dk4p[did]*(conj(u12sd[bthreadId])*(rd[bthreadId*nc]-rgd[bthreadId*nc])
						+u11sd[bthreadId] *(rd[bthreadId*nc+1]-rgd[bthreadId*nc+1]));
#endif
		}
	}
}

void cuReunitarise(Complex *u11t, Complex * u12t,
		const sycl::nd_item<3> &item_ct1){
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
	const int gsize = item_ct1.get_group_range(2) *
		item_ct1.get_group_range(1) *
		item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2) *
		item_ct1.get_local_range(1) *
		item_ct1.get_local_range(0);
	const int blockId =
		item_ct1.get_group(2) +
		item_ct1.get_group(1) * item_ct1.get_group_range(2) +
		item_ct1.get_group_range(2) * item_ct1.get_group_range(1) *
		item_ct1.get_group(0);
	const int threadId =
		blockId * bsize +
		(item_ct1.get_local_id(0) * item_ct1.get_local_range(1) +
		 item_ct1.get_local_id(1)) *
		item_ct1.get_local_range(2) +
		item_ct1.get_local_id(2);
	for(int i=threadId; i<kvol*ndim; i+=gsize*bsize){
		//Declaring anorm inside the loop will hopefully let the compiler know it
		//is safe to vectorise aggessively
		double anorm=sqrt(conj(u11t[i])*u11t[i]+conj(u12t[i])*u12t[i]).real();
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
void cuNew_trial(double dt, double *pp, Complex *u11t, Complex *u12t,int mu,
		const sycl::nd_item<3> &item_ct1){
	char *funcname = "New_trial";
	const int gsize = item_ct1.get_group_range(2) *
		item_ct1.get_group_range(1) *
		item_ct1.get_group_range(0);
	const int bsize = item_ct1.get_local_range(2) *
		item_ct1.get_local_range(1) *
		item_ct1.get_local_range(0);
	const int blockId =
		item_ct1.get_group(2) +
		item_ct1.get_group(1) * item_ct1.get_group_range(2) +
		item_ct1.get_group_range(2) * item_ct1.get_group_range(1) *
		item_ct1.get_group(0);
	const int threadId =
		blockId * bsize +
		(item_ct1.get_local_id(0) * item_ct1.get_local_range(1) +
		 item_ct1.get_local_id(1)) *
		item_ct1.get_local_range(2) +
		item_ct1.get_local_id(2);
	for(int i=threadId;i<kvol;i+=gsize*bsize){
		//Sticking to what was in the FORTRAN for variable names.
		//CCC for cosine SSS for sine AAA for...
		//Re-exponentiating the force field. Can be done analytically in SU(2)
		//using sine and cosine which is nice
		double AAA = dt*sqrt(pp[i*nadj*ndim+mu]*pp[i*nadj*ndim+mu]\
				+pp[(i*nadj+1)*ndim+mu]*pp[(i*nadj+1)*ndim+mu]\
				+pp[(i*nadj+2)*ndim+mu]*pp[(i*nadj+2)*ndim+mu]);
		double CCC = sycl::cos(AAA);
		double SSS = dt * sycl::sin(AAA) / AAA;
		Complex a11 = CCC+I*SSS*pp[(i*nadj+2)*ndim+mu];
		Complex a12 = pp[(i*nadj+1)*ndim+mu]*SSS + I*SSS*pp[i*nadj*ndim+mu];
		//b11 and b12 are u11t and u12t terms, so we'll use u12t directly
		//but use b11 for u11t to prevent RAW dependency
		Complex b11 = u11t[i*ndim+mu];
		u11t[i*ndim+mu] = a11*b11-a12*conj(u12t[i*ndim+mu]);
		u12t[i*ndim+mu] = a11*u12t[i*ndim+mu]+a12*conj(b11);
	}
}

//Calling Functions
//================
void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,
		unsigned int *iu, unsigned int *id, Complex *gamval, int *gamin,
		double *dk4m, double *dk4p, Complex_f jqq, float akappa,
		sycl::range<3> dimGrid, sycl::range<3> dimBlock) {
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
	char *funcname = "Dslash";
	streams[0].memcpy(phi, r, kferm*sizeof(Complex));
	/*
DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuDslash(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,jqq, akappa,item_ct1);
			});
}
void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,
		unsigned int *iu, unsigned int *id, Complex *gamval, int *gamin,
		double *dk4m, double *dk4p, Complex_f jqq, float akappa,\ 
		sycl::range<3>
		dimGrid,
		sycl::range<3> dimBlock) {
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
	char *funcname = "Dslashd";
	streams[0].memcpy(phi, r, kferm*sizeof(Complex));
	/*
DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuDslashd(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,jqq, akappa,item_ct1);
			});
}
void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,
		unsigned int *iu, unsigned int *id, Complex *gamval, int *gamin,
		double *dk4m, double *dk4p, float akappa,\ 
		sycl::range<3>
		dimGrid,
		sycl::range<3> dimBlock) {
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
	char *funcname = "Hdslash";
	streams[0].memcpy(phi, r, kferm2*sizeof(Complex));
	/*
DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuHdslash(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p, akappa,item_ct1);
			});
}
void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,
		unsigned int *iu, unsigned int *id, Complex *gamval, int *gamin,
		double *dk4m, double *dk4p, float akappa,\ 
		sycl::range<3>
		dimGrid,
		sycl::range<3> dimBlock) {
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
	char *funcname = "Hdslashd";
	//Spacelike term
	streams[0].memcpy(phi, r, kferm2*sizeof(Complex));
	/*
DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuHdslashd(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p, akappa,item_ct1);
			});
}

//Float editions
void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,
		unsigned int *iu, unsigned int *id, Complex_f *gamval,
		int *gamin, float *dk4m, float *dk4p, Complex_f jqq,
		float akappa,	sycl::range<3> dimGrid, sycl::range<3> dimBlock) {
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
	char *funcname = "Dslash_f";
	streams[0].memcpy(phi, r, kferm*sizeof(Complex_f));
	streams[0].parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuDslash0_f(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,jqq, akappa,item_ct1);
			});
	streams[0].parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuDslash1_f(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,jqq, akappa,item_ct1);
			});
}
void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,
		unsigned int *iu, unsigned int *id, Complex_f *gamval,
		int *gamin, float *dk4m, float *dk4p, Complex_f jqq,
		float akappa,\ 
		sycl::range<3>
		dimGrid,
		sycl::range<3> dimBlock) {
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
	char *funcname = "Dslashd_f";
	streams[0].memcpy(phi, r, kferm*sizeof(Complex_f));
	streams[0].parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuDslashd0_f(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,jqq, akappa,item_ct1);
			});
	streams[0].parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuDslashd1_f(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,jqq, akappa,item_ct1);
			});
}
void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,
		unsigned int *iu, unsigned int *id, Complex_f *gamval,
		int *gamin, float *dk4m, float *dk4p, float akappa,\ 
		sycl::range<3> dimGrid,		sycl::range<3> dimBlock) {
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
	char *funcname = "Hdslash_f";
	streams[0].memcpy(phi, r, kferm*sizeof(Complex_f));
	const int bsize = dimGrid[2] * dimGrid[1] * dimGrid[0];
	const int shareSize= ndim*bsize*nc*sizeof(Complex_f);
	/*
DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuHdslash_f(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,
					akappa,item_ct1);
			});
}
void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t,
		Complex_f *u12t, unsigned int *iu, unsigned int *id,
		Complex_f *gamval, int *gamin, float *dk4m, float *dk4p,
		float akappa, sycl::range<3> dimGrid,
		sycl::range<3> dimBlock) {
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
	char *funcname = "Hdslashd_f";
	streams[0].memcpy(phi, r, kferm*sizeof(Complex_f));
	/*
DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuHdslashd_f(phi, r, u11t, u12t, iu, id, gamval, gamin, dk4m, dk4p,
					akappa,item_ct1);
			});
}

void cuReunitarise(Complex *u11t, Complex *u12t, sycl::range<3> dimGrid,
		sycl::range<3> dimBlock) {
	/*
DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> item_ct1) {
			cuReunitarise(u11t, u12t,item_ct1);
			});
	cudaDeviceSynchronise();
}
void cuNew_trial(double dt, double *pp, Complex *u11t, Complex *u12t,
		sycl::range<3> dimGrid, sycl::range<3> dimBlock) {
	for(int mu=0;mu<ndim;mu++)
		streams[0].parallel_for(
				sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
				[=](sycl::nd_item<3> item_ct1) {
				cuNew_trial(dt,pp,u11t, u12t,mu,item_ct1);
				});
}