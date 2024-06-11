/*
 * Code for bosonic observables
 * Basically polyakov loop and Plaquette routines
 */
#include	<par_mpi.h>
#include	<su2hmc.h>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
// #include <thrust/execution_policy.h>

void cuAverage_Plaquette(double *hgs, double *hgt, Complex_f *u11t,
                         Complex_f *u12t, unsigned int *iu,
                         sycl::range<3> dimGrid, sycl::range<3> dimBlock) {
//	float *hgs_d, *hgt_d;
	int device=-1;
        device = dpct::dev_mgr::instance().current_device_id();
        float *hgs_d, *hgt_d;
	//Thrust want things in a weird format for the reduction, thus we oblige
	cudaMallocAsync((void **)&hgs_d,kvol*sizeof(float),streams[0]);
	thrust::device_ptr<float> hgs_T = thrust::device_pointer_cast(hgs_d);
	cudaMallocAsync((void **)&hgt_d,kvol*sizeof(float),streams[1]);
	thrust::device_ptr<float> hgt_T = thrust::device_pointer_cast(hgt_d);

        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) {
                    (hgs_d, hgt_d, u11t, u12t, iu);
            });

        *hgs= (double)thrust::reduce(hgs_T,hgs_T+kvol,(float)0);
	*hgt= (double)thrust::reduce(hgt_T,hgt_T+kvol,(float)0);
	//Temporary holders to keep OMP happy.
	/*
	double hgs_t=0; double hgt_t=0;
#pragma omp parallel for simd reduction(+:hgs_t,hgt_t)
	for(int i=0;i<kvol;i++){
		hgs_t+=hgs_d[i]; hgt_t+=hgt_d[i];
	}
	*hgs=hgs_t; *hgt=hgt_t;
	*/
	cudaFreeAsync(hgs_d,streams[0]); cudaFreeAsync(hgt_d,streams[1]);
}
void cuPolyakov(Complex_f *Sigma11, Complex_f *Sigma12, Complex_f *u11t,
                Complex_f *u12t, sycl::range<3> dimGrid,
                sycl::range<3> dimBlock) {
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) {
                    (Sigma11, Sigma12, u11t, u12t);
            });
}
//CUDA Kernels
void cuAverage_Plaquette(float *hgs_d, float *hgt_d, Complex_f *u11t, Complex_f *u12t, unsigned int *iu,
                         const sycl::nd_item<3> &item_ct1){
	char *funcname = "cuSU2plaq";
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
        //TODO: Check if μ and ν loops inside of site loop is faster. I suspect it is due to memory locality.
	for(int i=threadId;i<kvol;i+=bsize*gsize){
		hgt_d[i]=0; hgs_d[i]=0;

		for(int mu=1;mu<ndim;mu++)
			for(int nu=0;nu<mu;nu++){
				//This is threadsafe as the μ and ν loops are not distributed across threads
				switch(mu){
					//Time component
					case(ndim-1):
						hgt_d[i] -= SU2plaq(u11t,u12t,iu,i,mu,nu);
						break;
						//Space component
					default:
						hgs_d[i] -=	SU2plaq(u11t,u12t,iu,i,mu,nu);
						break;
				}
			}
	}
}
float SU2plaq(Complex_f *u11t, Complex_f *u12t, unsigned int *iu, int i, int mu, int nu){
	/*
	 * Calculates the plaquette at site i in the μ-ν direction
	 *
	 * Parameters:
	 * ==========
	 * Complex u11t, u12t:	Trial fields
	 * unsignedi int *iu:	Upper halo indices
	 * int mu, nu:				Plaquette direction. Note that mu and nu can be negative
	 * 							to facilitate calculating plaquettes for Clover terms. No
	 * 							sanity checks are conducted on them in this routine.
	 *
	 * Returns:
	 * ========
	 * double corresponding to the plaquette value
	 *
	 */
	const char *funcname = "SU2plaq";
	int uidm = iu[mu+ndim*i]; 

	Complex_f Sigma11=u11t[i*ndim+mu]*u11t[uidm*ndim+nu]-u12t[i*ndim+mu]*conj(u12t[uidm*ndim+nu]);
	Complex_f Sigma12=u11t[i*ndim+mu]*u12t[uidm*ndim+nu]+u12t[i*ndim+mu]*conj(u11t[uidm*ndim+nu]);

	int uidn = iu[nu+ndim*i]; 
	Complex_f a11=Sigma11*conj(u11t[uidn*ndim+mu])+Sigma12*conj(u12t[uidn*ndim+mu]);
	Complex_f a12=-Sigma11*u12t[uidn*ndim+mu]+Sigma12*u11t[uidn*ndim+mu];

	Sigma11=a11*conj(u11t[i*ndim+nu])+a12*conj(u12t[i*ndim+nu]);
	//Not needed in final result as it traces out
	//Sigma12[i]=-a11[i]*u12t[i*ndim+nu]+a12*u11t[i*ndim+mu];
	return creal(Sigma11);
}
void cuPolyakov(Complex_f *Sigma11, Complex_f * Sigma12, Complex_f * u11t,Complex_f *u12t,
                const sycl::nd_item<3> &item_ct1){
	char * funcname = "cuPolyakov";
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
        for(int it=1;it<ksizet;it++)
	//RACE CONDITION? gsize*bsize?
		for(int i=threadId;i<kvol3;i+=gsize*bsize){
			int indexu=it*kvol3+i;
			Complex_f a11=Sigma11[i]*u11t[indexu*ndim+3]-Sigma12[i]*conj(u12t[indexu*ndim+3]);
			//Instead of having to store a second buffer just assign it directly
			Sigma12[i]=Sigma11[i]*u12t[indexu*ndim+3]+Sigma12[i]*conj(u11t[indexu*ndim+3]);
			Sigma11[i]=a11;
		}
}
