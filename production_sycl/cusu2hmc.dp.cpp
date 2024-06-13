#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <su2hmc.h>
#define MIN(x,y) (x<y?x:y)
#define MAX(x,y) (x>y?x:y)
sycl::range<3> dimBlockOne = sycl::range<3>(1, 1, 1);
sycl::range<3> dimGridOne = sycl::range<3>(1, 1, 1);
//Worst case scenario, each block contains 256 threads. This should be tuned later
sycl::range<3> dimBlock = sycl::range<3>(1, 1, 1);
//sycl::range<3> dimBlock = 1;
sycl::range<3> dimGrid = sycl::range<3>(1, 1, 1);
//sycl::range<3>	dimBlock=dimBlockOne; sycl::range<3> dimGrid=dimGridOne;
//sycl::queue streams[ndirac * ndim * nadj];
void blockInit(int x, int y, int z, int t, sycl::range<3> *dimBlock,
		sycl::range<3> *dimGrid) {

	char *funcname = "blockInit";

	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	dpct::device_info prop; 
	dpct::get_device_info(prop,dev_ct1);
	//Threads per block
	int tpb = prop.get_max_work_group_size() / 8;
	//Warp size
	int tpw = prop.get_max_sub_group_size();
	int bx=1;
	//Set bx to be the largest power of 2 less than x that fits in a block
	while(bx<x/2 && bx<tpb)
		bx*=2;
	int by=1;
	//Set by to be the largest power of 2 less than y such that bx*by fits in a block
	while(by<y/2 && bx*by<tpb)
		by*=2;

	if(bx*by>=128){
		*dimBlock = sycl::range<3>(1, by, bx);
		//If the block size neatly divides the lattice size we can create
		//extra blocks safely
		int res= (nx*ny%bx*by ==0) ? (nx*ny)/(bx*by) :1;
		*dimGrid=sycl::range<3>(nz,nt,1);
	}
	else{
		int bz=1;
		//Set by to be the largest power of 2 less than y such that bx*by fits in an optimal block
		while(bz<z/2 && bx*by*bz<tpb)
			bz*=2;
		*dimBlock = sycl::range<3>(bz, by, bx);

		//If we have an awkward block size then flag it.
		if(bx*by*bz%tpw!=0)
			fprintf(stderr,"Alert %i in %s: Suboptimal block size for warp size %d. bx=%d by=%d bz=%d\n",
					BLOCKALERT,	funcname, tpw, bx, by,bz);
		*dimGrid=sycl::range<3>(z/bz,nt,1);
	}
	printf("Block: (%d,%d,%d)\tGrid: (%d,%d,%d)\n", (*dimBlock)[2],
			(*dimBlock)[1], (*dimBlock)[0], (*dimGrid)[2], (*dimGrid)[1],
			(*dimGrid)[0]);
}
void	Init_CUDA(Complex *u11t, Complex *u12t,Complex *gamval, Complex_f *gamval_f, int *gamin, double*dk4m,\
		double *dk4p, unsigned int *iu, unsigned int *id){
	/*
	 * Initialises the GPU Components of the system
	 *
	 * Calls:
	 * ======
	 * Addrc, Check_addr, ran2, DHalo_swap_dir, Par_sread, Par_ranset, Reunitarise
	 *
	 * Globals:
	 * =======
	 * Complex gamval:		Gamma Matrices
	 * Complex_f gamval_f:	Float Gamma matrices:
	 *
	 * Parameters:
	 * ==========
	 * int istart:				Zero for cold, >1 for hot, <1 for none
	 * int ibound:				Periodic boundary conditions
	 * int iread:				Read configuration from file
	 * float beta:				beta
	 * float fmu:				Chemical potential
	 * float akappa:			
	 * Complex_f ajq:			Diquark source
	 * Complex *u11:			First colour field
	 * Complex *u12:			Second colour field
	 * Complex *u11t:			First colour trial field
	 * Complex *u11t:			Second colour trial field
	 * Complex_f *u11t_f:	First float trial field
	 * Complex_f *u12t_f:	Second float trial field
	 * double	*dk4m:
	 * double	*dk4p:
	 * float		*dk4m_f:
	 * float		*dk4p_f:
	 * unsigned int *iu:		Up halo indices
	 * unsigned int *id:		Down halo indices
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Init_CUDA";
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	//Set iu and id to mainly read in CUDA and prefetch them to the GPU
	/*
	cudaMemPrefetchAsync(iu,ndim*kvol*sizeof(int),device,stream);
	cudaMemPrefetchAsync(id,ndim*kvol*sizeof(int),device,streams[1]);
	cudaMemAdvise(iu,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);
	cudaMemAdvise(id,ndim*kvol*sizeof(int),cudaMemAdviseSetReadMostly,device);
	*/

	//Gamma matrices and indices on the GPU
	//	cudaMemcpy(gamin_d,gamin,4*4*sizeof(int),cudaMemcpyHostToDevice);
	//	cudaMemcpy(gamval_d,gamval,5*4*sizeof(Complex),cudaMemcpyHostToDevice);
	//	cudaMemcpy(gamval_f_d,gamval_f,5*4*sizeof(Complex_f),cudaMemcpyHostToDevice);
	/*
DPCT1063:13: Advice parameter is device-defined and was set to 0. You
may need to adjust it.
*/

/*
	dpct::get_device(dev_ct1).in_order_queue().mem_advise(gamin, 4 * 4 * sizeof(int), 0);
	dpct::get_device(dev_ct1).in_order_queue().mem_advise(gamval, 5 * 4 * sizeof(Complex), 0);

	//More prefetching and marking as read-only (mostly)
	//Prefetching Momentum Fields and Trial Fields to GPU
	dpct::get_device(dev_ct1).in_order_queue().mem_advise(dk4p, (kvol+halo)* sizeof(double), 0);
	dpct::get_device(dev_ct1).in_order_queue().mem_advise(dk4m, (kvol+halo)* sizeof(double), 0);
	cudaMemPrefetchAsync(dk4p,(kvol+halo)*sizeof(double),device,streams[2]);
	cudaMemPrefetchAsync(dk4m,(kvol+halo)*sizeof(double),device,streams[3]);
	cudaMemPrefetchAsync(u11t, ndim*kvol*sizeof(Complex),device,streams[4]);
	cudaMemPrefetchAsync(u12t, ndim*kvol*sizeof(Complex),device,streams[5]);
	*/
}
void cuReal_convert(float *a, double *b, int len, bool dtof,
		sycl::range<3> dimBlock, sycl::range<3> dimGrid) {
	/* 
	 * Kernel wrapper for conversion between sp and dp complex on the GPU.
	 */
	char *funcname = "cuComplex_convert";
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimBlock * dimGrid, dimGrid),
			[=](sycl::nd_item<3> item_ct1) {
			cuReal_convert(a,b,len,dtof,item_ct1);
			});
}
void cuComplex_convert(Complex_f *a, Complex *b, int len, bool dtof,
		sycl::range<3> dimBlock, sycl::range<3> dimGrid) {
	/* 
	 * Kernel wrapper for conversion between sp and dp complex on the GPU.
	 */
	char *funcname = "cuComplex_convert";
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimBlock * dimGrid, dimGrid),
			[=](sycl::nd_item<3> item_ct1) {
			cuReal_convert((float *)a,(double *)b,2*len,dtof,item_ct1);
			});
}
void cuFill_Small_Phi(int na, Complex *smallPhi, Complex *Phi,
		sycl::range<3> dimBlock, sycl::range<3> dimGrid) {
	/*
DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
limit. To get the device limit, query info::device::max_work_group_size.
Adjust the work-group size if needed.
*/
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimBlock * dimGrid, dimGrid),
			[=](sycl::nd_item<3> item_ct1) {
			cuFill_Small_Phi(na, smallPhi, Phi,item_ct1);
			});
}
void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table,
		unsigned int mu, sycl::range<3> dimBlock,
		sycl::range<3> dimGrid)
{
	char *funcname = "cuZ_gather";
	/*
DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimBlock * dimGrid, dimGrid),
			[=](sycl::nd_item<3> item_ct1) {
			cuC_gather(x, y, n, table, mu,item_ct1);
			});
}
void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table,
		unsigned int mu, sycl::range<3> dimBlock,
		sycl::range<3> dimGrid)
{
	char *funcname = "cuZ_gather";
	/*
DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimBlock * dimGrid, dimGrid),
			[=](sycl::nd_item<3> item_ct1) {
			cuZ_gather(x, y, n, table, mu,item_ct1);
			});
}
void cuUpDownPart(int na, Complex *X0, Complex *R1, sycl::range<3> dimBlock,
		sycl::range<3> dimGrid) {
	/*
DPCT1049:12: The work-group size passed to the SYCL kernel may exceed
the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue stream = dev_ct1.in_order_queue();
	
	dpct::get_in_order_queue().parallel_for(
			sycl::nd_range<3>(dimBlock * dimGrid, dimGrid),
			[=](sycl::nd_item<3> item_ct1) {
			cuUpDownPart(na, X0, R1,item_ct1);
			});
}
//CUDA Kernels
void cuReal_convert(float *a, double *b, int len, bool dtof,
		const sycl::nd_item<3> &item_ct1){
	char *funcname = "cuReal_convert";
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
	//Double to float
	if(dtof)
		for(int i = threadId; i<len;i+=gsize*bsize)
			a[i]=(float)b[i];
	//Float to double
	else
		for(int i = threadId; i<len;i+=gsize*bsize)
			b[i]=(double)a[i];
}
void cuFill_Small_Phi(int na, Complex *smallPhi, Complex *Phi,
		const sycl::nd_item<3> &item_ct1)
{
	/*Copies necessary (2*4*kvol) elements of Phi into a vector variable
	 *
	 * Globals:
	 * =======
	 * Phi:	  The source array
	 * 
	 * Parameters:
	 * ==========
	 * int na: flavour index
	 * Complex *smallPhi:	  The target array
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	const char *funcname = "cuFill_Small_Phi";
	//BIG and small phi index
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
	for(int i = threadId; i<kvol;i+=gsize*bsize)
		for(int idirac = 0; idirac<ndirac; idirac++)
			for(int ic= 0; ic<nc; ic++)
				//	  PHI_index=i*16+j*2+k;
				smallPhi[(i*ndirac+idirac)*nc+ic]=Phi[((na*kvol+i)*ngorkov+idirac)*nc+ic];
}
void cuC_gather(Complex_f *x, Complex_f *y, int n, unsigned int *table, unsigned int mu,
		const sycl::nd_item<3> &item_ct1)
{
	char *funcname = "cuZ_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
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
	for(int i = threadId; i<n;i+=gsize*bsize)
		x[i]=y[table[i*ndim+mu]*ndim+mu];
}
void cuZ_gather(Complex *x, Complex *y, int n, unsigned int *table, unsigned int mu,
		const sycl::nd_item<3> &item_ct1)
{
	char *funcname = "cuZ_gather";
	//FORTRAN had a second parameter m giving the size of y (kvol+halo) normally
	//Pointers mean that's not an issue for us so I'm leaving it out
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
	for(int i = threadId; i<n;i+=gsize*bsize)
		x[i]=y[table[i*ndim+mu]*ndim+mu];
}
void cuUpDownPart(int na, Complex *X0, Complex *R1,
		const sycl::nd_item<3> &item_ct1){

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
	//Up/down partitioning (using only pseudofermions of flavour 1)
	for(int i = threadId; i<kvol;i+=gsize*bsize)
		for(int idirac = 0; idirac < ndirac; idirac++){
			X0[((na*kvol+i)*ndirac+idirac)*nc]=R1[(i*ngorkov+idirac)*nc];
			X0[((na*kvol+i)*ndirac+idirac)*nc+1]=R1[(i*ngorkov+idirac)*nc+1];
		}
}


//DIRTY HACK
//void cudaDeviceSynchronise(){
//	stream.wait();
//}

