#include <cuda_complex.hpp>
#include <cuda.h>
#include <stdio.h>
#define complexd complex<double>
#define complexf complex<float>
__global__ void test_run(){
	complexd z1={1.0/(1+threadIdx.x+blockIdx.z),1.0*threadIdx.x+1+blockIdx.y};
	complexd z2={1.0/(1+threadIdx.y+blockIdx.z),1.0*threadIdx.y+1+blockIdx.x};
	complexd z3 = z1+z2;
	complexd z4 = z1 -z2;
	complexd z5 = conj(z2);
	complexd z6 = z1*z2;
	double x = norm(z1);
	complexd z8 = z1/z2;
	bool eq = (z3==z4);
	bool neq = (z3!=z4);
	
	double x1 = creal(z1);
	double x2 = cimag(z2);
	complexd z9 = x1+z1;
	complexd z10 = z1+x1;
	complexd z11 =x2-z2;
	complexd z12 = z2-x2;
	complexd z13 = x1*z1;
	complexd z14 = z1*x1;
	complexd z15 = x2/z2;
	complexd z16 = z2/x2;
	bool deq1 = (x1==z1);
	bool dneq1 = (x1!=z1);
	bool deq2 = (z2==x2);
	bool dneq2 = (z2!=x2);
	
	printf("For z1=%f+%fi and z2=%f+%fi, z13=creal(z1)*z2=%f+%fi\n",
	creal(z1),cimag(z1),creal(z2),cimag(z2),creal(z13),cimag(z13));
}
int main(int argc, char *argv[]){
	dim3 dimGrid(1024,1024,1024);
	dim3 dimBlock(16,16,2);
	test_run<<<dimGrid,dimBlock>>>();
	return 0;
}
