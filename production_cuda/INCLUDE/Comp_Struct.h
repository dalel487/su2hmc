#ifndef CMPLX_STRCT
#define CMPLX_STRCT
//This contains the common complex structure definition for both the C and CUDA
//Codes

typedef struct{
	double re;
	double im;
	//Assignment
	//=========
	inline __device__ complex& operator=(const complex &z){
		if(this != z){//Avoid self assignment
			this.re = z.re; this.im = z.im;
		}
		return *this;
	}
	inline __device__ complex& operator=(const double &x){
		this.re = x; this.im = 0;
		return *this;
	}
	inline __device__ complex& operator=(const int &n){
		this.re = (double)n; this.im = 0;
		return *this;
	}
}complex;
static const complex I   = {0.0,1.0};
#endif
