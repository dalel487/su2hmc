#ifndef CMPLX_STRCT
#define CMPLX_STRCT
//This contains the common complex structure definition for both the C and CUDA
//Codes

typedef struct{
	double re;
	double im;
}complex;
static const complex I   = {0.0,1.0};
#endif
