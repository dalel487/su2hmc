/**
 * @file matrices.h
 *
 * @brief Matrix multiplication and related declarations
 */
#ifndef MATRICES
#define MATRICES
#ifdef __NVCC__
#include <curand.h>
#endif
#include <par_mpi.h>
#include <su2hmc.h>
#if (defined __cplusplus)
extern "C"
{
#endif
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in double precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:		Gamma matrices rescaled by kappa
	 *	@param	gamin:		Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in double precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices rescaled by kappa
	 *	@param	gamin:		Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in double precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices rescaled by kappa
	 *	@param	gamin:	Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslash(Complex *phi, Complex *r, Complex *u11t[2],unsigned int *iu,unsigned  int *id,\
			Complex *gamval, int *gamin, double *dk[2], float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in double precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices rescaled by kappa
	 *	@param	gamin:	Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa);
	//Float version
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in single precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices rescaled by kappa
	 *	@param	gamin:		Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval,int *gamin, float *dk[2], Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in single precision.
	 *
	 * @param	phi:			The product
	 * @param	r:				The array being acted on by M
	 * @param	u11t:		First colour trial field
	 * @param	u12t:		Second colour trial field
	 *	@param	iu:			Upper halo indices
	 *	@param	id:			Lower halo indices
	 *	@param	gamval:	Gamma matrices rescaled by kappa
	 *	@param	gamin:		Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	jqq:			Diquark source
	 *	@param	akappa:		Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Dslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval,int *gamin, float *dk[2], Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in single precision.
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices rescaled by kappa
	 *	@param	gamin:	Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu,unsigned int *id,\
			Complex_f *gamval, int *gamin, float *dk[2], float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in single precision
	 *
	 * @param	phi:		The product
	 * @param	r:			The array being acted on by M
	 * @param	u11t:	First colour trial field
	 * @param	u12t:	Second colour trial field
	 *	@param	iu:		Upper halo indices
	 *	@param	id:		Lower halo indices
	 *	@param	gamval:	Gamma matrices rescaled by kappa
	 *	@param	gamin:	Indices for dirac terms
	 * @param	dk4m:				@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 * @param	dk4p:				@f$\left(1-\gamma_0\right)e^\mu@f$
	 *	@param	akappa:	Hopping parameter
	 *
	 * @return Zero on success, integer error code otherwise
	 */
	int Hdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[2],unsigned int *iu,unsigned int *id,\
			Complex_f *gamval,int *gamin, float *dk[2], float akappa);
#ifdef DIAGNOSTIC
	int Diagnostics(int istart, Complex *u11, Complex *u12,Complex *u11t, Complex *u12t, Complex_f *u11t_f, Complex_f *u12t_f,\
			unsigned int *iu, unsigned int *id, int *hu, int *hd, double *dk4m, double *dk4p,\
			float *dk4m_f, float *dk4p_f, int *gamin, Complex *gamval, Complex_f *gamval_f,\
			Complex_f jqq, float akappa, float beta, double ancg);
#endif

void Transpose_z(Complex *out, const int, const int);
void Transpose_c(Complex_f *out, const int, const int);
void Transpose_d(double *out, const int, const int);
void Transpose_f(float *out, const int, const int);
void Transpose_I(int *out, const int, const int);
void Transpose_U(unsigned int *out, const int, const int);

#ifdef __NVCC__
	//Calling Functions
	void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa,dim3 dimGrid, dim3 dimBlock);
	void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned int *id,\
			Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa, dim3 dimGrid, dim3 dimBlock);
	//Float version
	void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f, int *gamin, float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *u11t_f, Complex_f *u12t_f,unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin, float *dk4m_f, float *dk4p_f, Complex_f jqq_f, float akappa_f,\
			dim3 dimGrid, dim3 dimBlock);
	void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut_f[2],unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin, float *dk_f[2], float akappa_f,dim3 dimGrid, dim3 dimBlock);
	void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut_f[2],unsigned int *iu,unsigned int *id,\
			Complex_f *gamval_f,int *gamin, float *dk_f[2], float akappa_f, dim3 dimGrid, dim3 dimBlock);

/**
 * @brief In place transpose
 *
 * @param out: The array being transposed
 * @param fast_in:	The old outermost/fastest index
 * @param fast_out:	The new outermost/fastest index
 * @param dimGrid:	CUDA grid
 * @param dimBlock:	CUDA block
 */
	void cuTranspose_z(Complex *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	void cuTranspose_c(Complex_f *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	void cuTranspose_d(double *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	void cuTranspose_f(float *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	void cuTranspose_I(int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	void cuTranspose_U(unsigned int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
#endif
#if (defined __cplusplus)
}
#endif
#ifdef __CUDACC__
//Actual CUDA
__global__ void cuDslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
__global__ void cuDslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, Complex_f jqq, float akappa);
__global__ void cuHdslash(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa);
__global__ void cuHdslashd(Complex *phi, Complex *r, Complex *u11t, Complex *u12t,unsigned int *iu,unsigned  int *id,\
		Complex *gamval, int *gamin, double *dk4m, double *dk4p, float akappa);
//Float version
__global__ void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *u11t, Complex_f *u12t,unsigned int *iu, unsigned int *id,\
		__shared__ Complex_f *gamval, int *gamin, float *dk4m, float *dk4p, Complex_f jqq, float akappa);
__global__ void cuDslashd_f(Complex_f *phi, const Complex_f *r, const Complex_f *u11t, const Complex_f *u12t,const unsigned int *iu, const unsigned int *id,\
		__shared__ Complex_f *gamval_d,	int *gamin_d,	const float *dk4m, const float *dk4p, const Complex_f jqq, const float akappa);

__global__ void cuHdslash_f(Complex_f *phi, const Complex_f *r, const Complex_f *u11t, const Complex_f *u12t,const unsigned int *iu, const unsigned int *id,\
		const __shared__ Complex_f gamval[20], const __shared__ int gamin[16], const float *dk4m, const float *dk4p, const float akappa);
__global__ void cuHdslashd_f(Complex_f *phi, const Complex_f *r, const Complex_f *u11t, const Complex_f *u12t,const unsigned int *iu, const unsigned int *id,\
		const __shared__ Complex_f gamval[20], const __shared__ int gamin[16], const float *dk4m, const float *dk4p, const float akappa);
template <typename T> __global__ void Transpose(T *out, const T *in, const int fast_in, const int fast_out);
#endif
#endif
