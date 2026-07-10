/**
 * @file matrices.h
 *
 * @brief Matrix multiplication and related declarations
 *
 *	@defgroup Dslashes Fermion matrix products
 */
#pragma once
#ifdef __NVCC__
#include <curand.h>
#endif
#include <par_mpi.h>
#if (defined __cplusplus)
extern "C"
{
#endif
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return	Zero on success, integer error code otherwise
	 */
	int Dslash(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned  int *id,\
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return	Zero on success, integer error code otherwise
	 */
	int Dslashd(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in double precision
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:		The product
	 * @param[in]		r:			The array being acted on by M
	 * @param[in]		ut:		Gauge trial field
	 *	@param[in]		iu,id:	Upper/lower halo indices
	 *	@param[in]		gamval:	Gamma matrices rescaled by kappa
	 *	@param[in]		gamin:	Indices for dirac terms
	 * @param[in]		dk:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$
	 *	@param[in]		akappa:	Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return	Zero on success, integer error code otherwise
	 */
	int Hdslash(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned  int *id,\
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in double precision
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:		The product
	 * @param[in]	r:			The array being acted on by M
	 * @param[in]	ut:		Gauge field
	 *	@param[in]	iu,id:	Upper/lower halo indices
	 *	@param[in]	gamval:	Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:	Indices for dirac terms
	 * @param[in]	dk:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:	Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return	Zero on success, integer error code otherwise
	 */
	int Hdslashd(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned  int *id,\
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], float akappa);
	//Float version
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in single precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return	Zero on success, integer error code otherwise
	 */
	int Dslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20],const unsigned short gamin[16], float *dk[nc], Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in single precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return	Zero on success, integer error code otherwise
	 */
	int Dslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20],const unsigned short gamin[16], float *dk[nc], Complex_f jqq, float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M r@f$ in single precision
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:		The product
	 * @param[in]	r:			The array being acted on by M
	 * @param[in]	ut:		Gauge field
	 *	@param[in]	iu,id:	Upper/lower halo indices
	 *	@param[in]	gamval:	Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:	Indices for dirac terms
	 * @param[in]	dk:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:	Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return	Zero on success, integer error code otherwise
	 */
	int Hdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20], const unsigned short gamin[16], float *dk[nc], float akappa);
	/**
	 * @brief Evaluates @f$\Phi=M^\dagger r@f$ in single precision
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:		The product
	 * @param[in]	r:			The array being acted on by M
	 * @param[in]	ut:		Gauge field
	 *	@param[in]	iu,id:	Upper/lower halo indices
	 *	@param[in]	gamval:	Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:	Indices for dirac terms
	 * @param[in]	dk:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:	Hopping parameter
	 *
	 * @post		Result written to @p phi
	 * @return 	Zero on success, integer error code otherwise
	 */
	int Hdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20],const unsigned short gamin[16], float *dk[nc], float akappa);

	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void Transpose_z(Complex *out, const int, const int);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void Transpose_c(Complex_f *out, const int, const int);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void Transpose_d(double *out, const int, const int);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void Transpose_f(float *out, const int, const int);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void Transpose_I(int *out, const int, const int);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void Transpose_U(unsigned int *out, const int, const int);

#ifdef __NVCC__
	//Calling Functions
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuDslash(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], Complex_f jqq, float akappa,
			dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M^\dagger r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuDslashd(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], Complex_f jqq, float akappa,
			dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M^\dagger r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuHdslash(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], float akappa,dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M^\dagger r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuHdslashd(Complex *phi, Complex *r, Complex *ut[nc],unsigned int *iu,unsigned int *id,
			Complex gamval[20], const unsigned short gamin[16], double *dk[nc], float akappa, dim3 dimGrid, dim3 dimBlock);
	//Float version
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M^\dagger r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuDslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20], const unsigned short gamin[16], float *dk[nc], Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M^\dagger r@f$ in double precision.
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:			The product
	 * @param[in]	r:				The array being acted on by M
	 * @param[in]	ut:			Gauge field
	 *	@param[in]	iu,id:		Upper/lower halo indices
	 *	@param[in]	gamval:		Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:		Indices for dirac terms
	 * @param[in]	dk:			@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	jqq:			Diquark source
	 *	@param[in]	akappa:		Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuDslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20],const unsigned short gamin[16], float *dk[nc], Complex_f jqq, float akappa,\
			dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M r@f$ in single precision
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:		The product
	 * @param[in]	r:			The array being acted on by M
	 * @param[in]	ut:		Gauge field
	 *	@param[in]	iu,id:	Upper/lower halo indices
	 *	@param[in]	gamval:	Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:	Indices for dirac terms
	 * @param[in]	dk:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:	Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuHdslash_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20],const unsigned short gamin[16], float *dk[nc], float akappa,dim3 dimGrid, dim3 dimBlock);
	/**
	 * @brief GPU calling wrapper for @f$\Phi=M^\dagger r@f$ in single precision
	 * @ingroup Dslashes
	 *
	 * @param[out]	phi:		The product
	 * @param[in]	r:			The array being acted on by M
	 * @param[in]	ut:		Gauge field
	 *	@param[in]	iu,id:	Upper/lower halo indices
	 *	@param[in]	gamval:	Gamma matrices rescaled by kappa
	 *	@param[in]	gamin:	Indices for dirac terms
	 * @param[in]	dk:		@f$\left(1+\gamma_0\right)e^{-\mu}@f$ and @f$\left(1+\gamma_0\right)e^{+\mu}@f$
	 *	@param[in]	akappa:	Hopping parameter
	 *	@param[in]	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post		Result written to @p phi
	 */
	void cuHdslashd_f(Complex_f *phi, Complex_f *r, Complex_f *ut[nc],unsigned int *iu,unsigned int *id,\
			Complex_f gamval[20],const unsigned short gamin[16], float *dk[nc], float akappa, dim3 dimGrid, dim3 dimBlock);

	/**
	 * @brief	Sum all terms in an array of doubles
	 * @ingroup Helper
	 * @param[in]	input:	Input array
	 * @param[in]	n:			Number of terms
	 * @param[in]	stream:	What stream to use (useful for simultaneous reductions)
	 *
	 * @return	Sum of all terms in input
	 */
	double cureduce_sum_d(double *input, const unsigned int n,const unsigned short stream);
	/**
	 * @brief	Sum all terms in an array of floats
	 * @ingroup Helper
	 * @param[in]	input:	Input array
	 * @param[in]	n:			Number of terms
	 * @param[in]	stream:	What stream to use (useful for simultaneous reductions)
	 *
	 * @return	Sum of all terms in input
	 */
	float cureduce_sum_f(float *input, const unsigned int n,const unsigned short stream);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 * @param[in]		dimGrid:		CUDA grid layout
	 * @param[in]		dimBlock:	CUDA block layout
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void cuTranspose_z(Complex *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 * @param[in]		dimGrid:		CUDA grid layout
	 * @param[in]		dimBlock:	CUDA block layout
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void cuTranspose_c(Complex_f *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 * @param[in]		dimGrid:		CUDA grid layout
	 * @param[in]		dimBlock:	CUDA block layout
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void cuTranspose_d(double *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 * @param[in]		dimGrid:		CUDA grid layout
	 * @param[in]		dimBlock:	CUDA block layout
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void cuTranspose_f(float *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 * @param[in]		dimGrid:		CUDA grid layout
	 * @param[in]		dimBlock:	CUDA block layout
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void cuTranspose_I(int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	/**
	 * @brief In place transpose used to convert from AoS to SoA memory layout
	 * @ingroup Helper
	 *
	 * @param[in,out] out: 			The array being transposed
	 * @param[in]		fast_in:		The old outermost/fastest index
	 * @param[in]		fast_out:	The new outermost/fastest index
	 * @param[in]		dimGrid:		CUDA grid layout
	 * @param[in]		dimBlock:	CUDA block layout
	 *
	 * @post	Result overwrites existing @p out with transposed array
	 */
	void cuTranspose_U(unsigned int *out, const int fast_in, const int fast_out, const dim3 dimGrid, const dim3 dimBlock);
	/**
	 *	@brief Add a single to a double value, and save the output in the double array
	 *			For complex valued arrays, one may cast the complex<double> and complex<float> arrays to double and float
	 *			arrays, and use 2N for the array length instead.
	 * @ingroup Helper
	 *	
	 *	@param[in,out]	d:						Double array
	 *	@param[in]		f:						float array
	 *	@param[in]		n:						Array lengths
	 *	@param[in]	 	dimGrid,dimBlock:	CUDA grid/block
	 *
	 * @post	@p d now contains the sum. (i.e. result is stored in place)
	 */
	void cuMixed_Sumto(double *d, float *f,const unsigned int n,const dim3 dimGrid,const dim3 dimBlock);
#endif
#if (defined __cplusplus)
}
#endif
