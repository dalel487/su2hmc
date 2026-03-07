MF=Makefile
CC ?=			gcc
CXX ?=		g++
NVCC ?=		nvcc

IFLAGS=     -I../INCLUDE -I ./ -I../production_c $(shell gsl-config --cflags)\
				-I /opt/nvidia/hpc_sdk/Linux_x86_64/latest/cuda/include/ \
				-I /opt/nvidia/hpc_sdk/Linux_x86_64/latest/math_libs/include/ 

CFLAGS =	$(IFLAGS) -D__NVCC__ -D__RANLUX__  -DSA3AT -DINT_OMF2 -MMD

ifeq ($(CXX),icpx)
# NOTE: 20251101: Intel oneAPI 2024.0+ with NVCC 13 or earlier (possibly later, but they dont exist as of when I wrote
# this) compatibility issues on AVX10.2+ CPUs (e.g., Granite Rapids)
# If compilation fails with AVX10.2 intrinsic errors, replace -march=native with -march=skylake-avx512
# or use an older Intel compiler version that doesn't include AVX10.2 intrinsics.
    $(error As of 20251101, the libstdc++ headers break when compiling with Intel OneAPI. This is due to NVCC not supporting noexcept(true) in the header for rsqrt(), and Intel OneAPI (clang) being stricter than GCC. You may have more luck with a version of nvcc greater than 13.0 or an older libstdc++ library)
CFLAGS+=  -qopt-zmm-usage=high -MD -ipo -static-intel -static-libgcc \
			-march=native -DMKL_DIRECT_CALL -D__USE_MKL__  -qopenmp \
			-qopenmp-simd  -use-intel-optimized-headers -diag-disable=10441
else ifeq ($(CXX),g++)
CFLAGS+=		-march=native -fcommon -fopenmp -fopenmp-simd -static-libgcc -foffload=disable
else ifeq ($(CXX),nvc++)
    $(warning As of 20251101, the nvc/nvc++ compiler segfaults during the conjugate gradient. We recommend you use GNU instead. (echo CC+gcc, echo CXX=g++))
CFLAGS+=		-tp=native -mp
else ifeq ($(CXX),clang++)
    $(error As of 20251101, the libstdc++ headers break when compiling with clang. This is due to NVCC not supporting noexcept(true) in the header for rsqrt(), and clang being stricter than GCC. You may have more luck with a version of nvcc greater than 13.0 or an older libstdc++ library)
CFLAGS+=		-fveclib=AMDLIBM -march=native -fcommon -flto -fopenmp=libomp -fopenmp-simd -static-libgcc -Wno-exception-spec 
#NVCCFLAGS+= -D_GLIBCXX_USE_CXX11_ABI=0
else
    $(error Unknown compiler: $(CXX))
endif

ifdef __USE_MKL__
#TODO: Currently only configured for use with an Intel Compiler
CFLAGS+=	-qmkl -DMKL_DIRECT_CALL 
LFLAGS=	-L${MKLROOT}/lib/intel64 -limf #-DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64
else ifdef AOCL_ROOT
CFLAGS += -DAMD_BLAS -I ${AOCL_ROOT}/include
LFLAGS = -L${AOCL_ROOT}/lib/ -lblis  -lalm
else ifdef LIBSCI_BASE_DIR
CFLAGS += -DUSE_BLAS
#Automatic on LUMI?
#LFLAGS = ‐L${LIBSCI_BASE_DIR}/gnu/91/x86_64/lib/ -lsci_comp
else
CFLAGS += -DGSL_BLAS
LFLAGS	=	-lm $(shell gsl-config --libs)
endif
LFLAGS	+=	-lm $(shell gsl-config --libs-without-cblas)

CULFLAGS=	-L/opt/nvidia/hpc_sdk/Linux_x86_64/latest/cuda/lib64/ \
				-L/opt/nvidia/hpc_sdk/Linux_x86_64/latest/math_libs/lib64/ \
				-lcuda -lcudart -lcublas 

LFLAGS +=		-lstdc++ -lpthread -ldl #${CULFLAGS}

NVCCFLAGS+=	${IFLAGS} --resource-usage -err-no -w -arch=native --allow-unsupported-compiler #-expt-relaxed-constexpr 
			#Last flag is an experimental one used to call host functions in device code. Not sure
			#if it does what we want because the error message suggesting it came up when declaring
			#a11 and a12 in the Dslash routine
#Replace first line of NVCCFLAGS with below when done. Targets every platform since 2016
#NVCCFLAGS=	${IFLAGS}  -arch=compute_60 -code=sm_60,sm_61,sm_70,sm_75,sm_80,sm_86,sm_90 -err-no -w \

EXE=	su2hmc_CUDA

CSRC_DIR=	../production_c
CSRC=	\
	    coord.c \
	    congrad.c \
		force.c \
    	random.c \
		par_mpi.c \
		bosonic.c\
		fermionic.c \
		integrate.c 

CUSRC_DIR=	../production_cuda
CUSRC=	\
		cubosonic.cu\
		cuclover.cu\
		cumatrices.cu\
		cuforce.cu\
		cusu2hmc.cu

#CUSRC= CUDA_Compile.cu
CINC=	\
		su2hmc.h \
		sizes.h \
	    par_mpi.h \
    	random.h \
	    matrices.h \
    	coord.h \
	    errorcodes.h \
		 integrate.h

#
# No need to edit below this line
#

.SUFFIXES: .c .cu .h .hpp .o 
.PHONY: clean debug release profiler tar
C_SOURCES:= $(wildcard $(CSRC_DIR)/*.c)
CU_SOURCES:= $(wildcard $(CUSRC_DIR)/*.cu)
COBJ=	$(patsubst $(CSRC_DIR)/%.c, $(CSRC_DIR)/%.o, $(C_SOURCES))
CUOBJ= $(patsubst $(CUSRC_DIR)/%.cu, $(CUSRC_DIR)/%.o, $(CU_SOURCES))
CLIB= libc.a
CULIB= libcu.a
CULIB2= libcu2.a
#OBJ=	$(CLIB) $(CULIB2)
OBJ = $(COBJ) $(CUOBJ) 
DEPS=	$(OBJ:.o=.d)

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=	$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

profiler:	CFLAGS += -g #-D_DEBUGCG
profiler:	NVCCFLAGS += -Xptxas -O3,-v -lineinfo #-D_DEBUGCG #-keep
profiler:	release

release:	CFLAGS +=  -O3  #-fprofile-use=./gcc_pgo
release:	NVCCFLAGS +=  #--maxrregcount=64 -G -dopt=on
release:	$(EXE)

diagnostic: CSRC+= diagnostics.c
diagnostic: CFLAGS+=-DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
diagnostic: NVCCFLAGS+=-DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
diagnostic: debug

debug:		CFLAGS += -g -O0 -D_DEBUG -D_DEBUGCG #-DNO_TIME -DNO_SPACE
debug:		NVCCFLAGS += -Xptxas -O0,-v -G -g -D_DEBUG  -keep 
debug:		$(EXE)
		 
$(EXE): $(OBJ) ../main.c
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(IFLAGS) -ccbin $(CC) -Xcompiler "$(CFLAGS) -rdynamic" --compiler-options -D__NO_AVX10_2__  $(LFLAGS) $(CULFLAGS) $^ -o $@
%.o: $(CUSRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(IFLAGS) -ccbin $(CXX) -Xcompiler "$(CFLAGS)  -rdynamic" --compiler-options -D__NO_AVX10_2__ $< -dc $@
%.o: $(CSRC_DIR)%.c
	$(CC) $(CFLAGS) -std=gnu23 $(IFLAGS) -c $<  

clean:
	rm -f $(OBJ) $(COBJ) $(CUOBJ) $(CULIB) $(CLIB) $(TMP)  $(DEPS) *.cpp?.ii *.cudafe1.* *.*bin *.fatbin.c *.ptx *.sm_* *module_id \
	*dlink* *.ltoir *.d i?_out *.dbg

INP=	\
	midout \

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
