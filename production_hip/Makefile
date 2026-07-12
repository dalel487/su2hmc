MF=Makefile
CC ?=			gcc
CXX ?=		g++
NVCC ?=		hipcc

#TODO: HIPBLAS Path should be smarter
IFLAGS=     -I../INCLUDE -I ./ -I../production_c $(shell gsl-config --cflags)\
				-I /usr/include/hipblas -I /usr/include/hiprand

CFLAGS =	$(IFLAGS) $(shell hipconfig --cpp_config) -DUSE_GPU -D__RANLUX__ -DSA3AT -DINT_OMF2 -MMD -fPIC

ifeq ($(CXX),icpx)
# NOTE: 20251101: Intel oneAPI 2024.0+ with NVCC 13 or earlier (possibly later, but they dont exist as of when I wrote
# this) compatibility issues on AVX10.2+ CPUs (e.g., Granite Rapids)
# If compilation fails with AVX10.2 intrinsic errors, replace -march=native with -march=skylake-avx512
# or use an older Intel compiler version that doesn't include AVX10.2 intrinsics.
    $(error As of 20251101, the libstdc++ headers break when compiling with Intel OneAPI. This is due to NVCC not supporting noexcept(true) in the header for rsqrt(), and Intel OneAPI (clang) being stricter than GCC. You may have more luck with a version of nvcc greater than 13.0 or an older libstdc++ library)
$(info Host compiling with Intel Compiler)
CFLAGS+=  -qopt-zmm-usage=high -MD -ipo -static-intel -static-libgcc \
			-xHOST -DMKL_DIRECT_CALL -D__USE_MKL__  -qopenmp -D__NO_AVX10_2__\
			-qopenmp-simd  -use-intel-optimized-headers -diag-disable=10441
else ifneq (,$(filter g++%,$(CXX)))
$(info Host compiling with GCC)
CFLAGS+=		-march=native -fcommon -fopenmp -fopenmp-simd -static-libgcc #-foffload=disable
LFLAGS+=		-lgomp
else ifeq ($(CXX),nvc++)
    $(warning As of 20251101, the nvc/nvc++ compiler segfaults during the conjugate gradient. We recommend you use GNU instead. (echo CC+gcc, echo CXX=g++))
$(info Host compiling with NVIDIA C Compiler)
CFLAGS+=		-tp=native -mp
else ifneq (,$(filter clang++%,$(CXX)))
    $(error As of 20251101, the libstdc++ headers break when compiling with clang. This is due to NVCC not supporting noexcept(true) in the header for rsqrt(), and clang being stricter than GCC. You may have more luck with a version of nvcc greater than 13.0 or an older libstdc++ library)
$(info Host compiling with Clang)
CFLAGS+=		-fveclib=AMDLIBM -march=native -fcommon -flto -fopenmp=libomp -fopenmp-simd -static-libgcc -Wno-exception-spec 
#NVCCFLAGS+= -D_GLIBCXX_USE_CXX11_ABI=0
else
    $(error Unknown compiler: $(CXX))
endif

ifeq ($(CXX),icpx)
$(info Using MKL)
#TODO: Currently only configured for using MKL with an Intel Compiler
CFLAGS+=	-qmkl -DMKL_DIRECT_CALL 
LFLAGS+=	-L${MKLROOT}/lib/intel64 -limf -DUSE_MKL -DMKL_DIRECT_CALL #-DMKL_ILP64
else ifdef AOCL_ROOT
$(info Using AMD BLIS)
CFLAGS += -DAMD_BLAS -I ${AOCL_ROOT}/include
LFLAGS += -L${AOCL_ROOT}/lib/ -lblis  -lalm
else ifdef LIBSCI_BASE_DIR
$(info Using Cray BLAS)
CFLAGS += -DUSE_BLAS
#Automatic on LUMI?
#LFLAGS = ‐L${LIBSCI_BASE_DIR}/gnu/91/x86_64/lib/ -lsci_comp
else ifeq ($(shell pkg-config --exists openblas && echo yes),yes)
$(info Using OpenBLAS)
CFLAGS += -DOPENBLAS $(shell pkg-config --cflags openblas)
LFLAGS += $(shell pkg-config --libs openblas)
else
$(info Using GSL BLAS)
CFLAGS += -DGSL_BLAS
LFLAGS	+=	-lm $(shell gsl-config --libs)
endif
LFLAGS	+=	-lm $(shell gsl-config --libs-without-cblas)

CULFLAGS= -L/usr/lib64/rocm/lib -lamdhip64 -lhipblas

LFLAGS += -lpthread -ldl #${CULFLAGS}

NVCCFLAGS+=	${IFLAGS} -w --offload-arch=gfx1030  #-expt-relaxed-constexpr 

EXE=	su2hmc_HIP

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
COBJ=	$(patsubst $(CSRC_DIR)/%.c, ./%.o, $(C_SOURCES))
CUOBJ= $(patsubst $(CUSRC_DIR)/%.cu, ./%.o, $(CU_SOURCES))
CLIB= libc.a
CULIB= #libcu.a
CULIB2= #libhiprtc.a
#OBJ=	$(CLIB) $(CULIB2)
OBJ = $(COBJ) $(CUOBJ) ./main.o
DEPS=	$(OBJ:.o=.d)

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=	$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

profiler:	CFLAGS += -g #-D_DEBUGCG
profiler:	NVCCFLAGS += -Rpass-analysis=kernel-resource-usage #-D_DEBUGCG #-keep
profiler:	release

release:	CFLAGS +=  -O3  #-fprofile-use=./gcc_pgo
release:	NVCCFLAGS +=  #--maxrregcount=64 -G -dopt=on
release:	$(EXE)

diagnostic: CSRC+= diagnostics.c
diagnostic: CFLAGS+=-DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
diagnostic: NVCCFLAGS+=-DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
diagnostic: debug

debug:		CFLAGS += -g -O0 -D_DEBUG #-D_DEBUGCG #-DNO_TIME -DNO_SPACE
debug:		NVCCFLAGS += -Rpass-analysis-kernel-resource-usage -O0 -g -D_DEBUG  
debug:		$(EXE)
		 
$(EXE): $(OBJ) #../main.o
	$(NVCC) $(NVCCFLAGS) --hip-link -fgpu-rdc $(IFLAGS) $(LFLAGS) $(CULFLAGS) $^ -o $@
%.o: $(CUSRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -std=gnu++17 $(IFLAGS) $(filter-out -fopenmp -fopenmp-simd,$(CFLAGS))  $< -fgpu-rdc -c -o $@
%.o: $(CSRC_DIR)/%.c
	$(CC) $(CFLAGS) -foffload=disable -std=gnu23 $(IFLAGS) -c $<  
./main.o: ../main.c
	$(CC) $(CFLAGS) -foffload=disable -std=gnu23 $(IFLAGS) -c $<  

clean:
	rm -f $(OBJ) $(COBJ) $(CUOBJ) $(CULIB) $(CLIB) $(TMP)  $(DEPS) *.cpp?.ii *.cudafe1.* *.*bin *.fatbin.c *.ptx *.sm_* *module_id \
	*dlink* *.ltoir *.d i?_out *.dbg

INP=	\
	midout \

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
