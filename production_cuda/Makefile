MF=Makefile
CC ?=		gcc
CXX ?=		g++
NVCC ?=		nvcc

IFLAGS=     -I../INCLUDE -I ./ -I../production_c $(shell gsl-config --cflags)\
				-I /opt/nvidia/hpc_sdk/Linux_x86_64/latest/cuda/include/ \
				-I /opt/nvidia/hpc_sdk/Linux_x86_64/latest/math_libs/include/ 

CFLAGS =	$(IFLAGS) -D__NVCC__ -D__RANLUX__  -DSA3AT -DINT_OMF2\

ifeq ($(CXX),icpx)
CFLAGS+=  -qopt-zmm-usage=high -MD -ipo -static-intel -static-libgcc \
			-xHOST -DMKL_DIRECT_CALL -D__USE_MKL__  -qopenmp \
			-qopenmp-simd  -use-intel-optimized-headers -diag-disable=10441
else ifeq ($(CXX),g++)
CFLAGS+=		-march=native -fcommon -fopenmp -fopenmp-simd -static-libgcc -foffload=disable
else ifeq ($(CXX),nvc++)
CFLAGS+=		-tp=native -fast -Mvect=simd -mp -static
else ifeq ($(CXX),clang++)
CFLAGS+=		-fveclib=AMDLIBM -march=native -fcommon -flto -fopenmp=libomp -fopenmp-simd -static-libgcc
else
    $(error Unknown compiler: $(CXX))
endif

ifdef __USE_MKL__
#TODO: Currently only configured for use with an Intel Compiler
CFLAGS+=	-qmkl
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

NVCCFLAGS=	${IFLAGS} --resource-usage -err-no -w --allow-unsupported-compiler #-expt-relaxed-constexpr 
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

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=	$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

profiler:	CFLAGS += -g #-D_DEBUGCG
profiler:	NVCCFLAGS += -Xptxas -O3,-v -lineinfo #-D_DEBUGCG #-keep
profiler:	release

release:	CFLAGS +=  -Ofast -funroll-all-loops -fcf-protection=none #-fprofile-use=./gcc_pgo
release:	NVCCFLAGS += -arch=native #--maxrregcount=64 -G -dopt=on
release:	$(EXE)

diagnostic: CSRC+= diagnostics.c
diagnostic: CFLAGS+=-DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
diagnostic: NVCCFLAGS+=-DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
diagnostic: debug

debug:		CFLAGS += -g -O0 -D_DEBUG #-D_DEBUGCG -DDIAGNOSTIC -DNO_TIME -DNO_SPACE
debug:		NVCCFLAGS += -Xptxas -O0,-v -G -g -D_DEBUG -arch=sm_75 -keep 
debug:		$(EXE)
		 
$(EXE): $(OBJ) ../main.c
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(IFLAGS) -ccbin $(CC) -Xcompiler "$(CFLAGS) -std=gnu11 -rdynamic " $(LFLAGS) $(CULFLAGS) $^ -o $@
%.o: $(CUSRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(IFLAGS) -ccbin $(CXX) -Xcompiler "$(CFLAGS) -std=c++17 -rdynamic" $^ -dc $@
%.o: $(CSRC_DIR)%.c
	$(CC) $(CFLAGS) -std=gnu11 $(IFLAGS) -c $^  

clean:
	rm -f $(OBJ) $(COBJ) $(CUOBJ) $(CULIB) $(CLIB) $(TMP) *.cpp?.ii *.cudafe1.* *.*bin *.fatbin.c *.ptx *.sm_* *module_id \
	*dlink* *.ltoir *.d i?_out

INP=	\
	midout \

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
