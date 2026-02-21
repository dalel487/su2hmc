MF=Makefile
CC ?=			gcc
CXX ?=		g++
MPICC ?=		mpicc

IFLAGS=	-I../INCLUDE  $(shell gsl-config --cflags)

CFLAGS=  $(IFLAGS) -std=gnu23 -MMD  -D__RANLUX__ -DSA3AT -DINT_OMF2

# Configure MPI wrapper to use the specified compiler
# This sets the appropriate environment variable based on the MPI implementation
# We're going to set all of the most common ones so we don't have to check what's even being used
ifneq ($(CC),)
#OpenMPI
export OMPI_CC := $(CC)
#MPICH
export MPICH_CC := $(CC)
#Intel MPI
export I_MPI_CC := $(CC)
endif

ifneq ($(filter icc icx, $(CC)),)
			#-xSKYLAKE -axSKYLAKE-AVX512,CASCADELAKE,TIGERLAKE,COFFEELAKE,ALDERLAKE,RAPTORLAKE
CFLAGS += -qopt-zmm-usage=high -ipo -static-intel -static-libgcc \
			-xHOST -fsanitize=address\
			-D__USE_MKL__  -qopenmp  -inline-level=2 \
			-qopenmp-simd  -use-intel-optimized-headers -diag-disable=10441
__USE_MKL__ = 1
else ifeq ($(CC),gcc)
CFLAGS += -march=native -fcommon -fopenmp -fopenmp-simd -static-libgcc -foffload=disable
else ifeq ($(CC),nvc)
#    $(warning As of 20251101, the nvc/nvc++ compiler segfaults during the conjugate gradient. We recommend you use GNU instead. (echo CC+gcc, echo CC=g++))
CFLAGS += -tp=native -mp -mp=multicore 
else ifeq ($(CC),clang)
    #$(error As of 20251101, the libstdc++ headers break when compiling with clang. This is due to NVCC not supporting noexcept(true) in the header for rsqrt(), and clang being stricter than GCC. You may have more luck with a version of nvcc greater than 13.0 or an older libstdc++ library)
CFLAGS += -fveclib=AMDLIBM -march=native -fcommon -flto -fopenmp -fopenmp-simd -static-libgcc -Wno-exception
LFLAGS += -latomic
#NVCCFLAGS+= -D_GLIBCC_USE_CC11_ABI=0
else
    $(error Unknown compiler: $(CC))
endif

ifdef __USE_MKL__
#TODO: Currently only configured for use with an Intel Compiler
CFLAGS +=	-qmkl -DMKL_DIRECT_CALL -DMKL_ILP64  -I"${MKLROOT}/include"
LFLAGS +=	-L${MKLROOT}/lib/intel64 -limf #-DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64
else ifdef AOCL_ROOT
CFLAGS += -DAMD_BLAS -I ${AOCL_ROOT}/include
LFLAGS += -L${AOCL_ROOT}/lib/ -lblis  -lalm
else ifdef LIBSCI_BASE_DIR
CFLAGS += -DUSE_BLAS
#Automatic on LUMI?
#LFLAGS = ‐L${LIBSCI_BASE_DIR}/gnu/91/x86_64/lib/ -lsci_comp
else
CFLAGS += -DGSL_BLAS
LFLAGS +=	-lm $(shell gsl-config --libs)
endif

LFLAGS +=	$(shell gsl-config --libs-without-cblas)

EXE=	su2hmc

SRC= \
	coord.c \
	random.c \
	matrices.c \
	congrad.c \
	bosonic.c\
	fermionic.c\
	force.c\
	par_mpi.c \
	su2hmc.c \
	clover.c \
	integrate.c\

INC= \
	sizes.h \
	errorcodes.h \
	coord.h \
	random.h \
	matrices.h \
	par_mpi.h \
	integrate.h\
	clover.h \
	su2hmc.h \
#
# No need to edit below this line
#
.SUFFIXES: .c .o
.PHONY: clean debug release profiler tar

OBJ=	$(SRC:.c=.o)
DEPS=	$(OBJ:.o=.d)
EDT=	$(SRC:.c=.c~)
TMP=	$(MOD) $(EDT) core

-include $(DEPS)
all:	$(EXE)

profiler:	CFLAGS += -g #-D_DEBUG
profiler:	release

release:	CFLAGS +=  -O3 
						
release:	$(EXE)

diagnostic: SRC+= diagnostics.c
diagnostic: CFLAGS+=-DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
diagnostic: debug

debug:	CFLAGS += -g -O0 -D_DEBUG #-D_DEBUG_MPI -Wall #-D_DEBUGCG 
debug:	$(EXE)

$(EXE):	$(OBJ)
		$(MPICC) ../main.c -MP $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)  
%.o: %.c
	$(MPICC) $(CFLAGS) $(IFLAGS) -c $<  

clean:
		rm -f $(OBJ) $(TMP) $(DEPS)

INP=	\
		midout

tar:
	tar cvfz $(EXE).tar $(MF) $(SRC) $(INC) $(INP) $(EXE) $(DEPS)
