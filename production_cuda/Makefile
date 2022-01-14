MF=	Makefile

FC=		mpifort 
CC=		mpiicc
CXX=	mpiicpc
NVCC=	nvcc

IFLAGS=     -I../INCLUDE -I"${MKLROOT}/include" -I/opt/AMD/aocl/aocl-linux-gcc-3.0-6/include/\
				-I/usr/local/cuda/include/ -I/usr/include/c++/10/\
				-I/usr/x86_64-linux-gnu/include/c++/10/x86_64-linux-gnu/

CFLAGS=     $(IFLAGS) -std=gnu11 -qopt-zmm-usage=high -xSKYLAKE -axCASCADELAKE\
			-static-intel -static-libgcc -DMKL_DIRECT_CALL -D__INTEL_MKL__ -D__RANLUX__\
			-qopenmp -qno-openmp-offload -qopenmp-simd -qopenmp-link=static -DSA3AT \
			 -masm=att -fcommon
#CFLAGS=	$(IFLAGS) -DUSE_BLAS -fcommon -fpermissive \
		-march=skylake -mtune=cascadelake -static -D__RANLUX__\
		-fopenmp-simd -DSA3AT -flto\
#		-foffload=amdgcn-amdhsa="-march=gfx900" -fopenacc\

LFLAGS=		-liomp5 -lpthread -ldl -lgsl -lcudart -lcublas -dlink #-L${NVSHMEM_HOME}/lib -lnvshmem -lblis 

MKLFLAGS=	-L${MKLROOT}/lib/intel64 -qmkl #-DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64

CUFLAGS=	${IFLAGS}  -arch=compute_75 -code=sm_75 -err-no -w\
			--allow-unsupported-compiler #-expt-relaxed-constexpr 
			#Last flag is an experimental one used to call host functions in device code. Not sure
			#if it does what we want because the error message suggesting it came up when declaring
			#a11 and a12 in the Dslash routine

EXE=		cuda_su2hmc_CASCADE

CSRC_DIR=	../production_c
CSRC=	\
	    coord.c \
	    congrad.c \
		su2hmc.c \
    	random.c \
		par_mpi.c \

CUSRC_DIR=	../production_cuda
CUSRC=	\
		matrices.cu\
		force.cu\
		bosonic.cu\
		fermionic.cu

CINC=	\
		su2hmc.h \
		sizes.h \
	    par_mpi.h \
    	random.h \
	    matrices.h \
    	coord.h \
	    errorcodes.h \

CUINC= \
		cuda_complex.hpp\

#
# No need to edit below this line
#

.SUFFIXES: .c .cu .h .hpp .o
.PHONY: clean debug release profiler tar

COBJ=	$(CSRC:.c=.o)
CUOBJ=	$(CUSRC:.cu=.o)
OBJ=	$(COBJ) $(CUOBJ)  

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=	$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

profiler:	CFLAGS += -g
profiler:	CUFLAGS += -O0 -G -g 
profiler:	release

release:	CFLAGS +=  -O3 -inline-level=2 -qopt-zmm-usage=high
#release:	CFLAGS +=  -Ofast -funroll-all-loops -fcf-protection=none
release:	CUFLAGS += -O3 #-dlto 
release:	$(EXE)

debug:		CFLAGS += -O0 -g -D_DEBUG -rdynamic # -DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
debug:		CUFLAGS += -O0 -G -g -D_DEBUG 
debug:		$(EXE)

$(EXE): $(OBJ)
	$(NVCC) -ccbin ${CC} $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS)" $(LFLAGS) $^ -o $@
	#$(CXX) $(CFLAGS) $(MKLFLAGS) $(LFLAGS) $^ -o $@
$(CUOBJ): %.o: $(CUSRC_DIR)/%.cu
	$(NVCC) -ccbin ${CXX} $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS) " -dc $< 
$(COBJ): %.o: $(CSRC_DIR)/%.c
	$(NVCC) -ccbin ${CC}  $(CUFLAGS) -Xcompiler "$(CFLAGS) -x c++ $(MKLFLAGS)" -c $< 

clean:
	rm -f $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
