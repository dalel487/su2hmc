MF=	Makefile

FC=		mpifort 
CC=		mpiicc
CXX=	mpiicpc
NVCC=	nvcc

IFLAGS=     -I../INCLUDE -I"${MKLROOT}/include"


CFLAGS=      $(IFLAGS) -cc=icx -qmkl -std=gnu++11  -DUSE_RAN2 \
			 -xHOST -static-intel -static-libgcc -masm=att -fcommon\
			 -qopenmp-simd -fopenmp -qopenmp-link=static -DSA3AT -x c++\

LFLAGS=		-liomp5 -lpthread -lm -ldl -lcudart -lcublas -L${NVSHMEM_HOME}/lib -lnvshmem -wd10237

MKLFLAGS=	-L${MKLROOT}/lib/intel64 -qmkl -DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64

CUFLAGS=	${IFLAGS} -ccbin ${CC} -arch=compute_75 -code=sm_75 -err-no -w\
			--allow-unsupported-compiler #-expt-relaxed-constexpr 
			#Last flag is an experimental one used to call host functions in device code. Not sure
			#if it does what we want because the error message suggesting it came up when declaring
			#a11 and a12 in the Dslash routine

EXE=		cuda_su2hmc_CASCADE

CSRC_DIR=	../production_c

CSRC=	\
	    coord.c \
		su2hmc.c \
    	random.c \
		par_mpi.c \

CUSRC_DIR=	../production_cuda
CUSRC=	\
		matrices.cu\
		force.cu\
		fermionic.cu\
		bosonic.cu

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

COBJ=	$(CSRC:.c=.o)
CUOBJ=	$(CUSRC:.cu=.o)
OBJ=	$(COBJ) $(CUOBJ)  

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=	$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

.PHONY: release
release:	CFLAGS +=  -O3 -inline-level=2 -ipo -qopt-zmm-usage=high
release:	CUFLAGS += -O3 #-dlto 
release:	$(EXE)

.PHONY: debug 
debug:		CFLAGS += -O0 -g -D_DEBUG -rdynamic # -DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
debug:		CUFLAGS += -O0 -G -g -D_DEBUG 
debug:		$(EXE)

$(EXE): $(OBJ)
	#$(NVCC) $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS)" $(LFLAGS) $^ -o $@
	$(CXX) $(CFLAGS) $(MKLFLAGS) $(LFLAGS) $^ -o $@
$(CUOBJ): %.o: $(CUSRC_DIR)/%.cu
	$(NVCC) $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS)" -dlink $< 
$(COBJ): %.o: $(CSRC_DIR)/%.c
	$(NVCC) $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS)" -c $< 

.PHONY: clean 
clean:
	rm -f $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
