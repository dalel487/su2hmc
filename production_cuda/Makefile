MF=	Makefile

#FC= mpif77 
CC= mpiicc
NVCC = nvcc

#FFLAGS= -O3 -I /bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/finclude/
IFLAGS=     -I../INCLUDE -I"${MKLROOT}/include"


CFLAGS=      $(IFLAGS) -std=gnu99 -DHAVE_SSE2 -ipo -qopt-zmm-usage=high\
		 -xCOMMON-AVX512 -static-intel -static-libgcc -D__NVCC__\
		 -qopenmp-simd -qopenmp -qopenmp-link=static -DSA3AT -DUSE_RAN2\


LFLAGS=	 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
		${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group\
		-liomp5 -lpthread -lm -ldl #-wd10237

MKLFLAGS=	-mkl -DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64 
CUFLAGS=	${IFLAGS} -ccbin ${CC} -arch=compute_70 -code=compute_70,sm_70,sm_72,sm_75 --allow-unsupported-compiler
EXE=	cuda_su2hmc_CASCADE

CSRC= \
      coord.c \
	su2hmc.c \
      random.c \
	par_mpi.c \

CUSRC= \
	multiply.cu


CINC= \
	su2hmc.h \
	sizes.h \
      par_mpi.h \
      slash.h \
      random.h \
      multiply.h \
      coord.h \
      errorcodes.h \

CUINC= \


#
# No need to edit below this line
#

.SUFFIXES: .c .cu .h .o

COBJ=	 $(CSRC:.c=.o)
CUOBJ= $(CUSRC:.cu=.o)
OBJ=	 $(COBJ) $(CUOBJ)  

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

.PHONY: release
release:	CFLAGS +=  -O3 -inline-level=2 
release:	CUFLAGS += -O3
release:	$(EXE)

.PHONY: debug 
debug:	CFLAGS += -O0 -g -D_DEBUG   #-DDIAGNOSTIC -DNO_TIME -DNO_SPACE
debug:	CUFLAGS += -O0 -G -g -D_DEBUG 	
debug:	$(EXE)

$(EXE): $(OBJ)
	$(NVCC) $(CUFLAGS) -o $@ $^ $(LFLAGS)

$(COBJ): %.o: %.c
	$(NVCC) $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS)" -c $< 

$(CUOBJ): %.o: %.cu
	$(NVCC) $(CUFLAGS) -Xcompiler "$(CFLAGS)" -dc $< 

.PHONY: clean 
clean:
	rm -f $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
