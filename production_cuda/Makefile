MF=	Makefile

#FC= mpif77 
CC= mpiicc
NVCC = nvcc

#FFLAGS= -O3 -I /bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/finclude/
IFLAGS=     -I./INCLUDE -I../production_c/SFMT-src-1.5.1


CFLAGS=      $(IFLAGS) -mkl -DUSE_MKL -std=gnu99 -DHAVE_SSE2 -ipo -qopt-zmm-usage=high\
		 -xCOMMON-AVX512 -static-intel -static-libgcc -DMKL_DIRECT_CALL\
		 -qopenmp-simd -qopenmp -qopenmp-link=static -DSA3AT 

LFLAGS=	 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
		${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group\
		-liomp5 -lpthread -lm -ldl #-wd10237

CUFLAGS=	${IFLAGS} -ccbin ${CC} -arch=compute_70 -code=compute_70,sm_70,sm_72,sm_75
EXE=	cuda_su2hmc_CASCADE

CSRC= \
	su2hmc.c \
	par_mpi.c \
      coord.c \
      random.c \
      SFMT.c\

CUSRC= \
	cuda_init.cu \
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

.SUFFIXES:
.SUFFIXES: .c .o

COBJ=	 $(CSRC:.c=.o)
CUOBJ= $(CUSRC:.cu=.o)
OBJ=	 $(COBJ) $(CUOBJ)

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

.f.o:
	$(FC) $(FFLAGS) -c $<

all:	$(EXE)

release:	CFLAGS +=  -inline-level=2\
		-DMKL_ILP64  -I "${MKLROOT}/include"

release:	CUFLAGS += -O3 -Xcompiler "${CFLAGS} ${LFLAGS}"

release:	$(EXE)

debug:	CFLAGS += -g \
		-D_DEBUG -DUSE_RAN2 #-DDIAGNOSTIC -DNO_TIME -DNO_SPACE

debug:	CUFLAGS += -O0 -G -g -D_DEBUG -DUSE_RAN2 -Xcompiler "${CFLAGS} ${LFLAGS}"	

debug:	$(EXE)

$(EXE):	$(OBJ)
	$(NVCC) $(CUFLAGS) -o $@ $(OBJ)

$(OBJ):	$(MF) 
#$(INC)

clean:
	rm -f $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
