MF=	Makefile

#FC= mpif77 
CC= mpiicc
NVCC = nvcc

#FFLAGS= -O3 -I /bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/finclude/
IFLAGS=     -I./INCLUDE -I./SFMT-src-1.5.1

CFLAGS=      $(IFLAGS) -mkl -DUSE_MKL -std=gnu99 -DHAVE_SSE2 -ipo -qopt-zmm-usage=high\
		 -xCOMMON-AVX512 -static-intel -static-libgcc -DMKL_DIRECT_CALL -I_MPI_FABRICS=ofi\
		 -qopenmp-simd #-qopenmp -qopenmp-link=static -DSA3AT 

LFLAGS=	 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
		${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group\
		-liomp5 -lpthread -lm -ldl #-wd10237

EXE=	su2hmc

SRC= \
	su2hmc.c \
	par_mpi.c \
      coord.c \
      random.c \
      multiply.c \
      SFMT.c\

INC= \
	su2hmc.h \
	sizes.h \
      par_mpi.h \
      slash.h \
      random.h \
      multiply.h \
      coord.h \
      errorcodes.h \


#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .c .o

OBJ=	$(SRC:.c=.o)

#MOD=	$(SRC:.c=.mod)
MOD=	
EDT=	$(SRC:.c=.c~)
TMP=	$(MOD) $(EDT) core

.f.o:
	$(FC) $(FFLAGS) -c $<

all:	$(EXE)

release:	CFLAGS += -g  -O1  -inline-level=2\
		-DMKL_ILP64  -I"${MKLROOT}/include"
release:	$(EXE)

debug:	CFLAGS += -g -O3 \
		-D_DEBUG -DDIAGNOSTIC #-DNO_TIME -DNO_SPACE

	
debug:	$(EXE)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF) 
#$(INC)

clean:
	rm -f $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvf $(EXE).tar $(MF) $(SRC) $(INC) $(INP)
