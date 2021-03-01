MF=	Makefile

#FC= mpif77 
CC= mpiicc

#FFLAGS= -O3 -I /bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/finclude/
IFLAGS=     -I./INCLUDE -I./SFMT-src-1.5.1

CFLAGS=      $(IFLAGS) -mkl -std=gnu99 -DUSE_MKL -DHAVE_SSE2 -ipo  -qopt-zmm-usage=high\
		 -check-pointers=none  -xCORE-AVX512 -static-intel -heap-arrays

LFLAGS=	 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
		${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group\
#		-liomp5 -lpthread -lm -ldl #-wd10237

EXE=	su2hmc

SRC= \
	su2hmc.c \
	par_mpi.c \
      coord.c \
      random.c \
      slash.c \
      SFMT.c\

INC= \
	su2hmc.h \
	sizes.h \
      par_mpi.h \
      slash.h \
      random.h \
      slash.h \
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

release:	CFLAGS += -O3 -inline-level=2 -qopenmp -qopenmp-link=static\
		-DMKL_ILP64  -I"${MKLROOT}/include"
release:	$(EXE)

debug:	CFLAGS += -g -traceback -O3 -inline-level=2 -D_DEBUG -qopenmp -qopenmp-link=static
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
