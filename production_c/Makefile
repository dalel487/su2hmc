MF=	Makefile

CC= mpiicc

IFLAGS=     -I../INCLUDE 

CFLAGS=      $(IFLAGS) -mkl -DUSE_MKL -std=gnu11 -ipo -qopt-zmm-usage=high \
		 -xCOMMON-AVX512 -static-intel -static-libgcc -DMKL_DIRECT_CALL \
		 -qopenmp-simd -qopenmp -qopenmp-link=static -DSA3AT -DUSE_RAN2

LFLAGS=	-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64\
		-liomp5 -lpthread -lm -ldl 

EXE=	su2hmc

SRC= \
	su2hmc.c \
	force.c\
	bosonic.c\
	fermionic.c\
	par_mpi.c \
      coord.c \
      random.c \
      matrices.c \

INC= \
	su2hmc.h \
	sizes.h \
      par_mpi.h \
      slash.h \
      random.h \
      matrices.h \
      coord.h \
      errorcodes.h \

#
# No need to edit below this line
#

.SUFFIXES: .c .o
.PHONY: clean debug release advisor tar

OBJ=	$(SRC:.c=.o)
EDT=	$(SRC:.c=.c~)
TMP=	$(MOD) $(EDT) core

all:	$(EXE)

advisor:	CFLAGS += -g

advisor:	release

release:	CFLAGS +=  -O3  -inline-level=2\
		-DMKL_ILP64  -I"${MKLROOT}/include"# -DUSE_RAN2
release:	$(EXE)

debug:	CFLAGS += -g -O0 \
		-D_DEBUG -DUSE_RAN2 -DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
	
debug:	$(EXE)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvfz $(EXE).tar $(MF) $(SRC) $(INC) $(INP) $(EXE)
