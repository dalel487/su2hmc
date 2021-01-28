MF=	Makefile

#FC= mpif77 
CC= mpiicc

#FFLAGS= -O3 -I /bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/finclude/

CFLAGS=     -ipo -O0 -g -xCASCADELAKE -DHAVE_SSE2 -DDEBUG -qopt-zmm-usage=high -check-pointers=none \
		-I ./SFMT-src-1.5.1/ -mkl -inline-level=2 -std=gnu99 -DUSE_MKL 

LFLAGS=

EXE=	su2hmc.exe

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

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF) $(INC)

clean:
	rm -f $(EXE) $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvf $(EXE).tar $(MF) $(SRC) $(INC) $(INP)
