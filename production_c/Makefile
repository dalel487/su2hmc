MF=	Makefile

#FC= mpif77 
CC= mpiicc

#FFLAGS= -O3 -I /bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/finclude/
IFLAGS=     -I./INCLUDE -I./SFMT-src-1.5.1

CFLAGS=     -ipo -xCASCADELAKE -DHAVE_SSE2 -qopt-zmm-usage=high -check-pointers=none \
		$(IFLAGS) -mkl -std=gnu99 -DUSE_MKL 

LFLAGS=

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

release:	CFLAGS += -O3 -inline-level=2 -qopenmp
release:	$(EXE)

debug:	CFLAGS += -O0 -g -inline-level=0 -DDEBUG
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
