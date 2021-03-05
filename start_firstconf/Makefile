MF=	Makefile

#FC= mpixlf77_r -qsmp=omp
FC	= mpifort
FFLAGS= -Ofast  -I /bgsys/drivers/toolchain/V1R1M2/gnu-linux/lib/gcc/powerpc64-bgq-linux/4.4.6/finclude/

LFLAGS=

EXE=	su2hmc.exe

SRC= \
	su2hmc.f \
	par_mpi.f

INC= \
	precision.h \
	sizes.h \
	common_diquark.h \
	common_dirac.h \
	common_dum1.h \
	common_gauge.h \
	common_mat.h \
	common_neighb.h \
	common_para.h \
	common_param.h \
	common_pseud.h \
	common_trans.h \
	common_trial.h \
	common_trial_u11u12.h

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .f .o

OBJ=	$(SRC:.f=.o)

#MOD=	$(SRC:.f=.mod)
MOD=	
EDT=	$(SRC:.f=.f~)
TMP=	$(MOD) $(EDT) core

.f.o:
	$(FC) $(FFLAGS) -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF) $(INC)

clean:
	rm -f $(EXE) $(OBJ) $(TMP)

INP=	\
	midout \
	con

tar:
	tar cvf $(EXE).tar $(MF) $(SRC) $(INC) $(INP)
