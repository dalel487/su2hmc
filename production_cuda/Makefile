MF=	Makefile

#FC= mpif77 
CC= mpiicc
NVCC = nvcc

IFLAGS=     -I../INCLUDE -I"${MKLROOT}/include" -I./


CFLAGS=      $(IFLAGS) -std=gnu99 -ipo -qopt-zmm-usage=high\
		 -xCOMMON-AVX512 -static-intel -static-libgcc -masm=att\
		 -qopenmp-simd -qopenmp -qopenmp-link=static -DSA3AT \

LFLAGS=	 -L${MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl -lcudart #-wd10237

MKLFLAGS=	-mkl -DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64 
CUFLAGS=	${IFLAGS} -ccbin ${CC} -arch=compute_70 -code=sm_70,sm_72,sm_75,sm_80,sm_86 -err-no\

EXE=	cuda_su2hmc_CASCADE

CSRC_DIR=../production_c

CSRC= \
      coord.c \
	su2hmc.c \
      random.c \
	par_mpi.c \

CUSRC= CUDA_Compile.cu\
#	matrices.cu\
#	force.cu\
#	fermionic.cu\
#	bosonic.cu


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
	cuda_complex.hpp\


#
# No need to edit below this line
#

.SUFFIXES: .c .cu .h .hpp .o

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
	$(NVCC) $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS) -fPIC" $(LFLAGS) $^ -o $@
$(CUOBJ): %.o: %.cu
	$(NVCC) $(CUFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS) -fPIC" -x cu -dc $< 
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
