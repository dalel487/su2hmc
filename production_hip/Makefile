FC=		mpifort 
CC=		mpiicc 
CXX=		mpicxx 
HIPCC=	hipcc

IFLAGS=     -I../INCLUDE -I"${MKLROOT}/include" -I ./ \
				-I/usr/include/c++/10/\
				-I/usr/x86_64-linux-gnu/include/c++/10/x86_64-linux-gnu/\
				-I/opt/rocm/include

CFLAGS=	$(IFLAGS) -DAMD_BLAS -D__HIPCC__ -D__RANLUX__  -Ofast -O3  -fpic\
			-march=znver3 -mtune=znver4

ROCLFLAGS=	-L/opt/rocm/lib -lrocm-core -lrocblas #-L${NVSHMEM_HOME}/lib -lnvshmem -lblis 

LFLAGS=		-lpthread -ldl -lgsl ${ROCLFLAGS}

HIPCCFLAGS=	${IFLAGS}  -x hip --gpu-archtecture=gfx1034
			#Last flag is an experimental one used to call host functions in device code. Not sure
			#if it does what we want because the error message suggesting it came up when declaring
			#a11 and a12 in the Dslash routine
#Replace first line of HIPCCFLAGS with above when done. Targets every platform since 2016
#HIPCCFLAGS=	${IFLAGS}  -arch=compute_75 -code=sm_75 -err-no -w \

EXE=		su2hmc_HIP

CSRC_DIR=	../production_c
CSRC=	\
	    coord.c \
	    congrad.c \
		force.c \
    	random.c \
		par_mpi.c \
		bosonic.c\
		fermionic.c

HIPSRC_DIR=	../production_hip
HIPSRC=	\
		cubosonic.hip\
		cumatrices.hip\
		cuforce.hip\

#HIPSRC= HIPDA_Compile.hip
CINC=	\
		su2hmc.h \
		sizes.h \
	    par_mpi.h \
    	random.h \
	    matrices.h \
    	coord.h \
	    errorcodes.h \

#
# No need to edit below this line
#

.SUFFIXES: .c .hip .h .hpp .o 
.PHONY: clean debug release profiler tar
C_SOURCES:= $(wildcard $(CSRC_DIR)/*.c)
HIP_SOURCES:= $(wildcard $(HIPSRC_DIR)/*.hip)
COBJ=	$(patsubst $(CSRC_DIR)/%.c, $(CSRC_DIR)/%.o, $(C_SOURCES))
HIPOBJ= $(patsubst $(HIPSRC_DIR)/%.hip, $(HIPSRC_DIR)/%.o, $(HIP_SOURCES))
CLIB= libc.a
HIPLIB= libhip.a
HIPLIB2= libhip2.a
OBJ=	$(CLIB) $(HIPLIB2) #$(COBJ) $(HIPOBJ) 

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
HIPEDT=	$(HIPSRC:.hip=.hip~)
TMP=	$(MOD) $(CEDT) $(HIPEDT) core

profiler:	CFLAGS += -g
profiler:	HIPCCFLAGS += -O0 -G -g 
profiler:	release

release:	CFLAGS +=  -O3
release:	HIPCCFLAGS += -O3# -dlto 
release:	$(EXE)

debug:		CFLAGS += -O0 -g -D_DEBUG -rdynamic # -DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
debug:		HIPCCFLAGS += -O0 -G -g -D_DEBUG 
debug:		$(EXE)

		 
$(EXE): ../main.c $(OBJ)
	$(CXX) $(CFLAGS) -D__HIPCC__ $^ $(LFLAGS) $(ROCLFLAGS)  -o $@
#$(HIPLIB2): $(HIPLIB)
#	ar rc $@ $^ 
#$(HIPLIB): $(HIPOBJ)
#	$(HIPCC) -ccbin $(CXX) $(HIPCCFLAGS) -Xcompiler "$(CFLAGS)" $(HIPCCFLAGS)  $(LFLAGS) -dlto -dlink $^ -o $@ 
#$(CLIB):	$(COBJ) 
#	ar rc $@  $^
%.o: $(HIPSRC_DIR)/%.hip
	$(HIPCC) $(HIPCCFLAGS) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" $^ -dc $@ 
%.o: $(CSRC_DIR)/%.c
	$(CC) $(CFLAGS) $(IFLAGS) -c -x c++ $^  

clean:
	rm -f $(OBJ) $(COBJ) $(HIPOBJ) $(HIPLIB) $(CLIB) $(TMP)

INP=	\
	midout \

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(HIPSRC) $(CINC) $(HIPINC) $(INP)
