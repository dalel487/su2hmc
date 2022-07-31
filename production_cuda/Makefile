FC=		mpifort 
CC=		mpiicc 
CXX=	mpicxx 
NVCC=	nvcc

IFLAGS=     -I../INCLUDE -I"${MKLROOT}/include" -I ./ \
				-I/usr/include/c++/10/\
				-I/usr/x86_64-linux-gnu/include/c++/10/x86_64-linux-gnu/\
				-I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/include/ \
				-I/opt/nvidia/hpc_sdk/Linux_x86_64/2022/math_libs/include/ 

CFLAGS=	$(IFLAGS) -DGSL_BLAS -D__NVCC__ -D__RANLUX_  -Ofast -O3  -fpic\
			-march=skylake -mtune=cascadelake\

MKLFLAGS=	#-L${MKLROOT}/lib/intel64 -qmkl #-DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64
CULFLAGS=	-L/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/lib64/ \
				-L/opt/nvidia/hpc_sdk/Linux_x86_64/2022/math_libs/lib64/ \
				-lcuda -lcudart -lcublas #-L${NVSHMEM_HOME}/lib -lnvshmem -lblis 

LFLAGS=		-lpthread -ldl -lgsl ${CULFLAGS}

NVCCFLAGS=	${IFLAGS}  -gencode arch=compute_75,code=lto_75 -gencode arch=compute_80,code=sm_80 -err-no -w \
			-expt-relaxed-constexpr 
			#Last flag is an experimental one used to call host functions in device code. Not sure
			#if it does what we want because the error message suggesting it came up when declaring
			#a11 and a12 in the Dslash routine
#Replace first line of NVCCFLAGS with above when done. Targets every platform since 2016
#NVCCFLAGS=	${IFLAGS}  -arch=compute_75 -code=sm_75 -err-no -w \

EXE=		cuda_su2hmc

CSRC_DIR=	../production_c
CSRC=	\
	    coord.c \
	    congrad.c \
		force.c \
    	random.c \
		par_mpi.c \
		bosonic.c\
		fermionic.c

CUSRC_DIR=	../production_cuda
CUSRC=	\
		cubosonic.cu\
		cumatrices.cu\
		cuforce.cu\

#CUSRC= CUDA_Compile.cu
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

.SUFFIXES: .c .cu .h .hpp .o 
.PHONY: clean debug release profiler tar
C_SOURCES:= $(wildcard $(CSRC_DIR)/*.c)
CU_SOURCES:= $(wildcard $(CUSRC_DIR)/*.cu)
COBJ=	$(patsubst $(CSRC_DIR)/%.c, $(CSRC_DIR)/%.o, $(C_SOURCES))
CUOBJ= $(patsubst $(CUSRC_DIR)/%.cu, $(CUSRC_DIR)/%.o, $(CU_SOURCES))
CLIB= libc.a
CULIB= libcu.a
CULIB2= libcu2.a
OBJ=	$(CLIB) $(CULIB2) #$(COBJ) $(CUOBJ) 

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=	$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

profiler:	CFLAGS += -g
profiler:	NVCCFLAGS += -O0 -G -g 
profiler:	release

release:	CFLAGS +=  -O3
release:	NVCCFLAGS += -O3# -dlto 
release:	$(EXE)

debug:		CFLAGS += -O0 -g -D_DEBUG -rdynamic # -DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
debug:		NVCCFLAGS += -O0 -G -g -D_DEBUG 
debug:		$(EXE)

		 
$(EXE): ../main.c $(OBJ)
	$(CXX) $(CFLAGS) $(MKLFLAGS) -D__NVCC__ $^ $(LFLAGS) $(CULFLAGS)  -o $@
$(CULIB2): $(CULIB)
	ar rc $@ $^ 
$(CULIB): $(CUOBJ)
	$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS)" $(NVCCFLAGS)  $(LFLAGS) -dlto -dlink $^ -o $@ 
$(CLIB):	$(COBJ) 
	ar rc $@  $^
%.o: $(CUSRC_DIR)/%.cu
	$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) -Xcompiler "$(CFLAGS) $(MKLFLAGS)" $^ -dc $@ 
%.o: $(CSRC_DIR)/%.c
	$(CXX) $(CFLAGS) $(IFLAGS) $(MKLFLAGS) -c -x c++ $^  

clean:
	rm -f $(OBJ) $(COBJ) $(CUOBJ) $(CULIB) $(CLIB) $(TMP)

INP=	\
	midout \

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
