FC=		mpifort 
CC=		mpiicc 
CXX=	mpicxx -cxx=nvc++
NVCC=	nvcc

IFLAGS=     -I../INCLUDE -I"${MKLROOT}/include" -I ./ \
				-I/usr/include/c++/10/\
				-I/usr/x86_64-linux-gnu/include/c++/10/x86_64-linux-gnu/\

CFLAGS=	$(IFLAGS) -DGSL_BLAS -D__RANLUX__ -acc -fast -O4 -Mipa=fast -fpic\
			-Mcache_align -static -static-nvidia -cuda -cudalib -tp=skylake\
			-gpu=managed

MKLFLAGS=	#-L${MKLROOT}/lib/intel64 -qmkl #-DUSE_MKL -DMKL_DIRECT_CALL -DMKL_ILP64
CULFLAGS=	-L/opt/nvidia/hpc_sdk/Linux_x86_64/2022/math_libs/lib64\
				-lcudart_static -lcublas_static #-L${NVSHMEM_HOME}/lib -lnvshmem #-lblis 
LFLAGS=		-lpthread -ldl -lgsl  ${CULFLAGS}

NVCCFLAGS=	${IFLAGS}  -arch=compute_60 -code=sm_60,sm_61,sm_70,sm_75,sm_80,sm_86 -err-no -w \
			-expt-relaxed-constexpr --allow-unsupported-compiler
			#Last flag is an experimental one used to call host functions in device code. Not sure
			#if it does what we want because the error message suggesting it came up when declaring
			#a11 and a12 in the Dslash routine
#Replace first line of NVCCFLAGS with above when done. Targets every platform since 2016
#NVCCFLAGS=	${IFLAGS}  -arch=compute_75 -code=sm_75 -err-no -w \

EXE=		cuda_su2hmc

CSRC_DIR=	../production_c
CSRC=	\
		su2hmc.c\
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

COBJ=	$(CSRC:.c=.o)
CLIB= libc.a
CUOBJ= $(CUSRC:.cu=.o)
CULIB= libcu.a
OBJ=	$(COBJ) $(CULIB) # $(CLIB) 

#MOD=	$(CSRC:.c=.mod)
MOD=	
CEDT=	$(CSRC:.c=.c~)
CUEDT=	$(CUSRC:.cu=.cu~)
TMP=	$(MOD) $(CEDT) $(CUEDT) core

profiler:	CFLAGS += -g
profiler:	NVCCFLAGS += -O0 -G -g 
profiler:	release

release:	CFLAGS +=  -O3 -inline-level=2 -qopt-zmm-usage=high
release:	NVCCFLAGS += -O3 #-dlto 
release:	$(EXE)

debug:		CFLAGS += -O0 -g -D_DEBUG -rdynamic # -DDIAGNOSTIC #-DNO_TIME -DNO_SPACE
debug:		NVCCFLAGS += -O0 -G -g -D_DEBUG 
debug:		$(EXE)

$(EXE): $(OBJ)
	${CXX}  $(CFLAGS) -D__NVCC__$(CSRC_DIR)/su2hmc.c -o $@   $^ $(LFLAGS) $(CULFLAGS) 
$(CULIB): $(CUOBJ)
	$(NVCC) -ccbin ${CXX} $(NVCCFLAGS) -Xcompiler "$(CFLAGS) " $(LFLAGS)  $^ -lib --output-file $@
#$(CLIB):	$(COBJ) 
#	xiar  rc  $@  $^
$(CUOBJ): %.o: $(CUSRC_DIR)/%.cu
	$(NVCC) -ccbin ${CXX} $(NVCCFLAGS) -Xcompiler "$(CFLAGS) " -dc $< 
$(COBJ): %.o: $(CSRC_DIR)/%.c
	${CXX} $(CFLAGS) $(IFLAGS)  -D__NVCC__ -c $< 
	#$(NVCC) -ccbin ${CC}  $(NVCCFLAGS) -Xcompiler "$(CFLAGS) -x c++ $(MKLFLAGS)" -dc $< 

clean:
	rm -f $(OBJ) $(COBJ) $(CUOBJ) $(TMP)

INP=	\
	midout \

tar:
	tar cvf $(EXE).tar $(MF) $(CSRC) $(CUSRC) $(CINC) $(CUINC) $(INP)
