/**
 * @file su2hmc_to_HiRep.c
 *
 * @brief
 * Convert su2hmc gauge config -> HiRep format
 * Reads:  s_path  (little-endian: int32 nrank | u[0][V*ndim] cdouble | u[1][V*ndim] cdouble | int64 seed)
 * Writes: h_path  (big-endian:    NG,T,X,Y,Z int32 | plaquette double | gauge AoS)
 *
 * Assumes single rank (nrank==1). Multi-rank dumps need rank-merge first.
 * Permutes mu: su2hmc (x,y,z,t)=(0,1,2,3) -> HiRep (t,x,y,z)=(0,1,2,3).
 * Re-indexes sites: su2hmc x-fastest -> HiRep z-fastest.
 *
 * @author D. Lawlor/Claude Opus 4.7
 */
#include	<assert.h>
#include	<clover.h>
#include	<matrices.h>
#ifdef	USE_GPU
#include	<cuda_runtime.h>
cublasHandle_t cublas_handle;
cublasStatus_t cublas_status;
cudaMemPool_t mempool;
//Fix this later
#endif

int main(int argc, char *argv[]){
	const char funcname[] = "main";
	//Prevent any issues with MPI being initialised needlessly in Par_begin
#if(nproc>1)
#error	nproc>1 in sizes.h. Ensure npx=npy=npt=npt=1, and then compile.
#endif
	//Lattice layout setup. 	
	Par_begin(argc,argv);
#ifdef USE_GPU
	//CUBLAS Handle
	cublasCreate(&cublas_handle);
	//Set up grid and blocks
	blockInit(nx, ny, nz, nt, &dimBlock, &dimGrid);
	//CUDA device
	int device=-1;
	cudaGetDevice(&device);
	//For asynchronous memory, when CUDA syncs any unused memory in the pool is released back to the OS
	//unless a threshold is given. We'll base our threshold off of Congradp
	//12*kvol for the clover and 4*16*kvolHalo for the fermion fields 
	//Factor of 1.5 because we need it in single and double precsion should be plenty without being excessive.
	//Not everything has a halo so larger halos give us more headroom too.
	cudaDeviceGetDefaultMemPool(&mempool, device);
	int threshold=8*kfermHalo*sizeof(Complex);
	cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
#endif
	FILE *midout;
	const char *filename = (argc!=2) ?"midout":argv[1];
	char *fileop = "r";
	if( !(midout = fopen(filename, fileop) ) ){
		fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
				OPENERROR, funcname, filename, fileop);
		exit(OPENERROR);
	}

	float dt=0.004;//HMC Step size to keep the existing parser happy
	float beta = 1.7f; //Inverse gauge coupling
	float kappa = 0.1780f;//Hopping parameter
	float jqq = 0.0;//Diquark source
	float c_sw = 0.0;//Clover coefficient
	float mu = 0.0f; //Chemical potential
	float delb=0; //Legacy. Not used
	int istart = 1;//Start type (hot or cold). Just to keep the existing parser happy
	int icheck = 5; //How often are configurations saved (or read in this case)
	int iread = 0; //First config number to read in
	int stepl = 250;//HMC Step length
	int ntraj = 10;//Number of trajectories (NOT CONFIGS). 
	fscanf(midout, "%f %f %f %f %f %f %f %d %d %d %d %d", &dt, &beta, &kappa,\
			&jqq, &c_sw, &mu, &delb, &stepl, &ntraj, &istart, &icheck, &iread);
	fclose(midout);
	assert(stepl>0);	assert(ntraj>0);	  assert(istart>=0);  assert(icheck>0);  assert(iread>=0); 
	//Boundary condition
	const int ibound = -1;
	//How many configs to read
#ifdef _DEBUG
	printf("Reading %d configs\n",ntraj);
#endif

	//Gauge, trial and momentum fields 
	//You'll notice that there are two different allocation/free statements
	//One for CUDA and one for everything else depending on what's
	//being used
	/***	Let's take a quick moment to compare this to the analysis code.
	 *	The analysis code stores the gauge field as a 4 component real valued vector, whereas the produciton code
	 *	used two complex numbers.
	 *
	 *	Analysis code: u=(Re(u[0]),Im(u[1]),Re(u[1]),Im(u[0]))
	 *	Production code: u[0]=u[0]+I*u[3]	u[1]=u[2]+I*u[1]
	 *
	 */
	Complex *u[2], *ut[2],*Phi;
	Complex_f *ut_f[2];
	double *dk[2];
	float	*dk_f[2];
	//Halo index arrays
	unsigned int *iu, *id;
	//And clover arrays. These only get assigned if @f$c_\text{SW}>0@f$
	Complex *sigval; Complex_f *sigval_f; unsigned short *sigin;
#ifdef USE_GPU
	//Managed here because it's easier to fill them on CPU
	cudaMallocManaged((void**)&iu,ndim*kvol*sizeof(int),cudaMemAttachGlobal);
	cudaMallocManaged((void**)&id,ndim*kvol*sizeof(int),cudaMemAttachGlobal);

	cudaMallocManaged((void **)&dk[0],(kvolHalo)*sizeof(double),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&dk[1],(kvolHalo)*sizeof(double),cudaMemAttachGlobal);
#ifdef _DEBUG
	cudaMallocManaged((void **)&dk_f[0],(kvolHalo)*sizeof(float),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&dk_f[1],(kvolHalo)*sizeof(float),cudaMemAttachGlobal);
#else
	cudaMalloc((void **)&dk_f[0],(kvolHalo)*sizeof(float));
	cudaMalloc((void **)&dk_f[1],(kvolHalo)*sizeof(float));
#endif

	unsigned short	*gamin; Complex *gamval; Complex_f *gamval_f;
	cudaMallocManaged((void **)&gamin,4*4*sizeof(short),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&gamval,5*4*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&gamval_f,5*4*sizeof(Complex_f),cudaMemAttachGlobal);

	cudaMallocManaged((void **)&u[0],ndim*kvol*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&u[1],ndim*kvol*sizeof(Complex),cudaMemAttachGlobal);
	//Needs to be managed as fermionic.c still used them on CPU
	cudaMallocManaged((void **)&ut[0],ndim*(kvolHalo)*sizeof(Complex),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&ut[1],ndim*(kvolHalo)*sizeof(Complex),cudaMemAttachGlobal);
#ifdef _DEBUG
	cudaMallocManaged((void **)&ut_f[0],ndim*(kvolHalo)*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&ut_f[1],ndim*(kvolHalo)*sizeof(Complex_f),cudaMemAttachGlobal);
	cudaMallocManaged((void **)&Phi, nf*kferm*sizeof(Complex),cudaMemAttachGlobal);
#else
	cudaMalloc((void **)&ut_f[0],ndim*(kvolHalo)*sizeof(Complex_f));
	cudaMalloc((void **)&ut_f[1],ndim*(kvolHalo)*sizeof(Complex_f));
	cudaMalloc((void **)&Phi, nf*kferm*sizeof(Complex));
#endif
#else
	id = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));
	iu = (unsigned int*)aligned_alloc(AVX,ndim*kvol*sizeof(int));

	alignas(AVX) unsigned short gamin[16]; alignas(AVX) Complex gamval[20]; alignas(AVX) Complex_f gamval_f[20];

	dk[0] = (double *)aligned_alloc(AVX,(kvolHalo)*sizeof(double));
	dk[1] = (double *)aligned_alloc(AVX,(kvolHalo)*sizeof(double));
	dk_f[0] = (float *)aligned_alloc(AVX,(kvolHalo)*sizeof(float));
	dk_f[1] = (float *)aligned_alloc(AVX,(kvolHalo)*sizeof(float));

	u[0] = (Complex *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex));
	u[1] = (Complex *)aligned_alloc(AVX,ndim*kvol*sizeof(Complex));
	ut[0] = (Complex *)aligned_alloc(AVX,ndim*(kvolHalo)*sizeof(Complex));
	ut[1] = (Complex *)aligned_alloc(AVX,ndim*(kvolHalo)*sizeof(Complex));
	ut_f[0] = (Complex_f *)aligned_alloc(AVX,ndim*(kvolHalo)*sizeof(Complex_f));
	ut_f[1] = (Complex_f *)aligned_alloc(AVX,ndim*(kvolHalo)*sizeof(Complex_f));
#endif
	//Get nearest neigbour addresses
	Addrc(iu, id);
	//And confirm they're legit
	Check_addr(iu, ksize, ksizet, 0, kvolHalo);
	Check_addr(id, ksize, ksizet, 0, kvolHalo);
	//Free unneccessary arrays from lattice setup.
	free(hd); free(hu); free(pcoord);

	/**
	 * @subsection Initialisation
	 *
	 * Changing the value of istart in the input parameter file gives us the following start options. These are quoted
	 * from the FORTRAN comments
	 *
	 * istart < 0: Start from tape in FORTRAN?!? How old was this code? (depreciated, replaced with iread)
	 *
	 * istart = 0: Ordered/Cold Start
	 * 			For some reason this leaves the trial fields as zero in the FORTRAN code?
	 *
	 * istart > 0: Random/Hot Start
	 */
	Init(istart,ibound,iread,beta,mu,kappa,jqq,c_sw,u,ut,ut_f,gamval,gamval_f,gamin,dk,dk_f,iu,id);
	if(c_sw)
		Init_clover(&sigval,&sigval_f,&sigin,c_sw);

	/// @f$\sigma_{\mu\nu}@f$ if we're using clover fermions
#ifdef USE_GPU
	//GPU Initialisation stuff
	Init_CUDA(ut[0],ut[1],gamval,gamval_f,gamin,dk[0],dk[1],iu,id);//&dimBlock,&dimGrid);
#endif
	//Send trials to accelerator for reunitarisation
	Reunitarise(ut);
	//Get trials back
#ifdef USE_GPU
#if(nproc>1) //Memcpy routines need to be strided if there is a halo since the lattice is not contiguous in memory
	for(unsigned short mu=0;mu<ndim;mu++){
		cudaMemcpyAsync(u[0]+kvol*mu, ut[0]+kvolHalo*mu, kvol*sizeof(Complex),cudaMemcpyDefault,streams[mu]);
		cudaMemcpyAsync(u[1]+kvol*mu, ut[1]+kvolHalo*mu, kvol*sizeof(Complex),cudaMemcpyDefault,streams[mu]);
	}
#else
	cudaMemcpyAsync(u[0], ut[0], ndim*kvol*sizeof(Complex),cudaMemcpyDefault,streams[0]);
	cudaMemcpyAsync(u[1], ut[1], ndim*kvol*sizeof(Complex),cudaMemcpyDefault,streams[1]);
#endif
#else
	for(unsigned short mu=0;mu<ndim;mu++){
		memcpy(u[0]+kvol*mu, ut[0]+kvolHalo*mu, kvol*sizeof(Complex));
		memcpy(u[1]+kvol*mu, ut[1]+kvolHalo*mu, kvol*sizeof(Complex));
	}
#endif
#ifdef USE_GPU
	cudaDeviceSynchronise();
#endif
	//Prepare file names
	char suffix[FILELEN]="";
	int buffer; char buff2[7];
	//Add script for extracting correct mu, j etc.
	buffer = (int)round(100*beta);
	sprintf(buff2,"b%03d",buffer);
	strcat(suffix,buff2);
	//κ
	buffer = (int)round(10000*kappa);
	sprintf(buff2,"k%04d",buffer);
	strcat(suffix,buff2);
	//μ
	buffer = (int)round(1000*mu);
	sprintf(buff2,"mu%04d",buffer);
	strcat(suffix,buff2);
	//J
	buffer = (int)round(1000*jqq);
	sprintf(buff2,"j%03d",buffer);
	strcat(suffix,buff2);
	//c_sw
	if(c_sw){
		buffer = (int)round(100*c_sw);
		sprintf(buff2,"c%03d",buffer);
		strcat(suffix,buff2);
	}
	//nx
	sprintf(buff2,"s%02d",nx);
	strcat(suffix,buff2);
	//nt
	sprintf(buff2,"t%02d",nt);
	strcat(suffix,buff2);
	for(unsigned int i=iread;i<=ntraj;i+=icheck){
		const unsigned int conf = i;

		printf("Reading config %d\n",conf);
		Par_sread(conf, beta, mu, kappa, jqq,c_sw,u[0],u[1],ut[0],ut[1]);
		Reunitarise(ut);
		Trial_Exchange(ut,ut_f);
#ifdef USE_GPU
		cudaDeviceSynchronise();
#endif

		//Polyakov loop
		double poly = Polyakov(ut_f);
		double hg, avplaqs, avplaqt;
		//Plaquettes
		Average_Plaquette(&hg,&avplaqs,&avplaqt,ut_f,iu,beta);
		//Fermionic observable setup
		double pbp=0; Complex qq=0; Complex qbqb=0;
		double endenf=0, denf=0;
		//Fun facts
		int itercg=0;
		int measure_check=Measure(&pbp,&endenf,&denf,&qq,&qbqb,respbp,&itercg,ut,ut_f,iu,id,\
				gamval,gamval_f,gamin,sigval,sigval_f,sigin,dk,dk_f,jqq,kappa,c_sw,Phi);
#pragma omp parallel for
		for(unsigned short j=0; j<3; j++)
			switch(j)
			{
				case(0):
					{
						FILE *fortout;
						char fortname[FILELEN] = "ext_fermi.";
						strcat(fortname,suffix);
						const char *fortop= (i==0) ? "w" : "a";
						if(!(fortout=fopen(fortname, fortop) )){
							fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
									OPENERROR, funcname, fortname, fortop);
#if(nproc>1)
							MPI_Abort(comm,OPENERROR);
#else
							exit(OPENERROR);
#endif
						}
						if(i==0)
							fprintf(fortout, "pbp\tendenf\tdenf\n");
						if(measure_check)
							fprintf(fortout, "%e\t%e\t%e\n", NAN, NAN, NAN);
						else
							fprintf(fortout, "%e\t%e\t%e\n", pbp, endenf, denf);
						fclose(fortout);
						break;
					}
				case(1):
					//The original code implicitly created these files with the name
					//fort.XX where XX is the file label
					//from FORTRAN. This was fort.12
					{
						FILE *fortout;
						char fortname[FILELEN] = "ext_bose."; 
						strcat(fortname,suffix);
						const char *fortop= (i==0) ? "w" : "a";
						if(!(fortout=fopen(fortname, fortop) )){
							fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
									OPENERROR, funcname, fortname, fortop);
						}
						if(i==0)
							fprintf(fortout, "avplaqs\tavplaqt\tpoly\n");
						fprintf(fortout, "%e\t%e\t%e\n", avplaqs, avplaqt, poly);
						fclose(fortout);
						break;
					}
				case(2):
					{
						FILE *fortout;
						char fortname[FILELEN] = "ext_diq.";
						strcat(fortname,suffix);
						const char *fortop= (i==0) ? "w" : "a";
						if(!(fortout=fopen(fortname, fortop) )){
							fprintf(stderr, "Error %i in %s: Failed to open file %s for %s.\nExiting\n\n",\
									OPENERROR, funcname, fortname, fortop);
#if(nproc>1)
							MPI_Abort(comm,OPENERROR);
#else
							exit(OPENERROR);
#endif
						}
						if(i==0)
							fprintf(fortout, "Re(qq)\n");
						if(measure_check)
							fprintf(fortout, "%e\n", NAN);
						else
							fprintf(fortout, "%e\n", creal(qq));
						fclose(fortout);
						break;
					}
				default: break;
			}
	}
#ifdef USE_GPU
	//Make a routine that does this for us
	cudaFree(dk[0]); cudaFree(dk[1]);
	cudaFree(Phi); cudaFree(ut[0]); cudaFree(ut[1]);
	cudaFree(u[0]); cudaFree(u[1]);
	cudaFree(id); cudaFree(iu); 
	cudaFree(dk_f[0]); cudaFree(dk_f[1]); cudaFree(ut_f[0]); cudaFree(ut_f[1]);
	cudaFree(gamin); cudaFree(gamval); cudaFree(gamval_f);
	if(c_sw){
		cudaFree(sigval); cudaFree(sigval_f); cudaFree(sigin);
	}
	cublasDestroy(cublas_handle);
#else
	free(dk[0]); free(dk[1]);
	free(Phi); free(ut[0]); free(ut[1]);
	free(u[0]); free(u[1]);
	free(id); free(iu);
	free(dk_f[0]); free(dk_f[1]); free(ut_f[0]); free(ut_f[1]);
	if(c_sw){
		free(sigval); free(sigval_f); free(sigin);
	}
#endif
#ifdef __RANLUX__
	gsl_rng_free(ranlux_instd);
#endif

	return 0;
}
