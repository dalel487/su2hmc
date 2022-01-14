#include <par_mpi.h>

//NOTE: In FORTRAN code everything was capitalised (despite being case insensitive)
//C is case sensitive, so the equivalent C command has the case format MPI_Xyz_abc
//Non-commands (like MPI_COMM_WORLD) don't always change

MPI_Comm comm = MPI_COMM_WORLD;

int Par_begin(int argc, char *argv[]){
	/* Initialises the MPI configuration
	 *
	 * Parameters:
	 * ---------
	 * int argc	Number of arguments given to the programme
	 * char *argv[]	Array of arguments
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise.
	 */

	char *funcname = "Par_begin";
	int size, commcart;
	if(MPI_Init(&argc, &argv)){
		fprintf(stderr, "Error %i in %s: Failed to initialise MPI\nExiting\n\n", NO_MPI_INIT, funcname);
		exit(NO_MPI_INIT);
	}

	if(MPI_Comm_rank(comm, &rank)){
		fprintf(stderr, "Error %i in %s: Failed to find rank.\nExiting...\n\n", NO_MPI_RANK, funcname);
		MPI_Finalise();
		exit(NO_MPI_RANK);
	}
	if(MPI_Comm_size(comm, &size)){
		fprintf(stderr, "Error %i in %s: Failed to find size\nExiting...\n\n", NO_MPI_SIZE, funcname);
		MPI_Finalise();
		exit(NO_MPI_SIZE);
	}
	//If size isn't the same as the max allowed number of processes, then there's a problem somewhere.
	if(size!=nproc){
		fprintf(stderr, "Error %i in %s: For process %i, size %i is not equal to nproc %i.\n"
				"Exiting...\n\n", SIZEPROC, funcname, rank, size, nproc);
		MPI_Finalise();
		exit(SIZEPROC);
	}
	//gsize is the size of the system, lsize is the size of each MPI Grid
	gsize[0]=nx; gsize[1]=ny; gsize[2]=nz; gsize[3]=nt;
	lsize[0]=ksizex; lsize[1]=ksizey; lsize[2]=ksizez; lsize[3]=ksizet;

	//Topology layout
	int cartsize[ndim] __attribute__((aligned(AVX)));
	cartsize[0]=npx; cartsize[1]=npy; cartsize[2]=npz; cartsize[3]=npt;

	//For the topology, says if each dimension is periodic or not
	//Probably for us everything will be but using the four vector
	//gives the choice at least
	int periods[ndim] __attribute__((aligned(AVX)));
#pragma unroll
	for(int i=0; i<ndim; i++)
		periods[i] = TRUE;
	//Not going to change the rank order
	int reorder = FALSE;
	//Declare the topology
	MPI_Cart_create(comm, ndim, cartsize, periods, reorder, &commcart);

	//Get nearest neighbours of processors
#pragma unroll
	for(int i= 0; i<ndim; i++)
		MPI_Cart_shift(commcart, i, 1, &pd[i], &pu[i]);
	//Get coordinates of processors in the grid
#ifdef __INTEL_MKL__
	pcoord = (int*)mkl_malloc(ndim*nproc*sizeof(int),AVX);
#else
	pcoord = (int*)aligned_alloc(AVX,ndim*nproc*sizeof(int));
#endif
	for(int iproc = 0; iproc<nproc; iproc++){
		MPI_Cart_coords(commcart, iproc, ndim, pcoord+iproc*ndim);
#pragma omp simd aligned(pcoord:AVX)
		for(int idim = 0; idim<ndim; idim++){
			pstart[idim][iproc] = pcoord[idim+ndim*iproc]*lsize[idim];
			pstop[idim][iproc]  = pstart[idim][iproc] + lsize[idim];
		}
	}
#ifdef _DEBUG
	if(!rank)
		printf("Running on %i processors.\n Grid layout is %ix%ix%ix%i\n",
				nproc, npx,npy,npz,npt);
	printf("Rank: %i pu: %i %i %i %i pd: %i %i %i %i\n", rank, pu[0], pu[1], pu[2], pu[3],
			pd[0], pd[1], pd[2], pd[3]);
#endif
	return 0;
}	
int Par_sread(const int iread, const double beta, const double fmu, const double akappa, const double ajq){
	/*
	 * Reads and assigns the gauges from file
	 *
	 * Parameters:
	 * ----------
	 *  None (file names are hardcoded in)
	 *
	 *  Returns:
	 *  -------
	 *  Zero on success, integer error code otherwise
	 */
	char *funcname = "Par_sread";
	MPI_Status status;
	//For sending the seeds later
	MPI_Datatype MPI_SEED_TYPE = (sizeof(seed)==sizeof(int)) ? MPI_INT:MPI_LONG;
	//We shall allow the almighty master thread to open the file
#ifdef __INTEL_MKL__
	Complex *u1buff = (Complex *)mkl_malloc(kvol*sizeof(Complex),AVX);
	Complex *u2buff = (Complex *)mkl_malloc(kvol*sizeof(Complex),AVX);
#else
	Complex *u1buff = (Complex *)aligned_alloc(AVX,kvol*sizeof(Complex));
	Complex *u2buff = (Complex *)aligned_alloc(AVX,kvol*sizeof(Complex));
#endif
	if(!rank){
		//Containers for input. Only needed by the master rank
#ifdef __INTEL_MKL__
		Complex *u11Read = (Complex *)mkl_malloc(ndim*gvol*sizeof(Complex),AVX);
		Complex *u12Read = (Complex *)mkl_malloc(ndim*gvol*sizeof(Complex),AVX);
#else
		Complex *u11Read = (Complex *)aligned_alloc(AVX,ndim*gvol*sizeof(Complex));
		Complex *u12Read = (Complex *)aligned_alloc(AVX,ndim*gvol*sizeof(Complex));
#endif
		static char gauge_file[FILELEN]="config.";
		int buffer; char buff2[7];
		//Add script for extrating correct mu, j etc.
		buffer = (int)(100*beta);
		sprintf(buff2,"b%03d",buffer);
		strcat(gauge_file,buff2);
		//κ
		buffer = (int)(10000*akappa);
		sprintf(buff2,"k%04d",buffer);
		strcat(gauge_file,buff2);
		//μ
		buffer = (int)(1000*fmu);
		sprintf(buff2,"mu%04d",buffer);
		strcat(gauge_file,buff2);
		//J
		buffer = (int)(100*ajq);
		sprintf(buff2,"j%02d",buffer);
		strcat(gauge_file,buff2);
		//nx
		sprintf(buff2,"s%02d",nx);
		strcat(gauge_file,buff2);
		//nt
		sprintf(buff2,"t%02d",nt);
		strcat(gauge_file,buff2);
		//nconfig
		char c[8];
		sprintf(c,".%06d", iread);
		strcat(gauge_file, c);

		char *fileop = "rb";
		printf("Opening gauge file on processor: %i\n",rank); 
		FILE *con;
		if(!(con = fopen(gauge_file, fileop))){
			fprintf(stderr, "Error %i in %s: Failed to open %s for %s.\
					\nExiting...\n\n", OPENERROR, funcname, gauge_file, fileop);
			MPI_Finalise();
			exit(OPENERROR);	
		}
		//TODO: SAFETY CHECKS FOR EACH READ OPERATION
		int old_nproc;
		//What was previously the the FORTRAN integer is now used to store the number of processors used to
		//generate the configuration
		fread(&old_nproc, sizeof(int), 1, con);
		if(old_nproc!=nproc)
			fprintf(stderr, "Warning %i in %s: Previous run was done on %i processors, current run uses %i.\n",\
					DIFNPROC,funcname,old_nproc,nproc);
		fread(u11Read, ndim*gvol*sizeof(Complex), 1, con);
		fread(u12Read, ndim*gvol*sizeof(Complex), 1, con);
		//The seed array will be used to gather and sort the seeds from each rank so they can be in a continuation run
		//If less processors are used then only nproc seeds are used (breaking the Markov Chain)
		//If more processors are used then we use the first seed to generate the rest as in Par_ranset
#ifdef __RANLUX__
		unsigned long *seed_array=(unsigned long*)calloc(nproc,sizeof(seed));
#elif defined __INTEL_MKL__ && !defined USE_RAN2
		int *seed_array=(int *)calloc(nproc,sizeof(seed));
#else
		long *seed_array=(long*)calloc(nproc,sizeof(seed));
#endif
		for(int i=0; i<fmin(old_nproc,nproc);i++)
			fread(seed_array+i, sizeof(seed), 1, con);
		fclose(con);
		//Any remaining processors get their initial value set as is done in Par_ranset
		for(int i=old_nproc; i<nproc; i++)
			seed_array[i] = seed_array[0]*(1.0f+8.0f*(float)i/(float)(size-1));
		if(!rank)
			seed=seed_array[0];
		for(int iproc = 1; iproc<nproc; iproc++)
			if(MPI_Send(&seed_array[iproc], 1, MPI_SEED_TYPE,iproc, 1, comm)){
				fprintf(stderr, "Error %i in %s: Failed to send seed to process %i.\nExiting...\n\n",
						CANTSEND, funcname, iproc);
				MPI_Finalise();
				exit(CANTSEND);
			}

		for(int iproc = 0; iproc < nproc; iproc++)
			for(int idim = 0; idim < ndim; idim++){
				int i = 0;
				//Index order is reversed from FORTRAN for performance
				//Going to split up assigning icoord[i] to reduce the
				//number of assignments.
				//We're weaving our way through the memory here, converting
				//between lattice and memory coordinates
				for(int ix=pstart[0][iproc]; ix<pstop[0][iproc]; ix++)
					for(int iy=pstart[1][iproc]; iy<pstop[1][iproc]; iy++)
						for(int iz=pstart[2][iproc]; iz<pstop[2][iproc]; iz++)
							for(int it=pstart[3][iproc]; it<pstop[3][iproc]; it++){
								//j is the relative memory index of icoord
								int j = Coord2gindex(ix,iy,iz,it);
								//ubuff[i]  = (ic == 0) ? u11read[j][idim] : u12read[j][idim];
								u1buff[i]=u11Read[idim*gvol+j];
								u2buff[i]=u12Read[idim*gvol+j];
								//C starts counting from zero, not 1 so increment afterwards or start at int i=-1
								i++;
							}
				if(i!=kvol){
					fprintf(stderr, "Error %i in %s: Number of elements %i is not equal to\
							kvol %i.\nExiting...\n\n", NUMELEM, funcname, i, kvol);
					MPI_Finalise();
					exit(NUMELEM);
				}
				if(!iproc){
#if (defined __INTEL_MKL__||defined USE_BLAS)
					cblas_zcopy(kvol,u1buff,1,u11+idim,ndim);
					cblas_zcopy(kvol,u2buff,1,u12+idim,ndim);
#else
#pragma omp simd aligned(u11,u12,u1buff,u2buff:AVX)
					for(i=0;i<kvol;i++){
						u11[i*ndim+idim]=u1buff[i];
						u12[i*ndim+idim]=u2buff[i];
					}
#endif
				}		
				else{
					//The master thread did all the hard work, the minions just need to receive their
					//data and go.
					if(MPI_Send(u1buff, kvol, MPI_C_DOUBLE_COMPLEX,iproc, 2*idim, comm)){
						fprintf(stderr, "Error %i in %s: Failed to send ubuff to process %i.\nExiting...\n\n",
								CANTSEND, funcname, iproc);
						MPI_Finalise();
						exit(CANTSEND);
					}
					if(MPI_Send(u2buff, kvol, MPI_C_DOUBLE_COMPLEX,iproc, 2*idim+1, comm)){
						fprintf(stderr, "Error %i in %s: Failed to send ubuff to process %i.\nExiting...\n\n",
								CANTSEND, funcname, iproc);
						MPI_Finalise();
						exit(CANTSEND);
					}
				}
			}
#ifdef __INTEL_MKL__
		mkl_free(u11Read); mkl_free(u12Read);
#else
		free(u11Read); free(u12Read);
#endif
		free(seed_array);
	}
	else{
		if(MPI_Recv(&seed, 1, MPI_SEED_TYPE, masterproc, 1, comm, &status)){
			fprintf(stderr, "Error %i in %s: Falied to receive seed on process %i.\nExiting...\n\n",
					CANTRECV, funcname, rank);
			MPI_Finalise();
			exit(CANTRECV);
		}
		for(int idim = 0; idim<ndim; idim++){
			//Receiving the data from the master threads.
			if(MPI_Recv(u1buff, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, 2*idim, comm, &status)){
				fprintf(stderr, "Error %i in %s: Falied to receive u11 on process %i.\nExiting...\n\n",
						CANTRECV, funcname, rank);
				MPI_Finalise();
				exit(CANTRECV);
			}
			if(MPI_Recv(u2buff, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, 2*idim+1, comm, &status)){
				fprintf(stderr, "Error %i in %s: Falied to receive u12 on process %i.\nExiting...\n\n",
						CANTRECV, funcname, rank);
				MPI_Finalise();
				exit(CANTRECV);
			}
#if (defined __INTEL_MKL__||defined USE_BLAS)
			cblas_zcopy(kvol,u1buff,1,u11+idim,ndim);
			cblas_zcopy(kvol,u2buff,1,u12+idim,ndim);
#else
#pragma omp parallel for simd aligned(u11,u12,u1buff,u2buff:AVX)
			for(int i=0;i<kvol;i++){
				u11[i*ndim+idim]=u1buff[i];
				u12[i*ndim+idim]=u2buff[i];
			}
#endif
		}
	}
#ifdef __INTEL_MKL__
	mkl_free(u1buff); mkl_free(u2buff);
#else
	free(u1buff); free(u2buff);
#endif
	memcpy(u11t, u11, ndim*kvol*sizeof(Complex));
	memcpy(u12t, u12, ndim*kvol*sizeof(Complex));
	return 0;
}
int Par_psread(char *filename, double *ps){
	/* Reads ps from a file
	 * Since this function is very similar to Par_sread, I'm not really going to comment it
	 * check there if you are confused about things. 
	 *
	 * Parameters;
	 * ==========
	 * char 	*filename: The name of the file we're reading from
	 * double	ps:	The destination for the file's contents
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Par_psread";
	MPI_Status status;
#ifdef __INTEL_MKL__
	double *psbuff = (double*)mkl_malloc(nc*kvol*sizeof(double),AVX);
	double *gps= (double*)mkl_malloc(nc*gvol*sizeof(double),AVX);
#else
	double *psbuff = (double*)aligned_alloc(AVX,nc*kvol*sizeof(double));
	double *gps= (double*)aligned_alloc(AVX,nc*gvol*sizeof(double));
#endif
	FILE *dest;
	if(!rank){
		if(!(dest = fopen(filename, "rb"))){
			fprintf(stderr, "Error %i in %s: Failed to open %s.\nExiting...\n\n", OPENERROR, funcname, filename);
			MPI_Finalise();
			exit(OPENERROR); 
		}
		fread(&gps, sizeof(gps), 1, dest);	
		fclose(dest);

		int i;
		for(int iproc=0;iproc<nproc;iproc++){
			i = 0;
			for(int ix=pstart[0][iproc]; ix<pstop[0][iproc]; ix++)
				for(int iy=pstart[1][iproc]; iy<pstop[1][iproc]; iy++)
					for(int iz=pstart[2][iproc]; iz<pstop[2][iproc]; iz++)
						for(int it=pstart[3][iproc]; it<pstop[3][iproc]; it++){
							i++;
							//j is the relative memory index of icoord
							int j = Coord2gindex(ix,iy,iz,it);
							//ubuff[i]  = (ic == 0) ? u11read[j][idim] : u12read[j][idim];
							psbuff[i*nc]=gps[j*nc];
							psbuff[i*nc+1]=gps[j*nc+1];
						}
			//Think its kvol-1 in C as C indexes from 0 not 1
			if(i!=kvol-1){
				fprintf(stderr, "Error %i in %s: Number of elements %i is not equal to\
						kvol %i.\nExiting...\n\n", NUMELEM, funcname, i, kvol);
				MPI_Finalise();
				exit(NUMELEM);
			}
			if(!iproc)
				//Replacing loops with memcpy for performance
				memcpy(ps, psbuff, kvol*2*sizeof(double));
			else
				if(MPI_Ssend(psbuff, kvol, MPI_DOUBLE,iproc, tag, comm)){
					fprintf(stderr, "Error %i in %s: Failed to send psbuff to process %i.\nExiting...\n\n",
							CANTSEND, funcname, iproc);
					MPI_Finalise();
					exit(CANTSEND);
				}
		}
	}
	else
		if(MPI_Recv(psbuff, kvol, MPI_DOUBLE, masterproc, tag, comm, &status)){
			fprintf(stderr, "Error %i in %s: Falied to receive psbuff from process %i.\nExiting...\n\n",
					CANTRECV, funcname, masterproc);
			MPI_Finalise();
			exit(CANTRECV);
		}
#ifdef __INTEL_MKL__
	mkl_free(psbuff); mkl_free(gps);
#else
	free(psbuff); free(gps);
#endif
	return 0;
}
int Par_swrite(const int itraj, const int icheck, const double beta, const double fmu, const double akappa, 
		const double ajq){
	/*
	 * Modified from an original version of swrite in FORTRAN
	 *
	 * Copies u11 and u12 into arrays without halos which
	 * then get written to output
	 *
	 * Parameters:
	 * ----------
	 * int	itraj:	Current trajectory
	 * int	icheck:	How often to write gauge fields. Used to generate name of configuration
	 * double	beta
	 * double	fmu:		Fermion chemical potential
	 * double	akappa:
	 * double	ajq:
	 *
	 * Globals:
	 * -------
	 * Complex	*u11, u12:	Gauge fields
	 * int	seed
	 *
	 * Calls:
	 * ------
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "par_swrite";
	MPI_Status status;
	//Used for seed array later on
	MPI_Datatype MPI_SEED_TYPE = (sizeof(seed)==sizeof(int)) ? MPI_INT:MPI_LONG;
#ifdef __INTEL_MKL__
	Complex *u1buff = (Complex *)mkl_malloc(kvol*sizeof(Complex),AVX);
	Complex *u2buff = (Complex *)mkl_malloc(kvol*sizeof(Complex),AVX);
#else
	Complex *u1buff = (Complex *)aligned_alloc(AVX,kvol*sizeof(Complex));
	Complex *u2buff = (Complex *)aligned_alloc(AVX,kvol*sizeof(Complex));
#endif
#if _DEBUG
	char dump_prefix[FILELEN]="u11.";
	char dump_buff[32];
	sprintf(dump_buff,"r%01d_c%06d",rank,itraj);
	strcat(dump_prefix,dump_buff);
	FILE *gauge_dump=fopen(dump_prefix,"wb");
	//Print the local trial field in the order it is stored in memory.
	//This is not the same order as it is stored in secondary storage
	fwrite(u11,ndim*kvol*sizeof(Complex),1,gauge_dump);
	fclose(gauge_dump);
#endif
	if(!rank){
		//Array to store the seeds. nth index is the nth processor
#ifdef __RANLUX__
		unsigned long *seed_array=(unsigned long*)calloc(nproc,sizeof(seed));
#elif defined __INTEL_MKL__ && !defined USE_RAN2
		int *seed_array=(int *)calloc(nproc,sizeof(seed));
#else
		long *seed_array=(long*)calloc(nproc,sizeof(seed));
#endif
		seed_array[0]=seed;
		for(int iproc = 1; iproc<nproc; iproc++)
			if(MPI_Recv(&seed_array[iproc], 1, MPI_SEED_TYPE,iproc, 1, comm, &status)){
				fprintf(stderr, "Error %i in %s: Failed to receive seed from process %i.\nExiting...\n\n",
						CANTRECV, funcname, iproc);
				MPI_Finalise();
				exit(CANTSEND);
			}
#ifdef __INTEL_MKL__
		Complex *u11Write = (Complex *)mkl_malloc(ndim*gvol*sizeof(Complex),AVX);
		Complex *u12Write = (Complex *)mkl_malloc(ndim*gvol*sizeof(Complex),AVX);
#else
		Complex *u11Write = (Complex *)aligned_alloc(AVX,ndim*gvol*sizeof(Complex));
		Complex *u12Write = (Complex *)aligned_alloc(AVX,ndim*gvol*sizeof(Complex));
#endif
		//Get correct parts of u11read etc from remote processors
		for(int iproc=0;iproc<nproc;iproc++)
			for(int idim=0;idim<ndim;idim++){
				if(iproc){
					if(MPI_Recv(u1buff, kvol, MPI_C_DOUBLE_COMPLEX, iproc, 2*idim, comm, &status)){
						fprintf(stderr, "Error %i in %s: Falied to receive u11 from process %i.\nExiting...\n\n",
								CANTRECV, funcname, iproc);
						MPI_Finalise();
						exit(CANTRECV);
					}
					if(MPI_Recv(u2buff, kvol, MPI_C_DOUBLE_COMPLEX, iproc, 2*idim+1, comm, &status)){
						fprintf(stderr, "Error %i in %s: Falied to receive u12 from process %i.\nExiting...\n\n",
								CANTRECV, funcname, iproc);
						MPI_Finalise();
						exit(CANTRECV);
					}
				}
				else{
					//No need to do MPI Send/Receive on the master rank
					//Array looping is slow so we use memcpy instead
#if (defined __INTEL_MKL__||defined USE_BLAS)
					cblas_zcopy(kvol,u11+idim,ndim,u1buff,1);
					cblas_zcopy(kvol,u12+idim,ndim,u2buff,1);
#else
#pragma omp parallel for simd aligned(u11,u12,u1buff,u2buff:AVX)
					for(int i=0;i<kvol;i++){
						u1buff[i]=u11[i*ndim+idim];
						u2buff[i]=u12[i*ndim+idim];
					}
#endif
#if _DEBUG
					char part_dump[FILELEN]="";
					strcat(part_dump,dump_prefix);
					sprintf(dump_buff,"_d%d",idim);
					strcat(part_dump,dump_buff);
					FILE *pdump=fopen(part_dump,"wb");
					fwrite(u1buff,ndim*kvol*sizeof(Complex),1,pdump);
					fclose(pdump);
#endif
				}
				int i=0;
							for(int it=pstart[3][iproc]; it<pstop[3][iproc]; it++)
						for(int iz=pstart[2][iproc]; iz<pstop[2][iproc]; iz++)
					for(int iy=pstart[1][iproc]; iy<pstop[1][iproc]; iy++)
				for(int ix=pstart[0][iproc]; ix<pstop[0][iproc]; ix++){
								//j is the relative memory index of icoord
								int j = Coord2gindex(ix, iy, iz, it);
								u11Write[idim*gvol+j] = u1buff[i];	
								u12Write[idim*gvol+j] = u2buff[i];	
								//C starts counting from zero, not 1 so increment afterwards or start at int i=-1
								i++;
							}
				if(i!=kvol){
					fprintf(stderr, "Error %i in %s: Number of elements %i is not equal to\
							kvol %i.\nExiting...\n\n", NUMELEM, funcname, i, kvol);
					MPI_Finalise();
					exit(NUMELEM);
				}
			}
#ifdef __INTEL_MKL__
		mkl_free(u1buff); mkl_free(u2buff);
#else
		free(u1buff); free(u2buff);
#endif

		char gauge_title[FILELEN]="config.";
		int buffer; char buff2[7];
		//Add script for extrating correct mu, j etc.
		buffer = (int)(100*beta);
		sprintf(buff2,"b%03d",buffer);
		strcat(gauge_title,buff2);
		//κ
		buffer = (int)(10000*akappa);
		sprintf(buff2,"k%04d",buffer);
		strcat(gauge_title,buff2);
		//μ
		buffer = (int)(1000*fmu);
		sprintf(buff2,"mu%04d",buffer);
		strcat(gauge_title,buff2);
		//J
		buffer = (int)(100*ajq);
		sprintf(buff2,"j%02d",buffer);
		strcat(gauge_title,buff2);
		//nx
		sprintf(buff2,"s%02d",nx);
		strcat(gauge_title,buff2);
		//nt
		sprintf(buff2,"t%02d",nt);
		strcat(gauge_title,buff2);

		char gauge_file[FILELEN];
		strcpy(gauge_file,gauge_title);
		char c[8];
		sprintf(c,".%06d", itraj);
		strcat(gauge_file, c);
		printf("Gauge file name is %s\n", gauge_file);
		printf("Writing the gauge file on processor %i.\n", rank);
		FILE *con;
		char *fileop = "wb";
		if(!(con=fopen(gauge_file, fileop))){
			fprintf(stderr, "Error %i in %s: Failed to open %s for %s.\
					\nExiting...\n\n", OPENERROR, funcname, gauge_file, fileop);
			MPI_Finalise();
			exit(OPENERROR);	
		}
		//TODO: SAFETY CHECKS FOR EACH WRITE OPERATION
		//Write the number of processors used in the previous run. This takes the place of the FORTRAN integer rather nicely
		fwrite(&size,sizeof(int),1,con);
		fwrite(u11Write, ndim*gvol*sizeof(Complex), 1, con);
		fwrite(u12Write, ndim*gvol*sizeof(Complex), 1, con);
		//TODO
		//Make a seed array, where the nth component is the seed on the nth rank for continuation runs.
		fwrite(seed_array, nproc*sizeof(seed), 1, con);
		fclose(con);
#ifdef __INTEL_MKL__
		mkl_free(u11Write); mkl_free(u12Write);
#else
		free(u11Write); free(u12Write);
#endif
		free(seed_array);
	}
	else{
		if(MPI_Send(&seed, 1, MPI_SEED_TYPE, masterproc, 1, comm)){
			fprintf(stderr, "Error %i in %s: Falied to send u11 from process %i.\nExiting...\n\n",
					CANTSEND, funcname, rank);
			MPI_Finalise();
			exit(CANTSEND);
		}
		for(int idim = 0; idim<ndim; idim++){
#if (defined __INTEL_MKL__||defined USE_BLAS)
			cblas_zcopy(kvol,u11+idim,ndim,u1buff,1);
			cblas_zcopy(kvol,u12+idim,ndim,u2buff,1);
#else
#pragma omp parallel for simd aligned(u11,u12,u1buff,u2buff:AVX)
			for(int i=0;i<kvol;i++){
				u1buff[i]=u11[i*ndim+idim];
				u2buff[i]=u12[i*ndim+idim];
			}
#endif
#ifdef _DEBUG
			char part_dump[FILELEN]="";
			strcat(part_dump,dump_prefix);
			sprintf(dump_buff,"_d%d",idim);
			strcat(part_dump,dump_buff);
			FILE *pdump=fopen(part_dump,"wb");
			fwrite(u1buff,ndim*kvol*sizeof(Complex),1,pdump);
			fclose(pdump);
#endif
			int i=0;
			if(MPI_Send(u1buff, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, 2*idim, comm)){
				fprintf(stderr, "Error %i in %s: Falied to send u11 from process %i.\nExiting...\n\n",
						CANTSEND, funcname, rank);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Send(u2buff, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, 2*idim+1, comm)){
				fprintf(stderr, "Error %i in %s: Falied to send u12 from process %i.\nExiting...\n\n",
						CANTSEND, funcname, rank);
				MPI_Finalise();
				exit(CANTSEND);
			}
		}
#ifdef __INTEL_MKL__
		mkl_free(u1buff); mkl_free(u2buff); 
#else
		free(u1buff); free(u2buff);
#endif
	}
	return 0;
}
//To be lazy, we've got modules to help us do reductions and broadcasts with a single argument
//rather than type them all every single time

inline int Par_isum(int *ival){
	/*
	 * Performs a reduction on a double ival to get a sum which is
	 * then distributed to all ranks.
	 *
	 * Parameters:
	 * -----------
	 * double *ival: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * Returns:
	 * --------
	 * Zero on success. Integer error code otherwise.
	 *
	 */
	char *funcname = "Par_isum";
	//Container to receive data.
	int *itmp;

	if(MPI_Allreduce(ival, itmp, 1, MPI_DOUBLE, MPI_SUM, comm)){
		fprintf(stderr,"Error %i in %s: Couldn't complete reduction for %i.\nExiting...\n\n", REDUCERR, funcname, *ival);
		MPI_Finalise();
		exit(REDUCERR);	
	}
	return 0;
}
inline int Par_dsum(double *dval){
	/*
	 * Performs a reduction on a double dval to get a sum which is
	 * then distributed to all ranks.
	 *
	 * Parameters:
	 * -----------
	 * double *dval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * Returns:
	 * --------
	 * Zero on success. Integer error code otherwise.
	 *
	 */
	char *funcname = "Par_dsum";
	//Container to receive data.
	double dtmp;

	if(MPI_Allreduce(dval, &dtmp, 1, MPI_DOUBLE, MPI_SUM, comm)){
		fprintf(stderr,"Error %i in %s: Couldn't complete reduction for %f.\nExiting...\n\n", REDUCERR, funcname, *dval);
		MPI_Finalise();
		exit(REDUCERR);	
	}
	*dval = dtmp;
	return 0;
}
inline int Par_fsum(float *fval){
	/*
	 * Performs a reduction on a double dval to get a sum which is
	 * then distributed to all ranks.
	 *
	 * Parameters:
	 * -----------
	 * double *dval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * Returns:
	 * --------
	 * Zero on success. Integer error code otherwise.
	 *
	 */
	char *funcname = "far_dsum";
	//Container to receive data.
	float ftmp;

	if(MPI_Allreduce(fval, &ftmp, 1, MPI_FLOAT, MPI_SUM, comm)){
		fprintf(stderr,"Error %i in %s: Couldn't complete reduction for %f.\nExiting...\n\n", REDUCERR, funcname, *fval);
		MPI_Finalise();
		exit(REDUCERR);	
	}
	*fval = ftmp;
	return 0;
}
inline int Par_csum(Complex_f *cval){
	/*
	 * Performs a reduction on a Complex zval to get a sum which is
	 * then distributed to all ranks.
	 *
	 * Parameters:
	 * -----------
	 * Complex_f *cval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * Returns:
	 * --------
	 * Zero on success. Integer error code otherwise.
	 *
	 */
	char *funcname = "Par_csum";
	//Container to receive data.
	Complex_f ctmp;

	if(MPI_Allreduce(cval, &ctmp, 1, MPI_C_FLOAT_COMPLEX, MPI_SUM, comm)){
#ifndef __NVCC__
		fprintf(stderr, "Error %i in %s: Couldn't complete reduction for %f+%f i.\nExiting...\n\n",
				REDUCERR, funcname, creal(*cval), cimag(*cval));
#endif
		MPI_Finalise();
		exit(REDUCERR);	
	}
	*cval = ctmp;
	return 0;
}
inline int Par_zsum(Complex *zval){
	/*
	 * Performs a reduction on a Complex zval to get a sum which is
	 * then distributed to all ranks.
	 *
	 * Parameters:
	 * -----------
	 * Complex *zval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * Returns:
	 * --------
	 * Zero on success. Integer error code otherwise.
	 *
	 */
	char *funcname = "Par_zsum";
	//Container to receive data.
	Complex ztmp;

	if(MPI_Allreduce(zval, &ztmp, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm)){
#ifndef __NVCC__
		fprintf(stderr, "Error %i in %s: Couldn't complete reduction for %f+%f i.\nExiting...\n\n",
				REDUCERR, funcname, creal(*zval), cimag(*zval));
#endif
		MPI_Finalise();
		exit(REDUCERR);	
	}
	*zval = ztmp;
	return 0;
}
inline int Par_icopy(int *ival){
	/*
	 * Broadcasts an integer to the other processes
	 *
	 * Parameters:
	 * ----------
	 * int ival
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Par_icopy";
	if(MPI_Bcast(ival,1,MPI_INT,masterproc,comm)){
		fprintf(stderr, "Error %i in %s: Failed to broadcast %i from %i.\nExiting...\n\n",
				BROADERR, funcname, *ival, rank);
		MPI_Finalise();
		exit(BROADERR);
	}
	return 0;
}
inline int Par_dcopy(double *dval){
	/*
	 * Broadcasts an double to the other processes
	 *
	 * Parameters:
	 * ----------
	 * double dval
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Par_dcopy";
	if(MPI_Bcast(dval,1,MPI_DOUBLE,masterproc,comm)){
		fprintf(stderr, "Error %i in %s: Failed to broadcast %f from %i.\nExiting...\n\n",
				BROADERR, funcname, *dval, rank);
		MPI_Finalise();
		exit(BROADERR);
	}
	return 0;
}
inline int Par_zcopy(Complex *zval){
	/*
	 * Broadcasts a Complex value to the other processes
	 *
	 * Parameters:
	 * ----------
	 * Complex *zval
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Par_zcopy";
	if(MPI_Bcast(zval,1,MPI_C_DOUBLE_COMPLEX,masterproc,comm)){
#ifndef __NVCC__
		fprintf(stderr, "Error %i in %s: Failed to broadcast %f+i%f from %i.\nExiting...\n\n",
				BROADERR, funcname, creal(*zval), cimag(*zval), rank);
#endif
		MPI_Finalise();
		exit(BROADERR);
	}
	return 0;
}

/*	Code for swapping halos.
 *	In the original FORTRAN there were seperate subroutines for up and down halos
 *	To make code maintainence easier I'm going to impliment this with switches
 *	and common functions
 *	We will define in su2hmc UP and DOWN. And add a parameter called layer to 
 *	functions. layer will be used to tell us if we wanted to call the up FORTRAN
 *	function or DOWN FORTRAN function
 */

inline int ZHalo_swap_all(Complex *z, int ncpt){
	/*
	 * Calls the functions to send data to both the up and down halos
	 *
	 * Parameters:
	 * -----------
	 * Complex z:	The data being sent
	 * int	ncpt:	Good Question
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "ZHalo_swap_all";

	//FORTRAN called zdnhaloswapall and zuphaloswapall here
	//Those functions looped over the directions and called zXXhaloswapdir
	//As the only place they are called in the FORTRAN code is right here,
	//I'm going to omit them entirely and just put the direction loop here
	//instead

	for(int mu=0; mu<ndim; mu++){
		ZHalo_swap_dir(z, ncpt, mu, DOWN);
		ZHalo_swap_dir(z, ncpt, mu, UP);			
	}
	return 0;
}
int ZHalo_swap_dir(Complex *z, int ncpt, int idir, int layer){
	/*
	 * Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * Parameters:
	 * -----------
	 *  Complex	*z:	The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 *  int		ncpt: The size of something else above. 	
	 *  int		idir:	The axis being moved along in C Indexing
	 *  int		layer:	Either DOWN (0) or UP (1)
	 *
	 *  Returns:
	 *  -------
	 *  Zero on success, Integer Error code otherwise
	 */
	char *funcname = "CHalo_swap_dir";
	MPI_Status status;
	if(layer!=DOWN && layer!=UP){
		fprintf(stderr, "Error %i in %s: Cannot swap in the direction given by %i.\nExiting...\n\n",
				LAYERROR, funcname, layer);
		MPI_Finalise();
		exit(LAYERROR);
	}
#ifdef __INTEL_MKL__
	Complex *sendbuf = (Complex *)mkl_malloc(halo*ncpt*sizeof(Complex), AVX);
#else
	Complex *sendbuf = (Complex *)aligned_alloc(AVX,halo*ncpt*sizeof(Complex));
#endif
	//How big is the data being sent and received
	int msg_size=ncpt*halosize[idir];
	//In each case we set up the data being sent then do the exchange
	switch(layer){
		case(DOWN):
			if(halosize[idir]+h1u[idir]>kvol+halo){
				fprintf(stderr, "Error %i in %s: Writing a message of size %i to flattened index %i will cause "\
						"a memory leak on rank %i.\nExiting...\n\n"
						,BOUNDERROR, funcname, msg_size, ncpt*h1u[idir], rank);
				MPI_Finalise();
				exit(BOUNDERROR);
			}
#pragma omp parallel for if(halosize[idir]>2048)
			for(int ihalo = 0; ihalo < halosize[idir]; ihalo++)
#pragma omp simd aligned(z, sendbuf:AVX)
				for(int icpt = 0; icpt <ncpt; icpt++)
					sendbuf[ihalo*ncpt+icpt]=z[ncpt*hd[ndim*ihalo+idir]+icpt];
			//For the zdnhaloswapdir we send off the down halo and receive into the up halo
			if(MPI_Isend(sendbuf, msg_size, MPI_C_DOUBLE_COMPLEX, pd[idir], tag, comm, &request)){
				fprintf(stderr,"Error %i in %s: Failed to send off the down halo from rank %i to rank %i.\nExiting...\n"
						,CANTSEND, funcname, rank, pd[idir]);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Recv(&z[ncpt*h1u[idir]], msg_size, MPI_C_DOUBLE_COMPLEX, pu[idir], tag, comm, &status)){
				fprintf(stderr,"Error %i in %s: Rank %i failed to receive into up halo from rank %i.\nExiting...\n",
						CANTRECV, funcname, rank, pu[idir]);
				MPI_Finalise();
				exit(CANTRECV);
			}
			break;
		case(UP):
			if(halosize[idir]+h1d[idir]>kvol+halo){
				fprintf(stderr, "Error %i in %s: Writing a message of size %i to flattened index %i will cause "\
						"a memory leak on rank %i.\nExiting...\n\n"
						,BOUNDERROR, funcname, msg_size, ncpt*h1d[idir], rank);
				MPI_Finalise();
				exit(BOUNDERROR);
			}
#pragma omp parallel for if(halosize[idir]>2048)
			for(int ihalo = 0; ihalo < halosize[idir]; ihalo++)
#pragma omp simd aligned(z, sendbuf:AVX)
				for(int icpt = 0; icpt <ncpt; icpt++)
					sendbuf[ihalo*ncpt+icpt]=z[ncpt*hu[ndim*ihalo+idir]+icpt];
			//For the zuphaloswapdir we send off the up halo and receive into the down halo
			if(MPI_Isend(sendbuf, msg_size, MPI_C_DOUBLE_COMPLEX, pu[idir], 0, comm, &request)){
				fprintf(stderr,"Error %i in %s: Failed to send off the up halo from rank %i to rank %i.\nExiting...\n",
						CANTSEND, funcname, rank, pu[idir]);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Recv(&z[ncpt*h1d[idir]], msg_size, MPI_C_DOUBLE_COMPLEX, pd[idir], tag, comm, &status)){
				fprintf(stderr,"Error %i in %s: Rank %i failed to receive into doww halo from rank %i.\nExiting...\n",
						CANTRECV, funcname, rank, pd[idir]);
				MPI_Finalise();
				exit(CANTRECV);
			}
			break;
	}
	MPI_Wait(&request, &status);
#ifdef __INTEL_MKL__
	mkl_free(sendbuf);
#else
	free(sendbuf);
#endif
	return 0;
}
inline int CHalo_swap_all(Complex_f *c, int ncpt){
	/*
	 * Calls the functions to send data to both the up and down halos
	 *
	 * Parameters:
	 * -----------
	 * Complex z:	The data being sent
	 * int	ncpt:	Good Question
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "ZHalo_swap_all";

	//FORTRAN called zdnhaloswapall and zuphaloswapall here
	//Those functions looped over the directions and called zXXhaloswapdir
	//As the only place they are called in the FORTRAN code is right here,
	//I'm going to omit them entirely and just put the direction loop here
	//instead

	for(int mu=0; mu<ndim; mu++){
		CHalo_swap_dir(c, ncpt, mu, DOWN);
		CHalo_swap_dir(c, ncpt, mu, UP);			
	}
	return 0;
}
int CHalo_swap_dir(Complex_f *c, int ncpt, int idir, int layer){
	/*
	 * Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * Parameters:
	 * -----------
	 *  Complex	*z:	The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 *  int		ncpt: The size of something else above. 	
	 *  int		idir:	The axis being moved along in C Indexing
	 *  int		layer:	Either DOWN (0) or UP (1)
	 *
	 *  Returns:
	 *  -------
	 *  Zero on success, Integer Error code otherwise
	 */
	char *funcname = "CHalo_swap_dir";
	MPI_Status status;
	if(layer!=DOWN && layer!=UP){
		fprintf(stderr, "Error %i in %s: Cannot swap in the direction given by %i.\nExiting...\n\n",
				LAYERROR, funcname, layer);
		MPI_Finalise();
		exit(LAYERROR);
	}
#ifdef __INTEL_MKL__
	Complex_f *sendbuf = (Complex_f *)mkl_malloc(halo*ncpt*sizeof(Complex_f), AVX);
#else
	Complex_f *sendbuf = (Complex_f *)aligned_alloc(AVX,halo*ncpt*sizeof(Complex));
#endif
	//How big is the data being sent and received
	int msg_size=ncpt*halosize[idir];
	//In each case we set up the data being sent then do the exchange
	switch(layer){
		case(DOWN):
			if(halosize[idir]+h1u[idir]>kvol+halo){
				fprintf(stderr, "Error %i in %s: Writing a message of size %i to flattened index %i will cause "\
						"a memory leak on rank %i.\nExiting...\n\n"
						,BOUNDERROR, funcname, msg_size, ncpt*h1u[idir], rank);
				MPI_Finalise();
				exit(BOUNDERROR);
			}
#pragma omp parallel for if(halosize[idir]>2048)
			for(int ihalo = 0; ihalo < halosize[idir]; ihalo++)
#pragma omp simd aligned(c, sendbuf:AVX)
				for(int icpt = 0; icpt <ncpt; icpt++)
					sendbuf[ihalo*ncpt+icpt]=c[ncpt*hd[ndim*ihalo+idir]+icpt];
			//For the zdnhaloswapdir we send off the down halo and receive into the up halo
			if(MPI_Isend(sendbuf, msg_size, MPI_C_FLOAT_COMPLEX, pd[idir], tag, comm, &request)){
				fprintf(stderr,"Error %i in %s: Failed to send off the down halo from rank %i to rank %i.\nExiting...\n"
						,CANTSEND, funcname, rank, pd[idir]);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Recv(&c[ncpt*h1u[idir]], msg_size, MPI_C_FLOAT_COMPLEX, pu[idir], tag, comm, &status)){
				fprintf(stderr,"Error %i in %s: Rank %i failed to receive into up halo from rank %i.\nExiting...\n",
						CANTRECV, funcname, rank, pu[idir]);
				MPI_Finalise();
				exit(CANTRECV);
			}
			break;
		case(UP):
			if(halosize[idir]+h1d[idir]>kvol+halo){
				fprintf(stderr, "Error %i in %s: Writing a message of size %i to flattened index %i will cause "\
						"a memory leak on rank %i.\nExiting...\n\n"
						,BOUNDERROR, funcname, msg_size, ncpt*h1d[idir], rank);
				MPI_Finalise();
				exit(BOUNDERROR);
			}
#pragma omp parallel for if(halosize[idir]>2048)
			for(int ihalo = 0; ihalo < halosize[idir]; ihalo++)
#pragma omp simd aligned(c, sendbuf:AVX)
				for(int icpt = 0; icpt <ncpt; icpt++)
					sendbuf[ihalo*ncpt+icpt]=c[ncpt*hu[ndim*ihalo+idir]+icpt];
			//For the zuphaloswapdir we send off the up halo and receive into the down halo
			if(MPI_Isend(sendbuf, msg_size, MPI_C_FLOAT_COMPLEX, pu[idir], 0, comm, &request)){
				fprintf(stderr,"Error %i in %s: Failed to send off the up halo from rank %i to rank %i.\nExiting...\n",
						CANTSEND, funcname, rank, pu[idir]);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Recv(&c[ncpt*h1d[idir]], msg_size, MPI_C_FLOAT_COMPLEX, pd[idir], tag, comm, &status)){
				fprintf(stderr,"Error %i in %s: Rank %i failed to receive into doww halo from rank %i.\nExiting...\n",
						CANTRECV, funcname, rank, pd[idir]);
				MPI_Finalise();
				exit(CANTRECV);
			}
			break;
	}
#ifdef __INTEL_MKL__
	mkl_free(sendbuf);
#else
	free(sendbuf);
#endif
	MPI_Wait(&request, &status);
	return 0;
}
int DHalo_swap_dir(double *d, int ncpt, int idir, int layer){
	/*
	 * Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * Parameters:
	 * -----------
	 *  double	*d:	The data being moved about
	 *  int		ncpt:	No idea
	 *  int		idir:	The axis being moved along
	 *  int		layer:	Either DOWN (0) or UP (1)
	 *
	 *  Returns:
	 *  -------
	 *  Zero on success, Integer Error code otherwise
	 */
	char *funcname = "ZHalo_swap_dir";
	MPI_Status status;
#ifdef __INTEL_MKL__
	double *sendbuf =(double *) mkl_malloc(halo*ncpt*sizeof(double), AVX);
#else
	double *sendbuf = (double *)aligned_alloc(AVX,halo*ncpt*sizeof(double));
#endif
	if(layer!=DOWN && layer!=UP){
		fprintf(stderr, "Error %i in %s: Cannot swap in the direction given by %i.\nExiting...\n\n",
				LAYERROR, funcname, layer);
		MPI_Finalise();
		exit(LAYERROR);
	}
	//How big is the data being sent and received
	int msg_size=ncpt*halosize[idir];
	//Impliment the switch. The code is taken from the end of the dedicated functions in the FORTRAN code.
	switch(layer){
		case(DOWN):
			if(halosize[idir]+h1u[idir]>kvol+halo){
				fprintf(stderr, "Error %i in %s: Writing a message of size %i to flattened index %i will cause "\
						"a memory leak on rank %i.\nExiting...\n\n"
						,BOUNDERROR, funcname, msg_size, ncpt*h1u[idir], rank);
				MPI_Finalise();
				exit(BOUNDERROR);
			}
#pragma omp parallel for if(halosize[idir]>2048)
			for(int ihalo = 0; ihalo < halosize[idir]; ihalo++)
#pragma omp simd aligned(d,sendbuf:AVX)
				for(int icpt = 0; icpt <ncpt; icpt++)
					sendbuf[ihalo*ncpt+icpt]=d[ncpt*hd[ndim*ihalo+idir]+icpt];
			//For the cdnhaloswapdir we send off the down halo and receive into the up halo
			if(MPI_Isend(sendbuf, msg_size, MPI_DOUBLE, pd[idir], tag, comm, &request)){
				fprintf(stderr, "Error %i in %s: Failed to send off the down halo from rank %i to rank %i.\nExiting...\n\n",
						CANTSEND, funcname, rank, pd[idir]);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Recv(&d[ncpt*h1u[idir]], msg_size, MPI_DOUBLE, pu[idir], tag, comm, &status)){
				fprintf(stderr, "Error %i in %s: Rank %i failed to receive into up halo from rank %i.\nExiting...\n\n",
						CANTRECV, funcname, rank, pu[idir]);
				MPI_Finalise();
				exit(CANTRECV);
			}
		case(UP):
			if(halosize[idir]+h1d[idir]>kvol+halo){
				fprintf(stderr, "Error %i in %s: Writing a message of size %i to flattened index %i will cause "\
						"a memory leak on rank %i.\nExiting...\n\n"
						,BOUNDERROR, funcname, msg_size, ncpt*h1d[idir], rank);
				MPI_Finalise();
				exit(BOUNDERROR);
			}
#pragma omp parallel for if(halosize[idir]>2048)
			for(int ihalo = 0; ihalo < halosize[idir]; ihalo++)
#pragma omp simd aligned(d,sendbuf:AVX)
				for(int icpt = 0; icpt <ncpt; icpt++)
					sendbuf[ihalo*ncpt+icpt]=d[ncpt*hu[ndim*ihalo+idir]+icpt];
			//For the cuphaloswapdir we send off the up halo and receive into the down halo
			if(MPI_Isend(sendbuf, msg_size, MPI_DOUBLE, pu[idir], 0, comm, &request)){
				fprintf(stderr,"Error %i in %s: Failed to send off the up halo from rank %i to rank %i.\nExiting...\n\n",
						CANTSEND, funcname, rank, pu[idir]);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Recv(&d[ncpt*h1d[idir]], msg_size, MPI_DOUBLE, pd[idir], tag, comm, &status)){
				fprintf(stderr, "Error %i in %s: Rank %i failed to receive into doww halo from rank %i.\nExiting...\n\n",
						CANTRECV, funcname, rank, pd[idir]);
				MPI_Finalise();
				exit(CANTRECV);
			}
	}	
#ifdef __INTEL_MKL__
	mkl_free(sendbuf);
#else
	free(sendbuf);
#endif
	MPI_Wait(&request, &status);
	return 0;
}
int Trial_Exchange(){
	/*
	 *	Exchanges the trial fields. I noticed that this halo exchange was happening
	 *	even though the trial fields hadn't been updated. To get around this
	 *	I'm making a function that does the halo exchange and only calling it after
	 *	the trial fields get updated.
	 */
	char *funchame = "Trial_Exchange";
#ifdef __INTEL_MKL__
	Complex *z = (Complex *)mkl_malloc((kvol+halo)*sizeof(Complex),AVX);
#else
	Complex *z = (Complex *)aligned_alloc(AVX,(kvol+halo)*sizeof(Complex));
#endif
	for(int mu=0;mu<ndim;mu++){
		//Copy the column from u11t
#if (defined __INTEL_MKL__ || USE_BLAS)
		cblas_zcopy(kvol, &u11t[mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i*ndim+mu];
#endif
		//Halo exchange on that column
		ZHalo_swap_all(z, 1);
		//And the swap back
#if (defined __INTEL_MKL__ || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, &u11t[mu], ndim);
		//Repat for u12t
		cblas_zcopy(kvol, &u12t[mu], ndim, z, 1);
#else
		for(int i=0; i<kvol+halo;i++){
			u11t[i*ndim+mu]=z[i];
			z[i]=u12t[i*ndim+mu];
		}
#endif
		ZHalo_swap_all(z, 1);
#if (defined __INTEL_MKL__ || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, &u12t[mu], ndim);
#else
		for(int i=0; i<kvol+halo;i++)
			u12t[i*ndim+mu]=z[i];
#endif
	}
#ifdef __INTEL_MKL__
	mkl_free(z);
#else
	free(z);
#endif
#pragma omp parallel for simd
	for(int i=0;i<ndim*(kvol+halo);i++){
		u11t_f[i]=(Complex_f)u11t[i];
		u12t_f[i]=(Complex_f)u12t[i];
	}
	//Only upload the halo of the trial field, but the full single precision field
#pragma acc update device(u11t[ndim*kvol:ndim*(kvol+halo)],u12t[ndim*kvol:ndim*(kvol+halo)],\
		u11t_f[0:ndim*(kvol+halo)],u12t_f[0:ndim*(kvol+halo)])
	return 0;
}
int Par_tmul(Complex *z11, Complex *z12){
	/*
	 * Parameters:
	 * ===========
	 * Complex *z11
	 * Complex *z12
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise.
	 */
	MPI_Status status;
	char *funcname = "Par_tmul";
	Complex *a11, *a12, *t11, *t12;
	int i, itime;
	//If we're using mkl, the mkl_malloc helps ensure arrays align with 64-bit
	//byte boundaries to improve performance and enable AVX-512 instructions.
	//Otherwise, malloc is pretty useful
#ifdef __INTEL_MKL__
	a11=(Complex *)mkl_malloc(kvol3*sizeof(Complex), AVX);
	a12=(Complex *)mkl_malloc(kvol3*sizeof(Complex), AVX);
	t11=(Complex *)mkl_malloc(kvol3*sizeof(Complex), AVX);
	t12=(Complex *)mkl_malloc(kvol3*sizeof(Complex), AVX);
#else
	a11=(Complex *)aligned_alloc(AVX,kvol3*sizeof(Complex));
	a12=(Complex *)aligned_alloc(AVX,kvol3*sizeof(Complex));
	t11=(Complex *)aligned_alloc(AVX,kvol3*sizeof(Complex));
	t12=(Complex *)aligned_alloc(AVX,kvol3*sizeof(Complex));
#endif
	//Intitialise for the first loop
	memcpy(a11, z11, kvol3*sizeof(Complex));
	memcpy(a12, z12, kvol3*sizeof(Complex));

	//Since the index of the outer loop isn't used as an array index anywher
	//I'm going format it exactly like the original FORTRAN
#ifdef _DEBUG
	if(!rank) printf("Sending between halos in the time direction. For rank %i pu[3]=%i and pd[3] = %i\n",
			rank, pu[3], pd[3]);
#endif
	for(itime=1;itime<npt; itime++){
		memcpy(t11, a11, kvol3*sizeof(Complex));	
		memcpy(t12, a12, kvol3*sizeof(Complex));	
#ifdef _DEBUG
		if(!rank) printf("t11 and t12 assigned. Getting ready to send to other processes.\n");
#endif
		//Send results to other processes down the line
		//What I don't quite get (except possibly avoiding race conditions) is
		//why we send t11 and not a11. Surely by eliminating the assignment of a11 to t11 
		//and using a blocking send we would have one fewer loop to worry about and improve performance?
		if(MPI_Isend(t11, kvol3, MPI_C_DOUBLE_COMPLEX, pd[3], tag, comm, &request)){
			fprintf(stderr, "Error %i in %s: Failed to send t11 to process %i.\nExiting...\n\n",
					CANTSEND, funcname, pd[3]);
			MPI_Finalise();
			exit(CANTSEND);
		}
#ifdef _DEBUG
		printf("Sent t11 from rank %i to the down halo on rank %i\n", rank, pd[3]);
#endif
		if(MPI_Recv(a11, kvol3, MPI_C_DOUBLE_COMPLEX, pu[3], tag, comm, &status)){
			fprintf(stderr, "Error %i in %s: Failed to receive a11 from process %i.\nExiting...\n\n",
					CANTSEND, funcname, pu[3]);
			MPI_Finalise();
			exit(CANTSEND);
		}
#ifdef _DEBUG
		printf("Received t11 from rank %i in the up halo on rank %i\n",  pu[3], rank);
#endif
		MPI_Wait(&request, &status);
		if(MPI_Isend(t12, kvol3, MPI_C_DOUBLE_COMPLEX, pd[3], tag, comm, &request)){
			fprintf(stderr, "Error %i in %s: Failed to send t12 to process %i.\nExiting...\n\n",
					CANTSEND, funcname, pd[3]);
			MPI_Finalise();
			exit(CANTSEND);
		}
		if(MPI_Recv(a12, kvol3, MPI_C_DOUBLE_COMPLEX, pu[3], tag, comm, &status)){
			fprintf(stderr, "Error %i in %s: Failed to receive a12 from process %i.\nExiting...\n\n",
					CANTSEND, funcname, pu[3]);
			MPI_Finalise();
			exit(CANTSEND);
		}
#ifdef _DEBUG
		printf("Finished sending and receiving  on  rank %i\n",  rank);
#endif
		MPI_Wait(&request, &status);

		//Post-multiply current loop by incoming one.
		//This is begging to be done in CUDA or BLAS
#ifdef _OPENACC
#pragma acc parallel loop copy(t11[0:kvol3],t12[0:kvol3]) copyin(a11[0:kvol],a12[0:kvol])
#else
#pragma omp parallel for simd
#endif
		for(i=0;i<kvol3;i++){
			t11[i]=z11[i]*a11[i]-z12[i]*conj(a12[i]);
			t12[i]=z11[i]*a12[i]+z12[i]*conj(a11[i]);
		}
		memcpy(z11, t11, kvol3*sizeof(Complex));
		memcpy(z12, t12, kvol3*sizeof(Complex));
	}
#ifdef __INTEL_MKL__
	mkl_free(a11); mkl_free(a12);
	mkl_free(t11); mkl_free(t12);
#else
	free(a11); free(a12); free(t11); free(t12);
#endif
	return 0;
}
