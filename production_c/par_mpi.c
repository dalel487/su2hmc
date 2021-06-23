#include <par_mpi.h>

//NOTE: In FORTRAN code everything was capitalised (despite being case insensitive)
//C is case sensitive, so the equivalent C command has the case format MPI_Xyz_abc
//Non-commands (like MPI_COMM_WORLD) don't always change
int rank, size;

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
#ifdef USE_MKL
	pcoord = mkl_malloc(ndim*nproc*sizeof(int),AVX);
#else
	pcoord = malloc(ndim*nproc*sizeof(int));
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
int Par_sread(){
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
	//Containers for input
#ifdef USE_MKL
	complex *u11Read = mkl_malloc(ndim*gvol*sizeof(complex),AVX);
	complex *u12Read = mkl_malloc(ndim*gvol*sizeof(complex),AVX);
	complex *u1buff = mkl_malloc(kvol*sizeof(complex),AVX);
	complex *u2buff = mkl_malloc(kvol*sizeof(complex),AVX);
#else
	complex *u11Read = malloc(ndim*gvol*sizeof(complex));
	complex *u12Read = malloc(ndim*gvol*sizeof(complex));
	complex *u1buff = malloc(kvol*sizeof(complex));
	complex *u2buff = malloc(kvol*sizeof(complex));
#endif
	//	complex ubuff[kvol];
	int icoord[ndim];
	double seed;
	//We shall allow the almighty master thread to open the file
	if(!rank){
		printf("Opening gauge file on processor: %i",rank); 
		FILE *con = fopen("con", "rb");
		fread(&u11Read, sizeof(u11Read), 1, con);
		fread(&u12Read, sizeof(u12Read), 1, con);
		fread(&seed, sizeof(seed), 1, con);
		fclose(con);

		//Run over processors, dimensions and colours
		//Could be sped up with omp but parallel MPI_Sends is risky. 
		//#pragma omp parallel for collapse(2)
		for(int iproc = 0; iproc < nproc; iproc++)
			for(int idim = 0; idim < ndim; idim++){
				int i = 0;
				//Index order is reversed from FORTRAN for performance
				//Going to split up assigning icoord[i] to reduce the
				//number of assignments.
				//We're weaving our way through the memory here, converting
				//between lattice and memory coordinates
				for(int ix=pstart[0][iproc]; ix<pstop[0][iproc]; ix++){
					icoord[0]=ix;
					for(int iy=pstart[1][iproc]; iy<pstop[1][iproc]; iy++){
						icoord[1]=iy;
						for(int iz=pstart[2][iproc]; iz<pstop[2][iproc]; iz++){
							icoord[2]=iz;
							for(int it=pstart[3][iproc]; it<pstop[3][iproc]; it++){
								icoord[3]=it;
								i++;
								//j is the relative memory index of icoord
								int j = Coord2gindex(icoord);
								//ubuff[i]  = (ic == 0) ? u11read[j][idim] : u12read[j][idim];
								u1buff[i]=u11Read[idim*gvol+j];
								u2buff[i]=u12Read[idim*gvol+j];
							}
						}
					}
				}
				if(i+1!=kvol){
					fprintf(stderr, "Error %i in %s: Number of elements %i is not equal to\
							kvol %i.\nExiting...\n\n", NUMELEM, funcname, i, kvol);
					MPI_Finalise();
					exit(NUMELEM);
				}
				//Likewise, C indexes from zero so should just use iproc
				if(!iproc){
					u11[i*ndim+idim]=u1buff[i];
					u11t[i*ndim+idim]=u1buff[i];
					u12[i*ndim+idim]=u2buff[i];
					u12t[i*ndim+idim]=u2buff[i];
				}		
				else{
					//The master thread did all the hard work, the minions just need to receive their
					//data and go.
					if(MPI_Ssend(u1buff, kvol, MPI_C_DOUBLE_COMPLEX,iproc, tag, comm)){
						fprintf(stderr, "Error %i in %s: Failed to send ubuff to process %i.\nExiting...\n\n",
								CANTSEND, funcname, iproc);
						MPI_Finalise();
						exit(CANTSEND);
					}
					if(MPI_Ssend(u2buff, kvol, MPI_C_DOUBLE_COMPLEX,iproc, tag, comm)){
						fprintf(stderr, "Error %i in %s: Failed to send ubuff to process %i.\nExiting...\n\n",
								CANTSEND, funcname, iproc);
						MPI_Finalise();
						exit(CANTSEND);
					}
				}
			}
	}
	else{
		for(int idim = 0; idim<ndim; idim++){
			//Receiving the data from the master threads.
			if(MPI_Recv(u11+(kvol+halo)*idim, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, tag, comm, &status)){
				fprintf(stderr, "Error %i in %s: Falied to receive u11 from process %i.\nExiting...\n\n",
						CANTRECV, funcname, masterproc);
				MPI_Finalise();
				exit(CANTRECV);
			}
			if(MPI_Recv(u12+(kvol+halo)*idim, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, tag, comm, &status)){
				fprintf(stderr, "Error %i in %s: Falied to receive u12 from process %i.\nExiting...\n\n",
						CANTRECV, funcname, masterproc);
				MPI_Finalise();
				exit(CANTRECV);
			}
		}
	}
#ifdef USE_MKL
	mkl_free(u11Read); mkl_free(u12Read); mkl_free(u1buff); mkl_free(u2buff);
#else
	free(u11Read); free(u12Read); free(u1buff); free(u2buff);
#endif
	memcpy(u11t, u11, ndim*gvol*sizeof(complex));
	memcpy(u12t, u12, ndim*gvol*sizeof(complex));
	Par_dcopy(&seed);
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
#ifdef USE_MKL
	double *psbuff = mkl_malloc(nc*kvol*sizeof(double),AVX);
	double *gps= mkl_malloc(nc*gvol*sizeof(double),AVX);
#else
	double *psbuff = malloc(nc*kvol*sizeof(double));
	double *gps= malloc(nc*gvol*sizeof(double));
#endif
	int icoord[ndim];
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
			for(int ix=pstart[0][iproc]; ix<pstop[0][iproc]; ix++){
				icoord[0]=ix;
				for(int iy=pstart[1][iproc]; iy<pstop[1][iproc]; iy++){
					icoord[1]=iy;
					for(int iz=pstart[2][iproc]; iz<pstop[2][iproc]; iz++){
						icoord[2]=iz;
#pragma ivdep
						for(int it=pstart[3][iproc]; it<pstop[3][iproc]; it++){
							icoord[3]=it;
							i++;
							//j is the relative memory index of icoord
							int j = Coord2gindex(icoord);
							//ubuff[i]  = (ic == 0) ? u11read[j][idim] : u12read[j][idim];
							psbuff[i*nc]=gps[j*nc];
							psbuff[i*nc+1]=gps[j*nc+1];
						}}}}
			//Think its i+1 in C as C indexes from 0 not 1
			if(i+1!=kvol){
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
#ifdef USE_MKL
	mkl_free(psbuff); mkl_free(gps);
#else
	free(psbuff); free(gps);
#endif
	return 0;
}
int Par_swrite(const int itraj, const int icheck, const double beta, const double fmu, const double akappa, const double ajq){
	/*
	 * Modified from an original version of swrite in FORTRAN
	 *
	 * Copies u11 and u12 into arrays without halos which
	 * then get written to output
	 *
	 * Parameters:
	 * ----------
	 * int	isweep
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "par_swrite";
	int icoord[4], iproc, seed;
	if(!rank){
#ifdef USE_MKL
		complex *u11Write = mkl_malloc(ndim*gvol*sizeof(complex),AVX);
		complex *u12Write = mkl_malloc(ndim*gvol*sizeof(complex),AVX);
		complex *u1buff = mkl_malloc(kvol*sizeof(complex),AVX);
		complex *u2buff = mkl_malloc(kvol*sizeof(complex),AVX);
#else
		complex *u11Write = malloc(ndim*gvol*sizeof(complex));
		complex *u12Write = malloc(ndim*gvol*sizeof(complex));
		complex *u1buff = malloc(kvol*sizeof(complex));
		complex *u2buff = malloc(kvol*sizeof(complex));
#endif
		//Get correct parts of u11read etc from remote processors
		for(iproc=0;iproc<nproc;iproc++)
			for(int idim=0;idim<ndim;idim++){
				if(iproc){
					if(MPI_Recv(u1buff, kvol, MPI_C_DOUBLE_COMPLEX, iproc, tag, comm, &status)){
						fprintf(stderr, "Error %i in %s: Falied to receive u11 from process %i.\nExiting...\n\n",
								CANTRECV, funcname, iproc);
						MPI_Finalise();
						exit(CANTRECV);
					}
					if(MPI_Recv(u2buff, kvol, MPI_C_DOUBLE_COMPLEX, iproc, tag, comm, &status)){
						fprintf(stderr, "Error %i in %s: Falied to receive u12 from process %i.\nExiting...\n\n",
								CANTRECV, funcname, iproc);
						MPI_Finalise();
						exit(CANTRECV);
					}
				}
				else{
					//No need to do MPI Send/Receive on the master rank
					//Array looping is slow so we use memcpy instead
					memcpy(u1buff, u11+idim*(kvol+halo), kvol*sizeof(complex));
					memcpy(u2buff, u12+idim*(kvol+halo), kvol*sizeof(complex));
				}
				int i=0;
				//could move the ic check to here, but it will make the code look rather unsightly
				for(int ix=pstart[0][iproc]; ix<pstop[0][iproc]; ix++){
					icoord[0]=ix;
					for(int iy=pstart[1][iproc]; iy<pstop[1][iproc]; iy++){
						icoord[1]=iy;
						for(int iz=pstart[2][iproc]; iz<pstop[2][iproc]; iz++){
							icoord[2]=iz;
							for(int it=pstart[3][iproc]; it<pstop[3][iproc]; it++){
								icoord[3]=it;
								i++;
								//j is the relative memory index of icoord
								int j = Coord2gindex(icoord);
								//Since the for loop puts limits on ic we can skip the safety
								//check in the FORTRAN code 
								//	ubuff[i]  = (ic == 0) ? u11read[j][idim] : u12read[j][idim];
								//	Instead of looping through ic
								u11Write[idim*gvol+j] = u1buff[i];	
								u12Write[idim*gvol+j] = u2buff[i];	
							}}}}
				if(i!=kvol){
					fprintf(stderr, "Error %i in %s: Number of elements %i is not equal to\
							kvol %i.\nExiting...\n\n", NUMELEM, funcname, i, kvol);
					MPI_Finalise();
					exit(NUMELEM);
				}
			}

		FILE *con;
		static char gauge_title[FILELEN]="config.";
		if(itraj==icheck){
			int buffer; char buff2[7];
			//Add script for extrating correct mu, j etc.
			buffer = (int)(100*beta);
			if(buffer<10)
				sprintf(buff2,"b00%i",buffer);
			else if(buffer<100)
				sprintf(buff2,"b0%i",buffer);
			else
				sprintf(buff2,"b%i",buffer);
			strcat(gauge_title,buff2);
			//κ
			buffer = (int)(10000*akappa);
			if(buffer<10)
				sprintf(buff2,"k000%i",buffer);
			else if(buffer<100)
				sprintf(buff2,"k00%i",buffer);
			else if(buffer<1000)
				sprintf(buff2,"k0%i",buffer);
			else
				sprintf(buff2,"k%i",buffer);
			strcat(gauge_title,buff2);
			//μ
			buffer = (int)(1000*fmu);
			if(buffer<10)
				sprintf(buff2,"mu000%i",buffer);
			else if(buffer<100)
				sprintf(buff2,"mu00%i",buffer);
			else if(buffer<1000)
				sprintf(buff2,"mu0%i",buffer);
			else
				sprintf(buff2,"%i",buffer);
			strcat(gauge_title,buff2);
			buffer = (int)(100*ajq);
			if(buffer<10)
				sprintf(buff2,"j0%i",buffer);
			else
				sprintf(buff2,"j%i",buffer);
			strcat(gauge_title,buff2);
			if(nx<10)
				sprintf(buff2,"s0%i",nx);
			else
				sprintf(buff2,"s%i",nx);
			strcat(gauge_title,buff2);
			if(nt<10)
				sprintf(buff2,"t0%i",nt);
			else
				sprintf(buff2,"t%i",nt);
			strcat(gauge_title,buff2);
		}

		char *fileop = "wb";
		char gauge_file[FILELEN];
		strcpy(gauge_file,gauge_title);
		char c[8];
		if(itraj<10)
			sprintf(c,".00000%i", itraj);
		else	if(itraj<100)
			sprintf(c,".0000%i", itraj);
		else	if(itraj<1000)
			sprintf(c,".000%i", itraj);
		else	if(itraj<10000)
			sprintf(c,".00%i", itraj);
		else	if(itraj<10000)
			sprintf(c,".0%i", itraj);
		else
			sprintf(c,".%i", itraj);
		strcat(gauge_file, c);
		printf("Gauge file name is %s\n", gauge_file);
		printf("Writing the gauge file on processor %i.\n", rank);
		if(!(con=fopen(gauge_file, fileop))){
			fprintf(stderr, "Error %i in %s: Failed to open %s.\nExiting...\n\n", OPENERROR, funcname, gauge_file);
			MPI_Finalise();
			exit(OPENERROR);	
		}
		fwrite(u11Write, ndim*gvol*sizeof(complex), 1, con);
		fwrite(u12Write, ndim*gvol*sizeof(complex), 1, con);
		fwrite(&seed, sizeof(seed), 1, con);
		fclose(con);
#ifdef USE_MKL
		mkl_free(u11Write); mkl_free(u12Write); mkl_free(u1buff); mkl_free(u2buff);
#else
		free(u11Write); free(u12Write); free(u1buff); free(u2buff);
#endif
	}
	else{
		for(int idim = 0; idim<ndim; idim++){
			if(MPI_Send(u11+(kvol+halo)*idim, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, tag, comm)){
				fprintf(stderr, "Error %i in %s: Falied to send u11 from process %i.\nExiting...\n\n",
						CANTSEND, funcname, iproc);
				MPI_Finalise();
				exit(CANTSEND);
			}
			if(MPI_Send(u12+(kvol+halo)*idim, kvol, MPI_C_DOUBLE_COMPLEX, masterproc, tag, comm)){
				fprintf(stderr, "Error %i in %s: Falied to send u12 from process %i.\nExiting...\n\n",
						CANTSEND, funcname, iproc);
				MPI_Finalise();
				exit(CANTSEND);
			}
		}
	}
	return 0;
}
//To be lazy, we've got modules to help us do reductions and broadcasts with a single argument
//rather than type them all every single time

int Par_isum(int *ival){
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
int Par_dsum(double *dval){
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
int Par_zsum(complex *zval){
	/*
	 * Performs a reduction on a complex zval to get a sum which is
	 * then distributed to all ranks.
	 *
	 * Parameters:
	 * -----------
	 * complex *zval: The pointer to the element being summed, and
	 * 		the container for said sum.
	 *
	 * Returns:
	 * --------
	 * Zero on success. Integer error code otherwise.
	 *
	 */
	char *funcname = "Par_zsum";
	//Container to receive data.
	complex ztmp;

	if(MPI_Allreduce(zval, &ztmp, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm)){
		fprintf(stderr, "Error %i in %s: Couldn't complete reduction for %f+%f i.\nExiting...\n\n",
				REDUCERR, funcname, creal(*zval), cimag(*zval));
		MPI_Finalise();
		exit(REDUCERR);	
	}
	*zval = ztmp;
	return 0;
}
int Par_icopy(int *ival){
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
int Par_dcopy(double *dval){
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
int Par_zcopy(complex *zval){
	/*
	 * Broadcasts a complex value to the other processes
	 *
	 * Parameters:
	 * ----------
	 * complex *zval
	 *
	 * Returns:
	 * -------
	 * Zero on success, integer error code otherwise
	 */
	char *funcname = "Par_zcopy";
	if(MPI_Bcast(zval,1,MPI_C_DOUBLE_COMPLEX,masterproc,comm)){
		fprintf(stderr, "Error %i in %s: Failed to broadcast %f+i%f from %i.\nExiting...\n\n",
				BROADERR, funcname, creal(*zval), cimag(*zval), rank);
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

int ZHalo_swap_all(complex *z, int ncpt){
	/*
	 * Calls the functions to send data to both the up and down halos
	 *
	 * Parameters:
	 * -----------
	 * complex z:	The data being sent
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
int ZHalo_swap_dir(complex *z, int ncpt, int idir, int layer){
	/*
	 * Swaps the halos along the axis given by idir in the direction
	 * given by layer
	 *
	 * Parameters:
	 * -----------
	 *  complex	*z:	The data being moved about. It should be an array of dimension [kvol+halo][something else]
	 *  int		ncpt: The size of something else above. 	
	 *  int		idir:	The axis being moved along in C Indexing
	 *  int		layer:	Either DOWN (0) or UP (1)
	 *
	 *  Returns:
	 *  -------
	 *  Zero on success, Integer Error code otherwise
	 */
	char *funcname = "ZHalo_swap_dir";
	if(layer!=DOWN && layer!=UP){
		fprintf(stderr, "Error %i in %s: Cannot swap in the direction given by %i.\nExiting...\n\n",
				LAYERROR, funcname, layer);
		MPI_Finalise();
		exit(LAYERROR);
	}
#ifdef USE_MKL
	complex *sendbuf = mkl_malloc(halo*ncpt*sizeof(complex), AVX);
#else
	complex *sendbuf = malloc(halo*ncpt*sizeof(complex));
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
#pragma omp simd aligned(z:AVX, sendbuf:AVX)
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
#pragma omp simd aligned(z:AVX, sendbuf:AVX)
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
#ifdef USE_MKL
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
#ifdef USE_MKL
	double *sendbuf = mkl_malloc(halo*ncpt*sizeof(double), AVX);
#else
	double *sendbuf = malloc(halo*ncpt*sizeof(double));
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
#pragma ivdep
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
#pragma ivdep
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
#ifdef USE_MKL
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
#ifdef USE_MKL
	complex *z = mkl_malloc((kvol+halo)*sizeof(complex),AVX);
#else
	complex *z = malloc((kvol+halo)*sizeof(complex));
#endif
	for(int mu=0;mu<ndim;mu++){
		//Copy the column from u11t
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol, &u11t[mu], ndim, z, 1);
#else
		for(int i=0; i<kvol;i++)
			z[i]=u11t[i*ndim+mu];
#endif
		//Halo exchange on that column
		ZHalo_swap_all(z, 1);
		//And the swap back
#if (defined USE_MKL || USE_BLAS)
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
#if (defined USE_MKL || USE_BLAS)
		cblas_zcopy(kvol+halo, z, 1, &u12t[mu], ndim);
#else
		for(int i=0; i<kvol+halo;i++)
			u12t[i*ndim+mu]=z[i];
#endif
	}
#ifdef USE_MKL
	mkl_free(z);
#else
	free(z);
#endif
	return 0;
}
int Par_tmul(complex *z11, complex *z12){
	/*
	 * Parameters:
	 * ===========
	 * complex *z11
	 * complex *z12
	 *
	 * Returns:
	 * =======
	 * Zero on success, integer error code otherwise.
	 */
	char *funcname = "Par_tmul";
	complex *a11, *a12, *t11, *t12;
	int i, itime;
	//If we're using mkl, the mkl_malloc helps ensure arrays align with 64-bit
	//byte boundaries to improve performance and enable AVX-512 instructions.
	//Otherwise, malloc is pretty useful
#ifdef USE_MKL
	a11=(complex *)mkl_malloc(kvol3*sizeof(complex), AVX);
	a12=(complex *)mkl_malloc(kvol3*sizeof(complex), AVX);
	t11=(complex *)mkl_malloc(kvol3*sizeof(complex), AVX);
	t12=(complex *)mkl_malloc(kvol3*sizeof(complex), AVX);
#else
	a11=malloc(kvol3*sizeof(complex));
	a12=malloc(kvol3*sizeof(complex));
	t11=malloc(kvol3*sizeof(complex));
	t12=malloc(kvol3*sizeof(complex));
#endif
	//Intitialise for the first loop
	memcpy(a11, z11, kvol3*sizeof(complex));
	memcpy(a12, z12, kvol3*sizeof(complex));

	//Since the index of the outer loop isn't used as an array index anywher
	//I'm going format it exactly like the original FORTRAN
#ifdef _DEBUG
	if(!rank) printf("Sending between halos in the time direction. For rank %i pu[3]=%i and pd[3] = %i\n",
			rank, pu[3], pd[3]);
#endif
	for(itime=1;itime<npt; itime++){
		memcpy(t11, a11, kvol3*sizeof(complex));	
		memcpy(t12, a12, kvol3*sizeof(complex));	
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
#pragma omp parallel for simd
		for(i=0;i<kvol3;i++){
			t11[i]=z11[i]*a11[i]-z12[i]*conj(a12[i]);
			t12[i]=z11[i]*a12[i]+z12[i]*conj(a11[i]);
		}
		memcpy(z11, t11, kvol3*sizeof(complex));
		memcpy(z12, t12, kvol3*sizeof(complex));
	}
#ifdef USE_MKL
	mkl_free(a11); mkl_free(a12);
	mkl_free(t11); mkl_free(t12);
#else
	free(a11); free(a12); free(t11); free(t12);
#endif
	return 0;
}
