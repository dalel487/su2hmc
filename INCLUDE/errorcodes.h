/** 
 * @file errorcodes.h
 * @brief This header is intended to be a useful reference for error codes and their meanings.
 *
 * By placing all the error codes in one file I hope to make it easier to diagnose any
 * potential issues that crop up, and create a logical and consistent labelling system
 * for error codes.
 *
 * Error codes will take the following format:
 * XXYZZ
 * where XX is the category of code (File Related, Index Related etc.)
 * Y is the severity (0 for error, 1 for warning and more to be added later)
 * ZZ is the identifying code.
 * 
 * Hopefully two digits should be enough to cover all possibilities.
 *
 * The way I intend the codes to be used is
 * fprintf(stderr, "Error %i in %s: Description of what happened\nExiting...\n\n", CODE, funcname,\
 * 	anything else to be printed);
 * It goes without saying that the Exiting bit should only be used if actually exiting the programme
 * and can be replaced with other text. Same with the first word Error for warnings etc.
 */
#ifndef ERRORCODES
#define ERRORCODES

//File I/O Errors. Leading digits are 10
//======================================
//Errors:
//-------
#define	OPENERROR	10001
#define	READERROR	10002
#define	WRITERROR	10003
#define	ARGERROR		10004

//Warnings:
//---------
#define	OPENWARN		10101
#define	READWARN		10102
#define	WRITEWARN	10103
#define	ARGWARN		10104

/*
 *	@section memerr Memory Errors.
 *	Leading digits are 11
 */
//======================================
//Errors:
//-------
#define	INDTOCOORD	11001	////Issues converting index to coordinates
#define	COORDTOIND	11002	/////Issues converting coordinate to index
#define	BOUNDERROR	11003	////Accessing out of bounds element
#define	ARRAYLEN		11004 ////Impossible value for array length
#define	CPYERROR		11005 ////Copy failed
									//Warnings:
									//---------
#define	BOUNDWARN	11103	////Accessing an out of bounds element, but not a big enough problem to crash the programme.
#define	LIMWARN		11104	////Order of limits (x_min and x_max for example) is reversed.
#define	CPYWARN		11105 ////Copy failed

//MPI Errors. Leading digits are 12
//================================
//Errors:
//------
#define	NO_MPI_INIT	12001	//Failed to initialise MPI
#define	NO_MPI_RANK	12002	//Failed to get the rank of the process
#define	NO_MPI_SIZE	12003	//Failed to get the number of ranks
#define	SIZEPROC		12004	//
#define	NUMELEM		12005	//Failed to evaluate the number of elements
#define	CANTSEND		12006	//Couldn't send to another process
#define	CANTRECV		12007	//Couldn't receive from a process
#define	BROADERR		12008	//Couldn't broadcast to the processes
#define	REDUCERR		12009	//Couldn't carry out a reduction
#define	GATHERR		12010	//Couldn't complete a gather
									//Warnings:
									//---------
#define	DIFNPROC		12101	//Continuation run on a different number of ranks

//Halo Errors. Leading digits are 13
//=================================
//Errors:
//-------
#define	LAYERROR		13001	//Can't access a layer of a halo
#define	HALOLIM		13002	//Index goes beyond the halo

//Physics/Maths Errors. Leading digits are 14
//=================================
//Errors:
//-------
#define	DIVZERO		14001	//Not quite an indexing error, bu division by zero

//Warnings:
//--------
#define	ITERLIM		14101 //Exceeded max number of iterations
#define	FITWARN		14102 //Fitting function has repeated x value

//Alerts:
//------
#define	NOINIT		14201 //Not initialising the lattice

//CUDA Errors. Leading digits are 15
//==================================
//Errors:
//------
#define	BLOCKERROR	15001

//Warnings:
//------
#define	BLOCKWARN	15101

//ALERT:
//------
#define	BLOCKALERT	15201
#endif
