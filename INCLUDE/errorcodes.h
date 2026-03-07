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

/** @section ioerr File I/O Errors. Leading digits are 10
*/
//======================================
//Errors:
//-------
///@brief	Error opening file
#define	OPENERROR	10001
///@brief	Error reading file
#define	READERROR	10002
///@brief	Error writing to file
#define	WRITERROR	10003
///@brief	Error with argument
#define	ARGERROR		10004

//Warnings:
//---------
///@brief	Minor issue opening file
#define	OPENWARN		10101
///@brief	Minor issue reading file
#define	READWARN		10102
///@brief	Minor issue writing to file
#define	WRITEWARN	10103
///@brief	Minor argument issue
#define	ARGWARN		10104

/** @section memerr Memory Errors.
 *	Leading digits are 11
 */
//======================================
//Errors:
//-------
/// @brief Issues converting index to coordinates
#define	INDTOCOORD	11001	
/// @brief Issues converting coordinate to index
#define	COORDTOIND	11002	
/// @brief Accessing out of bounds element
#define	BOUNDERROR	11003	
/// @brief Impossible value for array length
#define	ARRAYLEN		11004 
/// @brief Copy failed
#define	CPYERROR		11005 
/// @brief Undefined stride
#define	STRDERROR	11006 

//Warnings:
//---------
/// @brief Accessing out of bounds element, but not a big enough problem to crash the programme.
#define	BOUNDWARN	11103	
/// @brief Order of limits (x_min and x_max for example) is reversed.
#define	LIMWARN		11104
/// @brief Copy failed
#define	CPYWARN		11105 

/** @section mpierr MPI Errors. Leading digits are 12
*/
//================================
//Errors:
//------
/// @brief Failed to initialise MPI
#define	NO_MPI_INIT	12001	
/// @brief Failed to get the rank of the process
#define	NO_MPI_RANK	12002
/// @brief Failed to get the number of ranks
#define	NO_MPI_SIZE	12003
/// @brief Communicator size does not match expected size
#define	SIZEPROC		12004	
/// @brief Failed to evaluate the number of elements
#define	NUMELEM		12005
/// @brief Couldn't send to another process
#define	CANTSEND		12006
/// @brief Couldn't receive from another process
#define	CANTRECV		12007
/// @brief Couldn't broadcast to the processes
#define	BROADERR		12008
/// @brief Couldn't carry out a reduction operation
#define	REDUCERR		12009
/// @brief Couldn't complete a gather operation
#define	GATHERR		12010

//Warnings:
//---------
/// @brief Continuation run on different grid size
#define	DIFNPROC		12101

/// @section Halo Errors. Leading digits are 13
//=================================
//Errors:
//-------
///brief Can't access a layer of a halo
#define	LAYERROR		13001	
/// @brief Index goes beyond the halo
#define	HALOLIM		13002

/// @section Physics/Maths Errors. Leading digits are 14
//=================================
//Errors:
//-------
/// @brief Division by zero
#define	DIVZERO		14001
/// @brief Gauge link reunitarisation failed
#define	REUNIERR		14002
/// @brief Failed to convert precision correctly
#define	CONVERR		14003
/// @brief Up/down partitioning failed
#define	UDPERR		14003
/// @brief Up/down partitioning failed
#define	SPHIERR		14004

//Warnings:
//--------
/// @brief Exceeded max number of iterations
#define	ITERLIM		14101
/// @brief Fitting function has repeated x value
#define	FITWARN		14102

//Alerts:
//------
/// @brief Not initialising the lattice
#define	NOINIT		14201

//CUDA Errors. Leading digits are 15
//==================================
//Errors:
//------
/// @brief Error with setting block size
#define	BLOCKERROR	15001

//Warnings:
//------
/// @brief Warning with block size
#define	BLOCKWARN	15101

//ALERT:
//------
/// @brief Alert with block size
#define	BLOCKALERT	15201

/// @section Other errors. Leading digits are 16
/// @brief Not implemented
#define	NOIMPL	16001
#endif
