#ifndef SLASH
#define SLASH
#include <par_mpi.h>
#include <su2hmc.h>

//D Slash Functions
//=================
int Dslash(complex phi[][ngorkov][nc], complex r[][ngorkov][nc]);
int Dslashd(complex phi[][ngorkov][nc], complex r[][ngorkov][nc]);
int Hdslash(complex phi[][ndirac][nc], complex r[][ndirac][nc]);
int Hdslashd(complex phi[][ndirac][nc], complex r[][ndirac][nc]);
#endif
