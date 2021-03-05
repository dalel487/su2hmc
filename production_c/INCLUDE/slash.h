#ifndef SLASH
#define SLASH
#include <par_mpi.h>
#include <su2hmc.h>

//D Slash Functions
//=================
int Dslash(complex *phi, complex *r);
int Dslashd(complex *phi, complex *r);
int Hdslash(complex *phi, complex *r);
int Hdslashd(complex *phi, complex *r);
#endif
