#ifndef	DECL
#define	DECL

//Temporary header storing all the function declarations, and what funtions they call in the comments.
//We'll sort everything out properly later


int Fill_Small_Phi(int na, complex *Phi, complex *smallPhi);
	  /*Copies necessary (2*4*kvol) elements of Phi into a vector variable.
	   * Depends on nothing else
	   */
double Norm_squared(complex *z, int n);
	   /* Takes a complex number vector of length n and finds the square of its 
	   * norm
	   * Depends on nothing else
	   */ 
#endif
