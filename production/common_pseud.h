c*******************************************************************
c
c    Common block definition of pseud label.
c
c*******************************************************************

      common /pseud/ Phi(2,8,kvol+halo,Nf),X0(2,4,kvol+halo,Nf)
      complex(kind=cmplxkind) Phi, X0
