c*******************************************************************
c
c    Common block definition of dum1 label.
c
c*******************************************************************

      common /dum1/ R1(2,8,kvol+halo),R(2,8,kvol+halo),ps(2,kvol+halo)
      real(kind=realkind) ps
      complex(kind=cmplxkind) R1,R
