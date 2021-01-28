c*******************************************************************
c
c    Common block definition of gauge label.
c
c*******************************************************************

      common /gauge/ u11(kvol+halo,ndim),u12(kvol+halo,ndim),seed
      complex(kind=cmplxkind) u11, u12
      real(kind=realkind) seed
