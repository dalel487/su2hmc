c*******************************************************************
c
c    Common block definition of trial label.
c
c*******************************************************************

      common /trial/ u11(kvol+halo,ndim),u12(kvol+halo,ndim),
     &     pp(ndim,3,kvol+halo)
      complex(kind=cmplxkind) u11,u12
      real(kind=realkind) pp
