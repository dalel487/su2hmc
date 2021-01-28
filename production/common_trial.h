c*******************************************************************
c
c    Common block definition of trial label.
c
c*******************************************************************

      common /trial/ u11t(kvol+halo,ndim),u12t(kvol+halo,ndim),
     &     pp(ndim,3,kvol+halo)
      complex(kind=cmplxkind) u11t,u12t
      real(kind=realkind) pp
