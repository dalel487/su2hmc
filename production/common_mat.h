c*******************************************************************
c
c    Common block definition of mat label.
c
c*******************************************************************

      common /mat/ dk4p(kvol+halo), dk4m(kvol+halo)
      real(kind=realkind) dk4p, dk4m
