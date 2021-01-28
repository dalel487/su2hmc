c*******************************************************************
c
c    Defines the precision on the complex and real variables.
c    Each subroutine includes these definitions using:
c    INCLUDE precision.h
c
c*******************************************************************

c     Double precision complex and real variables
      integer, parameter :: cmplxkind = kind((1.0d0,1.0d0))
      integer, parameter :: realkind = kind(1.0d0)
