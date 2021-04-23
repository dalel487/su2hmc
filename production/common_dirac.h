c*******************************************************************
c
c    Common block definition of dirac label.
c
c*******************************************************************

      common /dirac/ gamval(5,4), gamin(4,4)
      complex(kind=cmplxkind) gamval
      integer gamin
