!     sread
!
!     Modified from original version of sread.
!     Data in 'con' first read into arrays of the
!     correct size and then copied to the arrays with 
!     halos.
!----------------------------------------------------
      subroutine sread
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definition
      include "common_gauge.h"
      include "common_trial.h"
   
      complex(kind=cmplxkind) u11Read(kvol,ndim), u12Read(kvol,ndim)
      integer i,j

      open(unit=10,file='con',
     1     status='unknown',form='unformatted')
      read (10) u11Read,u12Read,seed
      close(10)

c     Copy u11Read and u12Read into correct part of u11 & u12
      do i=1,ndim
      do j=1,kvol
      u11(j,i) = u11Read(j,i)
      u12(j,i) = u12Read(j,i)
      u11t(j,i) = u11(j,i)
      u12t(j,i) = u12(j,i)
      enddo
      enddo

      return
      end
c
!---------------------------------------------------
!     swrite
!
!     Modified from original version of swrite.
!     u11 and u12 first copied into arrays without 
!     halos.
!     These are then written to the output file.
!----------------------------------------------------
      subroutine swrite(isweep)
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definition
      include "common_gauge.h"
      
      character*3 c
      complex(kind=cmplxkind) u11Write(kvol,ndim), u12Write(kvol,ndim)
      integer isweep,i,j

c     Copy the u11 and u12 (minus the halos) to u11Write and u12Write
      do i=1,ndim
      do j=1,kvol
      u11Write(j,i) = u11(j,i)
      u12Write(j,i) = u11(j,i)
      enddo
      enddo

      if (ismaster) write(*,*) 'swrite: no parallel version yet'

c      write(c,'(i3.3)') isweep
c      open(unit=31,file='con'//c,
c     1     status='unknown',form='unformatted')
c      write (31) u11Write,u12Write,seed
c      close(31)
      return
      end
c********************************************************************
          SUBROUTINE RANGET(SEED)
          implicit none
          DOUBLE PRECISION    SEED,     G900GT,   G900ST,   DUMMY
          SEED  =  G900GT()
          RETURN
          ENTRY RANSET(SEED)
          DUMMY  =  G900ST(SEED)
          RETURN
          END

          FUNCTION RANF()
          implicit none
          include "precision.h"
          include "sizes.h"
          real(kind=realkind) :: RANF
          real(kind=realkind), parameter :: TINY = 1.0e-15
          DOUBLE PRECISION    G900GT,   G900ST
          DOUBLE PRECISION    DS(2),    DM(2),    DSEED
          DOUBLE PRECISION    DX24,     DX48
          DOUBLE PRECISION    DL,       DC,       DU,       DR
          DATA      DS     /  1665 1885.D0, 286 8876.D0  /
          DATA      DM     /  1518 4245.D0, 265 1554.D0  /
          DATA      DX24   /  1677 7216.D0  /
          DATA      DX48   /  281 4749 7671 0656.D0  /
          DL  =  DS(1) * DM(1)
          DC  =  DINT(DL/DX24)
          DL  =  DL - DC*DX24
          DU  =  DS(1)*DM(2) + DS(2)*DM(1) + DC
          DS(2)  =  DU - DINT(DU/DX24)*DX24
          DS(1)  =  DL
          DR     =  (DS(2)*DX24 + DS(1)) / DX48
          if (DR.gt.1.0) then
             write(*,*) 'WARNING Capped ranf to 1.0'
             RANF = 1.0
          elseif (DR.le.0.0) then
             write(*,*) 'WARNING SMALL ranf', DR
             RANF = TINY
          else
             RANF=  DR
          endif
          RETURN
          ENTRY G900GT()
          G900GT  =  DS(2)*DX24 + DS(1)
          RETURN
          ENTRY G900ST(DSEED)
          DS(2)  =  DINT(DSEED/DX24)
          DS(1)  =  DSEED - DS(2)*DX24
          G900ST =  DS(1)
c          write(*,*) 'On proc ', procid, ', ds = ', ds(1), ds(2)
c          call flush(6)
          RETURN
          END
      subroutine ranfdump(ranval)

      implicit none

      include "precision.h"

      real(kind=realkind) ranval
      integer :: icount = 0
      save icount
      character(32) :: filename

      icount = icount + 1

      write(filename,fmt='(''ranf_'', i3.3, ''.dat'')') icount
      write(*,*) 'filename = ', filename

      call par_ranread(filename, ranval)

c      open(unit=10, file=filename, form='unformatted')
c      read(10) ranval
c      close(unit=10)

      return
      end
