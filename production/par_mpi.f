      subroutine par_begin()

      implicit none

      include "precision.h"
      include "sizes.h"

      integer size, commcart
      integer, dimension(ndim) :: cartsize
      logical :: reorder  = .false.
      logical periods(ndim)
      integer idim, iproc

      call MPI_INIT(ierr)

      comm = MPI_COMM_WORLD

      call MPI_COMM_RANK(comm, procid, ierr)

      masterproc = 0

      ismaster = .false.

      if (procid == masterproc) ismaster = .true.

      call MPI_COMM_SIZE(comm, size, ierr)

      if (size .ne. nproc) then
        write(*,*) 'Error on process ', procid, ': size = ', size, 
     1             ' not equal to nproc = ',nproc

        stop
      end if
      gsize(1) = nx
      gsize(2) = ny
      gsize(3) = nz
      gsize(4) = nt

      lsize(1) = ksizex
      lsize(2) = ksizey
      lsize(3) = ksizez
      lsize(4) = ksizet
      
      cartsize(1) = npx
      cartsize(2) = npy
      cartsize(3) = npz
      cartsize(4) = npt

      periods(1) = .true.
      periods(2) = .true.
      periods(3) = .true.
      periods(4) = .true.

      call MPI_CART_CREATE(comm, ndim, cartsize, periods,
     1                     reorder, commcart, ierr)

c      call MPI_COMM_SIZE(commcart, size, ierr)
c
c      write(*,*) 'procid = ', procid, ', size of cart = ', size

      do idim = 1, ndim
        call MPI_CART_SHIFT(commcart, idim-1, 1,
     1                      pd(idim), pu(idim), ierr)

      end do

c      do idim = 1, ndim
c
c        write(*,*) 'proc = ', procid,
c     1             ': pu(', idim, ') = ', pu(idim), 
c     1             ', pd(', idim, ') = ', pd(idim)
c
c      end do

! Now get coords of all procs in the grid

      do iproc = 1, nproc
        call MPI_CART_COORDS(commcart, iproc-1, ndim,
     1                pcoord(1,iproc), ierr)

        do idim = 1, ndim
          pstart(idim, iproc) = pcoord(idim,iproc)*lsize(idim)+1
          pstop (idim, iproc) = pstart(idim,iproc) + lsize(idim) - 1

c          write(*,*) 'iproc, idim, pstart, pstop = ',
c     1                iproc, idim, pstart(idim,iproc), pstop(idim,iproc)
        end do

      end do

      if (ismaster) then
        write(*,*) 'Running on ', nproc, ' processors'
        write(*,*) 'Processor grid: ', npx, ' x ', npy, ' x ', 
     1             npz, ' x ', npt
        call flush(6)
      end if

      return
      end

      subroutine par_end()

      implicit none

      include "precision.h"
      include "sizes.h"

      call MPI_FINALIZE(ierr)

      return
      end

c
!---------------------------------------------------
!     par_sread
!
!     Modified from original version of sread.
!     Data in 'con' first read into arrays of the
!     correct size and then copied to the arrays with 
!     halos.
!----------------------------------------------------
      subroutine par_sread()
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definition
      include "common_gauge.h"
      include "common_trial.h"
   
      complex(kind=cmplxkind) u11Read(gvol,ndim),
     1                        u12Read(gvol,ndim)
      integer i,j,idim,iproc

      integer ix, iy, iz, it, ic, icoord(ndim)

      complex(kind=cmplxkind) ubuff(kvol)

      if (ismaster) then

        write(*,*) 'opening gauge file on processor ', procid

        open(unit=10,file='con',
     1     status='unknown',form='unformatted')
        read (10) u11Read,u12Read,seed
        close(10)

c send correct parts of u11Read etc to remote processors

c     Copy u11Read and u12Read into correct part of u11 & u12

        do iproc = 1, nproc

c          write(*,*) 'reading u field for proc ', iproc-1
          call flush(6)

          do idim=1,ndim

            do ic = 1, nc

              i = 0

              do it = pstart(4,iproc), pstop(4,iproc)
                do iz = pstart(3,iproc), pstop(3,iproc)
                  do iy = pstart(2,iproc), pstop(2,iproc)
                    do ix = pstart(1,iproc), pstop(1,iproc)

                      icoord(1) = ix
                      icoord(2) = iy
                      icoord(3) = iz
                      icoord(4) = it

                      i = i + 1
                      call coord2gindex(icoord, j)

c                      write(*,*) 'ix, iy, iz, it, j = ',
c     1                            ix, iy, iz, it, j

                      if (ic == 1) then
                        ubuff(i) = u11read(j,idim)
                      else if (ic == 2) then
                        ubuff(i) = u12read(j,idim)
                      else
                        write(*,*) 'Illegal ic'
                        stop
                      end if

                    end do
                  end do
                end do
              end do

              if (i .ne. kvol) then
                write(*,*) 'proc ', procid, ': par_sread error'
                stop
              end if

              if (iproc-1 == masterproc) then
                do i = 1, kvol
                  if (ic == 1) then
                    u11(i,idim)  = ubuff(i)
                    u11t(i,idim) = ubuff(i)
                  else
                    u12(i,idim)  = ubuff(i)
                    u12t(i,idim) = ubuff(i)
                  end if
                end do

              else

c              write(*,*) 'proc ', procid, ' sending idim = ', idim,
c     1                   ', ic = ', ic, ' to proc ', iproc-1

                 call MPI_SSEND(ubuff, kvol, MPI_CMPLXKIND,
     1                          iproc-1, tag, comm, ierr)
                 call flush(6)

              end if

            end do
          end do
        end do

      else

        do idim = 1, ndim

c         write(*,*) 'proc ', procid, ' receiving idim = ', idim,
c     1                   ', ic = ', 1, ' from proc ', masterproc
c
c         call flush(6)

         call MPI_RECV(u11(1,idim), kvol, MPI_CMPLXKIND,
     1                masterproc, tag, comm, status, ierr)

c         write(*,*) 'proc ', procid, ' receiving idim = ', idim,
c     1                   ', ic = ', 2, ' from proc ', masterproc
c
c         call flush(6)
         
         call MPI_RECV(u12(1,idim), kvol, MPI_CMPLXKIND,
     1        masterproc, tag, comm, status, ierr)

        end do

        do idim = 1, ndim
           do i = 1, kvol
              u11t(i,idim)  = u11(i,idim)
              u12t(i,idim)  = u12(i,idim)
           end do
        end do
      end if

c chrisj
c      if (ismaster) write(*,*) 'proc ', procid, ' seed sending = ', seed
      call par_ccopy(seed)

c chrisj
c      write(*,*) 'proc ', procid, ' seed received = ', seed
c      write(*,*) 'proc ', procid, ' exiting par_sread'
c      flush(6)

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
      subroutine par_swrite(isweep)
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definition
      include "common_gauge.h"
      
      integer isweep

      complex(kind=cmplxkind) u11Write(gvol,ndim),
     1                        u12Write(gvol,ndim)
      integer i,j,idim,iproc

      integer ix, iy, iz, it, ic, icoord(ndim)

      complex(kind=cmplxkind) ubuff(kvol)

      character*3 c

      if (ismaster) then

        write(*,*) 'writing gauge file on processor ', procid

c receive correct parts of u11Read etc from remote processors

        do iproc = 1, nproc

c          write(*,*) 'receiving u field from proc ', iproc-1
c          call flush(6)

          do idim=1,ndim

            do ic = 1, nc

              if (iproc-1 == masterproc) then

                do i = 1, kvol
                  if (ic == 1) then
                    ubuff(i) = u11(i,idim)
                  else
                    ubuff(i) = u12(i,idim)
                  end if
                end do

              else

c                write(*,*) 'proc ', procid, ' receiving idim = ', idim,
c     1                ', ic = ', ic, ' from proc ', iproc-1
c                call flush(6)

                call MPI_RECV(ubuff, kvol, MPI_CMPLXKIND,
     1               iproc-1, tag, comm, status, ierr)

              end if

              i = 0

              do it = pstart(4,iproc), pstop(4,iproc)
                do iz = pstart(3,iproc), pstop(3,iproc)
                  do iy = pstart(2,iproc), pstop(2,iproc)
                    do ix = pstart(1,iproc), pstop(1,iproc)

                      icoord(1) = ix
                      icoord(2) = iy
                      icoord(3) = iz
                      icoord(4) = it

                      i = i + 1
                      call coord2gindex(icoord, j)

c                      write(*,*) 'ix, iy, iz, it, j = ',
c     1                            ix, iy, iz, it, j

                      if (ic == 1) then
                        u11write(j,idim) = ubuff(i)
                      else if (ic == 2) then
                        u12write(j,idim) = ubuff(i)
                      else
                        write(*,*) 'Illegal ic'
                        stop
                      end if

                   end do
                  end do
                end do
              end do

              if (i .ne. kvol) then
                write(*,*) 'proc ', procid, ': par_swrite error'
                stop
              end if

            end do
          end do
        end do

        write(c,'(i3.3)') isweep
        write(*,*) 'gauge file name is ', 'con'//c
        open(unit=31,file='con'//c,
     1       status='unknown',form='unformatted')
        write (31) u11Write,u12Write,seed
        close(31)

      else

        do idim = 1, ndim

c          write(*,*) 'proc ', procid, ' sending idim = ', idim,
c     1                   ', ic = ', 1, ' to proc ', masterproc
c          call flush(6)

          call MPI_SSEND(u11(1,idim), kvol, MPI_CMPLXKIND,
     1                masterproc, tag, comm, ierr)

c          write(*,*) 'proc ', procid, ' sending idim = ', idim,
c     1                   ', ic = ', 2, ' to proc ', masterproc
c          call flush(6)
         
          call MPI_SSEND(u12(1,idim), kvol, MPI_CMPLXKIND,
     1                masterproc, tag, comm, ierr)

        end do

      end if

      return
      end
c

      subroutine par_csum(cval)

      implicit none

      include "precision.h"
      include "sizes.h"

      real(kind=realkind) :: cval, ctmp

      call MPI_ALLREDUCE(cval, ctmp, 1,
     1                   MPI_REALKIND, MPI_SUM, comm, ierr)

      cval = ctmp

      return
      end


      subroutine par_zsum(zval)

      implicit none

      include "precision.h"
      include "sizes.h"

      complex(kind=cmplxkind) :: zval, ztmp

      call MPI_ALLREDUCE(zval, ztmp, 1,
     1                   MPI_CMPLXKIND, MPI_SUM, comm, ierr)

      zval = ztmp

      return
      end


      subroutine par_icopy(ival)

      implicit none

      include "precision.h"
      include "sizes.h"

      integer :: ival

      call MPI_BCAST(ival, 1, MPI_INTEGER, masterproc, comm, ierr)

      return
      end


      subroutine par_ccopy(cval)

      implicit none

      include "precision.h"
      include "sizes.h"

c      complex(kind=cmplxkind) :: cval
      complex(kind=realkind) :: cval
      call MPI_BCAST(cval, 1, MPI_REALKIND, masterproc, comm, ierr)

      return
      end


      subroutine par_zcopy(zval)

      implicit none

      include "precision.h"
      include "sizes.h"

      complex(kind=cmplxkind) :: zval

      call MPI_BCAST(zval, 1, MPI_CMPLXKIND, masterproc, comm, ierr)

      return
      end


      subroutine zhaloswapall(z, ncpt)
      implicit none
      include "precision.h"
      include "sizes.h"
      
      include "common_mat.h"
      include "common_para.h"
      include "common_dirac.h"
      include "common_neighb.h"

      integer idir, ncpt

      complex(kind=cmplxkind) z(ncpt, kvol+halo)
 
      call zdnhaloswapall(z, ncpt)
      call zuphaloswapall(z, ncpt)

      return
      end

      subroutine zdnhaloswapall(z, ncpt)
      implicit none
      include "precision.h"
      include "sizes.h"
      
      include "common_mat.h"
      include "common_para.h"
      include "common_dirac.h"
      include "common_neighb.h"

      integer idir, ncpt

      complex(kind=cmplxkind) z(ncpt, kvol+halo)
 
      do idir = 1, 4

        call zdnhaloswapdir(z, ncpt, idir)

      end do

      return
      end

      subroutine zuphaloswapall(z, ncpt)
      implicit none
      include "precision.h"
      include "sizes.h"

      include "common_mat.h"
      include "common_para.h"
      include "common_dirac.h"
      include "common_neighb.h"

      integer idir, ncpt

      complex(kind=cmplxkind) z(ncpt, kvol+halo)
 
      do idir = 1, 4

        call zuphaloswapdir(z, ncpt, idir)

      end do

      return
      end

      
      subroutine zdnhaloswapdir(z, ncpt, idir)
      implicit none
      include "precision.h"
      include "sizes.h"
      
      include "common_mat.h"
      include "common_para.h"
      include "common_dirac.h"
      include "common_neighb.h"

      integer coord1(4), coord2(4), ihalo, icpt

      integer idir, ncpt, i

      complex(kind=cmplxkind) z(ncpt, kvol+halo)

c
c  Declare storage locally on the stack - not pure f77
c

      complex(kind=cmplxkind) sendbuf(ncpt, halo)

c
c  Send off the down halo
c

      do ihalo = 1, halosize(idir)
        do icpt = 1, ncpt

          sendbuf(icpt, ihalo) = z(icpt, hd(ihalo, idir))

        end do
      end do

      call MPI_ISEND(sendbuf, ncpt*halosize(idir), MPI_CMPLXKIND,
     1               pd(idir), tag, comm, request, ierr)

c
c  Receive into up halo
c

      call MPI_RECV(z(1,h1u(idir)), ncpt*halosize(idir), MPI_CMPLXKIND,
     1              pu(idir), tag, comm, status, ierr)

      call MPI_WAIT(request, status, ierr)

c      do ihalo = 1, halosize(idir)
c
c        i = h1u(idir) + ihalo - 1
c
c        do icpt = 1, ncpt
c
c           z(icpt, i) = sendbuf(icpt, ihalo)
c  
c        end do
c
c      end do

      return
      end

      subroutine zuphaloswapdir(z, ncpt, idir)
      implicit none
      include "precision.h"
      include "sizes.h"

      include "common_mat.h"
      include "common_para.h"
      include "common_dirac.h"
      include "common_neighb.h"
      
      integer coord1(4), coord2(4), ihalo, icpt

      integer idir, ncpt, i

      complex(kind=cmplxkind) z(ncpt, kvol+halo)

      complex(kind=cmplxkind) sendbuf(ncpt,halo)

c
c  Send off the up halo
c

      do ihalo = 1, halosize(idir)
        do icpt = 1, ncpt

          sendbuf(icpt, ihalo) = z(icpt, hu(ihalo, idir))

        end do
      end do

      call MPI_ISEND(sendbuf, ncpt*halosize(idir), MPI_CMPLXKIND,
     1               pu(idir), tag, comm, request, ierr)

c
c  Receive into down halo
c

      call MPI_RECV(z(1,h1d(idir)), ncpt*halosize(idir), MPI_CMPLXKIND,
     1              pd(idir), tag, comm, status, ierr)

      call MPI_WAIT(request, status, ierr)

c      do ihalo = 1, halosize(idir)
c
c        i = h1d(idir) + ihalo - 1
c
c        do icpt = 1, ncpt
c
c           z(icpt, i) = sendbuf(icpt, ihalo)
c  
c        end do
c
c      end do

      return
      end

      subroutine cuphaloswapdir(x, ncpt, idir)
      implicit none
      include "precision.h"
      include "sizes.h"

      include "common_mat.h"
      include "common_para.h"
      include "common_dirac.h"
      include "common_neighb.h"

      integer coord1(4), coord2(4), ihalo, icpt

      integer idir, ncpt, i

      real(kind=realkind) x(ncpt, kvol+halo)

      real(kind=realkind) sendbuf(ncpt,halo)

c
c  Send off the up halo
c

      do ihalo = 1, halosize(idir)
        do icpt = 1, ncpt

          sendbuf(icpt, ihalo) = x(icpt, hu(ihalo, idir))

        end do
      end do

      call MPI_ISEND(sendbuf, ncpt*halosize(idir), MPI_REALKIND,
     1               pu(idir), tag, comm, request, ierr)

c
c  Receive into down halo
c

      call MPI_RECV(x(1,h1d(idir)), ncpt*halosize(idir), MPI_REALKIND,
     1              pd(idir), tag, comm, status, ierr)

      call MPI_WAIT(request, status, ierr)

c      do ihalo = 1, halosize(idir)
c
c        i = h1d(idir) + ihalo - 1
c
c        do icpt = 1, ncpt
c
c           x(icpt, i) = sendbuf(icpt, ihalo)
c  
c        end do
c
c      end do

      return
      end


      subroutine par_psread(filename, ps)

      implicit none

      include "precision.h"
      include "sizes.h"

      character(32) filename

      real(kind=realkind) ps(2,kvol+halo)
      real(kind=realkind) gps(2,gvol)

      integer i,j,idim,iproc

      integer ix, iy, iz, it, ic, icoord(ndim)

      real(kind=realkind) psbuff(2,kvol)

c halo entries for ps are not significant - just written for
c convenience


c      write(*,*) 'proc ', procid, ' in par_psread'
c      call flush(6)

      if (ismaster) then

        write(*,*) 'opening ps file on processor ', procid,
     1             ': ', filename

        open(unit=10,file=filename, form='unformatted')
        read (10) gps
        close(10)

c send correct parts to remote processors

        do iproc = 1, nproc

          i = 0

          do it = pstart(4,iproc), pstop(4,iproc)
            do iz = pstart(3,iproc), pstop(3,iproc)
              do iy = pstart(2,iproc), pstop(2,iproc)
                do ix = pstart(1,iproc), pstop(1,iproc)

                  icoord(1) = ix
                  icoord(2) = iy
                  icoord(3) = iz
                  icoord(4) = it

                  i = i + 1
                  call coord2gindex(icoord, j)

                  psbuff(1,i) = gps(1,j)
                  psbuff(2,i) = gps(2,j)
                end do
              end do
            end do
          end do

          if (i .ne. kvol) then
             write(*,*) 'proc ', procid, ': par_psread error'
             stop
          end if

          if (iproc-1 == masterproc) then

            do i = 1, kvol
              ps(1,i) = psbuff(1,i)
              ps(2,i) = psbuff(2,i)
            end do

          else

            call MPI_SSEND(psbuff, 2*kvol, MPI_REALKIND,
     1                     iproc-1, tag, comm, ierr)
          end if

        end do

      else

        call MPI_RECV(ps, 2*kvol, MPI_REALKIND,
     1                masterproc, tag, comm, status, ierr)

      end if

c      write(*,*) 'proc ', procid, ' exiting par_psread'
c      flush(6)

      end
      
      subroutine par_ranread(filename, ranval)

      implicit none

      include "precision.h"
      include "sizes.h"

      character(32) filename

      real(kind=realkind) ranval

      if (ismaster) then
        write(*,*) 'opening ranf file on processor ', procid,
     1             ': ', filename

        open(unit=10, file=filename, form='unformatted')
        read(10) ranval
        close(unit=10)
      end if

      call par_ccopy(ranval)

      return
      end

      subroutine par_ranset(seed)

      implicit none

      include "precision.h"
      include "sizes.h"

      double precision seed, rseed

c     create new seeds in range seed to 9*seed
c     having a range of 0*seed gave an unfortunate pattern
c     in the underlying value of ds(1) (it was always 10 times bigger
c     on the last processor). This does not appear to happen with 9.

      rseed = seed

      if (nproc .gt. 1) then
        rseed = seed * (1.0 + 8.0 * float(procid)/float(nproc-1))
      end if

      call ranset(rseed)

c      write(*,*) 'On proc ', procid, ', seed, rseed = ',
c     1            seed, rseed
c      call flush(6)

      return
      end

      function par_granf()

      implicit none

      include "precision.h"
      include "sizes.h"
      include "common_gauge.h"

      real(kind=realkind) :: par_granf, ran2

      if (procid .eq. masterproc) then
        par_granf = ran2(seed)
c        write(*,*) 'par_granf: ranf() gives ', par_granf
      end if

      call par_ccopy(par_granf)

c      write(*,*) 'on proc ', procid, ', par_granf = ', par_granf

      return
      end


      subroutine par_tmul(z11, z12)

      implicit none

      include "precision.h"
      include "sizes.h"

      complex(kind=cmplxkind) z11(kvol3),z12(kvol3)
      complex(kind=cmplxkind) a11(kvol3),a12(kvol3)
      complex(kind=cmplxkind) t11(kvol3),t12(kvol3)

      integer :: i, itime


c initialise for first loop

      do i = 1, kvol3
        a11(i) = z11(i)
        a12(i) = z12(i)
      end do

      do itime = 1, npt-1

c  copy previously received incoming data to outward temp buff

        do i = 1, kvol3
          t11(i) = a11(i)
          t12(i) = a12(i)
        end do

c        write(*,*) 'On proc ', procid, ' sending t11 = ', t11(1),
c     1             'to proc ', pd(4)

c  send results from other processors down the line

        call MPI_ISEND(t11, kvol3, MPI_CMPLXKIND, pd(4),
     1                 tag, comm, request, ierr)

        call MPI_RECV (a11, kvol3, MPI_CMPLXKIND, pu(4),
     1                 tag, comm, status, ierr)

        call MPI_WAIT(request, status, ierr)


c        write(*,*) 'On proc ', procid, ' received a11 = ', a11(1),
c     1             'from proc ', pu(4)

c        write(*,*) 'On proc ', procid, ' sending t12 = ', t12(1),
c     1             'to proc ', pd(4)

        call MPI_ISEND(t12, kvol3, MPI_CMPLXKIND, pd(4),
     1                 tag, comm, request, ierr)

        call MPI_RECV (a12, kvol3, MPI_CMPLXKIND, pu(4),
     1                 tag, comm, status, ierr)

        call MPI_WAIT(request, status, ierr)


c        write(*,*) 'On proc ', procid, ' received a12 = ', a12(1),
c     1             'from proc ', pu(4)

c Post-multiply current loop by the incoming one

        do i=1,kvol3
          t11(i)=z11(i)*a11(i) - z12(i)*conjg(a12(i))
          t12(i)=z11(i)*a12(i) + z12(i)*conjg(a11(i))
        end do

c swap back

        do i=1,kvol3
          z11(i) = t11(i)
          z12(i) = t12(i)
        end do

      end do

      return
      end
c********************************************************
c  dummy routine to fool the loader....
      subroutine flush(i)
      return
      end

