c*******************************************************************
c
c    Defines the constants of the code and other parameters 
c    for loop dimensions.
c    Each subroutine includes these definitions using:
c    INCLUDE sizes.h
c
c*******************************************************************

c Common block definition for parallel variables
      integer, parameter :: nx = 8 
      integer, parameter :: nt = 16 
c       integer, parameter :: nx = 4 
c       integer, parameter :: nt = 8	
c Keep original restriction of single spatial extent

      integer, parameter :: ny =  nx
      integer, parameter :: nz =  nx
      integer, parameter :: gvol  = nx*ny*nz*nt
      integer, parameter :: gvol3 = nx*ny*nz


c Comment out for testing purposes

c     integer, parameter :: npx = 8
c     integer, parameter :: npt = 4

      integer, parameter :: npx = 2
      integer, parameter :: npt = 1 
      integer, parameter :: nthreads = 1 

c Initially restrict to npz = npy = npx
c This allows us to have a single ksize variable

      integer, parameter :: npy = npx
      integer, parameter :: npz = npx

      integer, parameter :: nproc = npx*npy*npz*npt

c     Existing parameter definitions.
      integer, parameter :: ksizex=nx/npx
      integer, parameter :: ksizey=ny/npy
      integer, parameter :: ksizez=nz/npz

      integer, parameter :: ksize = ksizex

      integer, parameter :: ksizet=nt/npt
      integer, parameter :: Nf=1

      integer, parameter :: kvol=ksizet*ksizez*ksizey*ksizex
      integer, parameter :: kvol3=ksizez*ksizey*ksizex

      integer, parameter :: itermax=1000
c      integer, parameter :: niterc=2*gvol
ccc   jis: hard limit to avoid runaway trajectories
      integer, parameter :: niterc=10000
ccc   jis

      integer, parameter :: kmom=12*kvol
      integer, parameter :: kferm=16*kvol
      integer, parameter :: kferm2=8*kvol
    
      integer, parameter :: halox=ksizey*ksizez*ksizet
      integer, parameter :: haloy=ksizex*ksizez*ksizet
      integer, parameter :: haloz=ksizex*ksizey*ksizet
      integer, parameter :: halot=ksizex*ksizey*ksizez
      integer, parameter :: halo=2*(halox+haloy+haloz+halot)

      integer, parameter :: kfermHalo=16*(kvol+halo)
      integer, parameter :: kferm2Halo=8*(kvol+halo)
      integer, parameter :: kmomHalo=12*(kvol+halo)

      real(kind=realkind), parameter :: respbp=0.000001
      real(kind=realkind), parameter :: rescgg=0.000001
      real(kind=realkind), parameter :: rescga=0.000000001

c     Constants for dimensions.
      integer, parameter :: nc = 2
      integer, parameter :: nadj = 3
      integer, parameter :: ndirac = 4
      integer, parameter :: ndim = 4
      integer, parameter :: ngorkov = 8
  
c Parallel stuff

      include 'mpif.h'

      integer, parameter :: MPI_CMPLXKIND = MPI_DOUBLE_COMPLEX
      integer, parameter :: MPI_REALKIND  = MPI_DOUBLE_PRECISION

      integer pu(ndim), pd(ndim)
      integer procid
      integer comm
      integer ierr

      integer status(MPI_STATUS_SIZE)
      integer statarray(MPI_STATUS_SIZE, 2*ndim)
      integer, parameter :: tag = 0
      integer request
      integer reqarray(2*ndim)

      integer gsize(ndim)
      integer lsize(ndim)
      integer pcoord(ndim, nproc)
      integer pstart(ndim, nproc)
      integer pstop (ndim, nproc)
      logical ismaster
      integer masterproc

      common /par/ pu, pd, procid, comm,
     1             gsize, lsize, pcoord, pstart, pstop,
     1             ismaster, masterproc


