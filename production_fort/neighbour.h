c*******************************************************************
c
c    Common block definition of neighb label.
c
c*******************************************************************

      common /neighb/ id(kvol,ndim),iu(kvol,ndim),
     &        hu(halo,4), hd(halo,4), h1u(4), h1d(4),
     &        h2u(4), h2d(4), halosize(4)
      integer id, iu
      integer hu, hd, h1u, h1d, h2u, h2d, halosize
