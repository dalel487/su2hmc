c***********************************************************************
      subroutine dslash(Phi,R,u11,u12)
c
c     calculates Phi = M*R
c
      implicit none
c      include 'omp_lib.h'
      include "precision.h"
      include "sizes.h"
c     include common block definitions
      include "common_mat.h"
      include "common_para.h"
      include "common_diquark.h"
      include "common_neighb.h"
      include "common_dirac.h"
      
      complex(kind=cmplxkind) u11(kvol+halo,ndim),u12(kvol+halo,ndim)
      complex(kind=cmplxkind) Phi(2,8,kvol+halo),R(2,8,kvol+halo)

      integer igorkov,ic,i,idirac,igork,mu,igork1
c     write(6,*) 'hi from dslash'
c

c
c     Makes reference to:
c
c     R(.,.,iu(i,mu)), ncpt = 16, need zdnhaloswapall
c     R(.,.,id(i,mu)), ncpt = 16, need zuphaloswapall
c
c     ie for R, need zhaloswapall
c
c     Also, u(id(i,mu),mu) ncpt = 1
c
c     need to do a zuphaloswapdir(mu) for u(mu)
c
c     Also dk4p(id(i,4)) and dk4m(id(i,4)), ncpt = 1
c
c     need to do a zuphaloswapdir(4)
c     
c     
c

      call zhaloswapall(R, 16)

      do mu = 1, ndim
        call zuphaloswapdir(u11(1,mu), 1, mu)
        call zuphaloswapdir(u12(1,mu), 1, mu)
      end do

      call cuphaloswapdir(dk4p,1,4)
      call cuphaloswapdir(dk4m,1,4)

c$omp parallel do default(none) shared(phi,jqq,
c$omp& akappa,u11,u12,R,id,dk4p,iu,dk4m,gamval,gamin)
c$omp& private(idirac, igork1, igork)
c     mass term
      do i=1,kvol
         do igorkov=1,ngorkov
            do ic=1,nc
               Phi(ic,igorkov,i)=R(ic,igorkov,i)
            enddo
         enddo
c     
c     diquark term (antihermitian)
c     
         do idirac=1,ndirac
            igork=idirac+4
            do ic=1,nc
               Phi(ic,idirac,i)=Phi(ic,idirac,i)+
     &              conjg(jqq)*gamval(5,idirac)*R(ic,igork,i)
               Phi(ic,igork,i)=Phi(ic,igork,i)-
     &              jqq*gamval(5,idirac)*R(ic,idirac,i)
            enddo
         enddo
c
c     Wilson term
c
         do mu=1,3
            do igorkov=1,ngorkov
               Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &              -akappa*(      u11(i,mu)*R(1,igorkov,iu(i,mu))
     &              +u12(i,mu)*R(2,igorkov,iu(i,mu))
     &              +conjg(u11(id(i,mu),mu))*R(1,igorkov,id(i,mu))
     &              -u12(id(i,mu),mu) *R(2,igorkov,id(i,mu)))
               Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &              -akappa*(-conjg(u12(i,mu))*R(1,igorkov,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,igorkov,iu(i,mu))
     &              +conjg(u12(id(i,mu),mu))*R(1,igorkov,id(i,mu))
     &              +u11(id(i,mu),mu) *R(2,igorkov,id(i,mu)))
            enddo
         enddo
c     
c     Dirac term
c
         do mu=1,3
            do igorkov=1,ngorkov
               idirac=mod((igorkov-1),4)+1
               if(igorkov.le.4)then
                  igork1=gamin(mu,idirac)
               else
                  igork1=gamin(mu,idirac)+4
               endif
               Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &              +gamval(mu,idirac)*
     &              (          u11(i,mu)*R(1,igork1,iu(i,mu))
     &              +u12(i,mu)*R(2,igork1,iu(i,mu))
     &              -conjg(u11(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              +u12(id(i,mu),mu) *R(2,igork1,id(i,mu)))
               Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &              +gamval(mu,idirac)*
     &              (-conjg(u12(i,mu))*R(1,igork1,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,igork1,iu(i,mu))
     &              -conjg(u12(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              -u11(id(i,mu),mu) *R(2,igork1,id(i,mu)))
            enddo
         enddo
c
c  Timelike Wilson term
c
         do igorkov=1,4
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           -dk4p(i)*(u11(i,4)*R(1,igorkov,iu(i,4))
     &           +u12(i,4)*R(2,igorkov,iu(i,4)))
     &           -dk4m(id(i,4))*(conjg(u11(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           -u12(id(i,4),4) *R(2,igorkov,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           -dk4p(i)*(-conjg(u12(i,4))*R(1,igorkov,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igorkov,iu(i,4)))
     &           -dk4m(id(i,4))*(conjg(u12(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           +u11(id(i,4),4) *R(2,igorkov,id(i,4)))
         enddo
c     
         do igorkov=5,ngorkov
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           -dk4m(i)*(u11(i,4)*R(1,igorkov,iu(i,4))
     &           +u12(i,4)*R(2,igorkov,iu(i,4)))
     &           -dk4p(id(i,4))*(conjg(u11(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           -u12(id(i,4),4) *R(2,igorkov,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           -dk4m(i)*(-conjg(u12(i,4))*R(1,igorkov,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igorkov,iu(i,4)))
     &           -dk4p(id(i,4))*(conjg(u12(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           +u11(id(i,4),4) *R(2,igorkov,id(i,4)))
         enddo

c
c  Timelike Dirac term

         do igorkov=1,4
            idirac=igorkov
            igork1=gamin(4,idirac)
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           +dk4p(i)*
     &           (u11(i,4)*R(1,igork1,iu(i,4))
     &           +u12(i,4)*R(2,igork1,iu(i,4)))
     &           -dk4m(id(i,4))*
     &           (conjg(u11(id(i,4),4))*R(1,igork1,id(i,4))
     &           -u12(id(i,4),4) *R(2,igork1,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           +dk4p(i)*
     &           (-conjg(u12(i,4))*R(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igork1,iu(i,4)))
     &           -dk4m(id(i,4))*
     &           (conjg(u12(id(i,4),4))*R(1,igork1,id(i,4))
     &           +u11(id(i,4),4) *R(2,igork1,id(i,4)))
         enddo

c     

         do igorkov=5,ngorkov
            idirac=igorkov-4
            igork1=gamin(4,idirac)+4
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           +dk4m(i)*
     &           (u11(i,4)*R(1,igork1,iu(i,4))
     &           +u12(i,4)*R(2,igork1,iu(i,4)))
     &           -dk4p(id(i,4))*
     &           (conjg(u11(id(i,4),4))*R(1,igork1,id(i,4))
     &           -u12(id(i,4),4) *R(2,igork1,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           +dk4m(i)*
     &           (-conjg(u12(i,4))*R(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igork1,iu(i,4)))
     &           -dk4p(id(i,4))*
     &           (conjg(u12(id(i,4),4))*R(1,igork1,id(i,4))
     &           +u11(id(i,4),4) *R(2,igork1,id(i,4)))
         enddo
      enddo
c$omp end parallel do

c     
      return
      end
c***********************************************************************
      subroutine dslashd(Phi,R,u11,u12)
c
c     calculates Phi = Mdagger*R
c
      implicit none
c      include 'omp_lib.h'
      include "precision.h"
      include "sizes.h"
c     include common block definition
      include "common_mat.h"
      include "common_para.h"
      include "common_diquark.h"
      include "common_neighb.h"
      include "common_dirac.h"
      
      complex(kind=cmplxkind) u11(kvol+halo,ndim),u12(kvol+halo,ndim)
      complex(kind=cmplxkind) Phi(2,8,kvol+halo),R(2,8,kvol+halo)

      integer igorkov,ic,i,idirac,igork,mu,igork1

c     write(6,*) 'hi from dslashd'
c

c
c     Makes reference to:
c
c     R(.,.,iu(i,mu)), ncpt = 16, need zdnhaloswapall
c     R(.,.,id(i,mu)), ncpt = 16, need zuphaloswapall
c
c     ie for R, need zhaloswapall
c
c     Also, u(id(i,mu),mu) ncpt = 1
c
c     need to do a zuphaloswapdir(mu) for u(mu)
c
c     Also dk4p(id(i,4)) and dk4m(id(i,4)), ncpt = 1
c
c     need to do a cuphaloswapdir(4) SINCE THIS IS A REAL ARRAY!
c     
c     
c

      call zhaloswapall(R, 16)

      do mu = 1, ndim
        call zuphaloswapdir(u11(1,mu), 1, mu)
        call zuphaloswapdir(u12(1,mu), 1, mu)
      end do

      call cuphaloswapdir(dk4p,1,4)
      call cuphaloswapdir(dk4m,1,4)

c$omp parallel do default(none) shared(gamin,dk4p,jqq,
c$omp& dk4m,u11,u12,R,iu,id,akappa,gamval,Phi) private(idirac,
c$omp& igork1, igork)
c     mass term
      do i=1,kvol
         do igorkov=1,ngorkov
            do ic=1,nc
               Phi(ic,igorkov,i)=R(ic,igorkov,i)
            enddo
         enddo
c     
c     diquark term (antihermitian)
c     
         do idirac=1,4
            igork=idirac+4
            do ic=1,nc
               Phi(ic,idirac,i)=Phi(ic,idirac,i)-
     &              conjg(jqq)*gamval(5,idirac)*R(ic,igork,i)
               Phi(ic,igork,i)=Phi(ic,igork,i)+
     &              jqq*gamval(5,idirac)*R(ic,idirac,i)
            enddo
         enddo
c
c     Wilson term
c
         do mu=1,3
            do igorkov=1,ngorkov
               Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &              -akappa*(      u11(i,mu)*R(1,igorkov,iu(i,mu))
     &              +u12(i,mu)*R(2,igorkov,iu(i,mu))
     &              +conjg(u11(id(i,mu),mu))*R(1,igorkov,id(i,mu))
     &              -u12(id(i,mu),mu) *R(2,igorkov,id(i,mu)))
               Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &              -akappa*(-conjg(u12(i,mu))*R(1,igorkov,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,igorkov,iu(i,mu))
     &              +conjg(u12(id(i,mu),mu))*R(1,igorkov,id(i,mu))
     &              +u11(id(i,mu),mu) *R(2,igorkov,id(i,mu)))
            enddo
         enddo
c
c     Dirac term
c
         do mu=1,3
            do igorkov=1,ngorkov
               idirac=mod((igorkov-1),4)+1
               if(igorkov.le.4)then
                  igork1=gamin(mu,idirac)
               else
                  igork1=gamin(mu,idirac)+4
               endif
               Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &              -gamval(mu,idirac)*
     &              (          u11(i,mu)*R(1,igork1,iu(i,mu))
     &              +u12(i,mu)*R(2,igork1,iu(i,mu))
     &              -conjg(u11(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              +u12(id(i,mu),mu) *R(2,igork1,id(i,mu)))
               Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &              -gamval(mu,idirac)*
     &              (-conjg(u12(i,mu))*R(1,igork1,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,igork1,iu(i,mu))
     &              -conjg(u12(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              -u11(id(i,mu),mu) *R(2,igork1,id(i,mu)))
            enddo
         enddo
c     
c  Timelike Wilson term
c
         do igorkov=1,4
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           -dk4m(i)*(u11(i,4)*R(1,igorkov,iu(i,4))
     &           +u12(i,4)*R(2,igorkov,iu(i,4)))
     &           -dk4p(id(i,4))*(conjg(u11(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           -u12(id(i,4),4) *R(2,igorkov,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           -dk4m(i)*(-conjg(u12(i,4))*R(1,igorkov,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igorkov,iu(i,4)))
     &           -dk4p(id(i,4))*(conjg(u12(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           +u11(id(i,4),4) *R(2,igorkov,id(i,4)))
         enddo
c     
         do igorkov=5,ngorkov
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           -dk4p(i)*(u11(i,4)*R(1,igorkov,iu(i,4))
     &           +u12(i,4)*R(2,igorkov,iu(i,4)))
     &           -dk4m(id(i,4))*(conjg(u11(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           -u12(id(i,4),4) *R(2,igorkov,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           -dk4p(i)*(-conjg(u12(i,4))*R(1,igorkov,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igorkov,iu(i,4)))
     &           -dk4m(id(i,4))*(conjg(u12(id(i,4),4))*
     &           R(1,igorkov,id(i,4))
     &           +u11(id(i,4),4) *R(2,igorkov,id(i,4)))
         enddo
c     
c  Timelike Dirac term
c     

         do igorkov=1,4
            idirac=igorkov
            igork1=gamin(4,idirac)
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           -dk4m(i)*
     &           (u11(i,4)*R(1,igork1,iu(i,4))
     &           +u12(i,4)*R(2,igork1,iu(i,4)))
     &           +dk4p(id(i,4))*
     &           (conjg(u11(id(i,4),4))*R(1,igork1,id(i,4))
     &           -u12(id(i,4),4) *R(2,igork1,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           -dk4m(i)*
     &           (-conjg(u12(i,4))*R(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igork1,iu(i,4)))
     &           +dk4p(id(i,4))*
     &           (conjg(u12(id(i,4),4))*R(1,igork1,id(i,4))
     &           +u11(id(i,4),4) *R(2,igork1,id(i,4)))
         enddo
c     
         do igorkov=5,ngorkov
            idirac=igorkov-4
            igork1=gamin(4,idirac)+4
            Phi(1,igorkov,i)=Phi(1,igorkov,i)
     &           -dk4p(i)*
     &           (u11(i,4)*R(1,igork1,iu(i,4))
     &           +u12(i,4)*R(2,igork1,iu(i,4)))
     &           +dk4m(id(i,4))*
     &           (conjg(u11(id(i,4),4))*R(1,igork1,id(i,4))
     &           -u12(id(i,4),4) *R(2,igork1,id(i,4)))
            Phi(2,igorkov,i)=Phi(2,igorkov,i)
     &           -dk4p(i)*
     &           (-conjg(u12(i,4))*R(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igork1,iu(i,4)))
     &           +dk4m(id(i,4))*
     &           (conjg(u12(id(i,4),4))*R(1,igork1,id(i,4))
     &           +u11(id(i,4),4) *R(2,igork1,id(i,4)))
         enddo
      enddo
c$omp end parallel do

c     
      return
      end
c***********************************************************************
      subroutine hdslash(Phi,R,u11,u12)
c
c     calculates Phi = M*R
c
      implicit none
c      include 'omp_lib.h'
      include "precision.h"
      include "sizes.h"
c     include common block definition
      include "common_mat.h"
      include "common_para.h"
      include "common_neighb.h"
      include "common_dirac.h"
     
      complex(kind=cmplxkind) u11(kvol+halo,ndim),u12(kvol+halo,ndim)
      complex(kind=cmplxkind) Phi(2,4,kvol+halo),R(2,4,kvol+halo)

      integer idirac,ic,i,mu,igork1
c     write(6,*) 'hi from hdslash'
c

c
c     Makes references to
c
c     R(.,.,iu(i,mu)) and R(.,.,id(i,mu)) with ncpt = 8  ** NOT 16 **
c
c     need a zhaloswapall
c
c     also u(id(i,mu),mu)
c
c     so need a zuphaloswapdir(mu) for u(mu)
c
c     Also dk4p(id(i,4)) and dk4m(id(i,4)), ncpt = 1
c
c     need to do a cuphaloswapdir(4)
c
c     Actually, this routine does not seem to access dk4p(id(),..) so
c     one of those halo swaps could be omitted
c


      call zhaloswapall(R, 8)

      do mu = 1, ndim
        call zuphaloswapdir(u11(1,mu), 1, mu)
        call zuphaloswapdir(u12(1,mu), 1, mu)
      end do

      call cuphaloswapdir(dk4p,1,4)
      call cuphaloswapdir(dk4m,1,4)

c$omp parallel do default(none) shared(phi,
c$omp& akappa,u11,u12,R,id,dk4p,iu,dk4m,gamval,gamin)
c$omp& private(idirac,
c$omp& igork1)
c     mass term
      do i=1,kvol
         do idirac=1,ndirac
            do ic=1,nc
               Phi(ic,idirac,i)=R(ic,idirac,i)
            enddo
         enddo
c     
c     Wilson term
c     
         do idirac=1,ndirac
            do mu=1,3
               Phi(1,idirac,i)=Phi(1,idirac,i)
     &              -akappa*(      u11(i,mu)*R(1,idirac,iu(i,mu))
     &              +u12(i,mu)*R(2,idirac,iu(i,mu))
     &              +conjg(u11(id(i,mu),mu))*R(1,idirac,id(i,mu))
     &              -u12(id(i,mu),mu) *R(2,idirac,id(i,mu)))
               Phi(2,idirac,i)=Phi(2,idirac,i)
     &              -akappa*(-conjg(u12(i,mu))*R(1,idirac,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,idirac,iu(i,mu))
     &              +conjg(u12(id(i,mu),mu))*R(1,idirac,id(i,mu))
     &              +u11(id(i,mu),mu) *R(2,idirac,id(i,mu)))
            enddo
c     
            do mu=1,3
               igork1=gamin(mu,idirac)
               Phi(1,idirac,i)=Phi(1,idirac,i)
     &              +gamval(mu,idirac)*
     &              (          u11(i,mu)*R(1,igork1,iu(i,mu))
     &              +u12(i,mu)*R(2,igork1,iu(i,mu))
     &              -conjg(u11(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              +u12(id(i,mu),mu) *R(2,igork1,id(i,mu)))
               Phi(2,idirac,i)=Phi(2,idirac,i)
     &              +gamval(mu,idirac)*
     &              (-conjg(u12(i,mu))*R(1,igork1,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,igork1,iu(i,mu))
     &              -conjg(u12(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              -u11(id(i,mu),mu) *R(2,igork1,id(i,mu)))
            enddo
c     
            Phi(1,idirac,i)=Phi(1,idirac,i)
     &           -dk4p(i)*(u11(i,4)*R(1,idirac,iu(i,4))
     &           +u12(i,4)*R(2,idirac,iu(i,4)))
     &           -dk4m(id(i,4))*(conjg(u11(id(i,4),4))*
     &           R(1,idirac,id(i,4))
     &           -u12(id(i,4),4) *R(2,idirac,id(i,4)))
            Phi(2,idirac,i)=Phi(2,idirac,i)
     &           -dk4p(i)*(-conjg(u12(i,4))*R(1,idirac,iu(i,4))
     &           +conjg(u11(i,4))*R(2,idirac,iu(i,4)))
     &           -dk4m(id(i,4))*(conjg(u12(id(i,4),4))*
     &           R(1,idirac,id(i,4))
     &           +u11(id(i,4),4) *R(2,idirac,id(i,4)))
c     
            igork1=gamin(4,idirac)
            Phi(1,idirac,i)=Phi(1,idirac,i)
     &           +dk4p(i)*
     &           (u11(i,4)*R(1,igork1,iu(i,4))
     &           +u12(i,4)*R(2,igork1,iu(i,4)))
     &           -dk4m(id(i,4))*
     &           (conjg(u11(id(i,4),4))*R(1,igork1,id(i,4))
     &           -u12(id(i,4),4) *R(2,igork1,id(i,4)))
            Phi(2,idirac,i)=Phi(2,idirac,i)
     &           +dk4p(i)*
     &           (-conjg(u12(i,4))*R(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igork1,iu(i,4)))
     &           -dk4m(id(i,4))*
     &           (conjg(u12(id(i,4),4))*R(1,igork1,id(i,4))
     &           +u11(id(i,4),4) *R(2,igork1,id(i,4)))
         enddo
      enddo
c$omp end parallel do
c     
      return
      end
c***********************************************************************
      subroutine hdslashd(Phi,R,u11,u12)
c
c     calculates Phi = Mdagger*R
c
      implicit none
c      include 'omp_lib.h'
      include "precision.h"
      include "sizes.h"
c     include common block definition
      include "common_mat.h"
      include "common_para.h"
      include "common_neighb.h"
      include "common_dirac.h"
      
      complex(kind=cmplxkind) u11(kvol+halo,ndim),u12(kvol+halo,ndim)
      complex(kind=cmplxkind) Phi(2,4,kvol+halo),R(2,4,kvol+halo)

      integer idirac,ic,i,mu,igork1
c     write(6,*) 'hi from hdslashd'
c

c
c     Makes references to
c
c     R(.,.,iu(i,mu)) and R(.,.,id(i,mu)) with ncpt = 8  ** NOT 16 **
c
c     need a zhaloswapall
c
c     also u(id(i,mu),mu)
c
c     so need a zuphaloswapdir(mu) for u(mu)
c
c     Also dk4p(id(i,4)) and dk4m(id(i,4)), ncpt = 1
c
c     need to do a cuphaloswapdir(4)


      call zhaloswapall(R, 8)

      do mu = 1, 4
        call zuphaloswapdir(u11(1,mu), 1, mu)
        call zuphaloswapdir(u12(1,mu), 1, mu)
      end do

      call cuphaloswapdir(dk4p,1,4)
      call cuphaloswapdir(dk4m,1,4)


c$omp parallel do default(none) shared(gamin,dk4p,
c$omp& dk4m,u11,u12,R,iu,id,akappa,gamval,Phi) private(idirac,
c$omp& igork1)
c     mass term
      do i=1,kvol
         do idirac=1,ndirac
            do ic=1,nc
               Phi(ic,idirac,i)=R(ic,idirac,i)
            enddo
         enddo
c     
c     Wilson term
c     
         do idirac=1,ndirac
            do mu=1,3
               Phi(1,idirac,i)=Phi(1,idirac,i)
     &              -akappa*(      u11(i,mu)*R(1,idirac,iu(i,mu))
     &              +u12(i,mu)*R(2,idirac,iu(i,mu))
     &              +conjg(u11(id(i,mu),mu))*R(1,idirac,id(i,mu))
     &              -u12(id(i,mu),mu) *R(2,idirac,id(i,mu)))
               Phi(2,idirac,i)=Phi(2,idirac,i)
     &              -akappa*(-conjg(u12(i,mu))*R(1,idirac,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,idirac,iu(i,mu))
     &              +conjg(u12(id(i,mu),mu))*R(1,idirac,id(i,mu))
     &              +u11(id(i,mu),mu) *R(2,idirac,id(i,mu)))
            enddo
c     
            do mu=1,3
               igork1=gamin(mu,idirac)
               Phi(1,idirac,i)=Phi(1,idirac,i)
     &              -gamval(mu,idirac)*
     &              (          u11(i,mu)*R(1,igork1,iu(i,mu))
     &              +u12(i,mu)*R(2,igork1,iu(i,mu))
     &              -conjg(u11(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              +u12(id(i,mu),mu) *R(2,igork1,id(i,mu)))
               Phi(2,idirac,i)=Phi(2,idirac,i)
     &              -gamval(mu,idirac)*
     &              (-conjg(u12(i,mu))*R(1,igork1,iu(i,mu))
     &              +conjg(u11(i,mu))*R(2,igork1,iu(i,mu))
     &              -conjg(u12(id(i,mu),mu))*R(1,igork1,id(i,mu))
     &              -u11(id(i,mu),mu) *R(2,igork1,id(i,mu)))
            enddo
c     
            Phi(1,idirac,i)=Phi(1,idirac,i)
     &           -dk4m(i)*(u11(i,4)*R(1,idirac,iu(i,4))
     &           +u12(i,4)*R(2,idirac,iu(i,4)))
     &           -dk4p(id(i,4))*(conjg(u11(id(i,4),4))*
     &           R(1,idirac,id(i,4))
     &           -u12(id(i,4),4) *R(2,idirac,id(i,4)))
            Phi(2,idirac,i)=Phi(2,idirac,i)
     &           -dk4m(i)*(-conjg(u12(i,4))*R(1,idirac,iu(i,4))
     &           +conjg(u11(i,4))*R(2,idirac,iu(i,4)))
     &           -dk4p(id(i,4))*(conjg(u12(id(i,4),4))*
     &           R(1,idirac,id(i,4))
     &           +u11(id(i,4),4) *R(2,idirac,id(i,4)))
c     
            igork1=gamin(4,idirac)
            Phi(1,idirac,i)=Phi(1,idirac,i)
     &           -dk4m(i)*
     &           (u11(i,4)*R(1,igork1,iu(i,4))
     &           +u12(i,4)*R(2,igork1,iu(i,4)))
     &           +dk4p(id(i,4))*
     &           (conjg(u11(id(i,4),4))*R(1,igork1,id(i,4))
     &           -u12(id(i,4),4) *R(2,igork1,id(i,4)))
            Phi(2,idirac,i)=Phi(2,idirac,i)
     &           -dk4m(i)*
     &           (-conjg(u12(i,4))*R(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*R(2,igork1,iu(i,4)))
     &           +dk4p(id(i,4))*
     &           (conjg(u12(id(i,4),4))*R(1,igork1,id(i,4))
     &           +u11(id(i,4),4) *R(2,igork1,id(i,4)))
         enddo
      enddo
c$omp end parallel do
      
c     
      return
      end

