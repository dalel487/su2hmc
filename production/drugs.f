c
c     dSdpi=dSdpi-Re(X1*(d(Mdagger)dp)*X2) -- Yikes!
c    we're gonna need drugs for this one......
c

c
c  Makes references to X1(.,.,iu(i,mu)) AND X2(.,.,iu(i,mu))
c  as a result, need to swap the DOWN halos in all dirs for
c  both these arrays, each of which has 8 cpts
c

      call zdnhaloswapall(X1, 8)
      call zdnhaloswapall(X2, 8)

      do 21 i=1,kvol
      do 21 mu=1,3
      do 21 idirac=1,ndirac
      igork1=gamin(mu,idirac)
c
       dSdpi(mu,1,i)=dSdpi(mu,1,i)+akappa*real(zi*
     &(conjg(X1(1,idirac,i))*
     & (-conjg(u12(i,mu))*X2(1,idirac,iu(i,mu))
     &  +conjg(u11(i,mu))*X2(2,idirac,iu(i,mu)))
     &+conjg(X1(1,idirac,iu(i,mu)))*
     &       ( u12(i,mu) *X2(1,idirac,i)
     &  -conjg(u11(i,mu))*X2(2,idirac,i))
     &+conjg(X1(2,idirac,i))*
     &        (u11(i,mu) *X2(1,idirac,iu(i,mu))
     &        +u12(i,mu) *X2(2,idirac,iu(i,mu)))
     &+conjg(X1(2,idirac,iu(i,mu)))*
     &       (-u11(i,mu) *X2(1,idirac,i)
     &  -conjg(u12(i,mu))*X2(2,idirac,i))))
       dSdpi(mu,1,i)=dSdpi(mu,1,i)+real(zi*gamval(mu,idirac)*
     &(conjg(X1(1,idirac,i))*
     & (-conjg(u12(i,mu))*X2(1,igork1,iu(i,mu))
     &  +conjg(u11(i,mu))*X2(2,igork1,iu(i,mu)))
     &+conjg(X1(1,idirac,iu(i,mu)))*
     &       (-u12(i,mu) *X2(1,igork1,i)
     &  +conjg(u11(i,mu))*X2(2,igork1,i))
     &+conjg(X1(2,idirac,i))*
     &        (u11(i,mu) *X2(1,igork1,iu(i,mu))
     &        +u12(i,mu) *X2(2,igork1,iu(i,mu)))
     &+conjg(X1(2,idirac,iu(i,mu)))*
     &        (u11(i,mu) *X2(1,igork1,i)
     &  +conjg(u12(i,mu))*X2(2,igork1,i))))
c
      dSdpi(mu,2,i)=dSdpi(mu,2,i)+akappa*real(
     &(conjg(X1(1,idirac,i))*
     & (-conjg(u12(i,mu))*X2(1,idirac,iu(i,mu))
     &  +conjg(u11(i,mu))*X2(2,idirac,iu(i,mu)))
     &+conjg(X1(1,idirac,iu(i,mu)))*
     &       (-u12(i,mu) *X2(1,idirac,i)
     &  -conjg(u11(i,mu))*X2(2,idirac,i))
     &+conjg(X1(2,idirac,i))*
     &       (-u11(i,mu) *X2(1,idirac,iu(i,mu))
     &        -u12(i,mu) *X2(2,idirac,iu(i,mu)))
     &+conjg(X1(2,idirac,iu(i,mu)))*
     &        (u11(i,mu) *X2(1,idirac,i)
     &  -conjg(u12(i,mu))*X2(2,idirac,i))))
      dSdpi(mu,2,i)=dSdpi(mu,2,i)+real(gamval(mu,idirac)*
     &(conjg(X1(1,idirac,i))*
     & (-conjg(u12(i,mu))*X2(1,igork1,iu(i,mu))
     &  +conjg(u11(i,mu))*X2(2,igork1,iu(i,mu)))
     &+conjg(X1(1,idirac,iu(i,mu)))*
     &        (u12(i,mu) *X2(1,igork1,i)
     &  +conjg(u11(i,mu))*X2(2,igork1,i))
     &+conjg(X1(2,idirac,i))*
     &       (-u11(i,mu) *X2(1,igork1,iu(i,mu))
     &        -u12(i,mu) *X2(2,igork1,iu(i,mu)))
     &+conjg(X1(2,idirac,iu(i,mu)))*
     &       (-u11(i,mu) *X2(1,igork1,i)
     &  +conjg(u12(i,mu))*X2(2,igork1,i))))
c
      dSdpi(mu,3,i)=dSdpi(mu,3,i)+akappa*real(zi*
     &(conjg(X1(1,idirac,i))*
     &        (u11(i,mu) *X2(1,idirac,iu(i,mu))
     &        +u12(i,mu) *X2(2,idirac,iu(i,mu)))
     &+conjg(X1(1,idirac,iu(i,mu)))*
     & (-conjg(u11(i,mu))*X2(1,idirac,i)
     &        -u12(i,mu) *X2(2,idirac,i))
     &+conjg(X1(2,idirac,i))*
     &  (conjg(u12(i,mu))*X2(1,idirac,iu(i,mu))
     &  -conjg(u11(i,mu))*X2(2,idirac,iu(i,mu)))
     &+conjg(X1(2,idirac,iu(i,mu)))*
     & (-conjg(u12(i,mu))*X2(1,idirac,i)
     &        +u11(i,mu) *X2(2,idirac,i))))
      dSdpi(mu,3,i)=dSdpi(mu,3,i)+real(zi*gamval(mu,idirac)*
     &(conjg(X1(1,idirac,i))*
     &        (u11(i,mu) *X2(1,igork1,iu(i,mu))
     &        +u12(i,mu) *X2(2,igork1,iu(i,mu)))
     &+conjg(X1(1,idirac,iu(i,mu)))*
     &  (conjg(u11(i,mu))*X2(1,igork1,i)
     &        +u12(i,mu) *X2(2,igork1,i))
     &+conjg(X1(2,idirac,i))*
     &  (conjg(u12(i,mu))*X2(1,igork1,iu(i,mu))
     &  -conjg(u11(i,mu))*X2(2,igork1,iu(i,mu)))
     &+conjg(X1(2,idirac,iu(i,mu)))*
     &  (conjg(u12(i,mu))*X2(1,igork1,i)
     &        -u11(i,mu) *X2(2,igork1,i))))
21     continue
c
c 4th direction is special
      do 214 idirac=1,ndirac
      igork1=gamin(4,idirac)
      do 214 i=1,kvol
      dSdpi(4,1,i)=dSdpi(4,1,i)+real(zi*
     &(conjg(X1(1,idirac,i))*
     & (dk4m(i)*(-conjg(u12(i,4))*X2(1,idirac,iu(i,4))
     &           +conjg(u11(i,4))*X2(2,idirac,iu(i,4))))
     &+conjg(X1(1,idirac,iu(i,4)))*
     & (dk4p(i)*      (+u12(i,4) *X2(1,idirac,i)
     &           -conjg(u11(i,4))*X2(2,idirac,i)))
     &+conjg(X1(2,idirac,i))*
     & (dk4m(i)*       (u11(i,4) *X2(1,idirac,iu(i,4))
     &                 +u12(i,4) *X2(2,idirac,iu(i,4))))
     &+conjg(X1(2,idirac,iu(i,4)))*
     & (dk4p(i)*      (-u11(i,4) *X2(1,idirac,i)
     &           -conjg(u12(i,4))*X2(2,idirac,i)))))
      dSdpi(4,1,i)=dSdpi(4,1,i)+real(zi*
     &(conjg(X1(1,idirac,i))*
     & (dk4m(i)*(-conjg(u12(i,4))*X2(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*X2(2,igork1,iu(i,4))))
     &+conjg(X1(1,idirac,iu(i,4)))*
     & (-dk4p(i)*       (u12(i,4) *X2(1,igork1,i)
     &            -conjg(u11(i,4))*X2(2,igork1,i)))
     &+conjg(X1(2,idirac,i))*
     & (dk4m(i)*       (u11(i,4) *X2(1,igork1,iu(i,4))
     &                 +u12(i,4) *X2(2,igork1,iu(i,4))))
     &+conjg(X1(2,idirac,iu(i,4)))*
     & (-dk4p(i)*      (-u11(i,4) *X2(1,igork1,i)
     &            -conjg(u12(i,4))*X2(2,igork1,i)))))
c
      dSdpi(4,2,i)=dSdpi(4,2,i)+real(
     & conjg(X1(1,idirac,i))*
     & (dk4m(i)*(-conjg(u12(i,4))*X2(1,idirac,iu(i,4))
     &           +conjg(u11(i,4))*X2(2,idirac,iu(i,4))))
     &+conjg(X1(1,idirac,iu(i,4)))*
     & (dk4p(i)*      (-u12(i,4) *X2(1,idirac,i)
     &           -conjg(u11(i,4))*X2(2,idirac,i)))
     &+conjg(X1(2,idirac,i))*
     & (dk4m(i)*      (-u11(i,4) *X2(1,idirac,iu(i,4))
     &                 -u12(i,4) *X2(2,idirac,iu(i,4))))
     &+conjg(X1(2,idirac,iu(i,4)))*
     & (dk4p(i)*      ( u11(i,4) *X2(1,idirac,i)
     &           -conjg(u12(i,4))*X2(2,idirac,i))))
      dSdpi(4,2,i)=dSdpi(4,2,i)+real(
     & (conjg(X1(1,idirac,i))*
     & (dk4m(i)*(-conjg(u12(i,4))*X2(1,igork1,iu(i,4))
     &           +conjg(u11(i,4))*X2(2,igork1,iu(i,4))))
     &+conjg(X1(1,idirac,iu(i,4)))*
     & (-dk4p(i)*      (-u12(i,4) *X2(1,igork1,i)
     &            -conjg(u11(i,4))*X2(2,igork1,i)))
     &+conjg(X1(2,idirac,i))*
     & (dk4m(i)*      (-u11(i,4) *X2(1,igork1,iu(i,4))
     &                 -u12(i,4) *X2(2,igork1,iu(i,4))))
     &+conjg(X1(2,idirac,iu(i,4)))*
     & (-dk4p(i)*       (u11(i,4) *X2(1,igork1,i)
     &            -conjg(u12(i,4))*X2(2,igork1,i)))))
c
      dSdpi(4,3,i)=dSdpi(4,3,i)+real(zi*
     &(conjg(X1(1,idirac,i))*
     & (dk4m(i)*       (u11(i,4) *X2(1,idirac,iu(i,4))
     &                 +u12(i,4) *X2(2,idirac,iu(i,4))))
     &+conjg(X1(1,idirac,iu(i,4)))*
     & (dk4p(i)*(-conjg(u11(i,4))*x2(1,idirac,i)
     &                 -u12(i,4) *X2(2,idirac,i)))
     &+conjg(X1(2,idirac,i))*
     & (dk4m(i)* (conjg(u12(i,4))*X2(1,idirac,iu(i,4))
     &           -conjg(u11(i,4))*X2(2,idirac,iu(i,4))))
     &+conjg(X1(2,idirac,iu(i,4)))*
     & (dk4p(i)*(-conjg(u12(i,4))*X2(1,idirac,i)
     &                 +u11(i,4) *X2(2,idirac,i)))))
      dSdpi(4,3,i)=dSdpi(4,3,i)+real(zi*
     &(conjg(X1(1,idirac,i))*
     & (dk4m(i)*       (u11(i,4) *X2(1,igork1,iu(i,4))
     &                 +u12(i,4) *X2(2,igork1,iu(i,4))))
     &+conjg(X1(1,idirac,iu(i,4)))*
     & (-dk4p(i)*(-conjg(u11(i,4))*x2(1,igork1,i)
     &                  -u12(i,4) *X2(2,igork1,i)))
     &+conjg(X1(2,idirac,i))*
     & (dk4m(i)* (conjg(u12(i,4))*X2(1,igork1,iu(i,4))
     &           -conjg(u11(i,4))*X2(2,igork1,iu(i,4))))
     &+conjg(X1(2,idirac,iu(i,4)))*
     & (-dk4p(i)*(-conjg(u12(i,4))*X2(1,igork1,i)
     &                  +u11(i,4) *X2(2,igork1,i)))))
214   continue
c 
3     continue
c
      return
      end
