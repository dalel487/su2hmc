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
