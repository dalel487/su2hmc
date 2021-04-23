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
