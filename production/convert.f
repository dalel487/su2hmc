      program convert

      parameter(ksize=8,ksizet=16,kvol3=ksize*ksize*ksize,
     #     kvol=kvol3*ksizet)
      parameter(ksizet2=8,kvolsmall=ksizet2*ksize*ksize*ksize)
      complex*16 u11,u12
      common/gauge/ u11(kvol,4),u12(kvol,4),seed

      integer seed

      complex tmp11(kvolsmall,4),tmp12(kvolsmall,4)

      open(unit=10,file='con',status='unknown',form='unformatted')

      read (10) u11,u12,seed

      open(unit=31,file='con-n2',status='unknown',form='unformatted')

      write(6,*) seed

      do mu=1,4
         do it=1,ksizet2
            do is=1,kvol3
               i = is + kvol3*(it-1)
               tmp11(i,mu) = u11(i,mu)
               tmp12(i,mu) = u12(i,mu)
            enddo
         enddo
      enddo

      write(31) tmp11,tmp12

      do mu=1,4
         do it=1,ksizet2
            do is=1,kvol3
               i = is + kvol3*(it-1)
               i2 = is + kvol3*(it-1+ksizet2)
               tmp11(i,mu) = u11(i2,mu)
               tmp12(i,mu) = u12(i2,mu)
            enddo
         enddo
      enddo

      write(31) tmp11,tmp12

      write(31) seed
      close(31)

      stop
      end
