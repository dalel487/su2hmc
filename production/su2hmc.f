       program su2hmc
c*******************************************************************
c    Hybrid Monte Carlo algorithm for Two Color QCD with Wilson-Gor'kov fermions
c    based on the algorithm of Duane et al. Phys. Lett. B195 (1987) 216. 
c
c    There is "up/down partitioning": each update requires
c    one operation of congradq on complex*16 vectors to determine
c    (Mdagger M)**-1  Phi where Phi has dimension 4*kvol*nc*Nf - 
c    The matrix M is the Wilson matrix for a single flavor
c    there is no extra species doubling as a result
c
c    matrix multiplies done using routines hdslash and hdslashd
c
c    Hence, the number of lattice flavors Nf is related to the
c    number of continuum flavors N_f by
c                  N_f = 2 * Nf
c
c    Fermion expectation values are measured using a noisy estimator.
c    on the Wilson-Gor'kov matrix, which has dimension 8*kvol*nc*Nf
c    inversions done using congradp, and matrix multiplies with dslash,
c    dslashd
c
c    trajectory length is random with mean dt*iterl
c    The code runs for a fixed number iter2 of trajectories.
c
c    Phi: pseudofermion field 
c    bmass: bare fermion mass 
c    fmu: chemical potential 
c    actiona: running average of total action

c    Fermion expectation values are measured using a noisy estimator.
c    outputs:
c    fort.11   psibarpsi, energy density, baryon density
c    fort.12   spatial plaquette, temporal plaquette, Polyakov line
c    fort.13   real<qq>, real <qbar qbar>, imag <qq>= imag<qbar qbar>
c
c                                               SJH March 2005
c
c     Hybrid code, P.Giudice, May 2013
c
c*******************************************************************
      implicit none
c     include 'omp_lib.h'
      include "precision.h"
      include "sizes.h"

c     include common block definitions - header file for each label 
      include "common_mat.h"
      include "common_gauge.h" 
      include "common_trial.h"
      include "common_pseud.h"
      include "common_para.h"
      include "common_neighb.h"
      include "common_param.h"
      include "common_diquark.h"
      include "common_trans.h"
      include "common_dum1.h"

      complex(kind=cmplxkind) zi,qq,qbqb
      complex(kind=cmplxkind) a11,a12,b11,b12
      real(kind=realkind) dSdpi(ndim,3,kvol+halo)

      integer istart,iread,iwrite,iprint,icheck,iterl,iter2
      integer naccp,ipbp,itot,isweep,na,igorkov,ic,i,idirac
      integer iadj,mu,iter,itercg
      real(kind=realkind) dt,ajq,delb,traj,proby,actiona
      real(kind=realkind) vel2a,pbpa,endenfa,denfa,yav,yyav
      real(kind=realkind) athq,h0,s0,action,d,aaa,ccc,sss
      real(kind=realkind) ytest,ranf,par_granf,h1,s1,dh,ds,y,x,vel2,pbp
      real(kind=realkind) endenf,denf,hg,avplaqs,avplaqt,poly,atraj
      real(kind=realkind) av_for,elapsed,start_time

      complex(kind=cmplxkind) usum
      real(kind=realkind) norm
      integer idim, rank, ranksize

c
c Start parallel code
c

      call par_begin()

c*******************************************************************
c$omp parallel
      call mpi_comm_rank(mpi_comm_world, rank, ierr)
      call mpi_comm_size(mpi_comm_world, ranksize, ierr)
      
c      if(rank.eq.0 .and. omp_get_thread_num() .eq. 0) then
c         write(*,*) 'Num of Threads: ', omp_get_num_threads()
c         write(*,*) 'Num of Tasks : ', ranksize
c      endif
      flush(6)
c$omp end parallel

      write(*,871) rank, pu(1), pu(2), pu(3), pu(4),
     &pd(1), pd(2), pd(3), pd(4)
	flush(6)
871	format("rank:", 1x, i,1x, "pu:",1x,i,1x,i,1x,i,1x,i,1x,
     &"pd:",1x,i,1x,i,1x,i,1x,i) 
c*******************************************************************            

c*******************************************************************
c     input
c*******************************************************************
      ibound=-1

c cj   istart.lt.0 : start from tape
      istart=2
c      istart=-1
c cj
c     iread=1
      iread=0
      iwrite=1
      iprint=1  !measures
      icheck=5  !save conf
      zi=(0.0,1.0)
      tpi=2.0*acos(-1.0)
c*******************************************************************
c     end of input
c*******************************************************************
      if (ismaster) then
        open(unit=7,file='output',status='unknown')
        open(unit=25,file='midout',status='unknown')
c        open(unit=98,file='control',status='unknown')
        read(25,*) dt,beta,akappa,ajq,athq,fmu,delb,iterl,iter2
        close(25)
      end if

      seed=967580165
      if(iread.eq.1) then
        if (ismaster) write(*,*) 'Calling par_sread =  ', seed
        call par_sread()
      endif

      call par_ccopy(dt)
      call par_ccopy(beta)
      call par_ccopy(akappa)
      call par_ccopy(ajq)
      call par_ccopy(athq)
      call par_ccopy(fmu)
      call par_ccopy(delb)
      call par_icopy(iterl)
      call par_icopy(iter2)

      jqq=cmplx(ajq,0.0)*exp(cmplx(0.0,athq))
      call par_ranset(seed)
c*******************************************************************
c     initialisation
c     istart.lt.0 : start from tape
c     istart=0    : ordered start
c     istart=1    : random start
c*******************************************************************
      call init(istart)
#ifdef DIAGNOSTICS
!     OMP PARALLEL FOR SIMD COLLAPSE(2)
      DO 9026 MU= 1, NDIM
      DO 9026 I = 1, KVOL
      U11T(I,MU)=U11(I,MU)
      U12T(I,MU)=U12(I,MU)
9026  CONTINUE
      CALL DIAG(ISTART)
      CALL MPI_FINALIZE(IERR)
      GOTO 9025
#endif

c Initial measurements
c
      call polyakov(poly)
      call su2plaq(hg,avplaqs,avplaqt)

c      usum = 0.0
c
c      do idim = 1, ndim
c        do i = 1, kvol
c
c          usum = usum + u11(i, idim)
c          usum = usum + u12(i, idim)
c
c        end do
c      end do
c
c      write(*,*) 'on proc ', procid, ' <u> = ', usum
c      write(*,*) 'on proc ', procid, ' hg, <Ps>, <Pt>, <Poly> = ',
c     1                         hg, avplaqs, avplaqt, poly
c      call flush(6)
c
c      call par_zsum(usum)

      if (ismaster) then
c        write(*,*) '<u> = ', usum
        write(*,*) 'hg, <Ps>, <Pt>, <Poly> = ',
     1              hg, avplaqs, avplaqt, poly
        call flush(6)
      end if

c*******************************************************************
c     loop on beta
c*******************************************************************
c*******************************************************************
c     print heading
c*******************************************************************
      traj=iterl*dt
c     proby=1.0/float(iterl)
      proby=2.5/float(iterl)

      if (ismaster) then

c     write(6, 9001)ksize,ksizet,Nf,dt,traj,iter2,beta,akappa,fmu,
c    &   ajq,athq
      write(7, 9001)ksize,ksizet,Nf,dt,traj,iter2,beta,akappa,fmu,
     &   ajq,athq
9001  format(' ksize=',i3,' ksizet=',i3,' Nf=',i3,/
     1 ,' time step: dt=',f6.4,' trajectory length=',f9.6,/
     1 ,' # trajectories=',i6,' beta=',f9.6,/
     1 ,' kappa =',f9.6,' chemical potential=',f9.6,/
     1 ,' diquark source =',f9.6,' diquark phase angle =',f9.6//)
c     write(6,9004) rescgg,rescga,respbp
      write(7,9004) rescgg,rescga,respbp
9004  format(' Stopping residuals: guidance: ',e7.2,' acceptance: ',
     &     e7.2,' estimator: ',e7.2)
      call ranget(seed)
c     write(6,9002) seed
      write(7,*) 'seed =', seed
      call flush(7)
9002  format(' seed = ',i20)

      end if

c*******************************************************************
c       initialise for averages
c*******************************************************************
      actiona=0.0
      vel2a=0.0
      pbpa=0.0
      endenfa=0.0
      denfa=0.0
      ancg=0.0
      ancgh=0.0
      yav=0.0
      yyav=0.0 
      naccp=0
      ipbp=0
      itot=0
c*******************************************************************
c     start of classical evolution
c*******************************************************************
#ifdef SA3AT
      START_TIME=MPI_WTIME()
#endif
      do 601 isweep=1,iter2
      if (ismaster) then
        write(*,*) 'Starting isweep = ', isweep
        call flush(6)
      end if
c*******************************************************************
c     Pseudofermion heatbath: Phi=Mdagger R, where R is gaussian
c*******************************************************************
      do 2013 na=1,Nf
      do igorkov=1,ngorkov
      do ic=1,nc
      call gauss0(ps)
      do i=1,kvol
      R(ic,igorkov,i)=cmplx(ps(1,i),ps(2,i))
      enddo
      enddo
      enddo

c      call znorm2(norm, R, nc*ngorkov*kvol)
c
c      write(*,*) 'On proc ', procid, ', |R|^2 = ', norm
c      call par_csum(norm)
c
c      if (ismaster) write(*,*) '|R|^2 = ', norm
c
c      call flush(6)

c
      call dslashd(R1,R,u11,u12)
c

c      call znorm2(norm, R1, nc*ngorkov*kvol)
c
c      write(*,*) 'On proc ', procid, ', |R1|^2 = ', norm
c      call par_csum(norm)
c
c      if (ismaster) write(*,*) '|R1|^2 = ', norm
c
c      call flush(6)

      do 2012 i=1,kvol
      do 2012 igorkov=1,ngorkov
      do 2012 ic=1,nc
      Phi(ic,igorkov,i,na)=R1(ic,igorkov,i)
2012  continue
c
c  up/down partitioning - only use the pseudofermions of flavor 1
      do i=1,kvol
      do idirac=1,ndirac
      do ic=1,nc
      X0(ic,idirac,i,na)=R1(ic,idirac,i)
      enddo
      enddo
      enddo
c  
2013  continue
c*******************************************************************
c     heatbath for p
c*******************************************************************
      do iadj=1,nadj
      do mu=1,2
      call gaussp(ps)
      do i=1,kvol
      pp(mu,iadj,i)=ps(1,i)
      pp(mu+2,iadj,i)=ps(2,i)
      enddo
      enddo
      enddo
c*******************************************************************
c     initialise trial fields
c*******************************************************************
      do 2007 mu=1,ndim
      do 2007 i=1,kvol
      u11t(i,mu)=u11(i,mu)
      u12t(i,mu)=u12(i,mu)
2007  continue
c     
c      if (ismaster) write(*,*) 'Calling hamiltonian'
c      call flush(6)

      call hamilton(H0,S0,rescga,isweep)
       
c      write(*,*) 'On proc ', procid, ', H0, S0 = ', H0, S0
      if (ismaster) write(*,*) 'H0, S0 = ', H0, S0
c
c      call flush(6)
      
      if(isweep.eq.1) then
      action=S0/gvol
      endif 
c     goto 501
c*******************************************************************
c      half-step forward for p
c*******************************************************************
      call force(dSdpi,1,rescgg)
	av_for=0
	do 25 i = 1, kvol
	do 25 iadj = 1, nadj
	do 25 mu = 1, ndim
		av_for=av_for+dSdpi(mu,iadj,i)
25	continue
c	write(*,*) "Average force after trial init:", av_for/kmom
      d=dt*0.5
      do 2004 i=1,kvol
      do 2004 iadj=1,nadj
      do 2004 mu=1,ndim
      pp(mu,iadj,i)=pp(mu,iadj,i)-dSdpi(mu,iadj,i)*d
2004  continue
c*******************************************************************
c     start of main loop for classical time evolution
c*******************************************************************
      do 500 iter=1,itermax
        if (ismaster) then
          write(*,*) 'iter = ', iter
          call flush(6)
        end if
c
c  step (i) st(t+dt)=st(t)+p(t+dt/2)*dt;
c
      d=dt
      do 2001 i=1,kvol
      do 2001 mu=1,ndim
      AAA=d*sqrt(pp(mu,1,i)**2+pp(mu,2,i)**2+pp(mu,3,i)**2)
      CCC=cos(AAA)
      SSS=d*sin(AAA)/AAA
      a11=cmplx(CCC,pp(mu,3,i)*SSS)
      a12=cmplx(pp(mu,2,i)*SSS,pp(mu,1,i)*SSS)
      b11=u11t(i,mu)
      b12=u12t(i,mu)
      u11t(i,mu)=a11*b11-a12*conjg(b12)
      u12t(i,mu)=a11*b12+a12*conjg(b11)
2001  continue
      call reunitarise
c
c  step (ii)  p(t+3dt/2)=p(t+dt/2)-dSds(t+dt)*dt (1/2 step on last iteration)
c
      call force(dSdpi,0,rescgg)
#ifdef _DEBUG
      av_for=0
      do 26 i = 1, kvol
      do 26 iadj = 1, nadj
      do 26 mu = 1, ndim
      av_for=av_for+dSdpi(mu,iadj,i)
26    continue
      write(*,*) "Average force after trial update:", av_for/kmom
#endif
c
c test for end of random trajectory
c 
c      ytest=par_granf()
c      call ranfdump(ytest)
c      if((ytest.lt.proby.or.iter.ge.(iterl/5)*6)
c     &       .and.iter.ge.(iterl/5)*4)then
      if(iter.eq.iterl) then
      d=dt*0.5
      do 2005 i=1,kvol
      do 2005 iadj=1,nadj
      do 2005 mu=1,ndim
      pp(mu,iadj,i)=pp(mu,iadj,i)-d*dSdpi(mu,iadj,i)
2005  continue
      itot=itot+iter 
      goto 501
      else
      d=dt
      do 3005 i=1,kvol
      do 3005 iadj=1,nadj
      do 3005 mu=1,ndim
      pp(mu,iadj,i)=pp(mu,iadj,i)-d*dSdpi(mu,iadj,i)
3005  continue
      endif
c 
500   continue
c**********************************************************************
c  Monte Carlo step: accept new fields with probability=
c              min(1,exp(H0-H1))
c**********************************************************************
501   call reunitarise
      call hamilton(H1,S1,rescga,isweep)
      dH=H0-H1
      dS=S0-S1
      if (ismaster) then
         write(7,*) dH,dS
         call flush(7)
         write(*,*) 'dH=',dH,' dS=',dS
         call flush(6)
      endif
      y=exp(dH)      
      yav=yav+y 
      yyav=yyav+y*y 
      if(dH.lt.0.0)then
      x=par_granf()
	write(*,*) "x=", x
c      call ranfdump(x)
      if(x.gt.y)goto 600
      endif
c
c     step accepted: set s=st
c
      if (ismaster) then
        write(*,*) 'New config accepted!'
        call flush(6)
c       write(7,9023) seed
      endif
c     JIS 20100525: write config here to preempt troubles during measurement!
c     JIS 20100525: remove when all is ok....
      if((isweep/icheck)*icheck.eq.isweep)then
        call ranget(seed)
        call par_swrite(isweep)
      end if

      do 2006 mu=1,ndim
      do 2006 i=1,kvol
      u11(i,mu)=u11t(i,mu)
      u12(i,mu)=u12t(i,mu)
2006  continue      
      naccp=naccp+1
      action=S1/gvol
600   continue
      actiona=actiona+action 
      vel2=0.0
      do 457 i=1,kvol
      do 457 iadj=1,nadj
      do 457 mu=1,ndim
      vel2=vel2+pp(mu,iadj,i)*pp(mu,iadj,i)
457   continue
      call par_csum(vel2)
      vel2=vel2/(12*gvol)
      vel2a=vel2a+vel2
c
      if((isweep/iprint)*iprint.eq.isweep)then
      do 2066 mu=1,ndim
      do 2066 i=1,kvol
      u11t(i,mu)=u11(i,mu)
      u12t(i,mu)=u12(i,mu)
2066  continue      
      if (ismaster) then
         write(*,*) 'starting measurements'
         call flush(6)
      endif
      call measure(pbp,endenf,denf,qq,qbqb,respbp,itercg)
      if (ismaster) then
         write(*,*) 'finished measurements'
         call flush(6)
      endif
      pbpa=pbpa+pbp
      endenfa=endenfa+endenf
      denfa=denfa+denf
      ipbp=ipbp+1

      call su2plaq(hg,avplaqs,avplaqt)
      call polyakov(poly)

      if (ismaster) then
      write(11,*) pbp,endenf,denf
      write(13,*) real(qq)
      write(12,*) avplaqs,avplaqt,poly
c     write intermediate data to output in case of crash
      write(7,*) itercg, ancg, ancgh
      call flush(7)
      call flush(11)
      call flush(12)
      call flush(13)
      write(6,*) isweep,':  ',itercg, ancg, ancgh
      write(6,*) isweep,':  ',pbp,endenf,denf
      write(6,*) isweep,':  ',avplaqs,avplaqt,poly
      write(6,*) isweep,':  ',real(qq)
      write(6,*) isweep,':  ',qbqb
      call flush(6)
      endif
c
      if((isweep/icheck)*icheck.eq.isweep)then
      call ranget(seed)
      call par_swrite(isweep)
c     write(7,9023) seed
      endif
      endif
c
601   continue
c*******************************************************************
c     end of main loop
c*******************************************************************
#ifdef SA3AT
      IF(ISMASTER) THEN
      ELAPSED=MPI_WTIME()
      OPEN(653,FILE=,'./Bench_times.csv',ACCESS='APPEND')
      WRITE(653,*) NX,',',NT,',',KVOL,',',NTHREADS,',',ELAPSED,',',
     &ELAPSED/NTRAJ
      CLOSE(653)
      ENDIF
#endif
      actiona=actiona/iter2 
      vel2a=vel2a/iter2 
      pbpa=pbpa/ipbp
      endenfa=endenfa/ipbp
      denfa=denfa/ipbp
      ancg=ancg/(Nf*itot)
      ancgh=ancgh/(Nf*2*iter2)
      yav=yav/iter2
      yyav=yyav/iter2-yav*yav 
      yyav=sqrt(yyav/(iter2-1)) 
      atraj=dt*itot/iter2 

      if (ismaster) then

c*******************************************************************
c     print global averages
c*******************************************************************
c     write(6, 9022) iter2,naccp,atraj,yav,yyav,ancg,ancgh,
c    & pbpa,vel2a,actiona,endenfa,denfa
      write(7, 9022) iter2,naccp,atraj,yav,yyav,ancg,ancgh,
     & pbpa,vel2a,actiona,endenfa,denfa
9022  format(' averages for last ',i6,' trajectories',/ 
     & 1x,' # of acceptances: ',i6,' average trajectory length= ',f8.3/
     & 1x,' <exp-dH>=',e11.4,' +/-',e10.3/
     2 1x,' av. # congrad itr. guidance: ',f9.3,'; acceptance: ',f9.3//
     1 1x,' psibarpsi=',e11.3/
     & 1x,' mean square velocity=',e10.3,'; action per site=',e10.3/
     & 1x,' energy density=',e11.3,'; number density=',e11.3//) 
      write(7, 9024)
      write(7, 9024)
9024  format(1x)
c
      close(7)
      close(11)
      close(13)
      close(13)
      end if
c
      if(iwrite.eq.1) then
      call ranget(seed)
c      call swrite(50)
c      write(7,*) 'seed =', seed
9023  format(' seed = ',i20)
      endif

c
c End parallel code
c

      call par_end()
9025  end
c******************************************************************
c   calculate dSds at each intermediate time
c******************************************************************
      subroutine force(dSdpi,iflag,res1)
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definitions
      include "common_mat.h"
      include "common_neighb.h"
      include "common_para.h"
      include "common_pseud.h"
      include "common_param.h"
      include "common_trial_u11u12.h"
      include "common_dirac.h"
c     these don't exist in master common block definition
      common/vector/X1(2,4,kvol+halo)

      complex(kind=cmplxkind) X2(2,4,kvol+halo)
      complex(kind=cmplxkind) X1,zi
      real(kind=realkind) dSdpi(ndim,3,kvol+halo)
      
      integer iflag,na,idirac,ic,i,itercg,mu,igork1
      real(kind=realkind) res1
c
c     write(6,111)
111   format(' Hi from force')
c
      zi=(0.0,1.0)
c
#ifndef NO_GAUGE
#warning "Compiling gauge"
      call gaugeforce(dSdpi)
#endif
c     return
c
      do 3 na=1,Nf
c
c     X1=(Mdagger M)**-1 Phi
c
      do 333 i=1,kvol
      do 333 idirac=1,ndirac
      do 333 ic=1,nc
      X1(ic,idirac,i)=X0(ic,idirac,i,na)
333   continue
      if(iflag.eq.1) goto 335
      call congradq(na,res1,itercg)
      ancg=ancg+float(itercg)
      do 334 i=1,kvol
      do 334 idirac=1,ndirac
      do 334 ic=1,nc
      X0(ic,idirac,i,na)=2*X1(ic,idirac,i)-X0(ic,idirac,i,na)
334   continue
335   continue
c
c     X2=2*M X1   
c
      call hdslash(X2,X1,u11,u12)
      do i=1,kvol
      do idirac=1,ndirac
      do ic=1,nc
      X2(ic,idirac,i)=2*X2(ic,idirac,i)
      enddo
      enddo
      enddo
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

#ifndef NO_SPACE
#warning "Compiling space"
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
#endif
c
c 4th direction is special
#ifndef NO_TIME
#warning "Compiling time"
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
#endif
c 
3     continue
c
      return
      end
c******************************************************************
c   Evaluation of Hamiltonian function
c
******************************************************************
      subroutine hamilton(h,s,res2,isweep)
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definitions 
      include "common_param.h"
c     can't include common_trial_u11u12.h because pp is aliased.
      common/trial/u11(kvol+halo,ndim),u12(kvol+halo,ndim),
     &     pp(kmomHalo)
c     can't include common_pseud.h because Phi & X0 are aliased
      common/pseud/Phi(kfermHalo,Nf),X0(kferm2Halo,Nf)
      common/vector/ X1(kferm2Halo)      

      real(kind=realkind) pp
      complex(kind=cmplxkind) Phi,X0,X1
      complex(kind=cmplxkind) u11,u12
      complex(kind=cmplxkind) :: smallPhi(kferm2Halo)
      
      integer isweep,imom,na,iferm,itercg
      real(kind=realkind) h,s,res2,hp,hg,hf,avplaqs,avplaqt

c      if(ismaster) then 
c        write(6,111)
c      endif
111   format(' Hi from hamilton')
c
      hp=0.0
      hg=0.0
      hf=0.0
c
      do 22 imom=1,kmom
      hp=hp+pp(imom)*pp(imom)
22    continue
c
      hp=hp*0.5
c
      call su2plaq(hg,avplaqs,avplaqt)
c         
c     goto 3
      do 3 na=1,Nf
c
      do 333 iferm=1,kferm2
      X1(iferm)=X0(iferm,na)
333   continue
c
      call congradq(na,res2,itercg)
      ancgh=ancgh+float(itercg)
c

      !copy correct elements from Phi into smallPhi
      call fillSmallPhi(na,Phi,smallPhi)

      do 4 iferm=1,kferm2
      X0(iferm,na)=X1(iferm)
      hf=hf+conjg(smallPhi(iferm))*X1(iferm)
4     continue
c
3     continue
c

c     hg has already been summed within su2plaq

      call par_csum(hp)
      call par_csum(hf)

      h=hg+hp+hf
      if(ismaster) then 
      write(6,*) isweep,':  hg', hg,'   hf', hf,'   hp', hp,'   h',h
      endif
      s=hg+hf
c
      return
      end              
c******************************************************************
c    matrix inversion via conjugate gradient algorithm
c       solves (Mdagger)Mx=Phi, 
c  implements up/down partitioning
c******************************************************************
      subroutine congradq(na,res,itercg)
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definitions
      include "common_mat.h"
      include "common_diquark.h"
      include "common_trial_u11u12.h"
c     can't include common_para.h because of bbb is different name.
      common/para/bbb,akappa,fmu,ibound
c     can't include common_pseud.h because of aliasing
      common/pseud/Phi(kfermHalo,Nf),X0(kferm2Halo,Nf)
c     common_vector.h is not in master
      common/vector/x(kferm2Halo)
    
      real(kind=realkind) bbb,akappa,fmu
      integer ibound
      complex(kind=cmplxkind) Phi,X0,x
      complex(kind=cmplxkind) x1(kferm2Halo),x2(kferm2Halo)
      complex(kind=cmplxkind) p(kferm2Halo),r(kferm2)
      complex(kind=cmplxkind) :: smallPhi(kferm2Halo)

      integer itercg,niterx,i
      real(kind=realkind) na,res,resid,betad,alpha,alphad
      real(kind=realkind) fac,alphan,betan,beta
c      write(6,111)
111   format(' Hi from congradq')
c
      resid=kferm2*res*res
      itercg=0
c
      do 1 niterx=1,niterc
      itercg=itercg+1
      if(niterx.gt.1) goto 51
c
c   initialise p=x, r=Phi(na)
c

      !copy correct elements from Phi into smallPhi
      call fillSmallPhi(na,Phi,smallPhi)

      do 2 i=1,kferm2
      p(i)=x(i)
      r(i)=smallPhi(i)
2     continue
      betad=1.0
      alpha=1.0
51    alphad=0.0
c
c  x2=(M^daggerM+J^2)p in a single step
c
      call hdslash(x1,p,u11,u12)
      call hdslashd(x2,x1,u11,u12)
c
      fac=conjg(jqq)*jqq*akappa**2
c NB need factor of kappa^2 to normalise fields correctly
      do i=1,kferm2
      x2(i)=x2(i)+fac*p(i)
      enddo
c
      if(niterx.eq.1) goto 201
c
c   alpha=(r,r)/(p,(Mdagger)Mp)
c 
      alphad=0.0
      do 31 i=1,kferm2
      alphad=alphad+conjg(p(i))*x2(i)
31    continue
      call par_csum(alphad)
      alpha=alphan/alphad
c      
c   x=x+alpha*p
c
      do 4 i=1,kferm2
      x(i)=x(i)+alpha*p(i)
4     continue
201   continue
      do i=1,kferm2
      betan=betan+conjg(r(i))*r(i) 
	end do
c     
c   r=r-alpha*(Mdagger)Mp
c
      do 6 i=1,kferm2
      r(i)=r(i)-alpha*x2(i)
6     continue
c
c   beta=(r_k+1,r_k+1)/(r_k,r_k)
c
      betan=0.0 
      do 61 i=1,kferm2
      betan=betan+conjg(r(i))*r(i) 
61    continue 

c      write(*,*) 'Proc ', procid, ', iter ', niterx, ', betan = ', betan
c      call flush(6)

      call par_csum(betan)

c      if (ismaster) write(*,*) Iter ', niterx, ', betan = ', betan
c      call flush(6)

      beta=betan/betad
      betad=betan
      alphan=betan
c
      if(niterx.eq.1) beta=0.0
c
c   p=r+beta*p
c
      do 7 i=1,kferm2
      p(i)=r(i)+beta*p(i)
7     continue

      if(betan.lt.resid) then
	if(ismaster) then 
	write(*,648)  niterx, betan, resid
648      format('Iter (CG) = ',i, ' resid = ', f, ' toler = ',f)
         call flush(7)
	endif
	   goto 8
      endif
1     continue

      write(7,1000)
1000  format(' # iterations of congrad exceeds niterc')

8     continue

c      if (ismaster) then
c      endif

      return
      end      
c******************************************************************
c    matrix inversion via conjugate gradient algorithm
c       solves (Mdagger)Mx=Phi, 
c           NB. no even/odd partitioning
c******************************************************************
      subroutine congradp(na,res,itercg)
      implicit none
      include "precision.h"
      include "sizes.h"
 
c     include common block definitions
      include "common_mat.h"
      include "common_trial_u11u12.h"
c     can't include common_para.h because bbb is different name.
      common/para/bbb,akappa,fmu,ibound
c     can't include common_pseud.h because of aliasing
      common/pseud/Phi(kfermHalo,Nf),X0(kferm2Halo,Nf)
      common/vectorp/x(kfermHalo)

      real(kind=realkind) bbb,akappa,fmu
      integer ibound
      complex(kind=cmplxkind) Phi,X0,x
      complex(kind=cmplxkind) x1(kfermHalo),x2(kfermHalo)
      complex(kind=cmplxkind) p(kfermHalo),r(kferm)

      integer na,itercg,niterx,i
      real(kind=realkind) res,resid,betad,alpha,alphad
      real(kind=realkind) alphan,beta,betan
c     write(6,111)
111   format(' Hi from congradp')
c
      resid=kferm*res*res
      itercg=0
c
      do 1 niterx=1,niterc
      itercg=itercg+1
      if(niterx.gt.1) goto 51
c
c   initialise p=x, r=Phi(na)
c
      do 2 i=1,kferm
      p(i)=x(i)
      r(i)=Phi(i,na)
2     continue
      betad=1.0
      alpha=1.0
51    alphad=0.0
c
c  x1=Mp
c
      call dslash(x1,p,u11,u12)
c
      if(niterx.eq.1) goto 201
c
c   alpha=(r,r)/(p,(Mdagger)Mp)
c 
      alphad=0.0
      do 31 i=1,kferm
      alphad=alphad+conjg(x1(i))*x1(i)
31    continue
      call par_csum(alphad)
      alpha=alphan/alphad
c      
c   x=x+alpha*p
c
      do 4 i=1,kferm
      x(i)=x(i)+alpha*p(i)
4     continue
201   continue
c     
c   x2=(Mdagger)x1=(Mdagger)Mp
c
      call dslashd(x2,x1,u11,u12)
c
c   r=r-alpha*(Mdagger)Mp
c
      do 6 i=1,kferm
      r(i)=r(i)-alpha*x2(i)
6     continue
c
c   beta=(r_k+1,r_k+1)/(r_k,r_k)
c
      betan=0.0 
      do 61 i=1,kferm
      betan=betan+conjg(r(i))*r(i) 
61    continue 
      call par_csum(betan)
      beta=betan/betad
      betad=betan
      alphan=betan
c
      if(niterx.eq.1) beta=0.0
c
c   p=r+beta*p
c
      do 7 i=1,kferm
      p(i)=r(i)+beta*p(i)
7     continue

      if(betan.lt.resid) goto 8
1     continue

      write(7,1000)
1000  format(' # iterations of congrad exceeds niterc')

8     continue

      if (ismaster) then
         write(*,*) 'Iter (CG)  = ', niterx, ', resid = ',
     %        betan, ', toler = ',resid
         call flush(7)
      endif
      
      return
      end      
c******************************************************************
c   Calculate fermion expectation values via a noisy estimator
c   -matrix inversion via conjugate gradient algorithm
c       solves Mx=x1
c     (Numerical Recipes section 2.10 pp.70-73)   
c   uses NEW lookup tables **
c*******************************************************************
      subroutine measure(psibarpsi,endenf,denf,qq,qbqb,res,itercg)
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definitions
      include "common_param.h"
      include "common_mat.h"
      include "common_pseud.h"
      include "common_neighb.h"
      include "common_trial_u11u12.h"
      include "common_dirac.h"
c     common_vectorp.h don't exist
      common/vectorp/xi(2,8,kvol+halo)

      complex(kind=cmplxkind) x(2,8,kvol+halo), R1(2,8,kvol+halo)
      complex(kind=cmplxkind) xi,qq,qbqb
      real(kind=realkind) ps(2,kvol+halo)

      integer itercg,igorkov,ic,i,idirac,igork,igork1
      real(kind=realkind) psibarpsi,endenf,denf,res
      real(kind=realkind) xu,xd,xuu,xdd

c     write(6,*) 'hi from measure'
c
c     set up noise
c
      do 300 igorkov=1,ngorkov
      do 300 ic=1,2
      call gauss0(ps)
      do i=1,kvol
      xi(ic,igorkov,i)=cmplx(ps(1,i),ps(2,i))
      x(ic,igorkov,i)=xi(ic,igorkov,i)
      enddo
300   continue
c
c R1= Mdagger*xi
c
      call dslashd(R1,xi,u11,u12)
c
      do 302 i=1,kvol
      do 302 igorkov=1,ngorkov
      do 302 ic=1,nc
      Phi(ic,igorkov,i,1)=R1(ic,igorkov,i)
      xi(ic,igorkov,i)=R1(ic,igorkov,i)
302   continue
c  
c xi= (MdaggerM)**-1 * R1 
c
      call congradp(1,res,itercg) 
c
      psibarpsi=0.0
      do 303 i=1,kvol
      do 303 igorkov=1,ngorkov
      do 303 ic=1,nc
      psibarpsi=psibarpsi+conjg(x(ic,igorkov,i))*xi(ic,igorkov,i)
303   continue 

      call par_csum(psibarpsi)

      psibarpsi=psibarpsi/(4*gvol)
c
      qbqb=(0.0,0.0)
      qq=(0.0,0.0)
      do 304 i=1,kvol
      do 304 idirac=1,ndirac
      igork=idirac+4
      do 304 ic=1,nc
      qbqb=qbqb+gamval(5,idirac)*conjg(x(ic,idirac,i))*xi(ic,igork,i)
      qq=qq-gamval(5,idirac)*conjg(x(ic,igork,i))*xi(ic,idirac,i)
304   continue

      call par_csum(qq)
      call par_csum(qbqb)

      qq=(qq+qbqb)/(2*gvol)
c     qbqb=qbqb/kvol
c
      xu=0.0
      xd=0.0
      xuu=0.0
      xdd=0.0
c  

c
c     Makes reference to x(.,.,,iu(i,4)) which has 16 cpts
c     Also  refers to x(.,.,,id(i,4)) which has 16 cpts
c     Therefore need both halos to be sent in direction 4 ONLY
c     Also references u(id(i,4),4) so need halos to be
c     sent in UP direction 4
c     Same for dk4p and dk4m
c     Also need u fields
c     It is almost certain that the u values are OK from previous
c     swaps but should do it in case - only need mu=4 with dir=4
c     also refers to dk4p(id(i,4)) and same for dk4m
c     again, may be OK with the dks but do them anyway

      call zdnhaloswapdir(x, 16, 4)
      call zuphaloswapdir(x, 16, 4)

      call zuphaloswapdir(u11(1,4), 1, 4)
      call zuphaloswapdir(u12(1,4), 1, 4)

      call cuphaloswapdir(dk4p,1,4)
      call cuphaloswapdir(dk4m,1,4)

      do 600 i=1,kvol
      do 600 igorkov=1,4
      idirac=igorkov
      igork1=gamin(4,idirac)
      xu=xu+dk4p(id(i,4))*
     &(conjg(x(1,igorkov,id(i,4)))*
     &(-u11(id(i,4),4)*xi(1,igorkov,i)
     & -u12(id(i,4),4)*xi(2,igorkov,i)
     & +u11(id(i,4),4)*xi(1,igork1,i)
     & +u12(id(i,4),4)*xi(2,igork1,i))
     &+conjg(x(2,igorkov,id(i,4)))*
     &(+conjg(u12(id(i,4),4))*xi(1,igorkov,i)
     & -conjg(u11(id(i,4),4))*xi(2,igorkov,i)
     & -conjg(u12(id(i,4),4))*xi(1,igork1,i)
     & +conjg(u11(id(i,4),4))*xi(2,igork1,i)))
      xd=xd+dk4m(i)*
     &(conjg(x(1,igorkov,iu(i,4)))*
     & (conjg(u11(i,4))*xi(1,igorkov,i)
     &       -u12(i,4) *xi(2,igorkov,i)
     & +conjg(u11(i,4))*xi(1,igork1,i)
     &       -u12(i,4) *xi(2,igork1,i))
     &+conjg(x(2,igorkov,iu(i,4)))*
     & (conjg(u12(i,4))*xi(1,igorkov,i)
     &       +u11(i,4) *xi(2,igorkov,i)
     & +conjg(u12(i,4))*xi(1,igork1,i)
     &       +u11(i,4) *xi(2,igork1,i)))
600   continue
      do 700 i=1,kvol
      do 700 igorkov=5,ngorkov
      idirac=igorkov-4
      igork1=gamin(4,idirac)+4
      xuu=xuu-dk4m(id(i,4))*
     &(conjg(x(1,igorkov,id(i,4)))*
     &(-u11(id(i,4),4)*xi(1,igorkov,i)
     & -u12(id(i,4),4)*xi(2,igorkov,i)
     & +u11(id(i,4),4)*xi(1,igork1,i)
     & +u12(id(i,4),4)*xi(2,igork1,i))
     &+conjg(x(2,igorkov,id(i,4)))*
     &(+conjg(u12(id(i,4),4))*xi(1,igorkov,i)
     & -conjg(u11(id(i,4),4))*xi(2,igorkov,i)
     & -conjg(u12(id(i,4),4))*xi(1,igork1,i)
     & +conjg(u11(id(i,4),4))*xi(2,igork1,i)))
      xdd=xdd-dk4p(i)*
     &(conjg(x(1,igorkov,iu(i,4)))*
     & (conjg(u11(i,4))*xi(1,igorkov,i)
     &       -u12(i,4) *xi(2,igorkov,i)
     & +conjg(u11(i,4))*xi(1,igork1,i)
     &       -u12(i,4) *xi(2,igork1,i))
     &+conjg(x(2,igorkov,iu(i,4)))*
     & (conjg(u12(i,4))*xi(1,igorkov,i)
     &       +u11(i,4) *xi(2,igorkov,i)
     & +conjg(u12(i,4))*xi(1,igork1,i)
     &       +u11(i,4) *xi(2,igork1,i)))
700   continue
      endenf=(xu-xd-xuu+xdd)
      denf=(xu+xd+xuu+xdd)

      call par_csum(endenf)
      call par_csum(denf)

      endenf = endenf/(2*gvol)
      denf   =   denf/(2*gvol)

c
      return
      end

c
!---------------------------------------------------
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
c
      subroutine init(istart)
c*******************************************************************
c     sets initial values
c     istart=0 cold start
c     istart=1 hot start
c     istart<0 no initialisation
c*******************************************************************
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definition
      include "common_para.h"
      include "common_mat.h"
      include "common_gauge.h"
      include "common_trial.h"
      include "common_neighb.h"
      include "common_dirac.h"
      
      complex(kind=cmplxkind) one,zi

      integer istart,k,l,j,i,ic,ksize3,ku,idirac,mu,ind
      real(kind=realkind) chem1,chem2,ranf,ranf1,ranf2,ran2
c
      one=(1.0,0.0)
      zi=(0.0,1.0)
c*******************************************************************
c  calculate constants
c*******************************************************************
      call addrc

c     Now check that they are OK

      call checkaddr(id, ksize, ksizet, 1, kvol+halo)
      call checkaddr(iu, ksize, ksizet, 1, kvol+halo)


      chem1=exp(fmu)
      chem2=1.0/chem1
      do 1 k=1,ksizet
      do 1 l=1,ksize
      do 1 j=1,ksize
      do 1 i=1,ksize
      ic=(((k-1)*ksize+(l-1))*ksize+(j-1))*ksize+i
      dk4p(ic)=akappa*chem1
      dk4m(ic)=akappa*chem2
1     continue
      if(ibound.eq.-1) then
c  Only do the multiplication by -1 if we are at the edge in time dir
c       write(*,*) 'On procid ', procid, ', pcoord(4) = ',
c     1             pcoord(4, procid+1)
        if (pcoord(4, procid+1) == npt-1) then
c          write(*,*) 'Implementing antiperiodic bcs on proc ', procid
          ksize3=ksize*ksize*ksize
          do 320 k=1,ksize3
            ku=kvol-ksize3+k
            dk4p(ku)=-dk4p(ku)
            dk4m(ku)=-dk4m(ku)
 320      continue
        endif
      endif
c*******************************************************************
c    setup Dirac algebra
c*******************************************************************
c
c     gamma_1
c
      gamval(1,1)=-zi
      gamval(1,2)=-zi
      gamval(1,3)= zi
      gamval(1,4)= zi
c
      gamin(1,1)=4
      gamin(1,2)=3
      gamin(1,3)=2
      gamin(1,4)=1
c
c     gamma_2
c
      gamval(2,1)=-one
      gamval(2,2)= one
      gamval(2,3)= one
      gamval(2,4)=-one
c
      gamin(2,1)=4
      gamin(2,2)=3
      gamin(2,3)=2
      gamin(2,4)=1
c
c     gamma_3
c
      gamval(3,1)=-zi
      gamval(3,2)= zi
      gamval(3,3)= zi
      gamval(3,4)=-zi
c
      gamin(3,1)=3
      gamin(3,2)=4
      gamin(3,3)=1
      gamin(3,4)=2
c
c     gamma_4
c
      gamval(4,1)= one
      gamval(4,2)= one
      gamval(4,3)= one
      gamval(4,4)= one
c
      gamin(4,1)=3
      gamin(4,2)=4
      gamin(4,3)=1
      gamin(4,4)=2
c
c     gamma_5 = gamma_1 * gamma_2 * gamma_3 * gamma_4
c
      gamval(5,1)= one
      gamval(5,2)= one
      gamval(5,3)=-one
      gamval(5,4)=-one
c
      do idirac=1,ndirac
      do mu=1,5
      gamval(mu,idirac)=gamval(mu,idirac)*akappa
      enddo
      enddo
c
      if(istart.lt.0) return
c
c     initialise gauge fields
c
      if(istart .ge. 1)goto 40
c     (else cold start)
      do 10 mu=1,ndim
      do 10 ind=1,kvol
      u11(ind,mu)=(1.0,0.0)
      u12(ind,mu)=(0.0,0.0)
10    continue
      return
c
40    continue
      do 61 ind=1,kvol
      do 61 mu=1,ndim
c      call random_number(ranf1)
c      call random_number(ranf2)
c      RANF1 = RAN2(SEED)
c      RANF2 = RAN2(SEED)
c      u11t(ind,mu)=cmplx(2.0*ranf1-1.0,2.0*ranf2-1.0)
      u11t(ind,mu)=cmplx(2.0*ran2(seed)-1.0,2.0*ran2(seed)-1.0)
c      call random_number(ranf1)
c      call random_number(ranf2)
c      RANF1 = RAN2(SEED)
c      RANF2 = RAN2(SEED)
      u12t(ind,mu)=cmplx(2.0*ran2(seed)-1.0,2.0*ran2(seed)-1.0)
61    continue
      call reunitarise
      do 62 mu=1,ndim
      do 62 ind=1,kvol
      u11(ind,mu)=u11t(ind,mu)
      u12(ind,mu)=u12t(ind,mu)
62    continue
      return
      end
c
      subroutine addrc
      implicit none
      include "precision.h"
      include "sizes.h"

      integer iaddr
      integer ihaloaddr

c*******************************************************************
c
c     loads the addresses required during the update
c
c*******************************************************************
c     include common block definition
      include "common_neighb.h"

      integer j1,j2,j3,j4,ic,i
      integer ihd1, ihu1, ihd2, ihu2, ihd3, ihu3, ihd4, ihu4

      ihd1 = 0
      ihu1 = 0
      ihd2 = 0
      ihu2 = 0
      ihd3 = 0
      ihu3 = 0
      ihd4 = 0
      ihu4 = 0

c     Now do the lookups appropriate for overindexing into halos
c     Order is dnx, upx, dny, upy, dnz, upz, dnt, upt

      h1d(1) = kvol + 1
      h2d(1) = h1d(1) + halox - 1

      h1u(1) = h2d(1) + 1
      h2u(1) = h1u(1) + halox - 1

      halosize(1) = halox

      h1d(2) = h2u(1) + 1
      h2d(2) = h1d(2) + haloy - 1

      h1u(2) = h2d(2) + 1
      h2u(2) = h1u(2) + haloy - 1

      halosize(2) = haloy

      h1d(3) = h2u(2) + 1
      h2d(3) = h1d(3) + haloz - 1

      h1u(3) = h2d(3) + 1
      h2u(3) = h1u(3) + haloz - 1

      halosize(3) = haloz

      h1d(4) = h2u(3) + 1
      h2d(4) = h1d(4) + halot - 1

      h1u(4) = h2d(4) + 1
      h2u(4) = h1u(4) + halot - 1

      halosize(4) = halot

c      do i = 1, 4
c        write(*,*) 'hd(', i, '): ', h1d(i), ' -> ', h2d(i)
c        write(*,*) 'hu(', i, '): ', h1u(i), ' -> ', h2u(i)
c      end do

c     Variables are:
c
c     h1d(mu) = starting  point in tail of down halo in direction mu
c     h2d(mu) = finishing point in tail of down halo in direction mu
c
c     h1u(mu) = starting  point in tail of up   halo in direction mu
c     h2u(mu) = finishing point in tail of up   halo in direction mu
c
c     hd(i,mu) = index in core of point that should be packed into the
c                ith location of the down halo in direction mu
c
c     hu(i,mu) = index in core of point that should be packed into the
c                ith location of the up   halo in direction mu
c
c     Note that hd and hu should be used for PACKING before SENDING
c
c     Unpacking would be done with a loop over ALL the core sites with
c     reference to normal dn/up lookups, ie we DO NOT have a list of
c     where in the halo the core point i should go
c
c     Halo points are ordered "as they come" in the linear loop over
c     core sites
c     C NOTE: Reverse order of indices when translating?
c

      do 40 j4=1,ksizet
      do 40 j3=1,ksize
      do 40 j2=1,ksize
      do 40 j1=1,ksize

      ic=(((j4-1)*ksize+(j3-1))*ksize+(j2-1))*ksize+j1

      if (j1 .ne. 1) then
        call ia(j1-1,j2,j3,j4,iaddr)
      else
        ihd1 = ihd1 + 1

        if (ihd1 .gt. halo) then
          write(*,*) 'ihd1 too big'
          stop
        endif

        hd(ihd1, 1) = ic
        iaddr = h1d(1) + ihd1 - 1
      end if

      id(ic,1) = iaddr

      if (j1 .ne. ksize) then
        call ia(j1+1,j2,j3,j4,iaddr)
      else
        ihu1 = ihu1 + 1

        if (ihu1 .gt. halo) then
          write(*,*) 'ihu1 too big'
          stop
        endif

        hu(ihu1, 1) = ic
        iaddr = h1u(1) + ihu1 - 1
      end if

      iu(ic,1) = iaddr

      if (j2 .ne. 1) then
        call ia(j1,j2-1,j3,j4,iaddr)
      else
        ihd2 = ihd2 + 1

        if (ihd2 .gt. halo) then
          write(*,*) 'ihd2 too big'
          stop
        endif

        hd(ihd2, 2) = ic
        iaddr = h1d(2) + ihd2 - 1
      end if

      id(ic,2) = iaddr

      if (j2 .ne. ksize) then
        call ia(j1,j2+1,j3,j4,iaddr)
      else
        ihu2 = ihu2 + 1

        if (ihu2 .gt. halo) then
          write(*,*) 'ihu2 too big'
          stop
        endif

        hu(ihu2, 2) = ic
        iaddr = h1u(2) + ihu2 - 1
      end if

      iu(ic,2) = iaddr

      if (j3 .ne. 1) then
        call ia(j1,j2,j3-1,j4,iaddr)
      else
        ihd3 = ihd3 + 1

        if (ihd3 .gt. halo) then
          write(*,*) 'ihd3 too big'
          stop
        endif

        hd(ihd3, 3) = ic
        iaddr = h1d(3) + ihd3 - 1
      end if

      id(ic,3) = iaddr

      if (j3 .ne. ksize) then
        call ia(j1,j2,j3+1,j4,iaddr)
      else
        ihu3 = ihu3 + 1

        if (ihu3 .gt. halo) then
          write(*,*) 'ihu3 too big'
          stop
        endif

        hu(ihu3, 3) = ic
        iaddr = h1u(3) + ihu3 - 1
      end if

      iu(ic,3) = iaddr

      if (j4 .ne. 1) then
        call ia(j1,j2,j3,j4-1,iaddr)
      else
        ihd4 = ihd4 + 1

        if (ihd4 .gt. halo) then
          write(*,*) 'ihd4 too big'
          stop
        endif

        hd(ihd4, 4) = ic
        iaddr = h1d(4) + ihd4 - 1
      end if

      id(ic,4) = iaddr

      if (j4 .ne. ksizet) then
        call ia(j1,j2,j3,j4+1,iaddr)
      else
        ihu4 = ihu4 + 1

        if (ihu4 .gt. halo) then
          write(*,*) 'ihu4 too big'
          stop
        endif

        hu(ihu4, 4) = ic
        iaddr = h1u(4) + ihu4 - 1
      end if

      iu(ic,4) = iaddr

  40  continue

c      write(*,*) 'ihd1 = ', ihd1, ', check = ', h2d(1) - h1d(1) + 1
c      write(*,*) 'ihu1 = ', ihu1, ', check = ', h2u(1) - h1u(1) + 1

c      write(*,*) 'ihd2 = ', ihd2, ', check = ', h2d(2) - h1d(2) + 1
c      write(*,*) 'ihu2 = ', ihu2, ', check = ', h2u(2) - h1u(2) + 1

c      write(*,*) 'ihd3 = ', ihd3, ', check = ', h2d(3) - h1d(3) + 1
c      write(*,*) 'ihu3 = ', ihu3, ', check = ', h2u(3) - h1u(3) + 1

c      write(*,*) 'ihd4 = ', ihd4, ', check = ', h2d(4) - h1d(4) + 1
c      write(*,*) 'ihu4 = ', ihu4, ', check = ', h2u(4) - h1u(4) + 1
#ifdef DIAGNOSTICS
!	$OMP PARALLEL SECTIONS
!	$OMP PARALLEL SECTION
      DO 2132 IC = 1,KVOL
      WRITE(7101,7003) ID(IC,1), ID(IC,2), ID(IC,3), ID(IC,4)
2132  CONTINUE
!	$OMP PARALLEL SECTION
      DO 2133 IC = 1,KVOL
      WRITE(7102,7003) IU(IC,1), IU(IC,2), IU(IC,3), IU(IC,4)
2133  CONTINUE
!	$OMP END SECTIONS
7003  FORMAT(I,1X,I,1X,I,1X,I)
#endif
      return
      end
c
      subroutine ia(i1,i2,i3,i4,nnn)
      implicit none
      include "precision.h"
      include "sizes.h"
c*******************************************************************
c    21st century
c     address calculator
c
c*******************************************************************
      integer i1,i2,i3,i4,nnn
      integer n1,n2,n3,n4
      n1=i1
      n2=i2
      n3=i3 
      n4=i4
      if(n1.le.0) then
      n1=n1+ksize
      go to 4
      endif
      if(n1.gt.ksize) n1=n1-ksize
c
   4  if(n2.le.0) then
      n2=n2+ksize
      go to 8
      endif
      if(n2.gt.ksize) n2=n2-ksize
c  
   8  if(n3.le.0) then
      n3=n3+ksize 
      go to 12
      endif
      if(n3.gt.ksize) n3=n3-ksize
c
  12  if(n4.le.0) then
      n4=n4+ksizet
      goto 16
      endif
      if(n4.gt.ksizet) n4=n4-ksizet
c
  16  nnn=(((n4-1)*ksize+(n3-1))*ksize+(n2-1))*ksize+n1
      return
      end
c**********************************************************************
c calculate vector of gaussian random numbers with unit variance
c to refresh momenta
c   Numerical Recipes pp.203
c**********************************************************************
      subroutine gaussp(ps)
      implicit none
      include "precision.h"
      include "sizes.h"
c     include common block definition
      include "common_gauge.h" 
      include "common_trans.h"
 
      integer :: icount = 0
      save icount
      character(32) :: filename

      real(kind=realkind) ps(2,kvol+halo)
      real(kind=realkind) ranf,theta,r,rand,ran2
      integer il
c       r was added by me for debugging      
c      if(ismaster) then
c      write(6,1)
c      endif
1     format(' Hi from gaussp')
      do il=1,kvol
c     Previously the programme used RANF() as defined later on
c     But this was generating really weird outputs so is
c     being replaced by the built in PRNG for now.
c1000  call random_number(r)
1000  ps(2,il)=sqrt(-2.0*log(ran2(seed)))
c      if(ismaster) then
c              write(*,*) "r=", r, "sqrt(-2.0*log((rand()))=", ps(2,il)
c      endif
c      do 1001 il=1,kvol
c      theta=tpi*rand()
c      call random_number(theta)
c      theta = ran2(seed)
      theta=tpi*ran2(seed)
      ps(1,il)=ps(2,il)*sin(theta)
      ps(2,il)=ps(2,il)*cos(theta)
c1001  continue 
      end do

c      icount = icount + 1
c
c      write(filename,fmt='(''gaussp_'', i3.3, ''.dat'')') icount
c      write(*,*) 'filename = ', filename
c
c      call par_psread(filename, ps)

c      open(unit=10, file=filename, form='unformatted')
c      read(10) ps
c      close(unit=10)

      return
      end      
c**********************************************************************
c calculate vector of gaussian random numbers with unit variance
c to generate pseudofermion fields R
c   Numerical Recipes pp.203
c**********************************************************************
      subroutine gauss0(ps)
      implicit none
      include "precision.h"
      include "sizes.h"
c     include common block definition
      include "common_gauge.h" 
      include "common_trans.h"

      integer :: icount = 0
      save icount
      character(32) :: filename

      real(kind=realkind) ps(2,kvol+halo)
      real(kind=realkind) ranf,theta,ran2,r
      integer il

c      write(6,1)
1     format(' Hi from gauss0')
c      do 1000 il=1,kvol
      do il = 1, kvol
c     Previously the programme used RANF() as defined later on
c     But this was generating really weird outputs so is
c     being replaced by the built in PRNG for now.
c     Because recycling is good we'll use ranf as a variable
c     instead of a function here. What could go wrong!
c1000  call random_number(ranf)
c1000  r = ran2(seed)
      ps(2,il)=sqrt(-log(ran2(seed)))
c      do 1001 il=1,kvol
c      theta=tpi*ranf()
c      call random_number(theta)
      theta = tpi*ran2(seed)
      ps(1,il)=ps(2,il)*sin(theta)
      ps(2,il)=ps(2,il)*cos(theta)
c 1001  continue 
      end do

c      icount = icount + 1
c
c      write(filename,fmt='(''gauss0_'', i3.3, ''.dat'')') icount
c      write(*,*) 'filename = ', filename
c
c
c      call par_psread(filename, ps)

c      open(unit=10, file=filename, form='unformatted')
c      read(10) ps
c      close(unit=10)

      return
      end      
c********************************************************************
      subroutine reunitarise
      implicit none
      include "precision.h"
      include "sizes.h"

c     include common block definition
      include "common_trial_u11u12.h"

      integer mu,i
      real(kind=realkind) anorm
c
      do 1 mu=1,ndim
      do 1 i=1,kvol
c
      anorm=sqrt(conjg(u11(i,mu))*u11(i,mu)
     &          +conjg(u12(i,mu))*u12(i,mu))
      u11(i,mu)=u11(i,mu)/anorm
      u12(i,mu)=u12(i,mu)/anorm
c
1     continue
      return
      end
c******************************************************************
c   calculate gauge action ** USING NEW LOOKUP TABLES **
c   follows routine 'qedplaq' in QED3 code
c******************************************************************
      subroutine su2plaq(hg,avplaqs,avplaqt)
      implicit none
      include "precision.h"
      include "sizes.h"
c     include common block definition
      include "common_neighb.h"
      include "common_para.h"
      include "common_trial_u11u12.h"

      complex(kind=cmplxkind) Sigma11(kvol),Sigma12(kvol)
      complex(kind=cmplxkind) a11(kvol),a12(kvol)

      real(kind=realkind) hg,avplaqs,avplaqt,hgs,hgt
      integer mu,nu,i
c
      hgs=0.0
      hgt=0.0
c
c     Do equivalent of halo swap

c     idim is shift direction
c     mu is u index


      do mu = 1, ndim

        call zdnhaloswapall(u11(1,mu), 1)
        call zdnhaloswapall(u12(1,mu), 1)

      end do


      do 1 mu=1,ndim
      do 1 nu=1,mu-1
c
      do 10 i=1,kvol
      Sigma11(i)= u11(i,mu)*u11(iu(i,mu),nu)
     &           -u12(i,mu)*conjg(u12(iu(i,mu),nu))
      Sigma12(i)= u11(i,mu)*u12(iu(i,mu),nu)
     &           +u12(i,mu)*conjg(u11(iu(i,mu),nu))
10    continue
c
      do 11 i=1,kvol
      a11(i)= Sigma11(i)*conjg(u11(iu(i,nu),mu))
     &       +Sigma12(i)*conjg(u12(iu(i,nu),mu))
      a12(i)=-Sigma11(i)*u12(iu(i,nu),mu)
     &       +Sigma12(i)*u11(iu(i,nu),mu)
11    continue
c
      do 12 i=1,kvol
      Sigma11(i)= a11(i)*conjg(u11(i,nu))
     &           +a12(i)*conjg(u12(i,nu))
c     Sigma12(i)=-a11(i)*u12(i,nu)
c    &           +a12(i)*u11(i,nu)
12    continue
c
c   S_g = -beta/2 * real(tr(Sigma))
c   average plaquette = 1/2 real(tr(Sigma))
c
      if(mu.le.3)then
      do i=1,kvol
      hgs=hgs-real(Sigma11(i))
      enddo
      else
      do i=1,kvol
      hgt=hgt-real(Sigma11(i))
      enddo
      endif
c
1     continue
c
      call par_csum(hgs)
      call par_csum(hgt)

      avplaqs=-hgs/(gvol*3)
      avplaqt=-hgt/(gvol*3)
      hg=(hgs+hgt)*beta
      if(ismaster) then
       write(*,*) "hgs=", hgs, "hgt=", hgt, "hg=", hg
      endif
      return
      end
c******************************************************************
c   calculate polyakov loop
c******************************************************************
      subroutine polyakov(poly)
      implicit none
      include "precision.h"
      include "sizes.h"
      include "common_trial_u11u12.h"

      complex(kind=cmplxkind) Sigma11(kvol3),Sigma12(kvol3)
      complex(kind=cmplxkind) a11(kvol3),a12(kvol3)

      integer :: indexu,i,it
      real(kind=realkind) poly
c
c     Changed this routine.
c     u11 and u12 now defined as normal ie (kvol+halo,4).
c     Copy of Sigma11 and Sigma12 is changed so that it copies
c     in blocks of ksizet.
c     Variable indexu also used to select correct element of u11 and u12 
c     in loop 10 below.

c
c Change the order of multiplication so that it can
c be done in parallel. Start at t=1 and go up to t=T:
c previously started at t+T and looped back to 1, 2, ... T-1
c

      do i=1,kvol3
      Sigma11(i) = u11(i,4)
      Sigma12(i) = u12(i,4)
      enddo

c
      do 1 it=2,ksizet
c
      do 10 i=1,kvol3
      indexu = (i+((it-1)*kvol3))
      a11(i)=Sigma11(i)*u11(indexu,4)-Sigma12(i)*conjg(u12(indexu,4))
      a12(i)=Sigma11(i)*u12(indexu,4)+Sigma12(i)*conjg(u11(indexu,4))
10    continue
c
      do 11 i=1,kvol3
      Sigma11(i)=a11(i)
      Sigma12(i)=a12(i)
11    continue
c
1     continue
c

c
c Multiply this partial loop with contributions of other processors in
c the timelike dimension
c
      call par_tmul(Sigma11, Sigma12)

      poly=0.0
      do 12 i=1,kvol3
      poly=poly+real(Sigma11(i))
12    continue

c
c All processors have the value for the complete polyakov line at
c all spatial sites. Need to globally sum over spatial processors
c but not across time as these are duplicates
c Simply zero the value for all but t=0
c
c This is a bit of a HACK

      if (pcoord(4,procid+1) .ne. 0) poly = 0.0

      call par_csum(poly)

      poly=poly/gvol3
c
      return
      end
c******************************************************************
c   calculate dSdpi due to Wilson action at each intermediate time
c
c******************************************************************
      subroutine gaugeforce(dSdpi)
      implicit none
      include "precision.h"
      include "sizes.h"
c     include common block definitions
      include "common_neighb.h"
      include "common_para.h"
      include "common_trial_u11u12.h"

      real(kind=realkind) dSdpi(ndim,3,kvol+halo)

      complex(kind=cmplxkind) a11(kvol),a12(kvol)
      complex(kind=cmplxkind) Sigma11(kvol),Sigma12(kvol)
      integer ish(kvol)
      complex(kind=cmplxkind) u11sh(kvol+halo), u12sh(kvol+halo)

      integer mu,i,nu
c     write(6,*) 'hi from gaugeforce'
c

c
c  Zero halos PURELY FOR DEBUGGING just in case they happen to
c  have correct data on entry
c
c      do mu = 1, 4
c        do i = kvol+1, kvol+halo
c          u11(i,mu) = 0.0
c          u12(i,mu) = 0.0
c        end do
c      end do
c

       do mu = 1, ndim
        call zhaloswapall(u11(1,mu), 1)
        call zhaloswapall(u12(1,mu), 1)
      end do

      do 1 mu=1,ndim
c
      do i=1,kvol
      Sigma11(i)=(0.0,0.0)
      Sigma12(i)=(0.0,0.0)
      enddo
c
c
      do 2 nu=1,ndim
c
      if(nu.eq.mu) goto2
c
c  first the +nu staple...
c
      do 10 i=1,kvol
      a11(i)= u11(iu(i,mu),nu)*conjg(u11(iu(i,nu),mu))
     &       +u12(iu(i,mu),nu)*conjg(u12(iu(i,nu),mu))     
      a12(i)=-u11(iu(i,mu),nu)*u12(iu(i,nu),mu)
     &       +u12(iu(i,mu),nu)*u11(iu(i,nu),mu)
10    continue
c
      do 11 i=1,kvol
      Sigma11(i)=Sigma11(i)
     &       +a11(i)*conjg(u11(i,nu))+a12(i)*conjg(u12(i,nu))
      Sigma12(i)=Sigma12(i)
     &       -a11(i)*u12(i,nu)+a12(i)*u11(i,nu)
11    continue
c
c  ... and then the -nu staple
c
c      call gather(kvol,ish,id(1,nu),iu(1,mu))

      call zgather(kvol, u11sh, kvol+halo, u11(1,nu), id(1,nu))
      call zgather(kvol, u12sh, kvol+halo, u12(1,nu), id(1,nu))

      call zdnhaloswapdir(u11sh, 1, mu)
      call zdnhaloswapdir(u12sh, 1, mu)

c
      do 20 i=1,kvol
      a11(i)= conjg(u11sh(iu(i,mu)))*conjg(u11(id(i,nu),mu))
     &       -      u12sh(iu(i,mu)) *conjg(u12(id(i,nu),mu))
      a12(i)=-conjg(u11sh(iu(i,mu)))*u12(id(i,nu),mu)
     &       -      u12sh(iu(i,mu)) *u11(id(i,nu),mu)
20    continue
c    
      do 21 i=1,kvol
      Sigma11(i)=Sigma11(i)
     &      +a11(i)*u11(id(i,nu),nu)
     &      -a12(i)*conjg(u12(id(i,nu),nu))
      Sigma12(i)=Sigma12(i)
     &      +a11(i)*u12(id(i,nu),nu)
     &      +a12(i)*conjg(u11(id(i,nu),nu))
21    continue
2     continue
c
c
      do 30 i=1,kvol
      a11(i)=u11(i,mu)*Sigma12(i)+u12(i,mu)*conjg(Sigma11(i))
      a12(i)=u11(i,mu)*Sigma11(i)+conjg(u12(i,mu))*Sigma12(i)
30    continue
c
      do 31 i=1,kvol
      dSdpi(mu,1,i)=beta*aimag(a11(i))
      dSdpi(mu,2,i)=beta* real(a11(i))
      dSdpi(mu,3,i)=beta*aimag(a12(i))
31    continue
1     continue
c
      return
      end


      subroutine zgather(n, x, m, y, table)
      implicit none
      include "precision.h"
      include "sizes.h"
      integer n, m, i
      complex(kind=cmplxkind) x(n), y(m)
      integer table(n)
c
      do i=1, n
      x(i)= y(table(i))
      enddo
c
      return
      end
C========================================================================
C
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
c     write(*,*) 'On proc ', procid, ', ds = ', ds(1), ds(2)
c     call flush(6)
      RETURN
      END
      ! ran2 random generator program !
      ! This function returns a random number between 0 ans 1.
      ! The function is now threadsafe. Below is an example of how to ! use it.
      !
      ! REAL X
      ! INTEGER SEED
      ! X = RAN2(SEED)
      !


      FUNCTION RAN2(IDUM)
      IMPLICIT NONE
      INCLUDE "precision.h"
      INTEGER IDUM,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
      REAL(KIND=REALKIND) :: RAN2,AM,EPS,RNMX
      PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,
     &IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,IR2=3791,
     &NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2E-7,RNMX=1.-EPS)
      INTEGER IDUM2,J,K,IV(NTAB),IY
      SAVE IV,IY,IDUM2
      DATA IDUM2/123456789/, IV/NTAB*0/, IY/0/
!    $OMP THREADPRIVATE(IDUM2,IV,IY)
      IF (IDUM.LE.0) THEN 
            IDUM=MAX(-IDUM,1) 
            IDUM2=IDUM

            DO J=NTAB+8,1,-1
                  K=IDUM/IQ1 
                  IDUM=IA1*(IDUM-K*IQ1)-K*IR1 
                  IF (IDUM.LT.0) IDUM=IDUM+IM1 
                  IF (J.LE.NTAB) IV(J)=IDUM
            END DO 
            IY=IV(1)
      ENDIF

      K=IDUM/IQ1 
      IDUM=IA1*(IDUM-K*IQ1)-K*IR1
      IF (IDUM.LT.0) IDUM=IDUM+IM1 
      K=IDUM2/IQ2 
      IDUM2=IA2*(IDUM2-K*IQ2)-K*IR2 
      IF (IDUM2.LT.0) IDUM2=IDUM2+IM2
      J =1+IY/NDIV
      IY=IV(J)-IDUM2
      IV(J)=IDUM 
      IF(IY.LT.1)IY=IY+IMM1 
      RAN2=MIN(AM*IY,RNMX)
      RETURN
      END FUNCTION RAN2
C***********************************************************************
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
#ifndef NO_SPACE
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
#endif
c
c  Timelike Wilson term
c
#ifndef NO_TIME
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
#endif
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
#ifndef NO_SPACE
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
#endif
c     
c  Timelike Wilson term
c
#ifndef NO_TIME
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
#endif
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
#ifndef NO_SPACE
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
#endif
#ifndef NO_TIME
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
#endif
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
#ifndef NO_SPACE
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
#endif
#ifndef NO_TIME
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
#endif
         enddo
      enddo
c$omp end parallel do
      
c     
      return
      end

!---------------------------------------------
!     fillSmallPhi                         
!
!     Routine to copy the necessary (2*4*kvol) 
!     elements of Phi into a vector varaible.
!---------------------------------------------
      subroutine fillSmallPhi(na, Phi, smallPhi)
      implicit none
      include "precision.h"
      include "sizes.h"
      complex(kind=cmplxkind), intent(in) :: Phi(kfermHalo,Nf)
      integer, intent(in) :: na
      complex(kind=cmplxkind), intent(out) :: smallPhi(kferm2Halo)
      integer indexSmallPhi, indexPhi
      integer i,j,k
      indexSmallPhi = 1
      do i = 1, kvol
      do j = 1, ndirac
      do k = 1, nc
      indexPhi = (16*(i-1)) + (2*(j-1)) + k
      smallPhi(indexSmallPhi) = Phi(indexPhi,na)
      indexSmallPhi = indexSmallPhi + 1
      enddo
      enddo
      enddo

      return
      end

      subroutine checkaddr(table, lns, lnt, imin, imax)
      implicit none
      include "precision.h"
      include "sizes.h"

      integer lns, lnt, imin, imax
      integer table(lns*lns*lns*lnt, ndim)

      integer itable, iaddr, idim

      integer ntable
      ntable = lns*lns*lns*lnt

c      write(*,*) 'Checking table'

       do idim = 1, ndim
        do itable = 1, ntable

          iaddr = table(itable, idim)

          if ((iaddr .lt. imin) .or. (iaddr .gt. imax)) then
            write(*,*) 'Table error!'
            write(*,*) 'lns, lnt, imin, imax = ', lns, lnt, imin, imax
            write(*,*) 'table(', itable, ', ', idim, ') = ',
     &                  iaddr
            stop
          end if

       end do
      end do

      return
      end

      subroutine index2lcoord(index, coord)
      implicit none
      include "precision.h"
      include "sizes.h"

      integer index
      integer coord(4)

      integer tindex

      tindex = index-1
      
      coord(1) = mod(tindex, ksizex) + 1
      tindex = tindex / ksizex

      coord(2) = mod(tindex, ksizey) + 1
      tindex = tindex / ksizey

      coord(3) = mod(tindex, ksizez) + 1
      tindex = tindex / ksizez

      coord(4) = tindex + 1

      return
      end

      subroutine index2gcoord(index, coord)
      implicit none
      include "precision.h"
      include "sizes.h"

      integer index
      integer coord(4)

      integer tindex

      tindex = index-1
      
      coord(1) = mod(tindex, nx) + 1
      tindex = tindex / nx

      coord(2) = mod(tindex, ny) + 1
      tindex = tindex / ny

      coord(3) = mod(tindex, nz) + 1
      tindex = tindex / nz

      coord(4) = tindex + 1

      return
      end

      subroutine coord2gindex(coord, index)
      implicit none
      include "precision.h"
      include "sizes.h"

      integer index
      integer coord(4)

      index = coord(1) + (coord(2)-1)*nx
     1                 + (coord(3)-1)*nx*ny
     1                 + (coord(4)-1)*nx*ny*nz

      return
      end

      subroutine testcoord
      implicit none
      include "precision.h"
      include "sizes.h"
      
      integer i
      integer coord(4)

      do i = 1, kvol
        call index2lcoord(i, coord)
        write(*,*) i, ' -> coord(',
     &    coord(1), ',', coord(2), ',', coord(3),',', coord(4)

      end do
      return
      end


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

      subroutine znorm2(norm, x, n)

      implicit none

      include "precision.h"

      complex(kind=cmplxkind), dimension(n) :: x
      real(kind=realkind) :: norm

      integer :: n, i

      norm = 0.0

      do i = 1, n
        norm = norm + x(i)*conjg(x(i))
      end do

      return
      end
