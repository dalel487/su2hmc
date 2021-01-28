#!/bin/bash -e

#PBS -q nodes128
#PBS -l walltime=24:00:00

# Set the BG options for the job
export BG_THREADLAYOUT=1
export BG_SHAREDMEMSIZE=32
export BG_ALLOW_CACHELINE_LOCKING=1
export BG_L2LOCK_L1_ONLY=1
export L1P_POLICY=std

#ulimit -s unlimited

DIR=$PBS_O_WORKDIR
cd $DIR

### make output files group read and writeable
umask 007

# Set the number of parallel tasks
export PTASKS=2048
export PTASKS_PER_NODE=16
export OMP_NUM_THREADS=4

###################################################

echo -n "START time  : " >> info.tmp
date >> info.tmp

# physical parameters ##
ss=16
tt=32
beta=1.7
kappa=0.1810
mu=0.50
j=0.02

# hmc parameters
dt=0.00172
iterlen=581

# run parameters
nconf=5 #total confs in this simul


# derived parameters
b=$(echo $beta | awk '{print $1*100}')
k=$(echo $kappa | awk '{print $1*10000}')
m=$(echo $mu | awk '{printf("%04d",$1*1000)}')
jj=$(echo $j | awk '{printf("%02d",$1*100)}')
suffix=b${b}k${k}mu${m}j${jj}s${ss}t${tt}

PROG=./su2hmc.exe
MPIRUN=/bgsys/drivers/ppcfloor/bin/runjob

cat << EOF > midout
$dt     $beta   $kappa $j   0.000000   $mu   0.5  $iterlen   $nconf
  dt     beta   akappa jqq  thetaq      fmu   aNf  iterl  iter2
EOF

# Run the program on the Blue Gene/Q:
$MPIRUN --block $PBS_QUEUE -n $PTASKS -p $PTASKS_PER_NODE --exe $PROG --env_all >> logsu2exe.log

#tidy up
mv fort.11 fermi.$suffix
mv fort.12 bose.$suffix
mv fort.13 diq.$suffix
mv output out.$suffix

cp con$(printf "%03d" $nconf) con.$suffix

#rm con

echo -n "END time  : " >> info.tmp
date >> info.tmp

exit 0
