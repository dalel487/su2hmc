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

THISJOB=su2production.sh 

# physical parameters ##
ss=16
tt=32
beta=1.7
kappa=0.1810
mu=0.15
j=0.04

# hmc parameters
dt=0.0032
iterlen=312

# run parameters
chain=0
nconf=10 #total confs in this simul
save=5   #save in su2code
end=1000 #tot confs at the end

# derived parameters
b=$(echo $beta | awk '{print $1*100}')
k=$(echo $kappa | awk '{print $1*10000}')
m=$(echo $mu | awk '{printf("%04d",$1*1000)}')
jj=$(echo $j | awk '{printf("%02d",$1*100)}')
suffix=b${b}k${k}mu${m}j${jj}s${ss}t${tt}

#REPOSITORY=/dirac1/work/dp006/dp006/giudice/SU2/Configs/V${ss}x${tt}/J${jj}/MU$m
REPOSITORY=/dirac1/work/dp006/dp006/dc-boz1/SU2/Configs/V${ss}x${tt}/J${jj}/MU$m

PROG=./su2hmc.exe
MPIRUN=/bgsys/drivers/ppcfloor/bin/runjob

cat << EOF > midout
$dt     $beta   $kappa $j   0.000000   $mu   0.5  $iterlen   $nconf
  dt     beta   akappa jqq  thetaq      fmu   aNf  iterl  iter2
EOF

# Run the program VN Mode on the Blue Gene/P:
ln -s con.$suffix con
$MPIRUN --block $PBS_QUEUE -n $PTASKS -p $PTASKS_PER_NODE --exe $PROG --env_all >> logsu2exe.log

#tidy up
cat fort.11 >> fermi.$suffix
cat fort.12 >> bose.$suffix
cat fort.13 >> diq.$suffix
cat output >> out.$suffix
cp con$(printf "%03d" $nconf) con.$suffix

rm fort.11 fort.12 fort.13 output con

# save configs to repository
if [ -f confignum.${suffix} ]; then
 start=$(cat confignum.${suffix})
else
 start=0
fi
stop=$((start+$nconf))
begin=$((((start+$save)/$save)*$save))
becf=$((begin-$start))
for ((cf=$becf,cfg=$begin;cfg<=$stop;cfg+=$save,cf+=$save)); do
  cp -p con$(printf "%03d" $cf) $REPOSITORY/config.${suffix}.$(printf "%d%05d" $chain $cfg)
done

echo $stop > confignum.${suffix}

if [ $stop -lt $end ]; then
  qsub $THISJOB
fi

echo " " >> logsu2exe.log
echo -n "Conf  : " >> logsu2exe.log
cat confignum.${suffix} >> logsu2exe.log
echo " " >> logsu2exe.log

echo -n "Conf  : " >> info.tmp
cat confignum.${suffix} >> info.tmp

echo -n "END time  : " >> info.tmp
date >> info.tmp

exit 0
