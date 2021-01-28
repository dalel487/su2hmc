#!/bin/bash

NAME=../out.b170k1810mu0500j02s16t32 
OUTPUT=accprob.dat

echo 'remove rubbish:'
rm  *~ $OUTPUT

if [ -e $NAME ]
then
  
    echo ' ' 
    
    echo 'File name:' $NAME
    
    echo 'Acc ratio -> ' $OUTPUT
    
#    awk 'BEGIN { ind=0; } /averages for last/ { nnow=$4 ; ind+=1; } /average traj/ {acnow=$4; printf("%5d %10.5f\n", ind, acnow/nnow ) } ' $NAME >> $OUTPUT
 
awk 'BEGIN { ind=0; ntot=0 } /averages for last/ { nnow=$4 ; ind+=1;} /average traj/ {acnow=$4; ntot=ntot+acnow/nnow; printf("%5d %10.5f\n", ind, acnow/nnow ) }; END { printf("\n tot average: %10.5f\n",ntot/ind)  } ' $NAME >> $OUTPUT
   
fi

exit 0
