#!/bin/bash

echo -n "Give me the number for log: "
read NUM

if [ -z  $NUM ]
then
echo "You have to give a number. Try again!"
exit 1
fi


mkdir log"$NUM"
rm -f con con???
mv su2production.sh.e???? ./log"$NUM"/
mv su2production.sh.o???? ./log"$NUM"/
mv *.b*k*mu*j*s*t* ./log"$NUM"/
mv info.tmp ./log"$NUM"/
mv logsu2exe.log ./log"$NUM"/
mv midout ./log"$NUM"/

exit 0
