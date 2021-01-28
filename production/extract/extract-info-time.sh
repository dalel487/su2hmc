#!/bin/bash

NAME=../info.tmp
OUTPUT=time.dat

echo 'remove rubbish:'
rm  *~ $OUTPUT

if [ -e $NAME ]
then
  
    echo ' ' 
    
    echo 'IN  :' $NAME
    
    echo 'OUT :' $OUTPUT
    
awk 'BEGIN {i=0; timetot=0; FS = "[ \t :]+" } /START/ {i=i+1;sday=$5; shour=$6; smin=$7; ssec=$8} /END/ {eday=$5; ehour=$6; emin=$7; esec=$8; time=(eday*24*60*60+ehour*60*60+emin*60+esec)-(sday*24*60*60+shour*60*60+smin*60+ssec); printf("%d %d\n",i,time); timetot=timetot+time } END { print i; print timetot/i}' $NAME


awk 'BEGIN {i=0; timetot=0; FS = "[ \t :]+" } /START/ {i=i+1;sday=$5; shour=$6; smin=$7; ssec=$8} /END/ {eday=$5; ehour=$6; emin=$7; esec=$8; time=(eday*24*60*60+ehour*60*60+emin*60+esec)-(sday*24*60*60+shour*60*60+smin*60+ssec); timetot=timetot+time } END { print timetot/i}' $NAME >> $OUTPUT


fi

exit 0
