#!/bin/bash
# Basic while loop
counter=1
while [ $counter -le 4 ]
do
  nr=$(printf "%03d" $counter)
  qrencode -s 6 -o QR_${nr}.png "Item ${nr}"
  convert -bordercolor black -border 1 QR_${nr}.png QR_${nr}.png
  ((counter++))
done
echo All done

