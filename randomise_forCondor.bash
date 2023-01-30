#!/bin/sh

inputFile=$1
design=$2
#mask=$3

datPath='/drive/folder/'

maskPath="/drive/maskFolder/"

# use this for running with a mask
# randomise -i ${datPath}${inputFile}/${inputFile}_4D.nii.gz -o ${datPath}${inputFile}/${design} -d ${datPath}${inputFile}/${design}.mat -t ${datPath}${inputFile}/${design}.con -n 5000 -T -m ${maskPath}${mask}

randomise -i ${datPath}${inputFile}/${inputFile}_4D.nii.gz -o ${datPath}${inputFile}/${design} -d ${datPath}${inputFile}/${design}.mat -t ${datPath}${inputFile}/${design}.con -n 5000 -T
