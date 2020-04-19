################################################
#
#  Developed by Dr. Renzhi Cao
#  Pacific Lutheran University
#  Email: caora@plu.edu
#  Website: https://cs.plu.edu/~caora/
#
###############################################

import os
import sys
from os import listdir
from os.path import isfile, join

if len(sys.argv)<5:
    print("This is a wrapper file to call my script and make predictions")
    print("python "+sys.argv[0]+" ProLanDO_oneFasta_model1.py ../data/TargetFiles ../FinalBestModel/training_DO_RNN_512_0.009000000000000001.model ../result/CaoLab_DO_model1")
    sys.exit(0)

scr = sys.argv[1]
input = sys.argv[2]
model = sys.argv[3]
modelNum = "1"
output = sys.argv[4]

os.system("mkdir "+output)

onlyfiles = [f for f in listdir(input) if isfile(join(input, f))]

for f in onlyfiles:
   pathIn = join(input,f)
   name = f.split(".")[1]
   pathOut = output+"/ProLanDO_"+f+"_do.txt"
   os.system("python "+scr+" "+model+" "+pathIn+" "+pathOut)

