#!/bin/bash
#$ -N conclusion
#$ -q lola
#$ -e salidas/err/conclusion/0015038_err.txt
#$ -o salidas/out/conclusion/0015038_out.txt
#$ -cwd

/anaconda/anaconda3/bin/python3 test/preprocessing/shading_attenuation_process.py