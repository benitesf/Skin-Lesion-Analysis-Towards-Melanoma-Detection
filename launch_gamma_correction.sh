#!/bin/bash
#$ -N segmentation
#$ -q lola
#$ -e salidas/err/preprocessing/gamma_correction_err.txt
#$ -o salidas/out/preprocessing/gamma_correction_out.txt
#$ -cwd

/anaconda/anaconda3/bin/python3 gamma_correction.py