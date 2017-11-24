#!/bin/bash
#$ -N preprocessing
#$ -q lola
#$ -e salidas/err/preprocessing_err.txt
#$ -o salidas/out/preprocessing_out.txt
#$ -cwd

/anaconda/anaconda3/bin/python3 preprocessing.py

