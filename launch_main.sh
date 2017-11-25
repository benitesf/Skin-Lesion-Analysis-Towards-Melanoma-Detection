#!/bin/bash
#$ -N segmentation
#$ -q lola
#$ -e salidas/err/segmentation_err.txt
#$ -o salidas/out/segmentation_out.txt
#$ -cwd

/anaconda/anaconda3/bin/python3 main.py
