#!/bin/bash
#$ -N segmentation
#$ -q lola
#$ -e salidas/err/red1/segmentation_err.txt
#$ -o salidas/out/red1/segmentation_out.txt
#$ -cwd

/anaconda/anaconda3/bin/python3 main.py
