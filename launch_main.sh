#!/bin/bash
#$ -N segmentation
#$ -q lola
#$ -e salidas/err/red2/segmentation_pre_err.txt
#$ -o salidas/out/red2/segmentation_pre_out.txt
#$ -cwd

/anaconda/anaconda3/bin/python3 main.py
