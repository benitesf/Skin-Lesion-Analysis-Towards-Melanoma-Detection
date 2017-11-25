#!/bin/bash
#$ -N segmentation
#$ -q lola
#$ -e salidas/err/segmentation_err.txt
#$ -o salidas/out/segmentation_out.txt
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=8G

/anaconda/anaconda3/bin/python3 main.py
