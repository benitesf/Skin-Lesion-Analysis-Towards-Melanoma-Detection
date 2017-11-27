#!/bin/bash
#$ -N segmentation
#$ -q lola
#$ -e salidas/err/segmentation_red1_pre_err_test.txt
#$ -o salidas/out/segmentation_red1_pre_out_test.txt
#$ -cwd

/anaconda/anaconda3/bin/python3 main.py
