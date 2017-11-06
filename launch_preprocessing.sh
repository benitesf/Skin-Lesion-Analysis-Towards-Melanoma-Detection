#!/bin/bash
#$ -N preprocessing
#$ -e salidas/err/preprocessing_err.txt
#$ -o salidas/out/preprocessing_out.txt
#$ -q v1,lola@lola01,lola@lola02
#$ -t 1-1
#$ -cwd

#DIR[${#DIR[@]}]=results/


##p=/home/miguel/ejemplocluster
##id=$JOB_ID.$SGE_TASK_ID

##mkdir /tmp/$id
##cd /tmp/$id

#mkdir results
##cp -r $p/* .
# copiar todos los datos a "." que es la carpeta del nodo

python3 preprocessing.py

#tar -czvf resultados$id.tar.gz *.jpg
#cp resultados$id.tar.gz $p/results/resultados$id.tar.gz
#cp salida$id.txt $p/salidas/out/salida$id.txt

#cd /tmp
#rm -r /tmp/$id


# t es el numero de tareas
# id se crea en tmp (en el nodo en que se este ejecutando)
# /tmp es el nodo