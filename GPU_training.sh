#!/bin/bash
#PBS -P cortical
#PBS -N Graiden_noise
#PBS -q defaultQ 
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb
#PBS -l walltime=167:59:59
#PBS -e PBSout/
#PBS -o PBSout/
##PBS -J 1-30

# make sure you have built a virtual system on python 3.6.5 
# and install pytorch just like how you install tensorflow on HPC:
# 	virtualenv --system-site-packages tf #tf is in your home directory
#	module load cuda/9.1.85 openmpi-gcc/3.0.0-cuda
#	pip install /usr/local/pytorch/torch-1.0.0a0+1a247f8.magma.cuda.9.1-cp36-cp36m-linux_x86_64.whl

cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
#params=`sed "${PBS_ARRAY_INDEX}q;d" job_params`
#param_array=( $params )
#python3 main.py --patch_size=13 --num_patches=1 --loc_hidden=256 --glimpse_hidden=128 --num_glimpses=10 --valid_size=0.1 --batch_size=256 --batchnorm_flag_phi=True --batchnorm_flag_l=True --batchnorm_flag_g=True --batchnorm_flag_h=True --glimpse_scale=1 --weight_decay=0.002 --dropout_phi=0.2 --dropout_l=0.3 --dropout_g=0.2 --dropout_h=0.3 --use_gpu=False --dataset_name='CIFAR' --train_patience=50 --epochs=500

#--batch_szie= --loc_hidden=192 --hidden_size=320 --glimpse_hidden= --num_glimpse= --glimpse_scale= --loss_fun_action= --loss_fun_baseline= 
#python mnist_class_11_WTA.py
python main.py 
python main.py --model='alexnet' --dataset='cifar10'