#!/bin/csh
#$ -N ResNet50
#$ -r Y
#$ -M sbanerj2@nd.edu
#$ -m abe
#$ -q gpu@qa-xp-013
#$ -l gpu_card=1

module load tensorflow/1.3
module load opencv/3.3

setenv CUDA_VISIBLE_DEVICES 1
python /afs/crc.nd.edu/user/s/sbanerj2/Private/UAV_NewThoughts/CIFAR100_classification/cifar100_resnet50.py
