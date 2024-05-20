#!/bin/sh

config=$1
gpus=$2
bs=$3
output=$4

if [ -z $config ]
then
    echo "No config file found! Run with "sh scripts/train.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [TRAINING_DATA] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh scripts/train.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [TRAINING_DATA] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh scripts/train.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [TRAINING_DATA] [OPTS]""
    exit 0
fi

shift 4
opts=${@}

python train_net.py --config $config \
 --num-gpus $gpus \
 SOLVER.IMS_PER_BATCH $bs \
 OUTPUT_DIR $output \
 $opts
    
