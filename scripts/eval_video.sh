#!/bin/sh

config=$1
gpus=$2
val_data=$3
test_num_class=$4
output=$5
weight=$6


if [ -z $config ]
then
    echo "No config file found! Run with "sh scripts/eval_video.sh [CONFIG] [NUM_GPUS] [VAL_DATA] [TEST_NUM_CLASS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh scripts/eval_video.sh [CONFIG] [NUM_GPUS] [VAL_DATA] [TEST_NUM_CLASS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $val_data ]
then
    echo "No val dataset found! Run with "sh scripts/eval_video.sh [CONFIG] [NUM_GPUS] [VAL_DATA] [TEST_NUM_CLASS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $test_num_class ]
then
    echo "Number of test classes not specified! Run with "sh scripts/eval_video.sh [CONFIG] [NUM_GPUS] [VAL_DATA] [TEST_NUM_CLASS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh scripts/eval_video.sh [CONFIG] [NUM_GPUS] [VAL_DATA] [TEST_NUM_CLASS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 6
opts=${@}

python train_net_video.py --config $config \
 --num-gpus $gpus \
 --eval-only \
 DATASETS.TEST $val_data \
 OUTPUT_DIR $output \
 TEST.TEST_NUM_CLASSES $test_num_class  \
 MODEL.WEIGHTS $weight
 $opts
python tools/mAP.py --et $val_data \
--dt $output


# python mAP.py --gt datasets/lvvis/val/val_instances.json \
#     --dt output/lvvis-video-base-openai/vote/11t5/inference/results.json \
# 	--et v \
# 	--nt results