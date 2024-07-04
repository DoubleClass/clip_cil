#!bin/bash
if [ "$1" != "" ]; then
    echo "Running on gpu: $1"
else
    echo "No gpu has been assigned."
fi


if [ "$2" = "cifar100" ]; then
        CUDA_VISIBLE_DEVICES=$1 python3 main.py --config-path configs/ \
        --config-name='cifar100.yaml'\
        dataset_root="../datasets/" \
        class_order="class_order/cifar100.yaml"  \
        log_path=$3 initial_increment=$4 increment=$5 reuse=$6 \
        ood=$7 linear_probe=$8 EPOCH=$9 seed=${10} custom=${11}
else
        echo "No scenario provided."  
fi
