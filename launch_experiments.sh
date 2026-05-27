#!/bin/bash

# Configuration
PROTOCOLS="t2.yaml"
SEEDS="44,54,66"
#EXPERIMENTS="yolo11_n_polyp.yaml,yolo11_n_polypTST.yaml,yolo11_n_tst.yaml,yolo11_n.yaml,yolo11_s_polyp.yaml,yolo11_s_polypTST.yaml,yolo11_s_tst.yaml,yolo11_s.yaml,yolo5_n_polyp.yaml,yolo5_n_tst.yaml,yolo5_n.yaml,yolo5_s_polyp.yaml,yolo5_s_tst.yaml,yolo5_s.yaml,yolo8_nano.yaml,yolo8_small.yaml"
EXPERIMENTS="yolo11_n.yaml,yolo11_n_tst.yaml"

echo "Starting Training Sweep..."
# The -m flag triggers the Multirun
python src/train.py -m \
    experiment=$EXPERIMENTS \
    protocol=$PROTOCOLS \
    params.seed=$SEEDS \
    params.experiment_name=T2 \

# params.weight_decay=0.001
echo "All experiments completed!"