#!/bin/bash

# Configuration
PROTOCOLS="t2"
SEEDS="42,54,66"
EXPERIMENTS="yolo12_n_polyp,yolo12_n"

echo "Starting Training Sweep..."
python src/preprocess.py
# The -m flag triggers the Multirun
python src/train.py -m \
    experiment=$EXPERIMENTS \
    params.protocol=$PROTOCOLS \
    params.seed=$SEEDS \
    params.experiment_name=yolo12_nT2 \
    params.lr=0.001 

# params.weight_decay=0.001
echo "All experiments completed!"
