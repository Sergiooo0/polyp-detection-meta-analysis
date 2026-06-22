#!/bin/bash
SEEDS="42,55,66"

#Collect all models .yaml files in the src/configs/experiments directory
FOLDER="src/configs/experiments"

files=()
for file in "$FOLDER"/*.yaml; do
    files+=("$(basename "$file")")
done

EXPERIMENTS=$(IFS=,; echo "${files[*]}")
#Uncomment the following line to run only specific experiments
#EXPERIMENTS="yolo11_n_tst.yaml,yolo11_n_polypTST.yaml,yolo11_s_polypTST.yaml,yolo11_s_tst.yaml,yolo5_n_tst.yaml,yolo5_s_tst.yaml"

python src/train.py -m \
    experiment=$EXPERIMENTS \
    params.seed=$SEEDS \
    params.experiment_name=polyp_detection