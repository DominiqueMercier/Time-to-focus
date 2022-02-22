#!/bin/bash
set -e -o pipefail

#./execute_randomization.sh |& tee ../../logs/execute_randomization_log.txt

cd ../scripts/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('anomaly_new')

MODELS=('AlexNet')
N_LAYERS=(1 3 5 9 13)

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        for N_LAYER in ${N_LAYERS[@]}
        do
            echo "=================================================="
            echo "Dataset: $DATASET | Model: $MODEL | NLayer: $N_LAYER"
            echo "=================================================="

            python -u main_classification.py --verbose \
                --dataset_name $DATASET --exp_path 'baseline' --runs 1 --architecture $MODEL \
                --standardize --validation_split 0.3 \
                --load_model --save_model --epochs 100 --batch_size 32 \
                --save_report --use_subset --subset_factor 1 --attr_name 'each_class' \
                --process_attributions --compute_attributions --save_memory \
                --randomize_model --randomize_top_down --randomize_n_layers $N_LAYER
        done
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="