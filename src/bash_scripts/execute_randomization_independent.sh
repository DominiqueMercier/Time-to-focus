#!/bin/bash
set -e -o pipefail

#./execute_randomization_independent.sh |& tee ../../logs/execute_randomization_independent_log.txt

cd ../scripts/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('CharacterTrajectories')

MODELS=('AlexNet')
LAYERS=(1 3 5 9 13)

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        for LAYER in ${LAYERS[@]}
        do
            echo "=================================================="
            echo "Dataset: $DATASET | Model: $MODEL | Layer: $LAYER"
            echo "=================================================="

            python -u main_classification.py --verbose \
                --dataset_name $DATASET --exp_path 'baseline' --runs 1 --architecture $MODEL \
                --standardize --validation_split 0.3 \
                --load_model --save_model --epochs 100 --batch_size 32 \
                --save_report --use_subset --subset_factor 1 --attr_name 'each_class' \
                --process_attributions --compute_attributions --save_memory \
                --randomize_model --randomize_top_down --randomize_ids $LAYER
        done
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="