#!/bin/bash
set -e -o pipefail

#./execute_randomized_data.sh |& tee ../../logs/execute_randomized_data_log.txt

cd ../scripts/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('CharacterTrajectories')

MODELS=('AlexNet')

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET | Model: $MODEL"
        echo "=================================================="

        python -u main_classification.py --verbose \
            --dataset_name $DATASET --exp_path 'randomized_data' --runs 1 --architecture $MODEL \
            --standardize --validation_split 0.3 --randomize_labels \
            --load_model --save_model --epochs 100 --batch_size 32 \
            --save_report --use_subset --subset_factor 1 --attr_name 'each_class' \
            --process_attributions --compute_attributions --save_memory
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="