#!/bin/bash
set -e -o pipefail

#./execute_baseline_extended.sh |& tee ../../logs/execute_baseline_extended_log.txt

cd ../scripts/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('Crop')

MODELS=('AlexNet')

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET | Model: $MODEL"
        echo "=================================================="

        python -u main_classification.py --verbose \
            --dataset_name $DATASET --exp_path 'baseline_ext' --runs 1 --architecture $MODEL \
            --standardize --validation_split 0.3 \
            --load_model --save_model --epochs 100 --batch_size 32 \
            --save_report --use_subset --subset_factor 100 --attr_name '100' \
            --process_attributions --compute_attributions --save_memory
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="