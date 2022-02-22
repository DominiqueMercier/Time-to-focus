#!/bin/bash
set -e -o pipefail

#./execute_statistics.sh |& tee ../../logs/execute_statistics_log.txt

cd ../scripts/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('anomaly_new' 'CharacterTrajectories' 'ECG5000' 'FaceDetection' 'FordA' 'UWaveGestureLibraryAll')

MODELS=('AlexNet')

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET | Model: $MODEL"
        echo "=================================================="

        python -u main_classification.py --verbose \
            --dataset_name $DATASET --exp_path 'baseline' --runs 1 --architecture $MODEL \
            --standardize --validation_split 0.3 --load_model --batch_size 32 \
            --use_subset --subset_factor 100 --evaluate_subset --save_subset_report \
            --create_comparer --comparer_exp_names 'baseline,randomized_data' --save_dicts --save_plots \
            --not_show_plots --compute_sensitivity --compute_time --compute_continuity --compute_correlation \
            --compute_modified_accs --compute_infidelity --compute_correlation_mat --compute_modified_acc_dict \
            --compute_agreements
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="