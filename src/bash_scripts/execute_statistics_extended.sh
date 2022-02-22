#!/bin/bash
set -e -o pipefail

#./execute_statistics_extended.sh |& tee ../../logs/execute_statistics_extended_log.txt

cd ../scripts/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

#DATASETS=('AsphaltPavementType' 'AsphaltRegularity' 'ElectricDevices' \
#            'HandOutlines' 'MedicalImages' 'MelbournePedestrian' 'NonInvasiveFetalECGThorax1' \
#            'PhalangesOutlinesCorrect' 'Strawberry' 'Wafer')
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
            --standardize --validation_split 0.3 --load_model --batch_size 32 \
            --use_subset --subset_factor 100 --evaluate_subset --save_subset_report \
            --create_comparer --comparer_exp_names 'baseline_ext' --save_dicts --save_plots \
            --not_show_plots --compute_modified_accs --compute_modified_acc_dict --compute_agreements
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="