import importlib
import os
import sys
from optparse import OptionParser

import numpy as np
import torch
from torchinfo import summary

############## Import modules ##############
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from modules.utils import mean_cr_utils, utils
from modules.networks.model import FCN, FDN, LSTM, AlexNet, Inception
from modules.networks import model_utils
from modules.datasets import dataset_utils, pkl_loader, ucr_loader
from modules.attributions.attribution_processor import ClassificationProcessor
from modules.attributions.attribution_config import \
    config as default_attr_config
from modules.attributions.attribution_comparer import (AttributionComparer,
                                                       gather_all_attrs,
                                                       shorten_attr_names)
from modules.attributions import plotter

def process(options):
    ########## Global settings #############
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir, result_dir = utils.maybe_create_dirs(
        options.dataset_name, root='../../', dirs=['models', 'results'], exp=options.exp_path, return_paths=True, verbose=options.verbose)

    ######### Dataset processing ###########
    dataset_dict = ucr_loader.get_datasets(options.root_path, prefix='**/')
    split_id = None
    try:
        trainX, trainY, testX, testY = ucr_loader.load_data(
            dataset_dict[options.dataset_name])
        valX, valY = None, None
    except:
        trainX, trainY, valX, valY, testX, testY = pkl_loader.load_data(os.path.join(
            options.root_path, options.dataset_name, options.dataset_file), is_channel_first=options.is_channel_first)
    if valX is not None:
        trainX, trainY, split_id = dataset_utils.fuse_train_val(
            trainX, trainY, valX, valY)
    trainX, trainY, testX, testY = dataset_utils.preprocess_data(
        trainX, trainY, testX, testY, normalize=options.normalize, standardize=options.standardize, channel_first=True)
    if split_id is None:
        trainX, trainY, valX, valY = dataset_utils.perform_datasplit(
            trainX, trainY, test_split=options.validation_split)
    else:
        trainX, trainY, valX, valY = dataset_utils.unfuse_train_val(
            trainX, trainY, split_id)
    channels, timesteps = trainX.shape[1:]
    n_classes = len(np.unique(trainY))

    if options.verbose:
        print('TrainX:', trainX.shape)
        print('ValX:', valX.shape)
        print('TestX:', testX.shape)
        print('Classes:', n_classes)

    ##### Subset creation for attr #########
    if options.use_subset:
        sub_testX, sub_testY, sub_ids = dataset_utils.select_subset(
            testX, testY, options.subset_factor)
    else:
        sub_testX, sub_testY, sub_ids = testX, testY, np.arange(
            testX.shape[0])

    ######## Label randomization ###########
    if options.randomize_labels:
        if options.verbose:
            print('Randomized Train labels and set val to train')
        trainY = dataset_utils.randomize_labels(trainY)
        valX, valY = trainX, trainY

    ######### Data loader creation #########
    trainloader = model_utils.create_dataloader(
        trainX, trainY, batch_size=options.batch_size, shuffle=True, drop_last=False, num_workers=8)
    valloader = model_utils.create_dataloader(
        valX, valY, batch_size=options.batch_size, shuffle=False, drop_last=False, num_workers=8)

    ######### Model architecture ###########
    architecture_func = {'AlexNet': AlexNet,
                         'FCN': FCN, 'FDN': FDN, 'LSTM': LSTM,
                         'Inception': Inception}

    ########## Run wise processing #########
    report_paths = []
    for run_id in range(options.runs):
        if options.verbose:
            print('Run %d / %d' % (run_id+1, options.runs))
        ####### Perform baseline model #########
        model_setup = options.architecture + '_batch-' + \
            str(options.batch_size) + '_run-' + str(run_id)
        model_setup_backup = model_setup
        model_path = os.path.join(
            model_dir, model_setup + '.pt') if options.save_model or options.load_model else None

        model = architecture_func[options.architecture](
            timesteps, channels, n_classes).to(device)
        if options.verbose:
            summary(model, input_size=(
                options.batch_size, channels, timesteps))
            pass

        if os.path.exists(model_path) and options.load_model:
            model.load_state_dict(torch.load(model_path))
        else:
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=5, verbose=options.verbose)

            model_utils.train(model, trainloader, valloader, options.epochs, optimizer, criterion,
                              lr_scheduler=scheduler, early_patience=10, path=model_path, verbose=options.verbose)
        model.eval()

        ######## Randomized modificaiton #######
        if options.randomize_model:
            layer_ids = np.fromstring(
                options.randomize_ids, dtype=int, sep=',')
            model = model_utils.randomize_layers(model, layer_ids=layer_ids, n_layers=options.randomize_n_layers,
                                                 top_down=options.randomize_top_down, verbose=options.verbose)

            model_setup = model_utils.get_randomized_path(model_setup_backup, layer_ids=options.randomize_ids, n_layers=options.randomize_n_layers,
                                                          top_down=options.randomize_top_down)

        ############# Evaluation ###############
        report_path = os.path.join(
            result_dir, model_setup + '_report.txt') if options.save_report or options.save_mcr else None
        outs = model(torch.Tensor(testX).to(device)).detach().to('cpu')
        preds = torch.argmax(outs, dim=1).numpy()
        utils.compute_classification_report(
            testY, preds, save=report_path, verbose=options.verbose, store_dict=True)
        if report_path is not None:
            report_paths.append(report_path.replace('.txt', '.pickle'))

        ############ Evaluate subset ###########
        if options.evaluate_subset or options.create_comparer:
            if options.use_subset:
                subset_folder = os.path.join(
                    result_dir, model_setup, 'subset_' + str(options.subset_factor))
            else:
                subset_folder = os.path.join(
                    result_dir, model_setup, 'complete')
            os.makedirs(subset_folder, exist_ok=True)

            subset_report_path = os.path.join(subset_folder, 'acc_report.txt') \
                if options.save_subset_report else None
            outs = model(torch.Tensor(sub_testX).to(device)).detach().cpu()
            preds = torch.argmax(outs, dim=1).numpy()
            utils.compute_classification_report(
                sub_testY, preds, save=subset_report_path, verbose=options.verbose, store_dict=True)

        ######### Attribution ##################
        if options.process_attributions:
            config_file = default_attr_config
            if options.attr_config is not None:
                config_spec = importlib.util.spec_from_file_location(
                "attr_config", options.attr_config)
                config_mod = importlib.util.module_from_spec(config_spec)
                config_spec.loader.exec_module(config_mod)
                config_file = config_mod.config
            attr_dir = None
            if not options.not_save_attributions:
                attr_dir = os.path.join(
                    model_dir, model_setup, options.attr_name)
                os.makedirs(attr_dir, exist_ok=True)
                if options.use_subset:
                    np.save(os.path.join(attr_dir, 'Sub_ids.npy'), sub_ids)
            attrProcessor = ClassificationProcessor(
                model, trainX.shape[1:], config_file, save_memory=options.save_memory, attr_dir=attr_dir,
                load=not options.compute_attributions, verbose=options.verbose)

            if options.compute_attributions:
                if options.verbose:
                    print('Use %s samples for attribution' % sub_ids.shape[0])
                attrProcessor.process_attributions(
                    sub_testX, sub_testY, attr_dir)

            if options.plot_attributions:
                attrProcessor.plot_approaches(
                    sub_testX, index=options.plot_index, not_show=options.not_show_plots, save_path=subset_folder if options.save_plots else None)

        #### Create temporarily processor ######
        if options.compute_sensitivity or options.compute_time:
            config_file = default_attr_config if options.attr_config is None else options.attr_config
            tmp_attrProcessor = ClassificationProcessor(
                model, trainX.shape[1:], config_file, save_memory=options.save_memory, load=False, verbose=options.verbose)

        ########### Compute Sensitivity ########
        if options.compute_sensitivity:
            x = sub_testX[options.plot_index:options.plot_index+1]
            y = sub_testY[options.plot_index:options.plot_index+1]
            tmp_attrProcessor.compute_sensitivity(
                x, y, perturb_radius=options.sensitivity_scale, n_perturb_samples=10, verbose=options.verbose)
            sensitivities = tmp_attrProcessor.gather_dict('sensitivity')

            if options.save_dicts:
                dict_path = os.path.join(
                    subset_folder, 'Attribution_sensitivity.txt')
                utils.get_pretty_dict(
                    sensitivities, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

            if options.save_plots or not options.not_show_plots:
                plotter.plot_bars(
                    sensitivities, 'sensitivity', ylabel='Sensitivity', not_show=options.not_show_plots, 
                    save_path=subset_folder if options.save_plots else None)

        ############## Compute Time ############
        if options.compute_time:
            x = sub_testX[options.plot_index:options.plot_index+1]
            y = sub_testY[options.plot_index:options.plot_index+1]
            tmp_attrProcessor.process_attributions(x, y)
            times = tmp_attrProcessor.gather_dict('time')

            if options.save_dicts:
                dict_path = os.path.join(
                    subset_folder, 'Attribution_time.txt')
                utils.get_pretty_dict(
                    times, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

            if options.save_plots or not options.not_show_plots:
                plotter.plot_bars(
                    times, 'time', ylabel='Seconds', not_show=options.not_show_plots, save_path=subset_folder if options.save_plots else None)

        ########### Attribution Comparer #######
        if options.create_comparer:
            attr_paths = gather_all_attrs(
                options.dataset_name, exp_names=options.comparer_exp_names, architectures=options.comparer_architectures,
                attr_paths=options.comparer_attr_paths, attr_names=options.comparer_attr_names, exclude=options.comparer_exclude)
            attr_names = shorten_attr_names(attr_paths)
            attrComparer = AttributionComparer()
            attrComparer.load_attributions(attr_paths, attr_names)
            attrComparer.group_by_method()

            ########### Plot comparer grid #########
            if options.plot_comparer_sample:
                attrComparer.plot_grid(sub_testX, options.plot_index, not_show=options.not_show_plots, save_path=subset_folder)

            ########### Compute Continuity #########
            if options.compute_continuity:
                continuities = attrComparer.compute_method_continuity(key='B_')
                if options.save_dicts:
                    dict_path = os.path.join(
                        subset_folder, 'Attribution_continuity.txt')
                    utils.get_pretty_dict(
                        continuities, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

                if options.save_plots or not options.not_show_plots:
                    plotter.plot_bars(
                        continuities, 'continuity', ylabel='Distance', not_show=options.not_show_plots, 
                        save_path=subset_folder if options.save_plots else None)

            ########### Compute correlation ########
            if options.compute_correlation:
                for perc in np.fromstring(options.correlation_perc, sep=','):
                    for corr in options.correlations.split(','):
                        attr_correlations = attrComparer.compute_corr_to_first(
                            map_ids=None, mode=corr, histogram=0, perc=perc)

                        if options.save_dicts:
                            dict_path = os.path.join(
                                subset_folder, 'Correlation_' + corr + '_perc-' + str(perc) + '.txt')
                            utils.get_pretty_dict(
                                attr_correlations, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

                        if options.save_plots or not options.not_show_plots:
                            plotter.plot_correlations(
                                attr_correlations, mode=corr, perc=perc, not_show=options.not_show_plots, 
                                save_path=subset_folder if options.save_plots else None)

            ########## Compute modified accs #######
            if options.compute_modified_accs:
                modified_accs = attrComparer.compute_modified_accs(
                    model, sub_testX, sub_testY, replace_strategy='zeros', perc=options.modified_accs_perc,
                    keep_smaller=options.modified_accs_keep_smaller)

                if options.save_dicts:
                    dict_path = os.path.join(subset_folder, 'Modified_Accuracies_perc-' + str(
                        options.modified_accs_perc) + '_keep_smaller-' + str(options.modified_accs_keep_smaller) + '.txt')
                    utils.get_pretty_dict(
                        modified_accs, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

                if options.save_plots or not options.not_show_plots:
                    plotter.plot_modified_accs(modified_accs, perc=options.modified_accs_perc, keep_smaller=options.modified_accs_keep_smaller,
                                               not_show=options.not_show_plots, save_path=subset_folder if options.save_plots else None)

            ########### Compute infidelity #########
            if options.compute_infidelity:
                infidalities = attrComparer.compute_infidelity(
                    model, sub_testX, sub_testY, scale=options.infidelity_scale, n_perturb_samples=100)

                if options.save_dicts:
                    dict_path = os.path.join(subset_folder, 'Infidelity.txt')
                    utils.get_pretty_dict(
                        infidalities, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

                if options.save_plots or not options.not_show_plots:
                    plotter.plot_infidalities(
                        infidalities, not_show=options.not_show_plots, save_path=subset_folder if options.save_plots else None)

            ####### Compute correlation matrix #####
            if options.compute_correlation_mat:
                for perc in np.fromstring(options.correlation_perc, sep=','):
                    for corr in options.correlations.split(','):
                        if perc < 100 and corr != 'jaccard':
                            continue
                        base_attr_corr_mat_pd = attrComparer.compute_corr_mat(
                            key='B_', mode=corr, perc=perc)
                        if options.save_dicts:
                            base_attr_corr_mat_dict = base_attr_corr_mat_pd.to_dict()
                            dict_path = os.path.join(
                                subset_folder, 'Correlation_Matrix_' + corr + '_perc-' + str(perc) + '_key-B.txt')
                            utils.get_pretty_dict(
                                base_attr_corr_mat_dict, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

                        if options.save_plots or not options.not_show_plots:
                            plotter.plot_correlation_matrix(
                                base_attr_corr_mat_pd, mode=corr, perc=perc, not_show=options.not_show_plots, 
                                save_path=subset_folder if options.save_plots else None)

            ######### Compute modified_dict ########
            if options.compute_modified_acc_dict:
                base_attr_names = [n for n in sorted(
                    attrComparer.attributions) if 'B_' in n]
                pr = np.fromstring(
                    options.compute_modified_acc_dict_ranges, sep=',')
                larger_dict, smaller_dict = np.arange(
                    pr[0], pr[1], pr[2]), np.arange(pr[3], pr[4], pr[5])
                modified_acc_dict = attrComparer.compute_modified_acc_dict(
                    model, sub_testX, sub_testY, base_attr_names, percs=[larger_dict, smaller_dict])

                if options.save_dicts:
                    dict_path = os.path.join(
                        subset_folder, 'Modified_Accuracies_Matrix.txt')
                    utils.get_pretty_dict(
                        modified_acc_dict, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

                if options.save_plots or not options.not_show_plots:
                    plotter.plot_masked_dict(
                        modified_acc_dict, not_show=options.not_show_plots, save_path=subset_folder if options.save_plots else None)

            ######## Compute required ratios ########
            if options.compute_agreements:
                base_attr_names = [n for n in sorted(
                    attrComparer.attributions) if 'B_' in n]
                agree = np.arange(*np.fromstring(options.agreement_percs, sep=','))
                agreement_dict = attrComparer.compute_agreement_dict(model, sub_testX, base_attr_names, agree, verbose=options.verbose)

                if options.save_dicts:
                    dict_path = os.path.join(
                        subset_folder, 'Modified_Agreement_Matrix.txt')
                    utils.get_pretty_dict(
                        agreement_dict, sort=True, save=dict_path if options.save_dicts else None, verbose=options.verbose)

                if options.save_plots or not options.not_show_plots:
                    plotter.plot_agreements(
                        agreement_dict, not_show=options.not_show_plots, save_path=subset_folder if options.save_plots else None)

    ###### Create mean eval report #########
    if options.save_mcr:
        mean_report_path = os.path.join(result_dir, report_paths[0].replace(
            '_run-0', '').replace('_report.txt', '_mean-report.txt'))
        mean_cr_utils.compute_meanclassification_report(
            report_paths, save=mean_report_path, verbose=options.verbose, store_dict=True)


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    ########## Global settings #############
    parser.add_option("--verbose", action="store_true",
                      dest="verbose", help="Flag to verbose")
    parser.add_option("--seed", action="store", type=int,
                      dest="seed", default=0, help="random seed")

    ######### Dataset processing ###########
    parser.add_option("--root_path", action="store", type=str,
                      dest="root_path", default="../../data/", help="Path that includes the different datasets")
    parser.add_option("--dataset_name", action="store", type=str,
                      dest="dataset_name", default="ElectricDevices", help="Name of the dataset folder")
    parser.add_option("--dataset_file", action="store", type=str,
                      dest="dataset_file", default="dataset.pickle", help="Name of the dataset file")
    parser.add_option("--normalize", action="store_true",
                      dest="normalize", help="Flag to normalize the data")
    parser.add_option("--standardize", action="store_true",
                      dest="standardize", help="Flag to standardize the data")
    parser.add_option("--validation_split", action="store", type=float,
                      dest="validation_split", default=0.3, help="Creates a validation set, set to zero to exclude validation set")
    parser.add_option("--is_channel_first", action="store_true",
                      dest="is_channel_first", help="Flag to process a dataset that is already channel first format")

    ######## Dataset modifications #########
    parser.add_option("--randomize_labels", action="store_true",
                      dest="randomize_labels", help="Flag to randomize labels")
    parser.add_option("--use_subset", action="store_true",
                      dest="use_subset", help="Flag to use a subset for later attribution")
    parser.add_option("--subset_factor", action="store", type=float,
                      dest="subset_factor", default=100, help="Creates a subset for later attribution processing")

    ######### Experiment details ###########
    parser.add_option("--runs", action="store", type=int,
                      dest="runs", default=1, help="Number of runs to execute")
    parser.add_option("--exp_path", action="store", type=str,
                      dest="exp_path", default=None, help="Sub-Folder for experiment setup")
    parser.add_option("--architecture", action="store", type=str,
                      dest="architecture", default='AlexNet', help="AlexNet, FCN, LSTM, FDN, Inception")

    ####### Perform baseline model #########
    parser.add_option("--load_model", action="store_true",
                      dest="load_model", help="Flag to load an existing model")
    parser.add_option("--save_model", action="store_true",
                      dest="save_model", help="Flag to save the model")
    parser.add_option("--epochs", action="store", type=int,
                      dest="epochs", default=100, help="Number of epochs")
    parser.add_option("--batch_size", action="store", type=int,
                      dest="batch_size", default=32, help="Batch size for training and prediction")

    ########## Model modifications #########
    parser.add_option("--randomize_model", action="store_true",
                      dest="randomize_model", help="Flag to randomize the model layers")
    parser.add_option("--randomize_top_down", action="store_true",
                      dest="randomize_top_down", help="Flag to randomize the model layers in top down order")
    parser.add_option("--randomize_n_layers", action="store", type=int,
                      dest="randomize_n_layers", default=None, help="Number of randomized layers")
    parser.add_option("--randomize_ids", action="store", type=str,
                      dest="randomize_ids", default="1", help="Ids to randomize based on order, comma separated")

    ############# Evaluation ###############
    parser.add_option("--save_report", action="store_true",
                      dest="save_report", help="Flag to save the evaluation report")
    parser.add_option("--save_mcr", action="store_true",
                      dest="save_mcr", help="Flag to save the mean evaluation report")

    ########## Evaluation subset ###########
    parser.add_option("--evaluate_subset", action="store_true",
                      dest="evaluate_subset", help="Flag evaluate the subset")
    parser.add_option("--save_subset_report", action="store_true",
                      dest="save_subset_report", help="Flag to save the subset report")

    ################ Save details ###########
    parser.add_option("--plot_index", action="store", type=int,
                      dest="plot_index", default=0, help="index to plot")
    parser.add_option("--save_dicts", action="store_true",
                      dest="save_dicts", help="Flag to save all dictionaries")
    parser.add_option("--save_plots", action="store_true",
                      dest="save_plots", help="Flag to save all plots")
    parser.add_option("--not_show_plots", action="store_true",
                      dest="not_show_plots", help="Flag to hide plots")

    ########## Attribution details #########
    parser.add_option("--process_attributions", action="store_true",
                      dest="process_attributions", help="Flag to process (save or load) attributions")
    parser.add_option("--not_save_attributions", action="store_true",
                      dest="not_save_attributions", help="Flag to not save attributions")
    parser.add_option("--attr_config", action="store", type=str,
                      dest="attr_config", default=None, help="Path to the attribution_config file")
    parser.add_option("--compute_attributions", action="store_true",
                      dest="compute_attributions", help="Flag to create new attributions")
    parser.add_option("--plot_attributions", action="store_true",
                      dest="plot_attributions", help="Flag to plot attributions")
    parser.add_option("--attr_name", action="store", type=str,
                      dest="attr_name", default="default", help="Name to identify attribution set")
    parser.add_option("--save_memory", action="store_true",
                      dest="save_memory", help="Flag to save memory")

    ########### Compute Sensitivity ########
    parser.add_option("--compute_sensitivity", action="store_true",
                      dest="compute_sensitivity", help="Flag to compute attribution sensitivity")
    parser.add_option("--sensitivity_scale", action="store", type=float,
                      dest="sensitivity_scale", default=0.02, help="Infidelity scale value")

    ############## Compute Time ############
    parser.add_option("--compute_time", action="store_true",
                      dest="compute_time", help="Flag compute the attribution times")

    ######### Attribution comparison #######
    parser.add_option("--create_comparer", action="store_true",
                      dest="create_comparer", help="Gathers all attributions for a dataset and creates comparer")
    parser.add_option("--comparer_exp_names", action="store", type=str,
                      dest="comparer_exp_names", default=None, help="Experiment names included in comparer")
    parser.add_option("--comparer_architectures", action="store", type=str,
                      dest="comparer_architectures", default=None, help="Architecture names included in comparer")
    parser.add_option("--comparer_attr_paths", action="store", type=str,
                      dest="comparer_attr_paths", default=None, help="Attribution exp names included in comparer")
    parser.add_option("--comparer_attr_names", action="store", type=str,
                      dest="comparer_attr_names", default=None, help="Attribution method names included in comparer")
    parser.add_option("--comparer_exclude", action="store", type=str,
                      dest="comparer_exclude", default=None, help="Setups excluded from comparer processing")

    ########### Plot comparer grid #########
    parser.add_option("--plot_comparer_sample", action="store_true",
                      dest="plot_comparer_sample", help="Plots the grid for a specific sample")

    ########### Compute Continuity #########
    parser.add_option("--compute_continuity", action="store_true",
                      dest="compute_continuity", help="Flag compute the attribution times")

    ########### Compute correlation ########
    parser.add_option("--compute_correlation", action="store_true",
                      dest="compute_correlation", help="Compute correlation of attribution maps")
    parser.add_option("--correlations", action="store", type=str,
                      dest="correlations", default="pearsonr,spearmanr,jaccard", help="Names of the correlations (pearsonr, spearmanr, jaccard)")
    parser.add_option("--correlation_perc", action="store", type=str,
                      dest="correlation_perc", default="10,20,100", help="Ranges correlations")

    ########## Compute modified accs #######
    parser.add_option("--compute_modified_accs", action="store_true",
                      dest="compute_modified_accs", help="Compute modified samples and accs")
    parser.add_option("--modified_accs_perc", action="store", type=int,
                      dest="modified_accs_perc", default=95, help="Perentage of data to keep")
    parser.add_option("--modified_accs_keep_smaller", action="store", type=int,
                      dest="modified_accs_keep_smaller", default=1, help="Flag to keep the smaller alues")

    ########### Compute infidelity #########
    parser.add_option("--compute_infidelity", action="store_true",
                      dest="compute_infidelity", help="Compute infidelity of attributions")
    parser.add_option("--infidelity_scale", action="store", type=float,
                      dest="infidelity_scale", default=0.2, help="Infidelity scale value")

    ####### Compute correlation matrix #####
    parser.add_option("--compute_correlation_mat", action="store_true",
                      dest="compute_correlation_mat", help="Compute corelation matrix for methods")

    ######### Compute modified_dict ########
    parser.add_option("--compute_modified_acc_dict", action="store_true",
                      dest="compute_modified_acc_dict", help="Compute modified samples and accs matrix")
    parser.add_option("--compute_modified_acc_dict_ranges", action="store", type=str,
                      dest="compute_modified_acc_dict_ranges", default="0,105,5,0,105,5", help="Ranges for modifications, (keep_larger, keep_smaller")

    ######## Compute required ratios #######
    parser.add_option("--compute_agreements", action="store_true",
                      dest="compute_agreements", help="Compute modified samples agreement ratio")
    parser.add_option("--agreement_percs", action="store", type=str,
                      dest="agreement_percs", default="90,105,5", help="Ranges for agreement keep_larger")

    # Parse command line options
    (options, args) = parser.parse_args()

    # print options
    print(options)
    process(options)
