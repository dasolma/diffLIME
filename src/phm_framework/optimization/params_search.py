import inspect
import argparse
import logging
import os, sys


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


from phm_framework.train_test.base import BaseTrainer
logging.basicConfig(level=logging.INFO)
logging.info("Working dir: " + os.getcwd())

SNRS = [-4, -2, 0, 2, 4, 6, 8, 10, None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-m", "--model", help="Model params", type=str, required=True)
    parser.add_argument("-d", "--dataset", help="Dataset params", type=str, required=True)
    parser.add_argument("-t", "--task", help="Dataset task", type=str, required=True)
    parser.add_argument("-c", "--cuda", help="Cuda visible", choices=["0", "1"], default="", required=False)
    parser.add_argument("-nc", "--ncpus", help="CPUs to take", type=int, required=False, default=4)
    parser.add_argument("-ng", "--ngpus", help="GPUs to take", type=int, required=False, default=2)
    parser.add_argument("-r", "--repetitions", help="Experiment repetitions", type=int, required=False, default=100)
    parser.add_argument("-b", "--debug", help="Debug mode", action='store_true')
    parser.add_argument("-o", "--output", help="Output dir", type=str, required=True)
    parser.add_argument("-snrs", "--snrs", help="List of SNR to apply", nargs='+', default=[None])
    parser.add_argument("-f", "--features", help="List of number of features to use", nargs='+', default=[787])
    parser.add_argument("-ir", "--imbalance_ratios", help="List of imbalance ratios to apply", required=False, nargs='+', default=[1])
    parser.add_argument("-aug", "--aug", help="Augmentation method list", nargs='+', default=[None])
    parser.add_argument("-fn", "--feature_noise", help="If add noise to features", action='store_true')
    parser.add_argument("-j", "--jobs", help="Number of process to use", type=int, default=4)
    parser.add_argument("-ncpt", "--ncpus_per_trial", help="CPUs to take per trial", type=int, required=False,
                        default=2)

    # Read arguments from command line
    args = parser.parse_args()

    logging.info("Params read")

    logging.info("GPU: " + str(args.cuda != ""))
    if args.cuda != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ncpus = args.ncpus
    logging.info(f"Limiting to tensorflow to use only {ncpus} threads")

    os.environ["OMP_NUM_THREADS"] = str(ncpus)
    os.environ["NUMEXPR_MAX_THREADS"] = str(ncpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncpus)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(ncpus)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(ncpus)

    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(
        ncpus
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        ncpus
    )
    tf.config.set_soft_device_placement(True)

    os.environ['RAY_memory_monitor_refresh_ms'] = "0"

    import phm_framework
    from phm_framework.data import datasets as phmd

    def train(config):

        new_keys = [k.split('__')[0] for k in config.keys() if '__' in k and k.split('__')[0] not in config]
        for k in new_keys:
            config[k] = {}

        for key in config.keys():
            if '__' in key:
                sect, param = key.split('__')
                config[sect][param] = config[key]

        config = {k: v for k, v in config.items() if '__' not in k}

        creator = get_model_creator()

        return phm_framework.optimization.utils.parameter_opt_cv(creator, config, trainer=BaseTrainer())


    def get_model_creator():
        net_creator_func = f"create_model"
        net_module = getattr(getattr(phm_framework, 'models'), args.model)
        creator = getattr(net_module, net_creator_func)

        return creator


    data_meta = phmd.read_meta(args.dataset)
    task = phmd.get_task(data_meta, args.task)

    #snrs = SNRS[:-1] if args.snr else [None]
    snrs = [float(snr) if snr is not None and snr != 'None' else None for snr in args.snrs]
    #imbalance_ratios = [1] if args.balanced else [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    imbalance_ratios = [float(b) for b in args.imbalance_ratios]
    #features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 787] if args.features is None else args.features
    features = [int(f) for f in args.features]
    if 'all' in args.aug:
        augmentations = ['interpolate', 'oversampling', 'smote', 'borderlinesmote', 'vae', 'ae', 'dpm']
    else:
        augmentations = [a if a != 'None' else None for a in args.aug]

    import ray
    from ray import tune
    from ray.tune.search import bayesopt

    ray.init(num_cpus=args.ncpus, num_gpus=args.ngpus)


    for aug in augmentations:

        for balance_ratio in imbalance_ratios:

            for snr in snrs:

                for nfeatures in features:

                    for signal_length in [2**10 * 10]:
                        logging.info(f"Training with {nfeatures} and a signal length of {signal_length} data points")

                        config = {

                            'model': {
                                'net': args.model,
                                'nestimators': 100,
                            },

                            'data': {
                                'dataset_name': args.dataset,
                                'dataset_target': args.task,
                                'preprocess': None,
                                "signal_length": signal_length,
                                "total_length": 2**10*10000,
                                "extract_features": True,
                                "number_of_features": nfeatures,
                                "snr": snr,
                                "balance_ratio": balance_ratio,
                                'augmentation': aug,
                                'feature_noise': args.feature_noise,
                                'persistance': False
                            },

                            'train': {
                                'timeout': 60 * 30,
                                'lr': 0.001,
                                'verbose': True,
                                'num_folds': task['num_folds'],
                                'repetitions': 5 if args.debug else args.repetitions,
                                'n_jobs': args.jobs,
                                'monitor': 'val_f1score',

                            },

                            'log': {
                                'directory': args.output,
                            },
                        }

                        space = {
                            'aug__depth': (3, 6),
                            'aug__kernels': (16, 256),
                            'aug__z_dim': (32, 256),
                            'aug__kde_bandwidth': (0.1, 1.0),
                            'aug__step_size_factor': (0.01, 0.5),
                            'aug__band_density_factor': (0.5, 2.0),
                        }

                        if task["nature"] == "time-series":
                            ts_len = min(2048, task['min_ts_len'] - 2)
                            config['train'].update({"ts_len": ts_len})

                        bayesopt = ray.tune.search.bayesopt.BayesOptSearch(
                            space=space, mode="max", metric="score",
                            random_search_steps=20)
                        csvlogger = ray.tune.logger.CSVLoggerCallback()


                        def trial_str_creator(trial):
                            trialname = args.model + "_" + args.dataset + "_" + args.task + "_" + trial.trial_id
                            return trialname

                        if args.debug:
                            train(config)

                        else:
                            analysis = ray.tune.run(
                                train,
                                name=args.model + "_" + args.dataset + "_" + args.task,
                                config=config,
                                resources_per_trial={'gpu': 1 if args.ngpus > 0 else 0, 'cpu': args.ncpus_per_trial},
                                num_samples=100,
                                search_alg=bayesopt,
                                callbacks=[csvlogger],
                                log_to_file=False,
                                trial_name_creator=trial_str_creator,
                                storage_path=os.path.join(args.output, "results/opt"),
                            )

                        #train(config)

