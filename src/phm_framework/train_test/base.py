import logging
import sys
import time
import traceback
from collections import defaultdict
import phmd
from phm_framework.logging import HASH_EXCLUDE, confighash, secure_decode, log_train
from phm_framework.utils import flat_dict
from phm_framework.train_test.utils import get_task
import pickle as pk
import numpy as np

logging.basicConfig(level=logging.INFO)

SNRS = [-4, -2, 0, 2, 4, 6, 8, 10, None]


class BaseTrainer:

    def train(self, model_creator, config, ifold, queue, debug, directory, timeout):
        logging.info('Starting training (fold %d) %s' % (ifold, config))

        try:
            context = {}
            context['config'] = config
            context['config']['train']['fold'] = ifold

            training_config = config['train']
            net_config = config['model']
            data_config = config['data']

            net_name = net_config['net']
            data_name = data_config['dataset_name']
            data_target = data_config['dataset_target']

            data_meta = phmd.read_meta(data_name)
            task = get_task(data_meta, data_target, model_creator)
            context['task'] = task

            csv_config = flat_dict(config.copy())
            csv_config['train__fold'] = ifold
            nhash = confighash(csv_config, exclude=HASH_EXCLUDE)
            arch_hash = confighash(csv_config, exclude=HASH_EXCLUDE + ["train__fold"])
            csv_config['run_hash'] = nhash
            csv_config['arch_hash'] = arch_hash

            import os
            import tensorflow as tf
            from phm_framework import models
            from phm_framework.optimization import hyper_parameters as hp

            # prepare output directory
            if not os.path.exists(directory):
                os.makedirs(directory)

            net_history = f"{directory}/net_{net_name}_{data_name}_{task['target']}_{arch_hash}_{nhash}_history.pk"

            # if already train, return saved history
            if os.path.exists(net_history):
                results = pk.load(open(net_history, 'rb'))
                queue.put((results, arch_hash))
                return

            # data reading and prepare data generators
            logging.info("Reading data")
            ts_len = secure_decode(training_config, "ts_len", dtype=int, task=task)
            context['config']['train']['ts_len'] = ts_len

            preprocess = hp.PREPROCESS[secure_decode(data_config, "preprocess", str, default='norm', task=task)]()
            context['config']['data']['preprocess'] = preprocess

            batch_size = secure_decode(training_config, "batch_size", dtype=int, task=task)
            context['config']['train']['batch_size'] = batch_size

            context['config']['data']['load_params'] = {
                "total_length": context['config']['data']['total_length'],
                "signal_length": context['config']['data']['signal_length'],
                "extract_features": context['config']['data']['extract_features'],
                "snr": context['config']['data']['snr'],
                "balance_ratio": context['config']['data']['balance_ratio'],
                "augmentation": context['config']['data']['augmentation'],
            }

            data = self.load_data(context)
            context['data'] = data

            logging.info("Finished Data reading")

            # create and compile model
            model_params = models.get_model_params(net_config, model_creator, task)
            csv_config.update(flat_dict({'model': model_params}))

            metric_results = defaultdict(lambda : [])
            random_states = np.random.randint(0, high=100000, size=(training_config['repetitions'],))
            for rep in range(training_config['repetitions']):
                model_params['random_state'] = random_states[rep]
                model_params['n_jobs'] = training_config['n_jobs']
                model = model_creator(**model_params)

                logging.info("Model created")
                context['model'] = model

                metrics = hp.get_loss(task)
                model.compile(metrics=metrics)

                # train
                start_time = time.time()
                results = model.fit(context)

                results['val_f1score'] = [2*p*r/ (p+r) for r, p in zip(results['val_recall'], results['val_precision'])]

                # save csv results
                metric_results['train__time'].append((time.time() - start_time))
                for k in results.keys():
                    if k.startswith('val'):
                        metric_results[k].append(results[k][-1])

                metric_results['val_f1score'].append(2 * metric_results['val_recall'][-1] * metric_results['val_precision'][-1] / (
                            metric_results['val_recall'][-1] + metric_results['val_precision'][-1]))

                #csv_config.update({k: results[k][-1] for k in results.keys() if k.startswith('val')})

                test_metrics = self.test(context)
                for name_metric, metric in test_metrics.items():
                    metric_results[name_metric].append(metric)
                    #results['name_metrics'] = [metric]

                metric_results['test_f1score'].append(2 * metric_results['test_recall'][-1] * metric_results['test_precision'][-1] / (
                            metric_results['test_recall'][-1] + metric_results['test_precision'][-1]))


            for metric, values in metric_results.items():
                csv_config[metric] = np.mean(values)

            csv_config["train__status"] = "FINISHED"

            results = {k: [np.mean(v)] for k, v in metric_results.items()}
            queue.put((results, arch_hash))

            log_train(csv_config, directory)

        except Exception as ex:
            if 'OOM' in str(ex):
                csv_config["train__status"] = "OOM ERROR"
            else:
                csv_config["train__status"] = "ERROR: " + str(ex)

            logging.error("Error: %s" % ex)
            logging.error(traceback.format_exc())
            sys.stdout.flush()
            queue.put(None)

            log_train(csv_config, directory)

    def load_data(self, context):
        data = phmd.load_data2train(context, return_test=True)
        return data

    def get_input_shape(self, train_gen):
        sample = train_gen[0][0]
        if isinstance(sample, list):
            input_shape = sample[0].shape[1:]
        else:
            input_shape = sample.shape[1:]
        return input_shape

    def test(self, context):
        results = context['model'].evaluate(context, set='test', verbose=True)

        return results

