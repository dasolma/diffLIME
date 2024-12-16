import os
from typing import Callable
import numpy as np
from ray import train as rtrain
import multiprocessing
import sys
import logging
import traceback
import phmd
from phm_framework.logging import secure_decode, get_best_info
from phm_framework.train_test.utils import get_task

logging.basicConfig(level=logging.INFO)

def parameter_opt_cv(model_creator: Callable,
                     experiment_config: dict = {},
                     trainer = None,
                     debug: bool = False):
    try:

        training_config = experiment_config['train']
        output_dir = experiment_config['log']['directory']

        net_name = experiment_config['model']['net']
        data_name = experiment_config['data']['dataset_name']
        target = experiment_config['data']['dataset_target']

        output_dir = os.path.join(output_dir, data_name, target, net_name)

        data_meta = phmd.read_meta(data_name)
        task = get_task(data_meta, target, model_creator)

        # min_score = config.pop('min_score')
        monitor = secure_decode(training_config, "monitor", str, default='val_loss', task=task, pop=False)
        timeout = secure_decode(training_config, "timeout", int, default=None, task=task)
        num_folds = secure_decode(training_config, 'num_folds', int, default=5, task=task, pop=False)

        experiment_config['train'] = training_config

        # wd = model_config.pop('working_dir')
        # os.chdir(wd)

        data = experiment_config.copy()
        data['model'] = model_creator.__name__
        data['folds'] = {}

        # cross-validation
        finish = False
        for ifold in range(num_folds):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=trainer.train, args=(model_creator, experiment_config, ifold,
                                                              queue, debug, output_dir, timeout))

            p.start()
            p.join()
            if p.is_alive():
                logging.info('Fold %d timeout' % ifold)
                p.terminate()
                p.join()

                finish = True
            else:
                r = queue.get()
                if r is None:
                    finish = True

                else:
                    data['folds'][ifold] = r[0]
                    arch_hash = r[1]

            if len(data['folds'].keys()) > 0:
                # compute the mean score
                epochs = [len(data['folds'][ifold][monitor]) for ifold in data['folds'].keys()]
                scores = [data['folds'][ifold][monitor][-1] for ifold in data['folds'].keys()]

                rtrain.report({"score": np.mean(scores), "mean_epochs": np.mean(epochs),
                               "std_score": np.std(scores)})

            elif finish:
                logging.info("Not finished any trial")
                rtrain.report({"score": 0, "mean_epochs": 0, "std_score": 0})

            if finish:
                logging.info("Finished train")
                return

            # compute the mean score
            epochs = [len(data['folds'][ifold][monitor]) for ifold in data['folds'].keys()]
            scores = [data['folds'][ifold][monitor][-1] for ifold in data['folds'].keys()]

        rtrain.report({"score": np.mean(scores), "mean_epochs": np.mean(epochs), "std_score": np.std(scores)})

    except Exception as ex:
        logging.error("Error: %s" % ex)
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        queue.put(None)

