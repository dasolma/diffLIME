import phm_framework as phmf
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import tensorflow as tf
import os
import logging
from phm_framework import scoring
from phm_framework.utils import flat_dict

logging.basicConfig(level=logging.INFO)
logging.info("Working dir: " + os.getcwd())

OUTPUT = [
    {
        'field': 'target',
        'value': 'rul',
        'output': 'relu'
    },
    {
        'field': 'type',
        'value': 'classification:binary',
        'output': 'sigmoid'
    },
    {
        'field': 'type',
        'value': 'classification:multiclass',
        'output': lambda task: 'sigmoid' if isinstance(task['target'], list) else 'softmax'
    },

]

LOSS_METRICS = [
    {
        'field': 'type',
        'value': 'regression',
        'output': ['mse',
                   lambda task: tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                   lambda task: tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                   lambda task: tf.keras.metrics.MeanAbsoluteError(name="mae"),
                   lambda task: scoring.SMAPE(name="smape"),
                   lambda task: scoring.NASAScore(name="nasa_score"),
                   ]
    },
    {
        'field': 'type',
        'value': 'classification:binary',
        'output': ["binary_crossentropy",
                   lambda task: tf.keras.metrics.BinaryAccuracy(name='acc'),
                   lambda task: phmf.models.metrics.Recall(2, mode='macro', name='recall'),
                   lambda task: phmf.models.metrics.Precision(2, mode='macro', name='precision')
                   ]
    },
    {
        'field': 'type',
        'value': 'classification:multiclass',
        'output': [lambda task: 'categorical_crossentropy' if isinstance(task['target'],
                                                                         list) else 'sparse_categorical_crossentropy',
                   lambda task: None if isinstance(task['target'], list) else tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                   lambda task: phmf.models.metrics.NonExclusiveRecall(len(task['target_labels']),
                                                                mode='macro', name='recall')
                                    if isinstance(task['target'], list)
                                    else phmf.models.metrics.Recall(len(task['target_labels']),
                                                                mode='macro', name='recall'),
                   lambda task: phmf.models.metrics.NonExclusivePrecision(len(task['target_labels']),
                                                                mode='macro', name='precision')
                                    if isinstance(task['target'], list)
                                    else phmf.models.metrics.Precision(len(task['target_labels']),
                                                                   mode='macro', name='precision')
         ]
    },
]

OUTPUT_DIM = [
    {
        'field': 'type',
        'value': 'regression',
        'output': 1
    },
    {
        'field': 'type',
        'value': 'classification:binary',
        'output': 1
    },
    {
        'field': 'type',
        'value': 'classification:multiclass',
        'output': lambda task: len(task['target_labels'])
    },

]



RANGES = {
    # nlp
    'nhideen_layers': (1, 7),
    'activation': (-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),

    # mscnn
    'kernel_size': (lambda task: min(3, len(task['features'])) + .3,
                    lambda task: min(10, len(task['features'])) + 0.99),
    'msblocks': (-0.49, 3.49),
    'block_size': (-0.49, 5.49),
    'f1': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(20, task['min_ts_len'])),
    'f2': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(20, task['min_ts_len'])),
    'f3': (lambda task: min(3, task['min_ts_len']),
           lambda task: min(20, task['min_ts_len'])),
    'filters': (16, 64),
    'conv_activation': (-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),

    'dilation_rate': (-0.49, 10.49),


    # rnn
    'cell_type': (-0.49, 1.49),
    'rnn_units': (32, 256),
    'bidirectional': (0, 1),

    # general
    'nblocks': (0.51, 4.49),
    'fc1': (16, 64),
    'fc2': (16, 64),
    'dense_activation': (-0.49, len(phmf.typing.ACTIVATIONS) + 0.49 - 1),
    'batch_normalization': (0, 1),

    # regularization
    'dropout': (0, 0.1),
    'l1': (0, 0.00001),
    'l2': (0, 0.00001),

    # transformers
    'nlayers': (0.51, 3.49),
    'segment_size': (0.05, 0.25),
    'model_dim': (8, 64),
    'num_heads': (8, 64),
    'mlp_dim': (16, 64),



}

class DummyPreprocess():

    def fit(self, X):
        pass

    def transform(self, X):
        return X

PREPROCESS = {
    None: DummyPreprocess,
    'norm': MinMaxScaler,
    'std': StandardScaler,
}



def update_dict(d1, d2, task):
    keys = list(d2.keys())

    if len(keys) == 0:
        return d1

    key = keys[0]
    value = d2[key]
    if key in d1:

        if isinstance(value, dict):
            d1[key] = update_dict(d1[key], value, task)

        elif callable(value):
            d1[key] = value(task)
        else:
            d1[key] = value

    else:
        if callable(value) and key != 'extra_callbacks':
            d1[key] = value(task)
        else:
            d1[key] = value

    d2 = d2.copy()
    del d2[key]

    value = d1[key]
    del d1[key]

    r = update_dict(d1, d2, task)
    r[key] = value

    return r



def get_config(task, rules):
    for rule in rules:
        if task[rule['field']] == rule['value']:
            o = rule['output']

            o = compute_value(o, task)

            return o


def compute_value(o, task):
    if isinstance(o, list):
        o = [e(task) if callable(e) else e for e in o]
    elif callable(o):
        o = o(task)
    return o


def get_loss(task):
    return get_config(task, LOSS_METRICS)


def get_output(task):
    return get_config(task, OUTPUT)


def get_output_dim(task):
    return get_config(task, OUTPUT_DIM)


def compute_ranges(task):
    return {k: (compute_value(v1, task), compute_value(v2, task)) for k, (v1, v2) in RANGES.items()}
