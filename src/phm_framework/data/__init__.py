import numpy as np
def prepare_data(X, signal_col, subsignal_length=1000, signal_max_length=20000):
    units_array = X.unit.values
    signal_array = X[signal_col].values
    targets = X[['unit', 'fault']].drop_duplicates().set_index('unit').fault.to_dict()

    N = signal_max_length // subsignal_length
    X = np.zeros((N * len(targets.keys()), subsignal_length))
    Y = np.zeros((N * len(targets.keys()), 0))

    NN = 0
    for i, unit in enumerate(targets.keys()):
        mask = units_array == unit
        signal = signal_array[mask]

        sml = (signal.shape[0] // subsignal_length) * subsignal_length
        n = min(sml // subsignal_length, N)
        sml = subsignal_length * n
        signal = signal[:sml]
        signal = signal.reshape((n, subsignal_length))

        X[NN:NN + n] = signal
        Y[NN:NN + n] = targets[unit]

        NN += n

    X = X[:NN]
    _xmin, _xmax = X.min(axis=1), X.max(axis=1)
    X = ((X.T - _xmin) / (_xmax - _xmin)).T

    Y = X[:NN]

    M = meta.get_attributes(X, n_jobs=8)

    return X, M, Y