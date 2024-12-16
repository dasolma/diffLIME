import random
import tensorflow as tf
import numpy as np
from collections.abc import Iterable
import logging

def _tuple(x):
    '''
        Checks if an object x is iterable
    '''
    if isinstance(x, Iterable): # If the object is iterable
        # x can be any type of object that can be iterated over, such as lists, tuples, sets, dictionaries
        return tuple(x) # Returns x as a tuple
    else: # If the object is not iterable
        return x # Returns x without modification

# Data train_generator
class Sequence(tf.keras.utils.Sequence):
    '''
        Generate data sequences for a single set of features and target
    '''
    def __init__(self, data, unit_cols, features_cols, target_col, batches_per_epoch=1000, batch_size=32,
                 extra_channel=False, ts_len=256, ts_consider=1., stride=1, transpose=False,
                 channel_last=False, unsupervised=False, normalize=False, target_filter=None):
        '''Class initialization'''

        if data is None: # Check if there is data
            return

        self.normalize = normalize
        self._min_max = None
        self._mean_std = None
        self.target_filter = target_filter
        if normalize:
            self._min_max = data[features_cols].values.min(), data[features_cols].values.max()
            self._mean_std = data[features_cols].values.mean(), data[features_cols].values.std()

        self.stride = stride # Step with which consecutive windows of the time series move
        self.batch_size = batch_size # Size of each batch
        self.batches_per_epoch = batches_per_epoch # Number of batches per epoch
        self.target_col = target_col # Column containing the target to be predicted
        self.units = [list(l) for l in data[unit_cols].drop_duplicates().values] # List of unique units in the data
        self.ts_consider = ts_consider # Proportion of the time series to consider

        units = str(self.units).replace('\n', ',')
        logging.info(f"Units: {units[:10]}. Total units: {len(self.units)}")

        self.extra_channel = extra_channel # Whether to add an extra channel in the data
        self.ts_len = ts_len # Length of the time series

        logging.info("Indexing features by unit") # Index the features by unit
        feature_cols = [c for c in data.columns if c not in unit_cols + [target_col]]
        #self.data = data.groupby(unit_cols) # Group the DataFrame `data` by the columns specified in `unit_cols`
        self.data = data[feature_cols].values
        # Allows performing operations on each group of units independently
        #self.units = [x for x in self.data.groups] # List of unique units
        self.units = data[unit_cols].values
        # Dictionary where the keys are units (converted to tuples) and the values are numpy arrays of
        # the target column data for each unit:
        #self.targets = self.data[target_col].unique()
        #self.target = {_tuple(x): self.data.get_group(x)[target_col].values for x in self.units}
        self.targets = data[target_col].values

        # Dictionary where the keys are units (converted to tuples) and the values are numpy arrays of
        # the feature column data for each unit:
        '''
        if transpose:
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values.T for x in self.units}
            self.nfeatures = self.data[list(self.data.keys())[0]].shape[-1]
        else:
            self.data = {_tuple(x): self.data.get_group(x)[features_cols].values for x in self.units}
            self.feature_cols = features_cols  # List of feature column names
            self.nfeatures = len(features_cols)  # Number of feature columns
        '''

        self.channel_last = channel_last
        self.unsupervised = unsupervised

    def __len__(self):
        '''
            Return the number of batches the train_generator should provide per epoch during the training
            of a model
        '''
        return self.batches_per_epoch

    def clone(self):
        seq = Sequence(None, None, None, None)

        seq.data = self.data
        seq.batch_size = self.batch_size
        seq.batches_per_epoch = self.batches_per_epoch
        seq.units = self.units
        seq.ts_consider = self.ts_consider
        seq.extra_channel = self.extra_channel
        seq.ts_len = self.ts_len
        seq.target = self.target
        seq.target_col = self.target_col
        seq.feature_cols = self.feature_cols # List of feature column names
        seq.stride = self.stride
        seq.nfeatures = self.nfeatures
        seq.channel_last = self.channel_last
        seq.unsupervised = self.unsupervised
        seq.target_filter = self.target_filter

        return seq

    def __getitem__(self, idx):
        '''
            Allows an instance of a class to act like a list or array, allowing access to its
            elements using indices
        '''
        ts_len = self.ts_len # Length of the time series

        # Initialize the feature matrix:
        if self.extra_channel: # Whether to add an extra channel in the data
            # X is initialized as a zero matrix with appropriate dimensions to contain the batch data
            # An extra dimension is added
            X = np.zeros(shape=(self.batch_size, 1, ts_len, 1))
        else: # If no extra channel is added in the data
            X = np.zeros(shape=(self.batch_size, 1, ts_len))

        # Initialize the target matrix:
        if isinstance(self.target_col, list): # If there are multiple target columns contained in a list
            # Y is initialized as a zero matrix for the targets
            Y = np.zeros(shape=(self.batch_size, len(self.target_col)))
        else: # If there is a single target column
            Y = np.zeros(shape=(self.batch_size,))

        # Creating the Batch:
        if self.target_filter is not None:
            target_filter = self.target_filter
            if not isinstance(self.target_filter, list):
                target_filter = [self.target_filter]
            units_indexes = [i for i, t in enumerate(self.targets) if t in target_filter]
        else:
            units_indexes = np.arange(0, len(self.units))

        for i in range(self.batch_size): # Iterate over the batch size
            unit_index = units_indexes[random.randint(0, len(units_indexes) - 1)] # Select a random unit
            T, k, ts = self.extract_ts(ts_len, unit_index) # Extract the time series and the target
            # Assign data to X:
            if self.extra_channel: # If an extra channel is added in the data
                X[i, :, :, 0] = ts
            else: # If no extra channel is added in the data
                X[i, :, :] = ts

            Y[i] = T

        if self.channel_last:
            X = np.moveaxis(X, 1, -1)

        # The batch of features (X) and targets (Y) is returned, both converted to float32 type:
        if self.normalize:
            X = (X - self._mean_std[0]) / self._mean_std[1]

        if self.unsupervised:
            return X.astype('float32'), X.astype('float32')
        else:
            return X.astype('float32'), Y.astype('float32')

    def extract_ts(self, ts_len, unit):
        '''
             Extract a time series of length `ts_len` from a specific unit of data
        '''

        # Retrieve the data for the unit:
        Db = self.data[unit] # Data corresponding to the specified unit
        T = self.targets[unit] # Targets corresponding to the specified unit
        # Ensure that the amount of data in the unit is at least equal to the required time series length:
        assert Db.shape[0] >= ts_len
        L = Db.shape[0] # Total length of the data series available for the specific unit

        # Select the starting point `k` of the time series within a unit of data:

        if self.ts_consider == 0: # If the proportion of the time series to consider is 0:
            # `k` is the last possible index to start a time series of length `ts_len` within `L` data points,
            # considering the stride:
            k = max(0, L - (ts_len * self.stride) - 1)
        else: # If the proportion of the time series to consider is specified
            # The starting point `Lini` will be the closest to the actual start of the series, the minimum between:
            # The start of the data window discounting a proportion `self.ts_consider` of the data from the
            # end of the series and the maximum possible index for the start of the time series
            Lini = min(int(L * (1 - self.ts_consider)), L - (ts_len * self.stride) - 1)
            # A random index `k` is selected within the range allowed by `Lini` and the maximum possible index:
            k = max(0, random.randint(Lini, L - (ts_len * self.stride) - 1))

        # Array of consecutive indexes with a fixed step between them:
        indexes = np.arange(k, k + (ts_len * self.stride), self.stride)

        # Adjust the indexes to ensure they are within valid limits:
        indexes = np.clip(indexes, 0, L - 1)
        ts = Db[indexes].T # Time series extracted from `Db` using the generated indexes and transposed
        # The resulting matrix is transposed so that the features (original columns of `Db`) become
        # the rows of the `ts` matrix

        return T, k, ts
        # Returns:
        # T: targets corresponding to the specified unit
        # k: starting point of the time series within a unit of data
        # ts: time series extracted from `Db` using the generated indexes and with appropriate dimensions
