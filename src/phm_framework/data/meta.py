import multiprocessing
import time
import numpy as np
import scipy
from scipy import signal, stats
import tqdm
import tensorflow as tf
from scipy import signal
import logging
from scipy.signal import detrend

from scipy.signal import find_peaks_cwt, ricker

logging.basicConfig(level=logging.INFO)


ATTRIBUTE_NAMES = ['stability', 'periodicity', 'peculiarity', 'oscilatlion', 'complexity', 'simetry', 'slope',
                   'informative', 'peaks', 'noise', 'dynamic_range', 'min_value', 'max_value', 'standard_deviation',
                   'variability']


def detect_outliers(data, m=2.):
    """
    Detect outliers in the given data using the modified Z-score method.

    Args:
    - data (np.ndarray): Input data.
    - m (float): Number of standard deviations to consider as a threshold for outliers.

    Returns:
    - np.ndarray: Boolean array indicating whether each data point is an outlier.
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(d))
    return s > m


def remove_outliers(a):
    """
    Remove outliers from the given array by replacing them with the minimum and maximum non-outlier values.

    Args:
    - a (np.ndarray): Input array.

    Returns:
    - np.ndarray: Array with outliers replaced.
    """
    outliers_mask = detect_outliers(a)
    nov = a[~outliers_mask]
    min_ov = outliers_mask & (a < nov.min())
    max_ov = outliers_mask & (a > nov.max())
    a[min_ov] = nov.min()
    a[max_ov] = nov.max()

    return a


def compute_metaattributes(gen, tslen):
    """
    Compute meta-attributes for each sample in a train_generator.

    Args:
    - gen: Data train_generator.
    - tslen (int): Length of the time series.

    Returns:
    - Tuple: List of samples and list of dictionaries containing meta-attributes for each sample.
    """
    attributes = []
    samples = []
    isample = 0
    for batch in tqdm.tqdm(gen):

        for x, y in zip(*batch):

            for i in range(x.shape[0]):
                s = x[i]

                att = {'attributes': [get_attributes_from_signal(s[i:i + 32]) for i in range(0, tslen, 32)]}

                att['y'] = y
                att['sample'] = isample
                att['feature'] = i
                att['signal'] = s
                attributes.append(att)
                print('.', end='')

            isample += 1

    return samples, attributes


def maxmin_attributes(attributes):
    """
    Compute the maximum, minimum, mean, and standard deviation of each attribute across all features.

    Args:
    - attributes (list): List of dictionaries containing meta-attributes for each sample.

    Returns:
    - dict: Dictionary containing maximum, minimum, mean, and standard deviation for each attribute.
    """
    N_FEATURES = max([a['feature'] for a in attributes]) + 1

    MAX_MIN = {}
    for att in ATTRIBUTE_NAMES:

        for ifeature in range(N_FEATURES):
            print(att, "series:", ifeature)
            data = np.array([a[att]
                             for d in attributes
                             for a in d['attributes']
                             if d['feature'] == ifeature])
            _max, _min = data.min(), data.max()
            _mean, _std = data.mean(), data.std()

            MAX_MIN[f"{att}_{ifeature}"] = (_max, _min, _mean, _std)

    return MAX_MIN


def calculate_coefficient_of_variation(x):
    """
    Calculate the coefficient of variation for a given signal.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Coefficient of variation.
    """
    # Calculate the mean and standard deviation of the signal x
    mean, std = np.mean(x), np.std(x)

    # Calculate the coefficient of variation of the signal x
    return std / (mean + 1e-12)


def calculate_entropy(x):
    """
    Calculate the entropy of a given signal.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Entropy of the signal.
    """
    # Calculate the probability distribution of the values in the signal x
    prob_distribution, _ = np.histogram(x, density=True)

    # Avoid the probability distribution containing zero values
    prob_distribution += 1e-12

    # Calculate the entropy of the signal x
    entropy = stats.entropy(prob_distribution)

    return entropy


def noise_ratio(a):
    """
    Calculate the signal-to-noise ratio (SNR) for a given signal.

    Args:
    - a (numpy.ndarray): Input signal.

    Returns:
    - float: Signal-to-noise ratio in dB.
    """
    # Calculate the residual (noise) by detrending the signal
    residual = detrend(a, type='linear')

    # Calculate the power of the signal and noise
    power_signal = np.var(a)  # Variance of the original signal (includes both signal and noise)
    power_noise = np.var(residual)  # Variance of the residual (noise)

    # Compute SNR in decibels
    snr = 10 * np.log10(power_signal / (power_noise ** 2))

    return snr


def extract_top_frequencies(time_series, sampling_rate=1, top_n=3):
    """
    Extracts the top N most representative frequencies from a time series.

    Parameters:
        time_series (np.ndarray): The input time series data.
        sampling_rate (float): Sampling rate of the series (default is 1 Hz).
        top_n (int): Number of top frequencies to extract (default is 3).

    Returns:
        list of tuples: [(frequency1, amplitude1), (frequency2, amplitude2), ...] sorted by amplitude.
    """
    # Length of the time series
    n = len(time_series)

    # Apply FFT and compute frequencies
    fft_result = np.fft.fft(time_series)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

    # Compute the amplitude spectrum
    amplitudes = np.abs(fft_result)

    # Exclude the DC component (frequency 0)
    nonzero_indices = np.where(frequencies > 0)
    frequencies = frequencies[nonzero_indices]
    amplitudes = amplitudes[nonzero_indices]

    # Find the top N frequencies based on amplitudes
    top_indices = np.argsort(amplitudes)[-top_n:][::-1]
    top_frequencies = [(frequencies[i], amplitudes[i]) for i in top_indices]

    return top_frequencies

def complexity_and_top_frequencies(s, sampling_rate=1, top_n=3):
    """
    Calculate the complexity of a given signal using the Fourier transform.

    Args:
    - s (numpy.ndarray): Input signal.

    Returns:
    - float: Complexity of the signal.
    """
    # Normalize the signal
    s = (s - s.min()) / (s.max() - s.min() + 1E-7)

    # Calculate the Fourier transform of the signal
    n = len(s)

    X = np.fft.fft(s)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)

    # Calculate the magnitude of the Fourier transform
    magnitude = np.abs(X)

    # Calculate the complexity of the signal as the sum of the magnitude of the Fourier transform
    # per unit of time
    complexity = np.sum(magnitude)


    # Exclude the DC component (frequency 0)
    nonzero_indices = np.where(frequencies > 0)
    frequencies = frequencies[nonzero_indices]
    amplitudes = magnitude[nonzero_indices]

    # Find the top N frequencies based on amplitudes
    top_indices = np.argsort(amplitudes)[-top_n:][::-1]
    top_frequencies = [frequencies[i] for i in top_indices]


    return complexity, top_frequencies


def calculate_oscillation(x):
    """
    Calculate the oscillation level of a given signal.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Oscillation level of the signal.
    """
    # Calculate the standard deviation of the signal
    std = np.std(x)

    # Calculate the arithmetic mean of the signal
    _mean = np.mean(x)

    # Calculate the oscillation level as the ratio between the standard deviation and the mean
    oscillation = std / (_mean + 1E-7)

    return np.abs(oscillation)


def calculate_stability_and_slope(x):
    """
    Calculate the stability of a given signal using linear regression.

    Args:
    - x (numpy.ndarray): Input signal.

    Returns:
    - float: Stability of the signal.
    """
    # Calculate the linear regression of the signal x
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, len(x)), y=x)

    # Calculate the stability as the absolute value of the coefficient of determination
    stability = np.abs(r_value ** 2)

    return stability, slope


def evaluate_periodicity(x, num_periods):
    """
    Evaluate the periodicity of a given signal.

    Args:
    - x (numpy.ndarray): Input signal.
    - num_periods (int): Number of periods to consider for periodicity evaluation.

    Returns:
    - float: Periodicity of the signal.
    """
    # Calculate the number of complete periods in the signal x
    period = len(x) // num_periods

    # Divide the signal x into as many complete periods as possible
    periods = np.split(x[:period * num_periods], num_periods)

    # Calculate the similarity between each pair of consecutive periods
    similarities = []
    for i in range(num_periods - 1):
        l, r = periods[i], periods[i + 1]
        if np.all(l == r):
            similarity = 1
        elif np.all(l == l[0]) and np.all(r == r[0]):
            similarity = 1
        elif np.all(l == l[0]) or np.all(r == r[0]):
            similarity = 0
        else:
            similarity = np.corrcoef(periods[i], periods[i + 1])[0, 1]
        similarities.append(similarity)

    # Calculate the periodicity as the mean of the similarities between consecutive periods
    periodicity = np.nanmean(np.abs(similarities))

    return periodicity

def number_cwt_peaks(x, n):
    """
    Number of different peaks in x.

    To estimamte the numbers of peaks, x is smoothed by a ricker wavelet for widths ranging from 1 to n. This feature
    calculator returns the number of peaks that occur at enough width scales and with sufficiently high
    Signal-to-Noise-Ratio (SNR)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: maximum width to consider
    :type n: int
    :return: the value of this feature
    :return type: int
    """
    return len(
        find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker)
    )


def get_attributes_from_signal(*s):
    if len(s) == 1:
        s = s[0]
    s = np.array(s)
    s2 = np.array([s, np.arange(0, len(s))])
    stability, slope = calculate_stability_and_slope(s)
    comp, top_freq = complexity_and_top_frequencies(s)
    top_freq = np.sort(top_freq)
    return list({
        'periodicity': evaluate_periodicity(s, 4),
        'stability': stability,
        'oscilatlion': calculate_oscillation(s),
        'complexity': comp,
        'noise': noise_ratio(s),
        'informative': calculate_entropy(s),
        'variability': calculate_coefficient_of_variation(s),
        'standard_deviation': np.std(s),
        'peculiarity': scipy.stats.kurtosis(s),
        'dynamic_range': abs(s.max() - s.min()),
        'simetry': abs(scipy.stats.skew(s)),
        'peaks': number_cwt_peaks(s, 10),
        'slope': slope,
        'f1': top_freq[0],
        'f2': top_freq[1],
        'f3': top_freq[2],
    }.values())


def get_attributes_from_signal_list(*sl):
    return [get_attributes_from_signal(*s) for s in sl]

def get_attributes(X, n_jobs=None):
    start_time = time.time()

    if n_jobs == 1:
        results = list(map(get_attributes_from_signal, list(X)))
    else:
        points = np.hstack((np.arange(0, X.shape[0], X.shape[0] // n_jobs), [X.shape[0]]))
        X = [X[i:j] for i, j in zip(points[:-1], points[1:])]
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = pool.starmap(get_attributes_from_signal_list, X)

        results = np.vstack(results)

    end_time = time.time()

    logging.info(f"Meta-attributes extracted in {end_time - start_time} seconds")

    return np.array(results)


def create_S2A_model(tslen, nattributes, lr, activation, output):
    """
    Create a Signal-to-Attribute (S2A) model.

    Args:
    - tslen (int): Length of the input time series.
    - nattributes (int): Number of attributes to predict.
    - lr (float): Learning rate for the optimizer.
    - activation (str): Activation function for hidden layers.
    - output (str): Activation function for the output layer.

    Returns:
    - tf.keras.models.Sequential: S2A model.
    """

    # slow and same performance
    #from phm_framework.nets.transformer import create_model
    #signal2attribute = create_model((tslen,), output, output_dim=nattributes)

    signal2attribute = tf.keras.models.Sequential()
    signal2attribute.add(tf.keras.Input(shape=(tslen,)))
    signal2attribute.add(tf.keras.layers.Dense(256, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(128, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(64, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(32, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(64, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(128, activation=activation))
    signal2attribute.add(tf.keras.layers.Dense(nattributes, activation=output))


    loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    opt = tf.keras.optimizers.Adam(lr=lr)
    signal2attribute.compile(optimizer=opt, loss=loss,
                             metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
                             )

    return signal2attribute
