import numpy as np
from sklearn.cluster import KMeans
from tslearn.metrics import dtw
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression

def sbd(x, y):
    """
    Shape-Based Distance (SBD) between two time series.
    It is based on the normalization of the cross-correlation.
    """
    # Normalized cross-correlation
    correlation = correlate(x - np.mean(x), y - np.mean(y), mode='full')
    correlation = correlation / (np.linalg.norm(x) * np.linalg.norm(y))

    # The SBD is 1 minus the maximum correlation
    return 1 - np.max(correlation)

def dtw_distance_matrix(X):
    """
    Computes the distance matrix using DTW (Dynamic Time Warping).

    Args:
        X (numpy.ndarray): Array of time series (n_series x n_timestamps).

    Returns:
        numpy.ndarray: Distance matrix of size (n_series x n_series).
    """
    n = len(X)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist_matrix[i, j] = dist_matrix[j, i] = dtw(X[i], X[j])  # Using DTW from tslearn
    return dist_matrix

def sbd_distance_matrix(X):
    """
    Computes the distance matrix using Shape-Based Distance (SBD).

    Args:
        X (numpy.ndarray): Array of time series (n_series x n_timestamps).

    Returns:
        numpy.ndarray: Distance matrix of size (n_series x n_series).
    """
    n = len(X)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist_matrix[i, j] = dist_matrix[j, i] = sbd(X[i], X[j])  # Using custom SBD function
    return dist_matrix

def calculate_centroids(X, labels, n_clusters, metric='dtw'):
    """
    Computes the centroids of clusters using a custom distance metric.

    Args:
        X (numpy.ndarray): Matrix of time series (n_series x n_timestamps).
        labels (numpy.ndarray): Cluster labels for each time series.
        n_clusters (int): Number of clusters.
        metric (str): Metric to use ('dtw' or 'sbd').

    Returns:
        centroids (numpy.ndarray): Array of centroids (n_clusters x n_timestamps).
        stds (numpy.ndarray): Array of standard deviations for each cluster (n_clusters x n_timestamps).
    """
    centroids = []
    stds = []

    for i in range(n_clusters):
        # Select time series belonging to cluster i
        cluster_series = X[labels == i]

        # Calculate mean and standard deviation for the cluster
        centroids.append(cluster_series.mean(axis=0))
        stds.append(cluster_series.std(axis=0))

    return np.array(centroids), np.array(stds)


def eliminar_pendiente(serie_temporal):
    """
    Modifies a time series to remove its linear trend (sets slope to 0).

    Args:
        serie_temporal (numpy.ndarray): 1D time series to process.

    Returns:
        numpy.ndarray: Detrended time series.
    """
    # Time indices (assuming uniform time steps)
    tiempo = np.arange(len(serie_temporal)).reshape(-1, 1)

    # Fit a linear regression model to the trend
    modelo = LinearRegression()
    modelo.fit(tiempo, serie_temporal)

    # Predict the trend and remove it
    tendencia = modelo.predict(tiempo)
    serie_sin_pendiente = serie_temporal - tendencia

    # Scale the detrended series to retain the original min-max range
    min_val, max_val = serie_temporal.min(), serie_temporal.max()
    serie_sin_pendiente = (serie_sin_pendiente - serie_sin_pendiente.min()) / \
                          (serie_sin_pendiente.max() - serie_sin_pendiente.min()) * (max_val - min_val) + min_val

    return serie_sin_pendiente


def clusterizar_series_temporales(series_temporales, n_clusters=3, metric='dtw'):
    """
    Clusters time series using KMeans and a custom distance metric (DTW or SBD).

    Args:
        series_temporales (numpy.ndarray): Matrix of time series (n_series x n_timestamps).
        n_clusters (int): Number of clusters to form.
        metric (str): Metric to use ('dtw' or 'sbd').

    Returns:
        labels (numpy.ndarray): Cluster labels assigned to each time series.
        centroids (numpy.ndarray): Centroids of the clusters.
        stds (numpy.ndarray): Standard deviations of the clusters.
    """
    # Compute the distance matrix using the selected metric
    if metric == 'dtw':
        dist_matrix = dtw_distance_matrix(series_temporales)
    elif metric == 'sbd':
        dist_matrix = sbd_distance_matrix(series_temporales)
    else:
        raise ValueError("Unsupported metric. Use 'dtw' or 'sbd'.")

    # Convert the distance matrix to a feature representation
    # by applying MDS or using the precomputed kernel directly
    from sklearn.manifold import MDS
    mds = MDS(n_components=n_clusters, dissimilarity="precomputed", random_state=42)
    embedded_data = mds.fit_transform(dist_matrix)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedded_data)

    # Compute cluster centroids
    centroids, stds = calculate_centroids(series_temporales, labels, n_clusters, metric)

    return labels, centroids, stds


def adjust_to_envelopes_preserving_shape(original_signal, upper_envelope, lower_envelope):
    """
    Adjusts a signal so that its local maxima and minima exactly match the envelopes,
    while preserving the original shape of the signal.

    Args:
        original_signal (numpy.ndarray): Original signal (1D).
        upper_envelope (numpy.ndarray): Upper envelope (1D).
        lower_envelope (numpy.ndarray): Lower envelope (1D).

    Returns:
        numpy.ndarray: Signal adjusted to respect the envelopes while preserving shape.
    """
    # Ensure the envelopes and signal have the same size
    if len(original_signal) != len(upper_envelope) or len(original_signal) != len(lower_envelope):
        raise ValueError("Envelopes and the original signal must have the same size.")

    step = 16
    adjusted_signal = np.copy(original_signal)
    for i in np.arange(0, adjusted_signal.shape[0], step):
        _min, _max = adjusted_signal[i:i + step].min(), adjusted_signal[i:i + step].max()

        adjusted_signal[i:i + step] = (adjusted_signal[i:i + step] - _min) / (_max - _min)
    scaled_signal = np.copy(adjusted_signal)

    _emin, _emax = lower_envelope.min(), upper_envelope.max()

    factor = 1 / (_emax - _emin)
    lower_envelope = (lower_envelope - _emin) * factor
    upper_envelope = (upper_envelope - _emin) * factor

    indexes = np.where(adjusted_signal >= 0.5)
    adjusted_signal[indexes] = adjusted_signal[indexes] * upper_envelope[indexes]

    indexes = np.where(adjusted_signal < 0.5)
    adjusted_signal[indexes] = 1 - ((1 - adjusted_signal[indexes]) * (1 - lower_envelope[indexes]))

    adjusted_signal = (adjusted_signal / factor) + _emin

    return adjusted_signal, scaled_signal


def adjust_slope(time_series, target_slope, time=None):
    """
    Adjusts a time series so that its regression slope matches the target slope.

    Args:
        time_series (numpy.ndarray): Original time series (1D).
        target_slope (float): Desired slope for the regression line.
        time (numpy.ndarray, optional): Time indices associated with the series. If not provided, a linear index is used.

    Returns:
        numpy.ndarray: Adjusted and scaled time series.
    """
    # Generate time indices if not provided
    if time is None:
        time = np.arange(len(time_series))

    # Calculate the current slope using linear regression
    time = time.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(time, time_series)
    current_slope = reg.coef_[0]

    # Calculate the necessary adjustment
    slope_adjustment = target_slope  # - current_slope

    adjustment = slope_adjustment * time.flatten()
    adjustment = adjustment - (adjustment.min() + (adjustment.max() - adjustment.min()) / 2)

    # Modify the series to achieve the desired slope
    adjusted_series = time_series + adjustment

    # Scale the adjusted series so its maxima and minima match the original series
    # original_min, original_max = time_series.min(), time_series.max()
    # adjusted_min, adjusted_max = adjusted_series.min(), adjusted_series.max()

    # Linear scaling
    scaled_series = adjusted_series
    # scaled_series = ((adjusted_series - adjusted_min) / (adjusted_max - adjusted_min)) * (original_max - original_min) + original_min

    return scaled_series
