import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from tslearn.metrics import dtw
from scipy.signal import correlate, find_peaks
from sklearn.linear_model import LinearRegression
from phm_framework.data import meta


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

    return labels, centroids, stds, kmeans


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


def extract_envelopes(time_series, time=None):
    """
    Extracts the upper and lower envelopes of a time series.

    Args:
        time_series (numpy.ndarray): Time series (1D).
        time (numpy.ndarray, optional): Times associated with the series. If not specified, a linear index is used.

    Returns:
        tuple: (upper_envelope, lower_envelope)
    """
    if time is None:
        time = np.arange(len(time_series))

    # Find maximum peaks (upper envelope)
    upper_peaks, _ = find_peaks(time_series)
    # Find minimum peaks (lower envelope)
    lower_peaks, _ = find_peaks(-time_series)

    # Interpolation to generate the envelopes
    upper_interp = interp1d(time[upper_peaks], time_series[upper_peaks],
                            kind="linear", bounds_error=False, fill_value="extrapolate")
    lower_interp = interp1d(time[lower_peaks], time_series[lower_peaks],
                            kind="linear", bounds_error=False, fill_value="extrapolate")

    upper_envelope = upper_interp(time)
    lower_envelope = lower_interp(time)

    return upper_envelope, lower_envelope


def assign_cluster_probabilities(new_series, centroids, metric='dtw', temperature=1.0):
    """
    Asigna probabilidades a cada cluster para nuevas series temporales.

    Args:
        new_series (numpy.ndarray): Nuevas series temporales (2D).
        centroids (numpy.ndarray): Centroides de los clusters.
        metric (str): La métrica a usar ('dtw' o 'sbd').
        temperature (float): Factor para ajustar la suavidad de las probabilidades.

    Returns:
        numpy.ndarray: Vector de probabilidades para cada cluster.
    """
    # Calcular la distancia entre la nueva serie y cada centroide
    if metric == 'dtw':
        distance_func = dtw
    elif metric == 'sbd':
        distance_func, _ = sbd
    else:
        raise ValueError("Métrica no soportada. Usa 'dtw' o 'sbd'.")
    distances = [[distance_func(e, c) for c in centroids] for e in new_series]
    distances = np.array(distances)

    # Convertir las distancias a probabilidades usando una distribución exponencial inversa
    scores = np.exp(-distances / temperature)
    probabilities = np.array([p / p.sum() for p in scores])

    return probabilities


def generate_distributions(X, centroids, chunks=10, top_n=5):
    envelopes = np.array([extract_envelopes(e) for e in X])
    frequences = np.array([[f for f, _ in meta.extract_top_frequencies(s, top_n=top_n)] for s in X])
    env_probs = assign_cluster_probabilities(envelopes, centroids)

    def generate_distributions_aux(X, env_probs, frequences, chunks=10):

        if len(frequences.shape) == 1 and frequences.shape[0] > 0:
            noise_ratios = np.array([meta.noise_ratio(s) for s in X])
            stability, slope = list(zip(*[meta.calculate_stability_and_slope(s) for s in X]))
            #entropy = list([meta.calculate_entropy(s) for s in X])
            #periodicity = list([meta.evaluate_periodicity(s, 10)])
            slope = np.array(slope)
            return [[{"N": frequences.shape[0],
                      "frec_dist": (frequences.mean(), frequences.std()),
                      "slope_dist": (slope.mean(), slope.std()),
                      "noise_dist": (noise_ratios.mean(), noise_ratios.std()),
                      #"entropy_dist": (entropy.mean(), entropy.std()),
                      #"periodicity_dist": (periodicity.mean(), periodicity.std()),
                      "env_probs": env_probs.mean(axis=0)
                      }]]
        elif len(frequences.shape) == 1 and frequences.shape[0] == 0:
            return [[{"frec_dist": (np.nan,)}]]
        elif len(frequences.shape) == 0 or frequences.shape[0] == 0:
            return [[{"frec_dist": (np.nan,)}]]
        else:
            freq = frequences[:, 0]
            freq_points = np.linspace(freq.min(), freq.max(), chunks)
            freq_ranges = list(zip(freq_points[:-1], freq_points[1:]))

            results = []
            for i, j in freq_ranges:
                mask = np.where((freq >= i) & (freq < j))
                for next_frec in generate_distributions_aux(X[mask],
                                                            env_probs[mask],
                                                            frequences[mask, 1:].squeeze()):
                    if (not np.isnan(next_frec[-1]["frec_dist"][0]) and
                            len(next_frec) == frequences.shape[1] - 1):
                        results.append([{"N": frequences.shape[0],
                                         "frec_dist": (freq.mean(), freq.std())
                                         }] + next_frec)

            return results

    return generate_distributions_aux(X, env_probs, frequences, chunks)

def add_noise(x, snr):
    """
    Adds Gaussian noise to a signal based on a specified Signal-to-Noise Ratio (SNR).

    Args:
        x (numpy.ndarray): Input signal (1D array or multidimensional array).
        snr (float): Desired Signal-to-Noise Ratio (SNR) in decibels (dB).

    Returns:
        numpy.ndarray: Signal with added noise.
    """
    # Convert SNR from decibels to a linear scale
    snr_linear = 10 ** (snr / 10.0)
    
    # Calculate the power of the signal
    xpower = np.mean(x ** 2, axis=0)
    
    # Calculate the required noise power to achieve the desired SNR
    npower = xpower / snr_linear

    # Generate a random mean for the noise centered around 0 with variance based on noise power
    center = np.random.normal(0, np.sqrt(npower))
    
    # Generate Gaussian noise with the calculated mean and variance
    noise = np.random.normal(center, np.sqrt(npower), x.shape)
    
    # Add the noise to the original signal
    noise_data = x + noise

    return noise_data


def generate_meta(s, centroids, top_n=5):
    """
    Generates metadata for a given time series.

    Args:
        s (numpy.ndarray): Time series data (1D array).
        centroids (numpy.ndarray): Centroids from the clustering process.
        top_n (int, optional): Number of most important frequencies to extract. Default is 5.

    Returns:
        list: Metadata values including:
            - Top `top_n` frequencies.
            - Slope of the signal.
            - Noise ratio.
            - Entropy of the signal.
            - Periodicity measure.
            - Index of the closest envelope cluster.
    """
    # Calculate stability and slope of the time series
    stability, slope = meta.calculate_stability_and_slope(s)
    
    # Calculate the noise ratio of the signal
    noise = meta.noise_ratio(s)
    
    # Extract the top `top_n` frequencies from the signal
    frecuencies = [f for f, _ in meta.extract_top_frequencies(s, top_n=top_n)]
    
    # Extract the upper and lower envelopes of the signal
    envelope = extract_envelopes(s)
    
    # Assign the envelope to the nearest cluster using probabilities
    ienvelope = assign_cluster_probabilities(np.array([envelope]), centroids).argmax()
    
    # Calculate the entropy of the signal
    entropy = meta.calculate_entropy(s)
    
    # Evaluate the periodicity of the signal
    periodicity = meta.evaluate_periodicity(s, 10)
    
    # Combine all calculated metadata into a single list
    return frecuencies + [slope, noise, entropy, periodicity, ienvelope]


def generate_synth_data(X, N=10000, chunks=10, top_n=5):
    """
    Generates synthetic time series data by clustering and sampling from distributions.

    Args:
        X (numpy.ndarray): Original dataset of time series (n_samples x n_timestamps).
        N (int, optional): Total number of synthetic samples to generate. Default is 10,000.
        chunks (int, optional): Number of chunks to divide data for distribution generation. Default is 10.
        top_n (int, optional): Number of most important frequencies to consider. Default is 5.

    Returns:
        tuple: 
            - (XX, EE, M): Synthetic dataset components:
                - XX (numpy.ndarray): Generated time series data (2N x n_timestamps).
                - EE (numpy.ndarray): Envelopes (upper and lower) for each series (2N x 2 x n_timestamps).
                - M (numpy.ndarray): Metadata including frequencies, slope, noise, entropy, periodicity, and envelope index.
            - (centroids, stds, kmeans): Clustering results:
                - centroids (numpy.ndarray): Cluster centroids.
                - stds (numpy.ndarray): Standard deviations for each cluster.
                - kmeans: KMeans model trained on the dataset.
    """
    # Number of samples to use for clustering
    N4cluster = 1000

    # Shuffle the indices of the dataset
    indexes = np.arange(1, X.shape[0])
    np.random.shuffle(indexes)

    # Extract envelopes from detrended signals for clustering
    envolventes = [extract_envelopes(eliminar_pendiente(X[indexes[i]])) for i in range(N4cluster)]
    series_temporales = np.array(envolventes)[:N4cluster, :, :]

    # Perform clustering on time series using K-means with DTW
    n_clusters = 10
    labels, centroids, stds, kmeans = clusterizar_series_temporales(series_temporales, n_clusters, metric='dtw')

    # Generate distributions for clusters
    distributions = generate_distributions(X, centroids, chunks, top_n)

    # Calculate the number of samples to generate for each cluster
    Ns = np.array([d[-1]['N'] for d in distributions])
    Ns = np.array(np.round((Ns / np.sum(Ns)) * N), dtype=int)

    # Initialize arrays for synthetic data, envelopes, and metadata
    XX = np.zeros((2 * N, X.shape[1]))  # Double the samples: original and noisy versions
    EE = np.zeros((2 * N, 2, X.shape[1]))  # Envelopes: upper and lower
    M = np.zeros((2 * N, top_n + 5))  # Metadata: frequencies, slope, noise, entropy, periodicity, envelope index

    # Time vector for synthetic signal generation
    time = np.linspace(0, 10, X.shape[1])

    # Index for generated samples
    i = 0

    # Generate synthetic data for each cluster
    for d, n in zip(distributions, Ns):
        for _ in range(n):
            # Sample properties for the synthetic signal
            frequencies = [np.random.normal(*f["frec_dist"]) for f in d]
            slope = np.random.normal(*d[-1]["slope_dist"])
            noise = np.random.normal(*d[-1]["noise_dist"])
            ienvelope = np.random.choice(np.arange(len(d[-1]["env_probs"])), p=d[-1]["env_probs"])
            eu, el = np.random.normal(centroids[ienvelope], stds[ienvelope] * 0.5)

            # Save envelopes for the signal
            EE[i, 0] = eu
            EE[i, 1] = el
            EE[i + 1, 0] = eu
            EE[i + 1, 1] = el

            # Generate the base synthetic signal
            s = np.zeros((128,))
            for f in frequencies:
                s += np.sin(time * f)

            # Adjust the signal to match the envelopes
            s, _ = adjust_to_envelopes_preserving_shape(s, eu, el)

            # Add slope to the signal
            s = adjust_slope(s, slope)

            # Save the clean signal and its metadata
            XX[i] = s
            M[i] = frequencies + [slope, 0,  # No noise for clean signal
                                  meta.calculate_entropy(s),
                                  meta.evaluate_periodicity(s, 10),
                                  ienvelope]

            # Add noise to the signal for a noisy version
            s_noisy = add_noise(np.copy(s), noise)

            # Save the noisy signal and its metadata
            XX[i + 1] = s_noisy
            M[i + 1] = frequencies + [slope, noise,
                                      meta.calculate_entropy(s_noisy),
                                      meta.evaluate_periodicity(s_noisy, 10),
                                      ienvelope]

            # Increment the index by 2 for the clean and noisy pair
            i += 2

            # Break if the dataset reaches the target size
            if i >= XX.shape[0]:
                break

        # Break outer loop if the dataset is complete
        if i >= XX.shape[0]:
            break

    # Return the generated synthetic dataset and clustering results
    return (XX, EE, M), (centroids, stds, kmeans)


def generate_synthetic_from_dpm(s, model, centroids, N=10, noise_ratio=0.8, forced_features=None):
    """
    Generates a synthetic signal from a trained Diffusion Probabilistic Model (DPM).

    Args:
        s (numpy.ndarray): The input signal (1D array).
        model: The trained DPM model used for synthesis.
        centroids (numpy.ndarray): Centroids from a clustering algorithm used to condition the model.
        N (int, optional): Number of diffusion steps for signal generation. Default is 10.
        noise_ratio (float, optional): Ratio of noise to add to the input signal. Default is 0.8.

    Returns:
        numpy.ndarray: The generated synthetic signal.
    """
    # Generate metadata (features) for the input signal based on centroids
    if forced_features is None:
        f = generate_meta(s, centroids)
    else:
        f = np.copy(forced_features)

    # Extract the upper and lower envelopes of the signal
    e = extract_envelopes(s)

    # Add Gaussian noise to the signal
    sn = s + np.random.normal(size=(s.shape[0],)) * noise_ratio
    sn = np.array([sn] * f.shape[0])

    # Reshape signal and metadata for model compatibility
    sn = np.expand_dims(sn, axis=[-1])  # Add batch and channel dimensions
    e = np.array(e).T  # Reshape envelopes for batching
    e = np.array([e]* f.shape[0])
    #f = np.expand_dims(f, axis=0)  # Reshape metadata for batching

    # Iteratively remove noise using the diffusion model
    for i in range(N, 0, -1):
        t = np.array([[i]])  # Current time step

        # Prepare inputs for the DPM model
        _input = model.prepare_inputs(sn, e, f, t)

        # Predict the noise at the current step
        pred_noises = model.network(_input, training=False)

        # Refine the noisy signal by subtracting predicted noise
        sn = sn - pred_noises.numpy()

    # Remove the batch and channel dimensions from the final signal
    sn = sn[:,:, 0]
    return sn

def plot_clusters(series_temporales, labels, centroids, stds, n_clusters):
    """
    Dibuja los clusters con una cuadrícula de 3 columnas y N filas.

    Args:
        series_temporales (numpy.ndarray): Array de series temporales.
        labels (numpy.ndarray): Etiquetas de los clusters.
        centroids (numpy.ndarray): Centroides de los clusters.
        n_clusters (int): Número de clusters.
    """
    # Determinar el número de filas y columnas
    n_cols = 3
    n_rows = int(np.ceil(n_clusters / n_cols))

    plt.figure(figsize=(15, n_rows * 5))  # Ajustar tamaño según filas

    for cluster_idx in range(n_clusters):
        plt.subplot(n_rows, n_cols, cluster_idx + 1)  # Configurar subplot

        s = centroids[cluster_idx, 0, :]
        std = stds[cluster_idx, 0, :]
        plt.plot(s)
        plt.fill_between(np.arange(s.shape[0]), s-std, s+std, alpha=0.2)

        s = centroids[cluster_idx, 1, :]
        std = stds[cluster_idx, 1, :]
        plt.plot(s)
        plt.fill_between(np.arange(s.shape[0]), s-std, s+std, alpha=0.2)

        plt.title(f"Cluster {cluster_idx}")
        plt.ylim(0, 1)
        plt.legend(loc="upper right")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

