import numpy as np
import matplotlib as mpl
from itertools import combinations
import ruptures as rpt
import random
from scipy.spatial.distance import euclidean
from scipy import stats
import os
import tempfile
from keras.models import load_model
import tqdm
import numpy as np
from scipy.signal import find_peaks

def segment_importances(signal):
    peaks = find_peaks(signal)[0]

    num_peaks = 20

    max_peaks = sorted(sorted(peaks, key=lambda x: signal[x])[-num_peaks:])

    peaks = find_peaks(-signal)[0]

    min_peaks = sorted(sorted(peaks, key=lambda x: -signal[x])[-num_peaks:])

    max_dist = 15

    aux_peaks = [max_peaks[0]]
    for peak in max_peaks[1:]:
        if peak - aux_peaks[-1] < max_dist and signal[peak] > signal[aux_peaks[-1]]:
            aux_peaks[-1] = peak
        elif peak - aux_peaks[-1] >= max_dist:
            aux_peaks.append(peak)

    max_peaks = aux_peaks  

    aux_peaks = [min_peaks[0]]
    for peak in min_peaks[1:]:
        if peak - aux_peaks[-1] < max_dist and signal[peak] < signal[aux_peaks[-1]]:
            aux_peaks[-1] = peak
        elif peak - aux_peaks[-1] >= max_dist:
            aux_peaks.append(peak)

    min_peaks = aux_peaks  

    peaks = sorted(min_peaks + max_peaks)
    segments = []
    for lpeak, rpeak in zip(peaks[:-1], peaks[1:]):
        if rpeak - lpeak == 1:
            if len(segments) == 0:
                segments.append(rpeak)
            else:
                segments[-1] = rpeak
        elif lpeak in max_peaks and rpeak in max_peaks:
            seg = signal[lpeak:rpeak]
            segments.append(lpeak + np.where(seg == seg.min())[0][0])
        elif lpeak in min_peaks and rpeak in min_peaks:
            seg = signal[lpeak:rpeak]
            segments.append(lpeak + np.where(seg == seg.max())[0][0])
        elif lpeak in max_peaks and rpeak in min_peaks:
            seg = np.cumsum(signal[lpeak+1:rpeak])
            segments.append(lpeak + np.where(seg == seg.max())[0][0])
        elif lpeak in min_peaks and rpeak in max_peaks:
            seg = np.cumsum(signal[lpeak+1:rpeak])
            segments.append(lpeak + np.where(seg == seg.min())[0][0])
    
    segments.append(signal.shape[0])
    
    aux = [segments[0]]
    for ini, end in zip(segments[:-1], segments[1:]):
        if end - ini < 10:
            aux[-1] = end
        else:
            aux.append(end)

    segments = aux
    
    return segments 


EPS = 1e-7

def validate_coherence(model, explainer, samples, targets, nstd=1.5, top_features=2, verbose=True):
    perturbed = []
    valid_idx = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i]
        exp = explainer.explain(xi)
        
        segments = segment_importances(exp)
        segments = list(zip(segments[:-1], segments[1:]))
        segment_means = [exp[ini:end].mean() for ini, end in segments]
        
        smin, smax = np.argmin(segment_means), np.argmax(segment_means)
        
        # remove that features
        xic = np.copy(xi).flatten()
        ssize = segments[smin][1] - segments[smin][0]
        xic[segments[smin][0]:segments[smin][1]] = xic[segments[smin][0]:segments[smin][1]] + np.random.normal(size=(ssize,)) * 0.3
        ssize = segments[smax][1] - segments[smax][0]
        xic[segments[smax][0]:segments[smax][1]] = xic[segments[smax][0]:segments[smax][1]] + np.random.normal(size=(ssize,)) * 0.3
        xic = xic.reshape(xi.shape)
        
     
        if not np.isnan(exp).any():
            valid_idx.append(i)
            perturbed.append(xic)

    perturbed = np.array(perturbed)
    samples = samples[valid_idx]
    targets = targets[valid_idx].astype(int)
   
    pred = model.predict(samples, verbose=0)[:, targets] 
    errors = 1 - pred ** 2

    
    exp_pred = model.predict(perturbed, verbose=0)[:, targets]
    exp_errors = 1 - exp_pred ** 2

    coherence_i = np.abs(errors - exp_errors)
    coherence = np.mean(coherence_i)

    return {
        'coherence': coherence,
        'completeness': np.mean(np.sum(exp_errors) / np.sum(errors)),
        'congruency': np.sqrt(np.mean((coherence_i - coherence) ** 2))
    }




def validate_stability(model, explainer, samples, verbose=True):
    """
    Similar objects must have similar explanations.
    """
    distances = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i]
        exp1 = explainer.explain(xi)
        exp2 = explainer.explain(xi + np.random.normal(size=xi.shape) * 0.01)
         

        distances.append(euclidean(exp1.flatten(), exp2.flatten()))


    return np.nanmean(distances)


def validate_selectivity(model, explainer, samples, samples_chunk=1, verbose=True):
    """
    The elimination of relevant variables must affect
    negatively to the prediction. To compute the selectivity
    the features are ordered from most to lest relevant.
    One by one the features are removed, set to zero for
    example, and the residual errors are obtained to get the
    area under the curve (AUC).
    """

    errors = []
    for i in tqdm.tqdm(range(len(samples) - 1), total=len(samples), disable=not verbose):
        dxs, des = [], []
        xi = samples[i]
        ei = explainer.explain(xi)
        if np.isnan(ei).any():
            continue
            
        segments = segment_importances(ei)
        segments = list(zip(segments[:-1], segments[1:]))
        segment_means = np.array([ei[ini:end].mean() for ini, end in segments])
        
        idxs = segment_means.argsort()[::-1]
        xs = [xi]
        
        xprime = np.copy(xi)
        for i in idxs:
            ssize = segments[i][1] - segments[i][0]
            xprime[segments[i][0]:segments[i][1]] = xprime[segments[i][0]:segments[i][1]] + np.random.normal(size=(ssize,)) * 0.3
            xs.append(xprime)
            xprime = np.copy(xprime)

        preds = model.predict(np.array(xs), batch_size=32, verbose=0)
        preds = preds[:, preds[0].argmax()]
        e = np.abs(preds[1:] - preds[:-1]) / (preds[0] + 1e-12)
        e = np.cumsum(e)
        e = 1 - (e / (e.max() + 1e-12))
        score = 1 - np.mean(e)

        errors.append(score)

    return np.nanmean(errors)

