import numpy as np
from sklearn.linear_model import Lasso, Ridge
from scipy.spatial.distance import cosine
import matplotlib
from phm_framework.data import meta, synthetic
from sklearn.preprocessing import MinMaxScaler


class LIME:

    def __init__(self, model, nsamples=1000, verbose=False):
        self.model = model
        self.nsamples = nsamples
        self.random_state = 666
        self.verbose = verbose
        self._data = None
        self._targets = None
        self._scaler = MinMaxScaler()

    def _get_weights(self):

        def distance_fn(x):
            ref = x[0]
            distance = np.zeros((x.shape[0],))
            for i in range(x.shape[0]):
                distance[i] = cosine(x[i], ref)
                #distance[i] = np.sum((x[i] * ref)**2)
            return distance

        distances = 1 - distance_fn(self._data)
        
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        
        sigma = 1.0
        
        #return np.exp(-distances ** 2 / (2 * sigma ** 2)) 
        return distances
    
    def __get_onehot(self, y, n):
        return [1 if i == y else 0 for i in range(n)]
    
    def prepare_data(self, signal):
        
        
        self._data = np.zeros((self.nsamples, signal.shape[0]))
        self._targets = np.zeros((self.nsamples,))
        
        
        self._data[0, :signal.shape[0]] = signal
        
        signal = np.squeeze(signal)
        probs = self.model.predict(np.array([signal]), verbose=0)
        prob = probs[0, probs.argmax()]
        klass = probs.argmax()
        
        source_klass = probs.argmax()
        self._targets[0] = prob
        source_prob = prob
        
        i = 1
        noise_range = np.abs(signal.max() - signal.min()) * 0.9
        
        for i in range(1, self.nsamples-1):
            self._data[i, :signal.shape[0]] = np.copy(signal) + np.random.uniform(-noise_range, noise_range, size=signal.shape)
       
    
        probs = self.model.predict(self._data[1:], verbose=0)
        prob = probs[:, klass]


        self._targets[1: ] = prob 
            
        return source_prob, klass


    def explain(self, signal):
        
        signal_length = signal.shape[0]
        
        source_prob, source_klass = self.prepare_data(signal)
        
        local_pred = None
        indexes= list(range(0, self._data.shape[0]))
        

        idxs = indexes
        model_regressor = Ridge(alpha=1, fit_intercept=True, 
                                random_state=self.random_state)

        weights = self._get_weights()
        model_regressor.fit(self._data[idxs], self._targets[idxs], 
                            sample_weight=weights[idxs])

        prediction_score = model_regressor.score(self._data[idxs], self._targets[idxs], 
                                                 sample_weight=weights[idxs])

        s = np.expand_dims(signal, axis=[0])
        local_pred = model_regressor.predict(s)
        
        
        exp = model_regressor.coef_
        
        #print((model_regressor.intercept_, prediction_score, local_pred, source_prob, source_klass))
        
        #return exp
        return (model_regressor.intercept_, exp, prediction_score, local_pred, 
                source_prob, source_klass)


class DiffLIME:

    def __init__(self, model, dpm_model, envelopes, nsamples=1000, verbose=False, smooth_signal_importances=False):
        self.model = model
        self.dpm_model = dpm_model
        self.centroids = envelopes
        self.nsamples = nsamples
        self.random_state = 666
        self.verbose = verbose
        self._data = None
        self._targets = None
        self._scaler = MinMaxScaler()
        self.smooth_signal_importances = smooth_signal_importances

    def _get_weights(self):

        def distance_fn(x):
            ref = x[0]
            distance = np.zeros((x.shape[0],))
            for i in range(x.shape[0]):
                distance[i] = cosine(x[i], ref)
                #distance[i] = np.sum((x[i] * ref)**2)
            return distance

        distances = 1 - distance_fn(self._data)
        
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        
        sigma = 1.0
        
        #return np.exp(-distances ** 2 / (2 * sigma ** 2)) 
        return distances
    
    def __get_onehot(self, y, n):
        return [1 if i == y else 0 for i in range(n)]
    
    def prepare_data(self, signal):
        dpm_model = self.dpm_model
        centroids = self.centroids
        
        
        self._data = np.zeros((self.nsamples, signal.shape[0] + 9 + 10))
        self._targets = np.zeros((self.nsamples,))
        
        
        meta_attr = synthetic.generate_meta(signal, centroids)

        self._data[0, :signal.shape[0]] = signal
        self._data[0, signal.shape[0]:] = meta_attr[:-1] + self.__get_onehot( meta_attr[-1], len(centroids))
        
        signal = np.squeeze(signal)
        probs = self.model.predict(np.array([signal]), verbose=0)
        prob = probs[0, probs.argmax()]
        klass = probs.argmax()
        
        source_klass = probs.argmax()
        self._targets[0] = prob
        source_prob = prob
        
        i = 1
        sn_meta_attr = np.zeros((self.nsamples-1, len(meta_attr)))
        for i in range(1, self.nsamples-1):
            # attribute noises
            sn_meta_attr[i, :9] = np.copy(meta_attr)[:9] + np.random.uniform(-0.5, 0.5, size=(9,))
            
            # envelope noise
            sn_meta_attr[i, -1] = np.random.choice(list(range(10)))
            
        # Nota: noise_ratio=0.02 y N = 1 mejor predicción, frecuencias siempre iguales.
        sn = synthetic.generate_synthetic_from_dpm(np.copy(signal), 
                                                   dpm_model, 
                                                   centroids, 
                                                   forced_features=sn_meta_attr, 
                                                   noise_ratio=0.2, 
                                                   N=1)

        probs = self.model.predict(sn, verbose=0)
        prob = probs[:, klass]


        sn_meta = [synthetic.generate_meta(s, centroids)[:-1] + self.__get_onehot(sn_meta_attr[i, -1], len(centroids))
                   for i, s in enumerate(sn)]

        self._data[1:, :signal.shape[0]] = sn
        self._data[1:, signal.shape[0]:] = sn_meta 
        self._targets[1: ] = prob
            
        self._data[:,  signal.shape[0]:] = self._scaler.fit_transform(self._data[:,  signal.shape[0]:])
            
        return source_prob, klass


    def explain(self, signal):
        
        signal_length = signal.shape[0]
        
        source_prob, source_klass = self.prepare_data(signal)
        
        local_pred = None
        indexes= list(range(0, 10))
        
        smeta = synthetic.generate_meta(signal, self.centroids)
        smeta = smeta[:-1] + self.__get_onehot(smeta[-1], len(self.centroids))
        s = np.concatenate((signal, smeta)).reshape(1,-1)
        s[:, signal_length:] = self._scaler.transform(s[:, signal_length:])
        
        for i in range(10, self._data.shape[0]):
            idxs = indexes + [i]
            model_regressor = Ridge(alpha=1, fit_intercept=True, 
                                    random_state=self.random_state)

            weights = self._get_weights()
            model_regressor.fit(self._data[idxs], self._targets[idxs], 
                                sample_weight=weights[idxs])

            prediction_score = model_regressor.score(self._data[idxs], self._targets[idxs], 
                                                     sample_weight=weights[idxs])
            

            nlocal_pred = model_regressor.predict(s)

            if (local_pred is not None and 
                np.abs(source_prob - nlocal_pred) < 0.2):
                #np.abs(source_prob - nlocal_pred) < np.abs(source_prob - local_pred)):
                local_pred = nlocal_pred
                indexes = idxs
            elif local_pred is None:
                local_pred = nlocal_pred
            else:
                continue
            
            if self.verbose:
                print(f'Intercept: {model_regressor.intercept_}')
                print(f'Local prediction: {local_pred}')
                print(f'Prediction score: {prediction_score}')
   

        idxs = indexes
        model_regressor = Ridge(alpha=1, fit_intercept=True, 
                                random_state=self.random_state)

        weights = self._get_weights()
        model_regressor.fit(self._data[idxs], self._targets[idxs], 
                            sample_weight=weights[idxs])

        prediction_score = model_regressor.score(self._data[idxs], self._targets[idxs], 
                                                 sample_weight=weights[idxs])

        local_pred = model_regressor.predict(s)
        
        
        def smooth_importances(importancias, window_size=3):
            return np.convolve(importancias, np.ones(window_size) / window_size, mode='same')

        
        exp = model_regressor.coef_
        # Suavizar las importancias
        if self.smooth_signal_importances:
            exp[:signal.shape[0]] = smooth_importances(exp[:signal.shape[0]], window_size=10)
        
        
        
        return (model_regressor.intercept_, exp, prediction_score, local_pred, 
                source_prob, source_klass)
