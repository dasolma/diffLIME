import numpy as np
from sklearn.linear_model import Lasso, Ridge
from scipy.spatial.distance import cosine

class DiffLIME:

    def __init__(self, model, dpm_model, envelopes, nsamples=1000, verbose=False):
        self.model = model
        self.dpm_model = dpm_model
        self.centroids = envelopes
        self.nsamples = nsamples
        self.random_state = 666
        self.verbose = verbose
        self._data = None
        self._targets = None

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

    def prepare_data(self, signal):
        dpm_model = self.dpm_model
        centroids = self.centroids
        
        
        self._data = np.zeros((self.nsamples, signal.shape[0]))
        self._targets = np.zeros((self.nsamples,))
        
        
        meta = synthetic.generate_meta(signal, centroids)

        self._data[0] = signal
        
        signal = np.squeeze(signal)
        probs = cwru_model.predict(np.array([signal]), verbose=0)
        prob = probs[0, probs.argmax()]
        
        source_klass = probs.argmax()
        self._targets[0] = prob
        #self._targets[0] = 1

        for i in range(self.nsamples-1):
            sn = synthetic.generate_synthetic_from_dpm(np.copy(signal), 
                                                       dpm_model, 
                                                       centroids, 
                                                       forced_features=meta, 
                                                       noise_ratio=0.02, 
                                                       N=1)

            probs = cwru_model.predict(np.array([sn]), verbose=0)
            prob = probs[0, klass]
            
            self._data[i+1] = sn
            self._targets[i+1] = prob
            #self._targets[i+1] = 1 if probs.argmax() == source_klass else 0


    def explain(self, signal):
        
        if self._data is None:
            self.prepare_data(signal)
        

        model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        
        weights = self._get_weights()
        model_regressor.fit(self._data, self._targets, sample_weight=weights)
        
        prediction_score = model_regressor.score(self._data, self._targets, sample_weight=weights)
        local_pred = model_regressor.predict(signal.reshape(1,-1))
        
        if self.verbose:
            print(f'Intercept: {model_regressor.intercept_}')
            print(f'Local prediction: {local_pred}')
            print(f'Prediction score: {prediction_score}')
            
        
        exp = sorted(enumerate(model_regressor.coef_), key=lambda x: np.abs(x[1]), reverse=True)
        return (model_regressor.intercept_, exp, prediction_score, local_pred)

