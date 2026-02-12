import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis


class AutoRegressiveHMM:
    def __init__(self, n_states=3, n_lags=3, n_iter=500):
        self.n_states = n_states
        self.n_lags = n_lags
        self.model = GaussianHMM(
            n_components=n_states, covariance_type='diag', n_iter=n_iter)
        self.scaler = StandardScaler()

    def create_lag_features(self, data):
        """Generate lag features for auto-regressive behavior using multiple features."""
        df = pd.DataFrame(
            data, columns=[f'Feature_{i}' for i in range(data.shape[1])])
        # If not enough rows for lagging, just return original data.
        if len(df) <= self.n_lags:
            return df.values

        for lag in range(1, self.n_lags + 1):
            for feature in df.columns:
                df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
        df.dropna(inplace=True)
        return df.value

    def fit(self, data):
        """Fit the ARLG-HMM model."""
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = self.create_lag_features(data)
        data = self.scaler.fit_transform(data)
        self.model.fit(data)

    def predict(self, data):
        """Predict hidden states for new data."""
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = self.create_lag_features(data)
        data = self.scaler.transform(data)
        return self.model.predict(data)

    def detect_anomalies(self, data, threshold=0.05):
        """Detect fraudulent transactions using HMM likelihood and Mahalanobis distance."""
        original_length = len(data)

        # Create lag features and scale the data
        lagged_data = self.create_lag_features(np.array(data))
        data_scaled = self.scaler.transform(lagged_data)

        # HMM Log Likelihood-Based Anomalies
        log_probs = np.asarray(self.model.score_samples(data_scaled)[0])
        anomaly_threshold = np.mean(log_probs) - (np.std(log_probs) * 2)
        hmm_anomalies = log_probs < anomaly_threshold

        # Mahalanobis Distance-Based Anomalies
        cov_matrix = np.cov(data_scaled.T)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        mean_vector = np.mean(data_scaled, axis=0)
        mahal_distances = np.array(
            [mahalanobis(row, mean_vector, inv_cov_matrix) for row in data_scaled])
        mahal_threshold = np.percentile(
            mahal_distances, 100 - (threshold * 100))
        mahal_anomalies = mahal_distances > mahal_threshold

        # Combine both anomaly detection methods
        combined_anomalies = hmm_anomalies | mahal_anomalies

        # Restore dropped rows with default False
        full_anomalies = np.full(original_length, False, dtype=bool)
        full_anomalies[-len(combined_anomalies):] = combined_anomalies

        return full_anomalies
