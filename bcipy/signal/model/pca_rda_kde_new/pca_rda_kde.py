import numpy as np
from bcipy.helpers.task import TrialReshaper
from bcipy.signal.model.base_model import SignalModel
from scipy.optimize import fmin_cobyla
from scipy.stats import iqr
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.neighbors import KernelDensity


class PcaRdaKdeModel(SignalModel):
    reshaper = TrialReshaper()

    def __init__(self, k_folds: int):
        self.k_folds = k_folds

    def fit(self, X, y):
        X, y = check_X_y(X, y, allow_nd=True)

        # First, fit gam and lam parameters using COBYLA
        lam_init = 0.9
        gam_init = 0.1
        folds = 10
        model = Pipeline(
            steps=[
                ("pca", ChannelwisePCA(num_channel=X.shape[1])),
                ("rda", RDA()),
            ]
        )

        def cost_fn(lam, gam):
            model.set_params(lam=lam, gam=gam)
            probs = cross_val_predict(model, X, y, cv=folds, n_jobs=-1, method="predict_proba")
            return roc_auc_score(y, probs[..., 1])

        best_lam, best_gam = fmin_cobyla(
            cost_fn,
            x0=(lam_init, gam_init),
            disp=0,
            cons=[
                lambda lam, _: lam - 1e-15,
                lambda _, gam: gam - 1e-15,
                lambda lam, _: 1 - lam,
                lambda _, gam: 1 - gam,
            ],
        )

        # Use optimized params
        model.set_params(rda__lam=best_lam, rda__gam=best_gam)

        # Save scores from PCA/RDA model
        probs = cross_val_predict(model, X, y, cv=folds, n_jobs=-1, method="predict_proba")

        # Fit RDA model a final time
        model.fit(probs, y)

        # Add KDE, and fit this to the saved scores above
        model.steps.append(("kde",))

        return self

    def predict_proba(self, X):
        raise NotImplementedError()


class ChannelwisePCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_channel, var_tol=1e-5):
        self.num_channel = num_channel
        self.var_tol = var_tol

    def fit(self, X, y=None):
        """X.shape == (trials, channels, samples)"""
        self.pca_list_ = [PCA(n_components=self.var_tol).fit(X[:, c, ...], y) for c in range(self.num_channel)]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return np.concatenate([self.pca_list_[c].transform(X[:, c, ...], y) for c in range(self.num_channel)], axis=1)


class RDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, lam=0.9, gam=0.1):
        self.lam = lam
        self.gam = gam

    def fit(self, X, y):
        # Check that X and y have correct shape
        raise NotImplementedError()

    def transform(self, X, y=None):
        return self.predict_proba(X, y)

    def predict(self, X, y=None):
        return self.predict_proba(X, y)

    def predict_proba(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X)
        raise NotImplementedError()


class KDE(BaseEstimator, TransformerMixin):
    def __init__(self, kernel="gaussian", num_channels=None, num_classes=2):
        self.num_channels = num_channels
        self.num_classes = num_classes

    def fit(self, X, y):
        bandwidth = 1
        if self.num_channels is not None:
            bandwidth = self._compute_bandwidth(X, self.num_channels)
        self.kde_list_ = []
        for c in range(self.num_classes):
            X_class = np.expand_dims(np.squeeze(X[y == c]), axis=1)
            self.kde_list_.append(KernelDensity(bandwidth=bandwidth, kernel=self.kernel).fit(X_class))
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        result = []
        for c in range(self.num_classes):
            X_class = np.expand_dims(np.squeeze(X[y == c]), axis=1)
            result.append(self.kde_list_[c].score_samples(X_class))

        breakpoint()  # TODO - transpose?
        return np.concatenate(result)

    def _compute_bandwidth(self, X, num_channels):
        return 1.06 * min(np.std(X), iqr(X) / 1.34) * np.power(num_channels, -0.2)
