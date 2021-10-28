from typing import Union
import numpy as np
from bcipy.helpers.task import TrialReshaper
from bcipy.signal.model.base_model import SignalModel
from scipy.optimize import fmin_cobyla
from scipy.stats import iqr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.neighbors import KernelDensity
from sklearn.covariance import OAS, empirical_covariance


class SimplerModel(SignalModel):
    reshaper = TrialReshaper()

    def fit(self, X, y):
        X, y = check_X_y(X, y, allow_nd=True)
        self.model_ = Pipeline(
            steps=[
                ("pca", ChannelwisePCA(num_channel=X.shape[1])),
                ("lda", LDA(solver="eigen", covariance_estimator=OAS(store_precision=False))),
                ("kde", KernelDensity(kernel="gaussian", bandwidth=0.1)),
            ]
        )
        self.model_.fit(X, y)

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, allow_nd=True)
        return self.model_.predict_proba(X)

    def load(self, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def evaluate(self, X, y):
        raise NotImplementedError()

    def score(self, X, y):
        check_is_fitted(self, "model_")
        return self.model_.score(X, y)


# This one is the beginning of an attempt at a refactored version of current model
# It seems likely that the RDA model is not significantly different than LDA with shrinkage
# (e.g. OAS or Ledoit-Wolf), and results in a ton of extra code that then contains bugs and is
# difficult to evaluate/unittest.
# Also - the use of RDA requires an extra cross-validation procedure. The current procedure is slow
# and seems a bit dubious in terms of a proper train/test split
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
                ("lda", LDA(covariance_estimator=OAS(store_precision=False))),
                ("kde", KernelDensity(kernel="gaussian", bandwidth=0.1)),
            ]
        )

        def cost_fn(lam_gam):
            model.set_params(rda__lam=lam_gam[0], rda__gam=lam_gam[1])
            probs = cross_val_predict(model, X, y, cv=folds, n_jobs=-1, method="predict_proba")
            return roc_auc_score(y, probs[..., 1])

        best_lam, best_gam = fmin_cobyla(
            cost_fn,
            x0=(lam_init, gam_init),
            disp=0,
            cons=[
                lambda lam_gam: lam_gam[0] - 1e-15,
                lambda lam_gam: lam_gam[1] - 1e-15,
                lambda lam_gam: 1 - lam_gam[0],
                lambda lam_gam: 1 - lam_gam[1],
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

    def predict(self, X):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def evaluate(self, X, y):
        raise NotImplementedError()


class RDA(BaseEstimator, TransformerMixin):
    def __init__(self, lam: float, gam: float):
        assert 0 <= lam <= 1 and 0 <= gam <= 1
        self.lam = lam
        self.gam = gam

    def fit(self, X, y=None):
        # TODO - Make custom covariance estimator that performs
        # the right mix of shrinkage and regularization using self.lam, self.gam
        X = check_array(X, allow_nd=True)
        cov = RegularizedCovariance(lam=self.lam, gam=self.gam).fit(X)
        raise NotImplementedError()


class RegularizedCovariance(BaseEstimator):
    """Computes the RDA covariance matrix for each class"""

    def __init__(self, lam: float, gam: float):
        assert 0 <= lam <= 1 and 0 <= gam <= 1
        self.lam = lam
        self.gam = gam

    def fit(self, X, y=None):
        num_features = X.shape[1]

        # 1. Fit the class-specific covariance estimates
        class_labels, class_counts = np.unique(y, return_counts=True)
        breakpoint()
        total_count = class_counts.sum()
        self.class_covs_ = []
        for idx, (class_label, class_count) in enumerate(zip(class_labels, class_counts)):
            self.class_covs_.append(empirical_covariance(X[y == class_label]) / class_count)

        # 2. Regularization, controlled by lambda parameter
        # Take convex combination between class-specific and pooled covariance estimates
        pooled = np.sum(self.class_covs_)
        for idx, (class_label, class_count) in enumerate(zip(class_labels, class_counts)):
            modified_count = (1 - self.lam) * class_count + self.lam * total_count
            regularized_est = (1 - self.lam) * self.class_covs_[idx] + self.lam * pooled
            self.class_covs_[idx] = regularized_est / modified_count

        # 3. Shrinkage, controlled by gamma parameter
        # Take convex combination between previous result and a diagonal covariance
        # (identity times the trace of the previous result)
        for idx, (class_label, class_count) in enumerate(zip(class_labels, class_counts)):
            first = (1 - self.gam) * self.class_covs_[idx]
            second = self.gam * np.eye(num_features) * np.trace(self.class_covs_[idx]) / num_features
            self.class_covs_[idx] = first + second

        breakpoint()
        self.covariance_ = None


class ChannelwisePCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_channel: int, n_components: Union[float, int] = 0.95):
        """Channelwise dimensionality reduction using PCA.

        Treat each channel independently, reducing the length of the time axis.
        The principal components used to reduce dimension are time-varying patterns.

        Args:
            num_channel (int): [description]
            n_components (Union[float, int], optional):
                If int, the number of components to keep.
                If float, the percentage of variance to keep.
        """
        self.num_channel = num_channel
        self.n_components = n_components

    def fit(self, X, y=None):
        """X.shape == (trials, channels, samples)"""
        check_array(X, allow_nd=True)
        assert X.ndim == 3 and X.shape[1] == self.num_channel
        self.pca_list_ = [PCA(n_components=self.n_components).fit(X[:, c, ...], y) for c in range(self.num_channel)]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return np.concatenate([self.pca_list_[c].transform(X[:, c, ...]) for c in range(self.num_channel)], axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def _get_explained_variance_ratios(self) -> np.ndarray:
        """Returns (num_channels, num_components) array, with percent explained variance for each PC."""
        return np.stack([self.pca_list_[c].explained_variance_ratio_ for c in range(self.num_channel)])


# class RDA(BaseEstimator, ClassifierMixin, TransformerMixin):
#     def __init__(self, lam=0.9, gam=0.1):
#         self.lam = lam
#         self.gam = gam

#     def fit(self, X, y):
#         """X.shape == (trials, channels, samples)"""
#         check_array(X, allow_nd=True)

#         raise NotImplementedError()

#     def transform(self, X, y=None):
#         return self.predict_proba(X, y)

#     def predict(self, X, y=None):
#         return self.predict_proba(X, y)

#     def predict_proba(self, X, y=None):
#         check_is_fitted(self)
#         X = check_array(X)
#         raise NotImplementedError()


class KDE(BaseEstimator, TransformerMixin):
    def __init__(self, kernel="gaussian", num_channels=None, num_classes=2):
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.kernel = kernel

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
        breakpoint()
        return np.concatenate(result)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def _compute_bandwidth(self, X, num_channels: int):
        return 1.06 * min(np.std(X), iqr(X) / 1.34) * np.power(num_channels, -0.2)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=10 * 128, n_classes=2)
    X = X.reshape((-1, 10, 128))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = SimplerModel()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
