import logging

import numpy as np
from sklearn.decomposition import PCA


class ChannelWisePrincipalComponentAnalysis:
    """Channel wise PCA application. Creates PCA objects respective to the
    number of channels.
    Attr:
        n_components(int): Number of components in PCA
        random_state(seed): Random state seed
        num_ch(int): number of channels in expected data
    """

    def __init__(self, n_components=None, random_state=None, num_ch=1):
        self.num_ch = num_ch
        self.list_pca = [PCA(n_components=n_components, random_state=random_state) for _ in range(self.num_ch)]
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"PCA. {n_components=}, {random_state=}, {num_ch=}")

    def fit(self, x, y=None):
        """Fit PCA to each channel of data.
        Args:
            x(ndarray[float]): C x N x k data array
            y(ndarray[int]): N x k observation (class) array
                N is number of samples k is dimensionality of features
                C is number of channels
        """
        for i in range(self.num_ch):
            self.list_pca[i].fit(x[i, :, :], y)

    def transform(self, x, y=None):
        """Apply fitted PCA to each channel.
        Args:
            x: data, with shape (channels, items, samples)
            y: labels (ignored)
        
        Returns:
            reduced dim data, shape (items, K), where K is the size after reducing and concatenating all channels
        """
        f_vector = []
        for i in range(self.num_ch):
            f_vector.append(self.list_pca[i].transform(x[i, :, :]))
        return np.concatenate(f_vector, axis=1)

    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)
