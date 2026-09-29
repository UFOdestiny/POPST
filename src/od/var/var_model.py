import numpy as np
from sklearn.decomposition import TruncatedSVD
from statsmodels.tsa.api import VAR as Var
from base.model import BaseModel


class VAR(BaseModel):
    """Vector AutoRegression on a low-rank (TruncatedSVD) factorisation of the
    flattened OD matrix.  Channel-aware: each mobility channel is decomposed and
    forecast independently, then stacked back into ``(T, N, N, D)``.
    """

    def __init__(self, node_num, input_dim, output_dim, k=6, **args):
        super().__init__(node_num, input_dim, output_dim, **args)
        self.k = k

    def _forecast_one(self, X_train, steps):
        """Forecast a single channel.  ``X_train`` is ``(T_train, N, N)`` ->
        returns ``(steps, N, N)``."""
        T_train, N, _ = X_train.shape
        X = X_train.reshape(T_train, N * N)

        svd = TruncatedSVD(n_components=self.k)
        F = svd.fit_transform(X)  # (T, k) temporal factors

        model = Var(F)
        res = model.fit(maxlags=self.k)
        fc = res.forecast(F[-res.k_ar :], steps=steps)  # (steps, k)

        V = svd.components_  # (k, N*N)
        Xhat = fc.dot(V).reshape(steps, N, N)
        return Xhat

    def forward(self, X_train, Y_test_len):
        """X_train ``(T_train, N, N, D)`` -> Y_pred ``(Y_test_len, N, N, D)``."""
        X_train = np.asarray(X_train)
        if X_train.ndim == 3:  # (T, N, N) single channel
            return self._forecast_one(X_train, Y_test_len)

        T_train, N, _, D = X_train.shape
        Y_pred = np.zeros((Y_test_len, N, N, D), dtype=np.float32)
        for c in range(D):
            Y_pred[..., c] = self._forecast_one(X_train[..., c], Y_test_len)
        return Y_pred
