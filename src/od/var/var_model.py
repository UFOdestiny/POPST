import numpy as np
from sklearn.decomposition import TruncatedSVD
from statsmodels.tsa.api import VAR as Var
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA
from base.model import BaseModel


class VAR(BaseModel):
    def __init__(self, node_num, input_dim, output_dim, k=6, **args):
        super().__init__(node_num, input_dim, output_dim, **args)
        self.k = k

    def forward(self, X_train, Y_test_len):
        """
        :param X_train: numpy array, shape (T_train, N, N)
        :param Y_test_len: int, 测试集长度
        :return: Y_pred, shape (Y_test_len, N, N)
        """
        T_train, N, _ = X_train.shape
        X = X_train.reshape(T_train, N * N)

        # 做低秩分解（SVD / PCA）
        svd = TruncatedSVD(n_components=self.k)
        F = svd.fit_transform(X)  # (T, k) 时间因子

        # 用 VAR 对因子做预测
        model = Var(F)
        res = model.fit(maxlags=self.k)
        # 预测未来 h 步因子
        fc = res.forecast(F[-res.k_ar :], steps=Y_test_len)  # shape (h, k)

        # 重构 OD 矩阵: X_hat = fc @ V^T
        V = svd.components_  # (k, N*N)
        Xhat = fc.dot(V)  # (h, N*N)
        Xhat = Xhat.reshape(Y_test_len, N, N)
        return Xhat
