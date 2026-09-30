from models.layers import calculate_random_walk_matrix
from models.layers import MDGCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseODModel
import numpy as np
from engine.recipe import ModelRecipe
from engine.trainer import BaseEngine_OD
from engine.metrics import zinb_mean, zinb_nll
from data.loader import load_adj_from_numpy
from data.graph import normalize_adj_mx


class NBNorm_ZeroInflated_T(nn.Module):
    """Temporal head emitting (n, p, pi) of a zero-inflated NB, one set per
    (horizon, node, destination) cell.  ``n = softplus``, ``p = sigmoid``,
    ``pi = sigmoid`` (Zhuang et al., KDD'21)."""

    def __init__(self, c_in, c_out, seq_len, horizon=1):
        super().__init__()
        self.horizon = horizon
        self.n_conv = nn.Conv2d(c_in, horizon * c_out, kernel_size=(seq_len, 1), bias=True)
        self.p_conv = nn.Conv2d(c_in, horizon * c_out, kernel_size=(seq_len, 1), bias=True)
        self.pi_conv = nn.Conv2d(c_in, horizon * c_out, kernel_size=(seq_len, 1), bias=True)
        self.out_dim = c_out

    def forward(self, x):  # x (B, T, N, F)
        x = x.permute(0, 2, 1, 3)  # (B, F, T, N) -> conv collapses the time axis
        n = F.softplus(self.n_conv(x))
        p = torch.sigmoid(self.p_conv(x))
        pi = torch.sigmoid(self.pi_conv(x))
        # (B, c_out, 1, N) -> (B, 1(=horizon), N, c_out)
        n = n.reshape(x.shape[0], self.horizon, self.out_dim, x.shape[-1])
        p = p.reshape_as(n)
        pi = pi.reshape_as(n)
        return n, p, pi


class NBNorm_ZeroInflated_S(nn.Module):
    """Spatial head emitting (n, p, pi) of a zero-inflated NB."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.n_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
        self.p_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
        self.pi_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
        self.out_dim = c_out

    def forward(self, x):  # x (B, F, N, horizon)
        n = F.softplus(self.n_conv(x))
        p = torch.sigmoid(self.p_conv(x))
        pi = torch.sigmoid(self.pi_conv(x))
        return n, p, pi


class ITCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation="relu"):
        super(ITCN, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        batch_size, seq_len, num_nodes, num_features = X.shape

        Xf = X
        inv_idx = torch.arange(Xf.size(1) - 1, -1, -1).long().to(device=Xf.device)
        Xb = Xf.index_select(1, inv_idx)

        Xf = Xf.permute(0, 2, 3, 1)
        Xb = Xb.permute(0, 2, 3, 1)

        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))
        outf = tempf + self.conv3(Xf)
        outf = outf.permute(0, 3, 1, 2)

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))
        outb = tempb + self.conv3b(Xb)
        outb = outb.permute(0, 3, 1, 2)

        rec = torch.zeros([batch_size, self.kernel_size - 1, self.out_channels, num_features]).to(
            device=Xf.device
        )
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)

        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(device=Xf.device)
        outb = outb.index_select(1, inv_idx)
        if self.activation == "relu":
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == "sigmoid":
            out = F.sigmoid(outf) + F.sigmoid(outb)
        else:
            out = outf + outb
        return out


class STZINB(BaseODModel):
    """Spatial-Temporal Zero-Inflated Negative Binomial network (Zhuang et al.,
    KDD 2021) for sparse OD demand.

    Twin branches — a temporal inception TCN and a spatial diffusion GCN — each
    emit zero-inflated NB parameters ``(n, p, pi)``; the two estimates are fused
    multiplicatively.  Trained by the ZINB negative log-likelihood (see
    :func:`engine.metrics.zinb_nll`); the point prediction is the ZINB mean
    ``E[y] = (1-pi)·n·(1-p)/p``.  Channel-as-batch over the 3 mobility channels
    (see :class:`models.base.BaseODModel`); driven by :class:`STZINB_Engine`.
    """

    cqr_compatible = False

    def __init__(
        self,
        A,
        node_num,
        hidden_dim_t,
        hidden_dim_s,
        rank_t,
        rank_s,
        num_timesteps_input,
        num_timesteps_output,
        device,
        input_dim,
        output_dim,
        seq_len,
        horizon,
        **args,
    ):
        super(STZINB, self).__init__(node_num, input_dim, output_dim, seq_len, horizon)

        self.TC1 = ITCN(node_num, hidden_dim_t, kernel_size=3)
        self.TC2 = ITCN(hidden_dim_t, rank_t, kernel_size=3, activation="linear")
        self.TC3 = ITCN(rank_t, hidden_dim_t, kernel_size=3)
        if seq_len < 3:
            raise ValueError("STZINB requires at least 3 history steps")
        self.TNB = NBNorm_ZeroInflated_T(hidden_dim_t, node_num, self.seq_len, horizon)

        self.SC1 = MDGCN(num_timesteps_input, hidden_dim_s, 3)
        self.SC2 = MDGCN(hidden_dim_s, rank_s, 2, activation="linear")
        self.SC3 = MDGCN(rank_s, hidden_dim_s, 2)
        self.SNB = NBNorm_ZeroInflated_S(hidden_dim_s, num_timesteps_output)

        self.A = A
        A_q = torch.from_numpy(calculate_random_walk_matrix(self.A).T.astype("float32"))
        A_h = torch.from_numpy(calculate_random_walk_matrix(self.A.T).T.astype("float32"))
        self.register_buffer("A_q", A_q)
        self.register_buffer("A_h", A_h)

    def forward(self, X, label=None):
        """X (B, T, N, N, D) -> (n, p, pi), each (B, horizon, N, N, D)."""
        X, b, d, squeeze_back = self._fold_channels(X)

        # temporal branch -> params over node_num destinations
        X_t = self.TC1(X)
        X_t = self.TC2(X_t)
        X_t = self.TC3(X_t)
        n_t, p_t, pi_t = self.TNB(X_t)

        # spatial branch
        X_s = self.SC1(X, self.A_q, self.A_h)
        X_s = self.SC2(X_s, self.A_q, self.A_h)
        X_s = self.SC3(X_s, self.A_q, self.A_h)
        n_s, p_s, pi_s = self.SNB(X_s)

        # fuse the two estimates (multiplicative, as in the reference)
        n = n_t * n_s
        p = p_t * p_s
        pi = pi_t * pi_s

        n = self._unfold_channels(n, b, d, squeeze_back)
        p = self._unfold_channels(p, b, d, squeeze_back)
        pi = self._unfold_channels(pi, b, d, squeeze_back)
        return n, p, pi


def setup(config, data_path, adj_path, node_num, device, logger):
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, "uqgnn")[0]
    return dict(gso=gso, device=device)


def build_model(config, node_num, **ctx):
    return STZINB(
        A=ctx["gso"],
        seq_len=config.data.seq_len,
        horizon=config.data.horizon,
        node_num=node_num,
        hidden_dim_t=config.model.params.hidden_dim_t,
        hidden_dim_s=config.model.params.hidden_dim_s,
        rank_t=config.model.params.rank_t,
        rank_s=config.model.params.rank_s,
        num_timesteps_input=config.data.seq_len,
        num_timesteps_output=config.data.horizon,
        device=ctx["device"],
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
    )



class STZINB_Engine(BaseEngine_OD):
    """Count-space ZINB training and evaluation for the OD adaptation of STZINB.

    Source: https://github.com/ZhuangDingyi/STZINB.
    Only normalized labels are inverse-transformed; (n,p,pi) already parameterize
    counts. Validation NLL selects checkpoints; point metrics use the ZINB mean."""

    def _to_counts(self, tensor):
        """Inverse-transform a tensor to the original count space when the data
        was normalised; the ZINB likelihood is defined over raw counts."""
        if self._normalize:
            tensor = self._inverse_transform(tensor, device=self._device)
        return tensor

    # -- training -------------------------------------------------------

    def train_batch(self):
        self.model.train()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n" + "=" * 25 + "   Training   " + "=" * 25
                )
            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])

            n, p, pi = self.model(X)
            # ZINB likelihood lives in the original count space; only the
            # label is inverse-transformed (n/p/pi are model outputs fit
            # directly against count-space labels, so zinb_mean(n,p,pi) is
            # already in count space and must not be inverse-transformed
            # again).
            label_c = self._to_counts(label)
            loss = zinb_nll(n, p, pi, label_c, null_val=mask_value)

            # Track point metrics against the ZINB mean for monitoring.
            with torch.no_grad():
                pred = zinb_mean(n, p, pi)
                self.metric.compute_one_batch(pred, label_c, mask_value, "train", value=loss)

            loss.backward()
            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)
            self._optimizer.step()
            self._iter_cnt += 1

    # -- evaluation -------------------------------------------------------

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        preds, labels, ns, ps, pis = [], [], [], [], []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                n, p, pi = self.model(X)
                # zinb_mean(n, p, pi) is already in count space (n/p/pi are
                # model outputs, not normalised data) — only the label needs
                # inverse-transforming back to count space to match it.
                pred = zinb_mean(n, p, pi)
                label = self._to_counts(label)

                if mode == "val":
                    mask_value = self._mask_value.to(pred.device)
                    nll = zinb_nll(n, p, pi, label, null_val=mask_value)
                    self.metric.compute_one_batch(pred, label, mask_value, "valid", value=nll)
                else:
                    preds.append(self._collect(pred).cpu())
                    labels.append(self._collect(label).cpu())
                    ns.append(self._collect(n).cpu())
                    ps.append(self._collect(p).cpu())
                    pis.append(self._collect(pi).cpu())

        if mode == "val":
            return

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        ns = torch.cat(ns, dim=0)
        ps = torch.cat(ps, dim=0)
        pis = torch.cat(pis, dim=0)

        if mode in {"test", "export"}:
            mask_value = torch.tensor(float("nan"))
            for i in range(self.model.horizon):
                nll = zinb_nll(
                    self._horizon_slice(ns, i),
                    self._horizon_slice(ps, i),
                    self._horizon_slice(pis, i),
                    self._horizon_slice(labels, i),
                    null_val=mask_value,
                )
                self.metric.compute_one_batch(
                    self._horizon_slice(preds, i),
                    self._horizon_slice(labels, i),
                    mask_value,
                    "test",
                    value=nll,
                )

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(preds, labels)


def get_recipe():
    return ModelRecipe(
        build_model=build_model,
        loss_fn="NLL",
        metric_list=["NLL", "MAE", "MAPE", "MSE", "RMSE"],
        od=True,
        od_cqr=True,
        engine_cls=STZINB_Engine,
        setup=setup,
    )
