import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
sys.path.append("/home/dy23a.fsu/st/")

from utils.args import get_data_path
from utils.generate import LogMinMaxScaler, LogScaler
from scipy.spatial import distance


def _pad_indices(indices, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    remainder = len(indices) % batch_size
    if remainder == 0:
        return indices
    num_padding = batch_size - remainder
    padding = np.repeat(indices[-1:], num_padding, axis=0)
    return np.concatenate([indices, padding], axis=0)


def _compute_num_batches(size, batch_size, droplast):
    if droplast:
        return size // batch_size
    return math.ceil(size / batch_size) if batch_size else 0


def _load_data_dir(data_path, years):
    return Path(data_path) / years


def _load_indices(data_dir, split):
    return np.load(data_dir / f"idx_{split}.npy")


class DataLoader(object):
    def __init__(
        self,
        data,
        idx,
        seq_len,
        horizon,
        bs,
        logger,
        name=None,
        pad_last_sample=False,
        droplast=False,
    ):
        if pad_last_sample:
            idx = _pad_indices(idx, bs)

        self.data = np.asarray(data)
        self.idx = np.asarray(idx)
        self.size = len(self.idx)
        self.bs = bs
        self.droplast = droplast
        self.num_batch = _compute_num_batches(self.size, self.bs, droplast)

        self.current_ind = 0
        loader_name = name or "loader"
        logger.info(f"{loader_name:5s} num: {self.idx.shape[0]},\tBatch num: {self.num_batch}")        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.idx = self.idx[perm]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = np.asarray(self.idx[start_ind:end_ind]).reshape(-1)
                if len(idx_ind) == 0:
                    break
                if self.droplast and len(idx_ind) < self.bs:
                    self.current_ind += 1
                    continue

                x_indices = idx_ind[:, None] + self.x_offsets
                y_indices = idx_ind[:, None] + self.y_offsets

                x = self.data[x_indices, ...].astype(np.float32, copy=False)
                y = self.data[y_indices, ...].astype(np.float32, copy=False)

                yield x, y
                self.current_ind += 1

        return _wrapper()


def load_dataset_plain(data_path, args, logger, drop=False):
    data_dir = _load_data_dir(data_path, args.years)
    ptr = np.load(data_dir / "his.npz")
    logger.info(f"{'Data shape':20s}: {ptr['data'].shape}")
    X = ptr["data"]
    xy = []
    for cat in ["train", "val", "test"]:
        idx = _load_indices(data_dir, cat)
        xy.append(X[idx])
    return xy, LogScaler()


def load_dataset(data_path, args, logger, drop=False):
    data_dir = _load_data_dir(data_path, args.years)
    ptr = np.load(data_dir / "his.npz")
    logger.info(f"{'Data shape':20s}: {ptr['data'].shape}")

    X = ptr["data"]

    dataloader = {}
    for cat in ["train", "val", "test"]:
        idx = _load_indices(data_dir, cat)
        dataloader[f"{cat}_loader"] = DataLoader(
            X, idx, args.seq_len, args.horizon, args.bs, logger, cat, droplast=drop
        )

    scaler = LogMinMaxScaler(ptr["min"], ptr["max"]) if "min" in ptr else LogScaler()

    return dataloader, scaler


class DataLoader_MPGCN(object):
    def __init__(
        self,
        data,
        idx,
        seq_len,
        horizon,
        bs,
        logger,
        adj_O,
        adj_D,
        name=None,
        pad_last_sample=False,
        droplast=False,
    ):
        if pad_last_sample:
            idx = _pad_indices(idx, bs)

        self.data = np.asarray(data)
        self.idx = np.asarray(idx)
        self.size = len(self.idx)
        self.bs = bs
        self.droplast = droplast
        self.num_batch = _compute_num_batches(self.size, self.bs, droplast)
        self.current_ind = 0
        loader_name = name or "loader"
        logger.info(f"{loader_name:5s} num: {self.idx.shape[0]:5d},\tBatch num: {self.num_batch:5d}")
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
        self.O = adj_O
        self.D = adj_D

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.idx = self.idx[perm]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = np.asarray(self.idx[start_ind:end_ind]).reshape(-1)
                if len(idx_ind) == 0:
                    break
                if self.droplast and len(idx_ind) < self.bs:
                    self.current_ind += 1
                    continue

                x_indices = idx_ind[:, None] + self.x_offsets
                y_indices = idx_ind[:, None] + self.y_offsets
                x = self.data[x_indices, ...].astype(np.float32, copy=False)
                y = self.data[y_indices, ...].astype(np.float32, copy=False)

                period = self.O.shape[-1]
                adj_index = np.mod(idx_ind, period)
                adj_o = self.O[:, :, adj_index].transpose(2, 0, 1)
                adj_d = self.D[:, :, adj_index].transpose(2, 0, 1)

                yield x, y, adj_o, adj_d
                self.current_ind += 1

        return _wrapper()


def construct_dyn_G(OD_data: np.array, perceived_period: int = 6):
    """Construct dynamic graphs based on OD history using vectorized cosine distance."""

    train_len = int(OD_data.shape[0] * 0.8)
    num_periods_in_history = train_len // perceived_period
    OD_history = OD_data[: num_periods_in_history * perceived_period, :, :, :]

    O_dyn_G, D_dyn_G = [], []
    for t in range(perceived_period):
        OD_t_avg = np.mean(OD_history[t::perceived_period, :, :, :], axis=0).squeeze(
            axis=-1
        )

        # Vectorized cosine distance computation
        O_G_t = distance.cdist(OD_t_avg, OD_t_avg, metric="cosine")
        D_G_t = distance.cdist(OD_t_avg.T, OD_t_avg.T, metric="cosine")

        O_dyn_G.append(O_G_t)
        D_dyn_G.append(D_G_t)

    return np.stack(O_dyn_G, axis=-1), np.stack(D_dyn_G, axis=-1)


def load_dataset_MPGCN(data_path, args, logger):
    data_dir = _load_data_dir(data_path, args.years)
    ptr = np.load(data_dir / "his.npz")
    logger.info(f"Data shape: {ptr['data'].shape}")
    X = ptr["data"]

    adjo, adjd = construct_dyn_G(X[..., np.newaxis])

    dataloader = {}
    for cat in ["train", "val", "test"]:
        idx = _load_indices(data_dir, cat)
        dataloader[f"{cat}_loader"] = DataLoader_MPGCN(
            X, idx, args.seq_len, args.horizon, args.bs, logger, adjo, adjd, cat
        )

    scaler = LogScaler()
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"Unable to load data {pickle_file}: {e}")
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    # base_dir = os.getcwd() + '/data/'
    base_dir = get_data_path()

    d = {
        # Flow Prediction: N -> N
        "Shenzhen": [base_dir + "shenzhen", base_dir + "shenzhen/adj.npy", 491],
        "Shenzhen2": [base_dir + "shenzhen2", base_dir + "shenzhen2/adj.npy", 491],
        "NYC": [base_dir + "nyc", base_dir + "nyc/adj.npy", 67],
        "NYC_Crash": [base_dir + "nyc_crash", base_dir + "nyc_crash/adj.npy", 42],
        "NYC_Combine": [base_dir + "nyc_combine", base_dir + "nyc_combine/adj.npy", 42],
        "Chicago": [base_dir + "chicago", base_dir + "chicago/adj.npy", 77],
        "NYISO": [base_dir + "nyiso", base_dir + "nyiso/adj.npy", 11],
        "CAISO": [base_dir + "caiso", base_dir + "caiso/adj.npy", 9],
        "Tallahassee": [base_dir + "tallahassee", base_dir + "tallahassee/adj.npy", 9],
        "NYISO_HDM": [base_dir + "nyiso_hdm", base_dir + "nyiso/adj.npy", 11],
        "CAISO_HDM": [base_dir + "caiso_hdm", base_dir + "caiso/adj.npy", 9],
        "Tallahassee_HDM": [
            base_dir + "tallahassee_hdm",
            base_dir + "tallahassee/adj.npy",
            9,
        ],
        "panhandle": [base_dir + "panhandle", base_dir + "panhandle/adj.npy", 924],

        "safegraph_fl": [base_dir + "safegraph_fl", base_dir + "safegraph_fl/adj.npy", 67],
        "safegraph_ca": [base_dir + "safegraph_ca", base_dir + "safegraph_ca/adj.npy", 58],
        "safegraph_tx": [base_dir + "safegraph_tx", base_dir + "safegraph_tx/adj.npy", 254],
        "safegraph_ny": [base_dir + "safegraph_ny", base_dir + "safegraph_ny/adj.npy", 62],

        # OD Prediction: N*N -> N*N
        "sz_taxi_od": [base_dir + "sz_taxi_od", base_dir + "shenzhen/adj.npy", 491],
        "sz_bike_od": [base_dir + "sz_bike_od", base_dir + "shenzhen/adj.npy", 491],
        "sz_subway_bike_od": [
            base_dir + "sz_subway_bike_od",
            base_dir + "shenzhen/adj.npy",
            491,
        ],
        "sz_subway_taxi_od": [
            base_dir + "sz_subway_taxi_od",
            base_dir + "shenzhen/adj.npy",
            491,
        ],
        "nyc_subway_bike_od": [
            base_dir + "nyc_subway_bike_od",
            base_dir + "nyc_taxi_od/adj.npy",
            67,
        ],
        "nyc_subway_taxi_od": [
            base_dir + "nyc_subway_taxi_od",
            base_dir + "nyc_taxi_od/adj.npy",
            67,
        ],
        "sz_dd_od": [base_dir + "sz_dd_od", base_dir + "shenzhen/adj.npy", 491],
        "sz_subway_od": [base_dir + "sz_subway_od", base_dir + "shenzhen/adj.npy", 491],
        "nyc_taxi_od": [base_dir + "nyc_taxi_od", base_dir + "nyc_taxi_od/adj.npy", 67],
        "nyc_bike_od": [base_dir + "nyc_bike_od", base_dir + "nyc_taxi_od/adj.npy", 67],
        "nyc_subway_od": [
            base_dir + "nyc_subway_od",
            base_dir + "nyc_taxi_od/adj.npy",
            67,
        ],
    }

    assert dataset in d.keys()
    return d[dataset]
