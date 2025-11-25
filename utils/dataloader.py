import multiprocessing as mp
import os
import pickle
import sys
import threading
import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
sys.path.append("/home/dy23a.fsu/st/")

from utils.args import get_data_path
from utils.generate import (
    LogMinMaxScaler,
    LogScaler,
    StandardScaler,
    StandardScaler_OD,
)
from scipy.spatial import distance


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
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)

        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        if not droplast and self.size % self.bs != 0:
            self.num_batch += 1

        self.current_ind = 0
        logger.info(f"{name} num: {self.idx.shape[0]}, Batch num: {self.num_batch}")

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.idx = self.idx[perm]

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, ...]
            y[i] = self.data[idx_ind[i] + self.y_offsets, ...]  # dimension

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind:end_ind, ...]

                x_shape = (
                    len(idx_ind),
                    self.seq_len,
                    self.data.shape[1],
                    self.data.shape[-1],
                )
                x_shared = mp.RawArray("f", int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype="f").reshape(x_shape)

                y_shape = (
                    len(idx_ind),
                    self.horizon,
                    self.data.shape[1],
                    self.data.shape[-1],
                )
                y_shared = mp.RawArray("f", int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype="f").reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = max(1, min(mp.cpu_count(), array_size // 2))
                chunk_size = (array_size + num_threads - 1) // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = min(start_index + chunk_size, array_size)
                    thread = threading.Thread(
                        target=self.write_to_shared_array,
                        args=(x, y, idx_ind, start_index, end_index),
                    )
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield x, y
                self.current_ind += 1

        return _wrapper()


def load_dataset_plain(data_path, args, logger, drop=False):
    ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
    logger.info(f"Data shape: {ptr['data'].shape}")
    X = ptr["data"]
    xy = []
    for cat in ["train", "val", "test"]:
        idx = np.load(os.path.join(data_path, args.years, f"idx_{cat}.npy"))
        xy.append(X[idx])
    return xy, LogScaler()


def load_dataset(data_path, args, logger, drop=False):
    ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
    logger.info(f"Data shape: {ptr['data'].shape}")

    X = ptr["data"]

    dataloader = {}
    for cat in ["train", "val", "test"]:
        idx = np.load(os.path.join(data_path, args.years, f"idx_{cat}.npy"))
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
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)

        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        if not droplast and self.size % self.bs != 0:
            self.num_batch += 1
        self.current_ind = 0
        logger.info(f"{name} num: {self.idx.shape[0]}, Batch num: {self.num_batch}")

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
        self.O = adj_O
        self.D = adj_D

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.idx = self.idx[perm]

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, ...]
            y[i] = self.data[idx_ind[i] + self.y_offsets, ...]  # dimension

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind:end_ind, ...]

                # print(idx_ind)
                adj_index = [i % 6 for i in idx_ind]

                x_shape = (
                    len(idx_ind),
                    self.seq_len,
                    self.data.shape[1],
                    self.data.shape[-1],
                )
                x_shared = mp.RawArray("f", int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype="f").reshape(x_shape)

                y_shape = (
                    len(idx_ind),
                    self.horizon,
                    self.data.shape[1],
                    self.data.shape[-1],
                )
                y_shared = mp.RawArray("f", int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype="f").reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = max(1, min(mp.cpu_count(), array_size // 2))
                chunk_size = (array_size + num_threads - 1) // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = min(start_index + chunk_size, array_size)
                    thread = threading.Thread(
                        target=self.write_to_shared_array,
                        args=(x, y, idx_ind, start_index, end_index),
                    )
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield x, y, self.O[:, :, adj_index].transpose(2, 0, 1), self.D[
                    :, :, adj_index
                ].transpose(2, 0, 1)
                self.current_ind += 1

        return _wrapper()


def construct_dyn_G(OD_data: np.array, perceived_period: int = 6):
    """Construct dynamic graphs based on OD history using vectorized distance computation."""
    from scipy.spatial.distance import cdist

    train_len = int(OD_data.shape[0] * 0.8)
    num_periods_in_history = train_len // perceived_period
    OD_history = OD_data[: num_periods_in_history * perceived_period, :, :, :]

    O_dyn_G, D_dyn_G = [], []
    for t in range(perceived_period):
        OD_t_avg = np.mean(OD_history[t::perceived_period, :, :, :], axis=0).squeeze(
            axis=-1
        )

        # Vectorized cosine distance computation
        O_G_t = cdist(OD_t_avg, OD_t_avg, metric="cosine")
        D_G_t = cdist(OD_t_avg.T, OD_t_avg.T, metric="cosine")

        O_dyn_G.append(O_G_t)
        D_dyn_G.append(D_G_t)

    return np.stack(O_dyn_G, axis=-1), np.stack(D_dyn_G, axis=-1)


def load_dataset_MPGCN(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
    logger.info(f"Data shape: {ptr['data'].shape}")
    X = ptr["data"]

    adjo, adjd = construct_dyn_G(X[..., np.newaxis])

    dataloader = {}
    for cat in ["train", "val", "test"]:
        idx = np.load(os.path.join(data_path, args.years, f"idx_{cat}.npy"))
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
