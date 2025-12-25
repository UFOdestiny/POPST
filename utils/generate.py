import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
sys.path.append(os.path.abspath(__file__ + "/../../"))
sys.path.append("/home/dy23a.fsu/st/")

from utils.args import get_data_path


def _resolve_device(data=None, device: Optional[Union[str, torch.device]] = None):
    if device is not None:
        return torch.device(device)
    if isinstance(data, torch.Tensor):
        return data.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class MinMaxScaler:
    def __init__(self, _min=None, _max=None):
        self.min = _min
        self.max = _max

    def fit(self, data):
        self.min = data.min()
        self.max = data.max()

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class StandardScaler:
    def __init__(self, mean=None, std=None, offset=None):
        """
        :param axis: 指定标准化的轴。
                     - axis=0：对每个特征进行标准化。
                     - axis=1：对每个样本的特征集合标准化。
                     - axis=2：对每个时间点的特征集合标准化。
        """
        if mean is not None:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)
            self.offset = torch.tensor(offset)
        else:
            self.mean = mean
            self.std = std
            self.offset = offset

    def fit(self, data):
        sequence, region, dim = data.shape
        temp_data = data.reshape(sequence * region, dim)
        self.mean = np.mean(temp_data, axis=0)
        self.std = np.std(temp_data, axis=0)

        # 确保标准差不为 0，避免除以 0 的情况
        self.std[self.std == 0] = 1.0

        # 计算标准化后的数据的最小值
        normalized_data = (temp_data - self.mean) / self.std
        self.offset = -np.min(normalized_data, axis=0)

        # self.offset+=1.5 # important

    def transform(self, data):
        if self.mean is None or self.std is None or self.offset is None:
            raise ValueError(
                "StandardScaler_OD is not fitted yet. Call 'fit' with training data first."
            )
        normalized_data = (data - self.mean) / self.std
        return normalized_data + self.offset

    def inverse_transform(self, data, device=None):
        if self.mean is None or self.std is None or self.offset is None:
            raise ValueError(
                "StandardScaler_OD is not fitted yet. Call 'fit' with training data first."
            )
        dev = _resolve_device(data, device)
        offset = self.offset.to(device=dev)
        std = self.std.to(device=dev)
        mean = self.mean.to(device=dev)

        if isinstance(data, torch.Tensor):
            unshifted_data = data - offset
            return unshifted_data * std + mean

        tensor_data = torch.tensor(data, device=dev)
        restored = (tensor_data - offset) * std + mean
        return restored.cpu().numpy()


class StandardScaler_OD:
    def __init__(self, mean=None, std=None, offset=None):
        """
        :param axis: 指定标准化的轴。
                     - axis=0：对每个特征进行标准化。
                     - axis=1：对每个样本的特征集合标准化。
                     - axis=2：对每个时间点的特征集合标准化。
        """
        if mean is not None:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)
            self.offset = torch.tensor(offset)
        else:
            self.mean = mean
            self.std = std
            self.offset = offset

    def fit(self, data):
        sequence, region, region2 = data.shape
        temp_data = data.reshape(sequence * region * region2, 1)
        self.mean = np.mean(temp_data, axis=0)
        self.std = np.std(temp_data, axis=0)

        # 确保标准差不为 0，避免除以 0 的情况
        self.std[self.std == 0] = 1.0

        # 计算标准化后的数据的最小值
        normalized_data = (temp_data - self.mean) / self.std
        self.offset = -np.min(normalized_data, axis=0)

        # self.offset+=1.5 # important

    def transform(self, data):
        if self.mean is None or self.std is None or self.offset is None:
            raise ValueError(
                "StandardScaler is not fitted yet. Call 'fit' with training data first."
            )
        normalized_data = (data - self.mean) / self.std
        return normalized_data + self.offset

    def inverse_transform(self, data, device=None):
        if self.mean is None or self.std is None or self.offset is None:
            raise ValueError(
                "StandardScaler is not fitted yet. Call 'fit' with training data first."
            )
        dev = _resolve_device(data, device)
        offset = self.offset.to(device=dev)
        std = self.std.to(device=dev)
        mean = self.mean.to(device=dev)

        if isinstance(data, torch.Tensor):
            unshifted_data = data - offset
            return unshifted_data * std + mean

        tensor_data = torch.tensor(data, device=dev)
        restored = (tensor_data - offset) * std + mean
        return restored.cpu().numpy()


class LogScaler:
    def __init__(self):
        self.base = 1

    def transform(self, data):
        return np.log(data + 1)

    def inverse_transform(self, data, device=None):
        if isinstance(data, np.ndarray):
            return np.exp(data) - 1
        dev = _resolve_device(data, device)
        return torch.exp(data).to(device=dev) - 1


class LogMinMaxScaler:
    def __init__(self, data_min=0, data_max=0):
        self.data_min_ = torch.tensor(data_min)
        self.data_max_ = torch.tensor(data_max)

    def fit(self, data):
        log_data = np.log1p(data)
        self.data_min_ = log_data.min()
        self.data_max_ = log_data.max()
        print(f"LogMinMaxScaler min: {self.data_min_}, max:{self.data_max_}")
        return self

    def transform(self, data):
        log_data = np.log1p(data)
        return (log_data - self.data_min_) / (self.data_max_ - self.data_min_)

    def inverse_transform(self, data, device=None):
        span = self.data_max_ - self.data_min_
        if isinstance(data, torch.Tensor):
            log_data = data * span + self.data_min_
            return torch.expm1(log_data.to(device=_resolve_device(data, device)))

        log_data = data * span.cpu().numpy() + self.data_min_.cpu().numpy()
        return np.expm1(log_data)


class RatioScaler:
    def __init__(self, ratio=1000):
        self.ratio = ratio

    def transform(self, data):
        return data / self.ratio

    def inverse_transform(self, data):
        return data * self.ratio


def split_dataset(data, args, tra_ratio, val_ratio, test_ratio):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)
    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0] - abs(max(y_offsets)))  # Exclusive
    print(f"Index min: {min_t}, max: {max_t}")
    idx = np.arange(min_t, max_t, 1)

    N = len(idx)
    val_ = int(round(val_ratio * N))
    test_ = int(round(test_ratio * N))
    train_ = N - val_ - test_

    idx_train = idx[:train_]
    idx_val = idx[train_ : train_ + val_]
    idx_test = idx[train_ + val_ :]
    idx_all = idx[:]
    return idx_train, idx_val, idx_test, idx_all


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    print(df.shape)
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]
    if add_time_of_day:
        time_ind = (
            df.index.values - df.index.values.astype("datetime64[D]")
        ) / np.timedelta64(1, "D")
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print("idx min & max:", min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def generate_flow(args):
    # data_path = "D:/OneDrive - Florida State University/datasets/shenzhen/shenzhen_1h/values_in.npy"
    # data_path = "D:/OneDrive - Florida State University/datasets/nyc/flow.npy"
    # data_path = "/blue/gtyson.fsu/dy23a.fsu/datasets/safegraph/pattern/panhandle.npy"

    data_path = "/blue/gtyson.fsu/dy23a.fsu/datasets/safegraph/tx/flow_tx.npy"

    # N * D * T
    data = np.load(data_path)
    print(f"data shape: {data.shape}")
    print(f"Original max: {data.max()}, min: {data.min()}, mean: {data.mean()}")

    if len(data.shape) == 3:
        data = data.transpose(2, 0, 1)
    else:
        data = data[:, np.newaxis].transpose(2, 0, 1)

    # split idx
    idx_train, idx_val, idx_test, idx_all = split_dataset(data, args, 0.8, 0.1, 0.1)

    # normalize
    # x_train = data[:idx_val[0] - args.seq_length_x, :, :]

    # hour day month
    # data_hdm = data[:, :, 1:]
    # data = np.expand_dims(data[:, :, 0], axis=-1)

    scaler = LogMinMaxScaler()  # LogScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    # scaler = StandardScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)
    # mean = scaler.mean
    # std = scaler.std
    # offset = scaler.offset

    # hour day month
    # data = np.concatenate([data, data_hdm], axis=-1)

    # print(mean)
    # print(std)
    # print(offset)

    print(f"Normalized max: {data.max()}, min: {data.min()}, mean: {data.mean()}")

    base_dir = _ensure_dir(Path(get_data_path()) / args.dataset / args.years)
    np.savez_compressed(
        base_dir / "his.npz",
        data=data,
        min=scaler.data_min_,
        max=scaler.data_max_,
    )
    _save_indices(base_dir, idx_train, idx_val, idx_test, idx_all)


def generate_od(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    data_path = "/blue/gtyson.fsu/dy23a.fsu/switch/nyc/taxi.npy"
    data_path = "/blue/gtyson.fsu/dy23a.fsu/switch/nyc/bike.npy"
    # data_path = "/blue/gtyson.fsu/dy23a.fsu/switch/nyc/subway.npy"

    # data_path = "/blue/gtyson.fsu/dy23a.fsu/switch/shenzhen/taxi.npy"
    # data_path2 = "/blue/gtyson.fsu/dy23a.fsu/switch/shenzhen/dd.npy"
    # data_path3 = "/blue/gtyson.fsu/dy23a.fsu/switch/shenzhen/bike.npy"

    data = np.load(data_path)
    data = data.transpose(2, 0, 1)  # [..., np.newaxis]

    data = data[:, ...]

    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0] - abs(max(y_offsets)))  # Exclusive
    print("idx min & max:", min_t, max_t)
    idx = np.arange(min_t, max_t, 1)

    print("final data shape:", data.shape, "idx shape:", idx.shape)
    num_samples = len(idx)
    num_train = round(num_samples * 0.8)
    num_val = round(num_samples * 0.1)
    print(num_train, num_val, num_samples - num_train - num_val)

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train : num_train + num_val]
    idx_test = idx[num_train + num_val :]
    idx_all = idx[:]
    print(data[0][-1])
    print("max, min, mean: ", data.max(), data.min(), data.mean())
    # normalize
    # x_train = data[:idx_val[0] - args.seq_length_x, :, :]

    # scaler = StandardScaler_OD()
    # scaler.fit(data)
    # data = scaler.transform(data)
    # mean = scaler.mean
    # std = scaler.std
    # offset = scaler.offset
    # print("mean, std, offset: ", mean, std, offset)
    # print("max, min, mean: ", data.max(), data.min(), data.mean())

    scaler = LogScaler()
    data = scaler.transform(data)
    print("max, min, mean: ", data.max(), data.min(), data.mean())
    d = scaler.inverse_transform(data)

    print(d[0][-1])

    base_dir = _ensure_dir(Path(get_data_path()) / args.dataset / args.years)
    print("save to: ", base_dir)
    np.savez_compressed(base_dir / "his.npz", data=data)
    _save_indices(base_dir, idx_train, idx_val, idx_test, idx_all)


def generate_od_2(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    # data_path1 = "/blue/gtyson.fsu/dy23a.fsu/switch/shenzhen/subway_.npy"
    # data_path2 = "/blue/gtyson.fsu/dy23a.fsu/switch/shenzhen/bike.npy"
    # data = np.load(data_path1).transpose(2, 0, 1)[:672,...]
    # data_2 = np.load(data_path2).transpose(2, 0, 1)[:672,...]

    data_path1 = "/blue/gtyson.fsu/dy23a.fsu/switch/nyc/subway.npy"
    data_path2 = "/blue/gtyson.fsu/dy23a.fsu/switch/nyc/bike.npy"
    data = np.load(data_path1).transpose(2, 0, 1)
    data_2 = np.load(data_path2).transpose(2, 0, 1)

    data = np.concatenate((data, data_2), axis=0)
    print(data.shape)
    # exit()

    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0] - abs(max(y_offsets)))  # Exclusive
    print("idx min & max:", min_t, max_t)
    idx = np.arange(min_t, max_t, 1)

    print("final data shape:", data.shape, "idx shape:", idx.shape)
    num_samples = len(idx)
    num_train = round(num_samples * 0.9)
    num_val = 287  # 168
    num_test = 287  # 168
    num_train = num_samples - num_test - num_val
    print(num_train, num_val, num_samples - num_train - num_val)

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train : num_train + num_val]
    idx_test = idx[num_train + num_val :]
    idx_all = idx[:]
    # print(data[0][-1])
    print("max, min, mean: ", data.max(), data.min(), data.mean())
    # normalize
    # x_train = data[:idx_val[0] - args.seq_length_x, :, :]

    # scaler = StandardScaler_OD()
    # scaler.fit(data)
    # data = scaler.transform(data)
    # mean = scaler.mean
    # std = scaler.std
    # offset = scaler.offset
    # print("mean, std, offset: ", mean, std, offset)
    # print("max, min, mean: ", data.max(), data.min(), data.mean())

    scaler = LogScaler()
    data = scaler.transform(data)
    print("max, min, mean: ", data.max(), data.min(), data.mean())
    d = scaler.inverse_transform(data)

    print(d[0][-1])

    base_dir = _ensure_dir(Path(get_data_path()) / args.dataset / args.years)
    print("save to: ", base_dir)
    np.savez_compressed(base_dir / "his.npz", data=data)
    _save_indices(base_dir, idx_train, idx_val, idx_test, idx_all)


def _save_indices(base_dir: Path, idx_train, idx_val, idx_test, idx_all):
    np.save(base_dir / "idx_train", idx_train)
    np.save(base_dir / "idx_val", idx_val)
    np.save(base_dir / "idx_test", idx_test)
    np.save(base_dir / "idx_all", idx_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="safegraph_tx", help="dataset name")
    parser.add_argument("--years", type=str, default="2018")
    parser.add_argument("--seq_length_x", type=int, default=7, help="sequence Length")
    parser.add_argument("--seq_length_y", type=int, default=3, help="sequence Length")
    parser.add_argument("--tod", type=int, default=1, help="time of day")
    parser.add_argument("--dow", type=int, default=1, help="day of week")

    args = parser.parse_args()
    generate_flow(args)
