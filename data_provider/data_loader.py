import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def _load_m4_wide(path):
    """Read a wide-format M4 CSV (rows = series, cols = timesteps).

    Both Hourly-train.csv and Hourly-test.csv use quoted values and have
    the series id ("H1", "H2", …) as the first column of each data row.
    Returns a list of float32 arrays, one per series, with NaN padding stripped.
    """
    df = pd.read_csv(path, header=0, index_col=0)
    series = []
    for _, row in df.iterrows():
        vals = pd.to_numeric(row, errors='coerce').dropna().values.astype(np.float32)
        series.append(vals)
    return series


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='ETTh1.csv', target='OT', scale=True):
        self.seq_len, self.pred_len = size[0], size[1]
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 12 / 4 / 4 months of hourly data
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        cols = df_raw.columns[1:]  # drop date column
        if self.features == 'S':
            cols = [self.target]
        df_data = df_raw[cols].values.astype(np.float32)

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]])
            df_data = self.scaler.transform(df_data)

        self.data_x = df_data[b1:b2]
        if self.features == 'MS':
            target_idx = list(df_raw.columns[1:]).index(self.target)
            self.data_y = df_data[b1:b2, target_idx:target_idx + 1]
        else:
            self.data_y = df_data[b1:b2]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data_x[index: index + self.seq_len]
        y = self.data_y[index + self.seq_len: index + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset_ETT_hour):
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 12 / 4 / 4 months of 15-min data
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        cols = df_raw.columns[1:]
        if self.features == 'S':
            cols = [self.target]
        df_data = df_raw[cols].values.astype(np.float32)

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]])
            df_data = self.scaler.transform(df_data)

        self.data_x = df_data[b1:b2]
        if self.features == 'MS':
            target_idx = list(df_raw.columns[1:]).index(self.target)
            self.data_y = df_data[b1:b2, target_idx:target_idx + 1]
        else:
            self.data_y = df_data[b1:b2]


class Dataset_Custom(Dataset):
    """General-purpose loader for Weather, Traffic, Electricity, Exchange."""

    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='weather.csv', target='OT', scale=True):
        self.seq_len, self.pred_len = size[0], size[1]
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        n = len(df_raw)
        # 70 / 10 / 20 split
        borders = [
            (0, int(n * 0.7)),
            (int(n * 0.7) - self.seq_len, int(n * 0.8)),
            (int(n * 0.8) - self.seq_len, n),
        ]
        b1, b2 = borders[self.set_type]

        # Drop date/time column (assumed first)
        date_cols = [c for c in df_raw.columns if c.lower() in ('date', 'datetime')]
        num_cols = [c for c in df_raw.columns if c not in date_cols]
        if self.features == 'S':
            num_cols = [self.target]
        df_data = df_raw[num_cols].values.astype(np.float32)

        if self.scale:
            train_b1, train_b2 = borders[0]
            self.scaler.fit(df_data[train_b1:train_b2])
            df_data = self.scaler.transform(df_data)

        self.data_x = df_data[b1:b2]
        if self.features == 'MS':
            target_idx = num_cols.index(self.target)
            self.data_y = df_data[b1:b2, target_idx:target_idx + 1]
        else:
            self.data_y = df_data[b1:b2]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data_x[index: index + self.seq_len]
        y = self.data_y[index + self.seq_len: index + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    """M4 Hourly dataset loader (wide-format, one series per row).

    Data layout expected under ``root_path/data_path/``:
        Hourly-train.csv  —  variable-length train series, NaN-padded
        Hourly-test.csv   —  fixed-length test horizon per series

    Splits:
        train / val  — sliding windows from the first 80 % / last 20 % of
                       each training series.
        test         — one sample per series: x = last ``seq_len`` steps of the
                       train series, y = first ``pred_len`` steps of the test file.

    Each series is z-score normalised independently (scaler fitted on the
    training portion only).  Output shape is always ``(T, 1)`` — M4 is
    univariate by construction.
    """

    def __init__(self, root_path, flag='train', size=None, features='S',
                 data_path='m4', target='OT', scale=True):
        self.seq_len = size[0]
        self.pred_len = size[1]
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        m4_dir = os.path.join(self.root_path, self.data_path)
        train_series = _load_m4_wide(os.path.join(m4_dir, 'Hourly-train.csv'))
        test_series = _load_m4_wide(os.path.join(m4_dir, 'Hourly-test.csv'))

        min_len = self.seq_len + self.pred_len
        xs, ys = [], []

        for train_s, test_s in zip(train_series, test_series):
            n = len(train_s)
            split = int(n * 0.8)  # 80 % train, 20 % val

            scaler = StandardScaler()
            if self.scale and split > 0:
                scaler.fit(train_s[:split].reshape(-1, 1))

            def _norm(arr, _scaler=scaler, _do=self.scale and split > 0):
                if _do:
                    return _scaler.transform(arr.reshape(-1, 1)).ravel()
                return arr

            if self.flag == 'test':
                if n < self.seq_len or len(test_s) < self.pred_len:
                    continue
                x = _norm(train_s[-self.seq_len:]).reshape(self.seq_len, 1)
                y = _norm(test_s[:self.pred_len]).reshape(self.pred_len, 1)
                xs.append(x)
                ys.append(y)

            else:
                if self.flag == 'train':
                    chunk = _norm(train_s[:split])
                else:  # val — overlap by seq_len for continuity
                    chunk = _norm(train_s[max(0, split - self.seq_len):])

                if len(chunk) < min_len:
                    continue

                for j in range(len(chunk) - self.seq_len - self.pred_len + 1):
                    xs.append(chunk[j: j + self.seq_len].reshape(self.seq_len, 1))
                    ys.append(chunk[j + self.seq_len: j + self.seq_len + self.pred_len]
                               .reshape(self.pred_len, 1))

        self.data_x = np.stack(xs, axis=0).astype(np.float32)
        self.data_y = np.stack(ys, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return torch.tensor(self.data_x[index]), torch.tensor(self.data_y[index])

    def inverse_transform(self, data):
        # Per-series scalers are not retained after construction; return as-is.
        return data
