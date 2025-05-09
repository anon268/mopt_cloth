import time
import os.path as osp
from random import shuffle

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    def __init__(self, buffer_size, name, save_path, mode="pos", restrain=False):
        super().__init__()
        self.buffer_size = buffer_size
        self.data_count = 0
        self.fillup_nb = 0
        self.name = name
        self.save_path = save_path
        self.isopen = True
        self.mode = mode
        self.restrain = restrain
        self.data = [
            torch.zeros((self.buffer_size, 1600, 3), dtype=torch.float),
            torch.zeros((self.buffer_size, 3), dtype=torch.float),
            torch.zeros((self.buffer_size), dtype=torch.float),
            torch.zeros((self.buffer_size, 1600, 3), dtype=torch.float),
            torch.zeros((self.buffer_size), dtype=torch.bool),
        ]
        self.idcs = None

    @property
    def fill_line(self):
        return self.data_count if self.fillup_nb == 0 else self.buffer_size

    def __add__(self, other):
        assert self.mode == other.mode
        assert len(self.data) == len(other.data)
        super().__add__(other)
        self.buffer_size += other.buffer_size
        self.data_count += other.data_count
        for i, d in enumerate(other.data):
            self.data[i] = torch.cat((self.data[i], d), dim=0)
        self.make_idcs()
        return self

    def __len__(self):
        if self.idcs is None:
            len = self.data_count
        else:
            len = self.idcs.shape[0]
        return len

    def __getitem__(self, idx):
        if self.idcs is not None:
            idx = self.idcs[idx]

        return tuple(d[idx] for d in self.data)

    def __str__(self):
        stats = self._get_fill_stats()
        return f"Buffer {self.name}: {stats[0]:.2f}% full ({len(self)} entries), refill {stats[1]:.2f}%, refilled {stats[2]} times"

    def make_idcs(self):
        if self.restrain:
            self.idcs = torch.nonzero(self.data[2][: self.fill_line] >= 0)[:, 0]
            if len(self.idcs) == 0:
                self.idcs = None
        else:
            self.idcs = None

    def get_mean_std(self, fpath):
        with h5py.File(fpath, "r") as f:
            pos = torch.tensor(np.array(f["pos"]))
            # cpos, _, _ = center_align(pos)
            cpos = pos
            std, mean = torch.std_mean(cpos.reshape(-1, 3), axis=0)
            mn, mx = (
                torch.min(cpos.reshape(-1, 3), dim=0).values,
                torch.max(cpos.reshape(-1, 3), dim=0).values,
            )
        return mean, std, mn, mx

    def _get_fill_stats(self):
        return (
            self.fill_line / self.buffer_size * 100,
            self.data_count / self.buffer_size * 100,
            self.fillup_nb,
        )

    def pre_fill(self, fpath):
        print("Pre-filling")
        with h5py.File(fpath, "r") as h5_file:
            all_pos = torch.tensor(np.array(h5_file["pos"]))
            filler_size = min(all_pos.shape[0], self.buffer_size - 1)
            self.data[0][:filler_size] = all_pos[:filler_size]
            del all_pos
            all_action = torch.tensor(np.array(h5_file["action"]))
            self.data[1][:filler_size] = all_action[:filler_size]
            del all_action
            all_rewards = torch.tensor(np.array(h5_file["rewards"]))
            self.data[2][:filler_size] = all_rewards.squeeze(-1)[:filler_size]
            del all_rewards
            all_npos = torch.tensor(np.array(h5_file["next_pos"]))
            self.data[3][:filler_size] = all_npos[:filler_size]
            del all_npos
            all_hasnext = torch.tensor(np.array(h5_file["has_next"]))
            self.data[4][:filler_size] = all_hasnext.squeeze(-1)[:filler_size]
            del all_hasnext

            self.data_count = filler_size
        print("Filled")
        if filler_size >= self.buffer_size - 1:
            self.fillup_nb += 1

    def save(self):
        self.isopen = False
        dest = osp.join(
            self.save_path,
            f"{self.name}_chunk{self.fillup_nb}_size{self.buffer_size}.hdf5",
        )
        cpt = 0
        while osp.isfile(dest):
            dest = osp.join(
                self.save_path,
                f"{self.name}_chunk{self.fillup_nb}_size{self.buffer_size}_{cpt}.hdf5",
            )
            cpt += 1

        with h5py.File(dest, "a") as f:
            f.create_dataset("pos", data=self.data[0])
            f.create_dataset("action", data=self.data[1])
            f.create_dataset("rewards", data=self.data[2])
            f.create_dataset("next_pos", data=self.data[3])
            f.create_dataset("has_next", data=self.data[4])
        self.isopen = True

    def append(self, data):
        for i, d in enumerate(data):
            self.data[i][self.data_count] = d
        self.data_count += 1

        if self.data_count >= self.buffer_size:
            self.save()
            time.sleep(0.1)
            while not self.isopen:
                print("Saving")
                time.sleep(1)
            self.data_count = 0
            self.fillup_nb += 1


class H5Dataset(Dataset):
    def __init__(self, h5_file, idcs, nfunc=1):
        super().__init__()
        self.rw_field = "rewards"
        if nfunc > 1:
            self.rw_field = "new_rewards"
        self.nfunc = nfunc
        self.load(h5_file, idcs)

    def load(self, h5_file, idcs):
        all_pos = np.array(h5_file["pos"])[idcs]
        all_npos = np.array(h5_file["next_pos"])[idcs]
        all_action = np.array(h5_file["action"])[idcs]
        all_rewards = np.array(h5_file[self.rw_field])[idcs]
        all_hasnext = np.array(h5_file["has_next"])[idcs]
        if len(all_hasnext.shape) == 2:
            all_hasnext = all_hasnext.squeeze(-1)
        if len(all_rewards.shape) == 2:
            pos_idx = np.nonzero(all_rewards[:, 0] >= 0)[0]
        else:
            pos_idx = np.nonzero(all_rewards >= 0)[0]
        if self.nfunc == 3:
            rw_select = np.array([0, 1, 5])
        else:
            rw_select = np.arange(self.nfunc)
        self.data = (
            torch.tensor(all_pos[pos_idx]),
            torch.tensor(all_action[pos_idx]),
            torch.tensor(all_rewards[pos_idx][:, rw_select]),
            torch.tensor(all_npos[pos_idx]),
        )

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return (
            self.data[0][idx],
            self.data[1][idx],
            self.data[2][idx],
            self.data[3][idx],
        )


class MultiH5Dataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return sum([ds.__len__() for ds in self.datasets])

    def __getitem__(self, idx):
        for ds in self.datasets:
            if idx - len(ds) < 0:
                return ds[idx]
            idx -= len(ds)


def make_split_ds(val_ratio, fpath, multi=1, train_percent=1):
    if isinstance(fpath, list):
        if len(fpath) == 1:
            return _make_split_ds(val_ratio, fpath[0], multi, train_percent)
        train_ds = []
        val_ds = []
        for f in fpath:
            t, v = _make_split_ds(val_ratio, f, multi, train_percent)
            train_ds.append(t)
            val_ds.append(v)
        return MultiH5Dataset(train_ds), MultiH5Dataset(val_ds)
    else:
        return _make_split_ds(val_ratio, [fpath], multi, train_percent)


def _make_split_ds(val_ratio, fpath, multi=1, train_percent=1):
    with h5py.File(fpath, "r") as f:
        tot_sz = f["action"].shape[0]
        val_sz = int(val_ratio * tot_sz)
        train_sz = tot_sz - val_sz
        idcs = list(range(tot_sz))
        shuffle(idcs)
        train_ds = H5Dataset(f, idcs[: int(train_percent * train_sz)], nfunc=multi)
        val_ds = H5Dataset(f, idcs[-val_sz:], nfunc=multi)
    return train_ds, val_ds
