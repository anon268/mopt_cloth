import os
import sys
import torch
import wandb
import numpy as np
import torch.nn as nn
import h5pickle as h5pkl
import matplotlib.pyplot as plt

from random import shuffle
from tqdm import tqdm
from models.unet import UNet
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from torchvision.tv_tensors import Mask


class H5Dataset(Dataset):
    def __init__(self, data, idxs):
        self.idxs = sorted(idxs)
        self.imgs = torch.from_numpy(data[0][self.idxs]).permute(0, 3, 1, 2)
        self.masks = torch.from_numpy(data[1][self.idxs])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return (self.imgs[idx], self.masks[idx])


def load_data(data_path):
    data = h5pkl.File(data_path, "r")
    _nspl = data["imgs"].shape[0]
    _id_list = list(range(_nspl))
    shuffle(_id_list)
    _tid, _vid = _id_list[: int(_nspl * 0.8)], _id_list[int(_nspl * 0.8) :]
    imgs = np.array(data["imgs"])
    masks = np.array(data["masks"])
    return H5Dataset((imgs, masks), _tid), H5Dataset((imgs, masks), _vid)


def validate(model, dl, wdb_run, device):
    val_transforms = v2.Compose(
        [
            v2.Resize((128, 128)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    _val_loss = []
    _inmask = []
    _outmask = []
    _loss = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for imgs, masks in tqdm(dl, "Validation"):
            masks = Mask(masks)
            imgs, masks = val_transforms(imgs, masks)
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            _val_loss.append(
                _loss(pred, masks).mean(dim=(0, 2, 3)).detach().cpu().numpy()
            )
            _inmask.append(
                np.stack(
                    [
                        _loss(pred[:, 0][masks[:, 0] > 0], masks[:, 0][masks[:, 0] > 0])
                        .mean()
                        .detach()
                        .cpu()
                        .numpy(),
                        _loss(pred[:, 1][masks[:, 1] > 0], masks[:, 1][masks[:, 1] > 0])
                        .mean()
                        .detach()
                        .cpu()
                        .numpy(),
                    ],
                    axis=0,
                )
            )
            _outmask.append(
                np.stack(
                    [
                        _loss(
                            pred[:, 0][masks[:, 0] == 0], masks[:, 0][masks[:, 0] == 0]
                        )
                        .mean()
                        .detach()
                        .cpu()
                        .numpy(),
                        _loss(
                            pred[:, 1][masks[:, 1] == 0], masks[:, 1][masks[:, 1] == 0]
                        )
                        .mean()
                        .detach()
                        .cpu()
                        .numpy(),
                    ],
                    axis=0,
                )
            )
        _val_loss = np.stack(_val_loss, axis=0).mean(axis=0)
        _inmask = np.stack(_inmask, axis=0).mean(axis=0)
        _outmask = np.stack(_outmask, axis=0).mean(axis=0)
        wdb_run.log(
            {
                "Validation/loss": _val_loss.mean(),
                "Validation/pick loss": _val_loss[0],
                "Validation/place loss": _val_loss[1],
                "Validation/pick loss in mask": _inmask[0],
                "Validation/place loss in mask": _inmask[1],
                "Validation/pick loss out mask": _outmask[0],
                "Validation/place loss out mask": _outmask[1],
            }
        )


def main():
    data_path = sys.argv[1]
    wdb_run = wandb.init(
        project="GRAVIS_transfer",
        entity="april-lab",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = load_data(data_path)
    model = UNet(3, 2).to(device)
    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(128, 128), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=16)
    opt = Adam(model.parameters(), lr=1e-4)
    loss = nn.MSELoss(reduction="none")
    loss2 = nn.MSELoss()

    for epoch_id in range(300):
        wdb_run.log({"Epoch": epoch_id})
        loss_arr = []
        validate(model, val_dl, wdb_run, device)
        for imgs, masks in tqdm(train_dl, "Training"):
            masks = Mask(masks)
            imgs, masks = transforms(imgs, masks)
            imgs, masks = imgs.to(device), masks.to(device)

            pred = model(imgs)
            opt.zero_grad()
            _l1 = loss(pred, masks).mean(dim=(0, 2, 3))
            _l2 = loss2(pred[masks == 0], masks[masks == 0])
            _l = _l1.mean()  # + _l2
            _l.backward()
            opt.step()
            loss_arr.append(_l1.detach().cpu().numpy())
            wdb_run.log(
                {
                    "Epoch": epoch_id,
                    "Training/batch loss": _l.mean().item(),
                    "Training/batch pick loss": _l1[0].item(),
                    "Training/batch place loss": _l1[1].item(),
                    "Training/batch seg loss": _l2.item(),
                }
            )
        _epoch_loss = np.stack(loss_arr, axis=0).mean(axis=0)
        wdb_run.log(
            {
                "Training/loss": _epoch_loss.mean(),
                "Training/pick loss": _epoch_loss[0],
                "Training/place loss": _epoch_loss[1],
            }
        )
        os.makedirs("/exp/unet", exist_ok=True)
        torch.save(model.state_dict(), "/exp/unet/ckpt_biggest.pth")


if __name__ == "__main__":
    main()
