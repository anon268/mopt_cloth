"""Agent class"""

import os
import os.path as osp
import pickle as pkl

import torch
import torch.nn.functional as F

from models.mopt import (
    ConvED,
    ConvDecoderLin,
)


class Agent(object):
    def __init__(
        self,
        name,
        task,
        image_size,
        device,
        pick_layers,
        place_layers,
        nfunc=1,
        encoder="conv",
    ):
        self.name = name
        self.task = task
        self.device = device
        self.image_size = image_size
        self.encoder = encoder
        self.nfunc = nfunc
        self.pick_layers = pick_layers
        self.place_layers = place_layers
        self.models_dir = osp.join("checkpoints", self.name)
        os.makedirs(self.models_dir, exist_ok=True)
        self.pick_model = None
        self.place_model = None

    def _attr_dict(self):
        attr = {
            "name": self.name,
            "task": self.task,
            "device": self.device,
            "image_size": self.image_size,
            "models_dir": self.models_dir,
            "nfunc": self.nfunc,
            "encoder": self.encoder,
            "pick_layers": self.pick_layers,
            "place_layers": self.place_layers,
        }
        return attr

    def _load_attr(self, attr):
        self.name = attr["name"]
        self.task = attr["task"]
        self.device = attr["device"]
        self.image_size = attr["image_size"]
        self.models_dir = attr["models_dir"]
        self.nfunc = attr["nfunc"]
        self.encoder = attr["encoder"]
        self.pick_layers = attr["encoder"]
        self.place_layers = attr["place_layers"]

        return attr

    def train(self):
        self.pick_model.train()
        self.place_model.train()

    def eval(self):
        self.pick_model.eval()
        self.place_model.eval()

    def show_param_count(self):
        pick_params = self.pick_model.count_params()
        place_params = self.place_model.count_params()
        print(f"Params:")
        print(f" Pick head: {pick_params}")
        print(f" Place head: {place_params}")
        print(f" Total: {pick_params + place_params}")
        return pick_params + place_params

    def get_params_list(self, lr):
        return [
            {"params": self.pick_model.parameters(), "lr": lr},
            {"params": self.place_model.parameters(), "lr": lr},
        ]

    def save(self, prefix=""):
        print(f"Saving: {self.name}, {self.models_dir}, {self.task}")
        os.makedirs(osp.join(self.models_dir, prefix), exist_ok=True)
        checkpoint = {
            "pick_model": self.pick_model.state_dict(),
            "place_model": self.place_model.state_dict(),
        }

        torch.save(checkpoint, osp.join(self.models_dir, prefix, "ckpt.pth"))
        attr = self._attr_dict()
        with open(osp.join(self.models_dir, prefix, "params.pkl"), "wb") as f:
            pkl.dump(attr, f)

    def load(self, path=None):
        if path is None:
            path = self.models_dir

        with open(osp.join(path, "params.pkl"), "rb") as f:
            attr = pkl.load(f)
            self._load_attr(attr)

        checkpoint = torch.load(osp.join(path, "ckpt.pth"))
        self.pick_model.load_state_dict(checkpoint["pick_model"])
        self.place_model.load_state_dict(checkpoint["place_model"])
        self.models_dir = osp.join("checkpoints", self.name)

    def clone_model(self, agent):
        self.pick_model.load_state_dict(agent.pick_model.state_dict())
        self.place_model.load_state_dict(agent.place_model.state_dict())

    def update_weights(self, agent, tau):
        for param, agent_param in zip(
            self.pick_model.parameters(), agent.pick_model.parameters()
        ):
            param.data.copy_((1 - tau) * param.data + tau * agent_param.data)
        for param, agent_param in zip(
            self.place_model.parameters(), agent.place_model.parameters()
        ):
            param.data.copy_((1 - tau) * param.data + tau * agent_param.data)


class MOptAgent(Agent):
    def __init__(
        self,
        name,
        task,
        image_size,
        device,
        pick_layers,
        place_layers,
        batch_norm=False,
        layer_norm=True,
        nfunc=1,
        encoder="conv",
    ):
        super().__init__(
            name,
            task,
            image_size,
            device,
            pick_layers,
            place_layers,
            nfunc=nfunc,
        )
        self.pick_model = ConvED(
            3,
            pick_layers,
            [42, 42],
            nfunc,
            0,
            batch_norm,
            layer_norm,
            encoder=encoder,
        ).to(device)
        self.place_model = ConvDecoderLin(
            self.pick_model.latent_sz + 2, place_layers, nfunc, batch_norm, layer_norm
        ).to(device)

    def act(self, pos, func_id=0, ret_q=False):
        pos = pos.reshape(40, 40, 3).to(self.device)
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        if self.encoder == "conv":
            pos = F.pad(pos, (1, 1, 1, 1), "constant", -1)
        q_pick, lat_feat = self.pick_model(pos)
        q_pick = q_pick[..., 4:-4, 4:-4].clone()  # .reshape(self.nfunc, -1)
        q_pick = q_pick[0, func_id].reshape(-1)
        lat_feat = lat_feat.reshape(1, -1)

        pick = torch.argmax(q_pick).unsqueeze(0)
        q_place = self.place_model(
            torch.cat(
                (
                    lat_feat,
                    torch.cat(
                        [
                            pick.div(40, rounding_mode="floor") / 20 - 1,
                            pick.remainder(40) / 20 - 1,
                        ],
                        dim=-1,
                    ).unsqueeze(0),
                ),
                dim=-1,
            )
        )
        q_place = q_place[0, func_id]
        place = (q_place == torch.max(q_place)).nonzero()
        place = -1 + 2 * place.roll(1) / (self.image_size - 1)

        idx = torch.randint(place.shape[0], (1,)).item()

        if ret_q:
            return (pick, place[idx], q_pick, q_place)

        return (pick, place[idx])

    def run_batch(self, states, actions):
        bs = states.shape[0]
        pick_nodes = actions[:, 0].float().unsqueeze(-1)
        if self.encoder == "conv":
            states = F.pad(states, (1, 1, 1, 1), "constant", -1)
        place_pos = ((self.image_size - 1) * (1 + actions[:, 1:]) / 2).int()

        q_pick, lat_feat = self.pick_model(states)
        q_pick = q_pick[..., 4:-4, 4:-4].clone().reshape(bs, self.nfunc, -1)
        lat_feat = lat_feat.reshape(bs, -1)
        val_pick = q_pick[range(bs), :, pick_nodes[:, 0].int()]

        q_place = self.place_model(
            torch.cat(
                (
                    lat_feat,
                    torch.cat(
                        [
                            pick_nodes.div(40, rounding_mode="floor") / 20 - 1,
                            pick_nodes.remainder(40) / 20 - 1,
                        ],
                        dim=-1,
                    ),
                ),
                dim=-1,
            )
        )
        val_place = q_place[range(bs), :, place_pos[:, 1], place_pos[:, 0]]

        return val_pick, val_place, q_pick, q_place

    def max_pick_batch(self, states):
        bs = states.shape[0]
        if self.encoder == "conv":
            states = F.pad(states, (1, 1, 1, 1), "constant", -1)
        q_pick, _, _, _ = self.pick_model(states)
        q_pick = q_pick[..., 4:-4, 4:-4].clone().reshape(bs, self.nfunc, -1)
        return q_pick.max(dim=-1).values

    def max_place_batch(self, states, actions):
        bs = states.shape[0]
        pick_nodes = actions[:, 0].float().unsqueeze(-1).to(self.device)
        if self.encoder == "conv":
            states = F.pad(states, (1, 1, 1, 1), "constant", -1)
        _, lat_feat = self.pick_model(states)
        lat_feat = lat_feat.reshape(bs, -1)
        q_place = self.place_model(
            torch.cat(
                (
                    lat_feat,
                    torch.cat(
                        [
                            pick_nodes.div(40, rounding_mode="floor") / 20 - 1,
                            pick_nodes.remainder(40) / 20 - 1,
                        ],
                        dim=-1,
                    ),
                ),
                dim=-1,
            )
        )
        return q_place.flatten(start_dim=2).max(dim=-1).values
