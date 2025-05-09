"""Offline training script.
"""

import os
import os.path as osp
import argparse
import wandb
import numpy as np

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt

from agent import MOptAgent
from data_utils import make_split_ds


TAU = 0.001
IMG_SIZE = 64
BATCH_SIZE = 2048
AGENT = "EDAgentLin"


def validate(agent, val_ds, device, gamma, args):
    mean = torch.tensor([0.0159, -0.0017, 0.0003], device="cuda:0")
    std = torch.tensor([0.0626, 0.0094, 0.0609], device="cuda:0")
    scale = std
    dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    with torch.no_grad():
        for _, (states, actions, rewards, next_states) in tqdm(
            enumerate(dataloader), desc="Validation"
        ):
            states, actions, rewards, next_states = (
                states.to(device),
                actions.to(device),
                rewards.to(device),
                next_states.to(device),
            )
            states = (states - mean) / scale
            states = states.reshape(-1, 40, 40, 3)
            states = states.permute(0, 3, 1, 2)

            next_states = (next_states - mean) / scale
            next_states = next_states.reshape(-1, 40, 40, 3)
            next_states = next_states.permute(0, 3, 1, 2)

            val_pick, val_place, q_pick, q_place, _, _ = agent.run_batch(
                states, actions
            )

            term_mask = (rewards >= (0.95 * 50)).float()

            pick_tgt = agent.max_place_batch(states, actions)
            qplus = agent.max_pick_batch(next_states)
            place_tgt = (1 - term_mask) * (rewards + qplus * gamma) + term_mask * (
                rewards / (1 - gamma)
            )

            place_loss = F.mse_loss(val_place, place_tgt, reduction="none").mean(dim=0)
            pick_loss = F.mse_loss(val_pick, pick_tgt, reduction="none").mean(dim=0)

            loss = 0.5 * pick_loss.mean() + 0.5 * place_loss.mean()
            val_dict = {
                "TRAIN_ONLINE/validation loss": loss,
                "TRAIN_ONLINE/validation pick loss": pick_loss[0],
                "TRAIN_ONLINE/validation place loss": place_loss[0],
                "Stats/Validation Q Place flatten": wandb.Histogram(
                    q_place[:, 0].flatten().detach().cpu().numpy()
                ),
                "Stats/Validation Q Pick flatten": wandb.Histogram(
                    q_pick[:, 0].flatten().detach().cpu().numpy()
                ),
            }
            if pick_loss.shape[0] > 1:
                val_dict.update(
                    {
                        "VALIDATION/validation pick loss flatten": pick_loss[0],
                        "VALIDATION/validation place loss flatten": place_loss[0],
                        "VALIDATION/validation pick loss fold flat": pick_loss[1],
                        "VALIDATION/validation place loss fold flat": place_loss[1],
                        "VALIDATION/validation pick loss fold diag": pick_loss[5],
                        "VALIDATION/validation place loss fold diag": place_loss[5],
                        "Stats/Validation Q Place fold horizontal": wandb.Histogram(
                            q_place[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Validation Q Place fold diagonal": wandb.Histogram(
                            q_place[:, 5].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Validation Q Pick fold horizontal": wandb.Histogram(
                            q_pick[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Validation Q Pick fold diagonal": wandb.Histogram(
                            q_pick[:, 5].flatten().detach().cpu().numpy()
                        ),
                    }
                )

            return val_dict


def sample_place_dist(agent, val_ds, device):
    mean = torch.tensor([0.0159, -0.0017, 0.0003], device="cuda:0")
    std = torch.tensor([0.0626, 0.0094, 0.0609], device="cuda:0")
    mn = torch.tensor([-0.1755, -0.0373, -0.1776], device="cuda:0")
    mx = torch.tensor([0.1765, 0.0579, 0.1769], device="cuda:0")
    scale = std
    dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    with torch.no_grad():
        states, actions, _, next_states = next(iter(dataloader))
        states, actions, next_states = (
            states.to(device),
            actions.to(device),
            next_states.to(device),
        )
        bs = states.shape[0]
        states = (states - mean) / scale
        states = states.reshape(-1, 40, 40, 3)
        states = states.permute(0, 3, 1, 2)
        next_states = (next_states - mean) / scale
        next_states = next_states.reshape(-1, 40, 40, 3)
        next_states = next_states.permute(0, 3, 1, 2)
        actions = actions.to(device)

        # action variation
        sel_states = states[torch.randint(bs, (1,))[0]].repeat(bs, 1, 1, 1)
        states_dist = agent.max_place_batch(sel_states, actions)

        # state variation
        sel_actions = actions[torch.randint(bs, (1,))[0]].repeat(bs, 1)
        actions_dist = agent.max_place_batch(states, sel_actions)

    return states_dist, actions_dist


def run_jobs(args):
    # Set the beginning of the agent name.
    name = args.task
    device = torch.device("cuda")
    mean = torch.tensor([0.0003, 0.0113, 0.0002], device="cuda:0")
    std = torch.tensor([0.0733, 0.0078, 0.0732], device="cuda:0")
    scale = std
    print(f"Dataset mean, std: {mean}, {std}")
    gamma = 0.9
    opt_iter = 0
    lr = args.learning_rate
    pick_layers = [64, 16, 8, 32, 32, 16, 8, 8, 8]
    place_layers = [16, 8, 8, 8]
    batch_norm = False
    layer_norm = True
    encoder = "conv"
    train_percent = 1.0

    train_ds, val_ds = make_split_ds(
        0.2, args.data_file.split(","), multi=args.nfunc, train_percent=train_percent
    )
    tds_s, vds_s = len(train_ds), len(val_ds)

    config = {
        "lr": lr,
        "tau": TAU,
        "gamma": gamma,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "pick_layers": pick_layers,
        "place_layers": place_layers,
        "agent": AGENT,
        "batch_norm": batch_norm,
        "layer_norm": layer_norm,
        "train_samples": tds_s,
        "val_samples": vds_s,
        "nfunc": args.nfunc,
        "encoder": encoder,
        "train percent": train_percent,
    }

    agent_cls = MOptAgent

    agent = agent_cls(
        name,
        args.task,
        image_size=IMG_SIZE,
        device=device,
        pick_layers=pick_layers,
        place_layers=place_layers,
        batch_norm=batch_norm,
        layer_norm=layer_norm,
        nfunc=args.nfunc,
        encoder=encoder,
    )
    n_params = agent.show_param_count()
    frozen = agent_cls(
        name,
        args.task,
        image_size=IMG_SIZE,
        device=device,
        pick_layers=pick_layers,
        place_layers=place_layers,
        batch_norm=batch_norm,
        layer_norm=layer_norm,
        nfunc=args.nfunc,
        encoder=encoder,
    )
    config["n_params"] = n_params
    log_writer = wandb.init(
        project="GRAVIS_policy",
        entity="april-lab",
        job_type=args.task,
        notes="Not residual decoder. train_percent 0.1",
        group=args.run_group,
        config=config,
    )

    print(f"Train ds size: {tds_s}, val_ds: {vds_s}")

    if len(args.load_model) > 0:
        agent.load(args.load_model)
        frozen.load(args.load_model)

    agent.train()
    frozen.eval()

    epoch_id = 0
    lr = args.learning_rate

    losses = []
    losses.append(
        lambda loss_data: F.mse_loss(
            loss_data["val_pick"], loss_data["pick_tgt"]
        ).unsqueeze(0)
    )
    losses.append(
        lambda loss_data: F.mse_loss(
            loss_data["val_place"], loss_data["place_tgt"]
        ).unsqueeze(0)
    )
    losses.append(
        lambda loss_data: (
            F.mse_loss(
                loss_data["bound"], torch.zeros_like(loss_data["bound"])
            ).unsqueeze(0)
            if loss_data["bound"] is not None
            else torch.tensor(0, device=device).unsqueeze(0)
        )
    )

    while epoch_id < args.num_epoch // train_percent:
        # train critic with online data
        dataloader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
        )
        optimizer = AdamW(agent.get_params_list(lr))
        print("Start place training")
        for i, (states, actions, rewards, next_states) in tqdm(
            enumerate(dataloader), desc="Training"
        ):
            wandb_dict = {}

            states, actions, rewards, next_states = (
                states.to(device),
                actions.to(device),
                rewards.to(device),
                next_states.to(device),
            )
            states = (states - mean) / scale
            states = states.reshape(-1, 40, 40, 3)
            states = states.permute(0, 3, 1, 2)

            next_states = (next_states - mean) / scale
            next_states = next_states.reshape(-1, 40, 40, 3)
            next_states = next_states.permute(0, 3, 1, 2)

            val_pick, val_place, q_pick, q_place = agent.run_batch(states, actions)

            term_mask = (rewards >= (0.95 * 50)).float()

            with torch.no_grad():
                pick_tgt = frozen.max_place_batch(states, actions)
                qplus = frozen.max_pick_batch(next_states)

            place_tgt = (1 - term_mask) * (rewards + qplus * gamma) + term_mask * (
                rewards / (1 - gamma)
            )

            loss_data = {
                "q_pick": q_pick,
                "val_pick": val_pick,
                "pick_tgt": pick_tgt,
                "q_place": q_place,
                "val_place": val_place,
                "place_tgt": place_tgt,
            }
            diff1 = q_place.flatten() - torch.ones_like(q_place.flatten()) * 50 / (
                1 - gamma
            )

            out_range = []
            if (diff1 > 0).any():
                out_range.append(diff1[diff1 > 0])
            diff2 = q_pick.flatten() - torch.ones_like(q_pick.flatten()) * 50 / (
                1 - gamma
            )
            if (diff2 > 0).any():
                out_range.append(diff2[diff2 > 0])
            if len(out_range) > 0:
                loss_data.update({"bound": torch.cat(out_range, dim=-1)})
            else:
                loss_data.update({"bound": None})

            loss_vals = []
            optimizer.zero_grad()
            for u, loss_fn in enumerate(losses):
                loss_i = loss_fn(loss_data)
                loss_vals.append(loss_i)
            torch.cat(loss_vals).sum().backward()
            optimizer.step()

            if opt_iter % 100 == 0:
                agent.eval()
                with torch.no_grad():
                    run_name = log_writer.name.replace(" ", "").replace("/", "")
                    val_dict = validate(agent, val_ds, device, gamma, args)
                    states_dist, actions_dist = sample_place_dist(agent, val_ds, device)
                    wandb_dict.update(val_dict)
                    wandb_dict.update(
                        {
                            "Stats/Place flatten state sentitvity": wandb.Histogram(
                                states_dist[:, 0].flatten().detach().cpu().numpy()
                            ),
                            "Stats/Place action sensitivity": wandb.Histogram(
                                actions_dist.flatten().detach().cpu().numpy()
                            ),
                        }
                    )
                    if states_dist.shape[1] == 3:
                        wandb_dict.update(
                            {
                                "Stats/Place fold horizontal state sentitvity": wandb.Histogram(
                                    states_dist[:, 1].flatten().detach().cpu().numpy()
                                ),
                                "Stats/Place fold diagonal state sentitvity": wandb.Histogram(
                                    states_dist[:, 2].flatten().detach().cpu().numpy()
                                ),
                            }
                        )
                    bs = q_pick.shape[0]
                    out_dir = osp.join(args.out_dir, run_name)
                    os.makedirs(osp.join(out_dir, "exp"), exist_ok=True)
                    os.makedirs(osp.join(out_dir, "exp/pick"), exist_ok=True)
                    os.makedirs(osp.join(out_dir, "exp/place"), exist_ok=True)
                    d_nb = torch.randint(bs, (1,))
                    val_map = q_pick[d_nb].detach().cpu().numpy()
                    pos = states.reshape(bs, -1, 3)[d_nb].detach().cpu().numpy()
                    np.savez(
                        osp.join(
                            out_dir,
                            f"exp/pick/IST_Qpick_{epoch_id}.npz",
                        ),
                        val_map=val_map,
                        pos=pos,
                    )
                    place_img = q_place[d_nb].detach().squeeze(0)
                    place_img = place_img.permute(1, 2, 0).detach().cpu().numpy()
                    plt.imshow(place_img, cmap="seismic")
                    plt.colorbar()
                    plt.savefig(
                        osp.join(
                            out_dir,
                            f"exp/place/IST_Qplace_{epoch_id}.png",
                        )
                    )
                    plt.clf()
                agent.train()

                run_name = log_writer.name.replace(" ", "").replace("/", "")
                agent.save(prefix=run_name)

            wandb_dict.update(
                {
                    f"TRAIN_ONLINE/epoch": epoch_id,
                    f"TRAIN_ONLINE/pick_loss": loss_vals[0],
                    f"TRAIN_ONLINE/place_loss": loss_vals[1],
                    f"TRAIN_ONLINE/size_loss": loss_vals[2],
                    f"TRAIN_ONLINE/kld_loss": loss_vals[3],
                    f"TRAIN_ONLINE/loss": sum(loss_vals),
                    "Stats/Val Pick flatten": wandb.Histogram(
                        val_pick[:, 0].flatten().detach().cpu().numpy()
                    ),
                    "Stats/Place tgt max no term": place_tgt[rewards < 0.95 * 50]
                    .max()
                    .detach()
                    .cpu()
                    .numpy(),
                    "Stats/Tgt Pick flatten": wandb.Histogram(
                        pick_tgt[:, 0].flatten().detach().cpu().numpy()
                    ),
                    "Stats/Q Pick flatten": wandb.Histogram(
                        q_pick[:, 0].flatten().detach().cpu().numpy()
                    ),
                    "Stats/Q Plus flatten (frozen)": wandb.Histogram(
                        qplus[:, 0].flatten().detach().cpu().numpy()
                    ),
                    "Stats/Q Place flatten": wandb.Histogram(
                        q_place[:, 0].flatten().detach().cpu().numpy()
                    ),
                    "Stats/Tgt Place flatten": wandb.Histogram(
                        place_tgt[:, 0].flatten().detach().cpu().numpy()
                    ),
                    "TRAIN_ONLINE/opt_iter": opt_iter,
                }
            )
            if place_tgt.shape[1] == 3:
                wandb_dict.update(
                    {
                        "Stats/Val Pick fold horizontal": wandb.Histogram(
                            val_pick[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Val Pick fold diagonal": wandb.Histogram(
                            val_pick[:, 5].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Tgt Pick fold horizontal": wandb.Histogram(
                            pick_tgt[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Tgt Pick fold diagonal": wandb.Histogram(
                            pick_tgt[:, 5].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Q Pick fold horizonal": wandb.Histogram(
                            q_pick[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Q Pick fold diagonal": wandb.Histogram(
                            q_pick[:, 5].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Q Plus fold horizontal (frozen)": wandb.Histogram(
                            qplus[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Q Plus fold diagonal (frozen)": wandb.Histogram(
                            qplus[:, 5].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Q Place fold horizontal": wandb.Histogram(
                            q_place[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Q Place fold diagonal": wandb.Histogram(
                            q_place[:, 5].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Tgt Place fold horizontal": wandb.Histogram(
                            place_tgt[:, 1].flatten().detach().cpu().numpy()
                        ),
                        "Stats/Tgt Place fold diagonal": wandb.Histogram(
                            place_tgt[:, 5].flatten().detach().cpu().numpy()
                        ),
                    }
                )
            if place_tgt[rewards >= (0.95 * 50)].numel() > 0:
                wandb_dict.update(
                    {
                        "Stats/Place tgt max term": place_tgt[rewards >= 0.95 * 50]
                        .flatten()
                        .max()
                        .detach()
                        .cpu()
                        .numpy()
                    }
                )
            log_writer.log(wandb_dict)

            opt_iter += 1
            frozen.update_weights(agent, TAU)

        epoch_id += 1


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument("--task", type=str, default="cloth-flatten")
    parser.add_argument("--nfunc", default=3, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=1,
        help="How many test do you need for inferring",
    )
    parser.add_argument("--load_model", default="")
    parser.add_argument("--run_group", default="")
    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--out_dir", default="./", type=str)
    args = parser.parse_args()

    run_jobs(args)


if __name__ == "__main__":
    main()
