import os
import time
import wandb
import random
import pickle
import argparse
import multiprocessing
import numpy as np
import cv2

from agent import IEAgent, ConvAgent
from data_utils import make_split_ds, center_align

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from matplotlib import colors
from matplotlib import pyplot as plt


IMG_SIZE = 128
UNATTAINABLE_COST = -1

def validate(agent, frozen, val_ds):
    dataloader = DataLoader(val_ds, batch_size=8192, shuffle=True, num_workers=8)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), desc='Validation'):
            pos = data[0].to(device)
            pos, r, t = center_align(pos)
            # pos = (pos - mean)/std
            next_pos = data[3].to(device)
            next_pos, next_r, next_t = center_align(next_pos)
            # next_pos = (next_pos - mean)/std
            # print(r.dtype, t.dtype, data[1].dtype)
            r, t = r.float(), t.float()
            action = data[1].float()
            place_act = torch.bmm(r[:, ::2, ::2], (action[:, 1:].to(device) - t[:, ::2]).unsqueeze(1).permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
            place_act = place_act / 2
            # place_act = torch.clip(place_act, -1, 1)
            agent_action = torch.cat((data[1][:, 0][:, None].to(device), place_act), dim=-1)
            data = (pos, agent_action, data[2], next_pos, data[4])
            val_pick, val_place, q_pick, q_place = agent.run_batch(*data)

            has_next = data[4].to(val_place.device)
            rewards = data[2].float().to(val_place.device)
            place_tgt = rewards

            pick_tgt = frozen.max_place_batch(data[0], data[1], has_next.shape[0])
            pick_tgt[~has_next * (rewards == -1)] = UNATTAINABLE_COST
            qplus = gamma * frozen.max_pick_batch(data[3], has_next.shape[0])
            place_tgt[has_next] += qplus[has_next]
            # What to do with -1 reward ?

            place_loss = F.mse_loss(val_place, place_tgt)
            pick_loss = F.mse_loss(val_pick, pick_tgt)

            loss = 0.5 * pick_loss + 0.5 * place_loss
            val_dict = {f'TRAIN_ONLINE/step {args.step} validation loss': loss,
                        f'TRAIN_ONLINE/step {args.step} validation pick loss': pick_loss,
                        f'TRAIN_ONLINE/step {args.step} validation place loss': place_loss,
                        'Stats/Validation Q Place': wandb.Histogram(q_place.flatten().detach().cpu().numpy()),
                        'Stats/Validation Q Pick': wandb.Histogram(q_pick.flatten().detach().cpu().numpy()),
                        'Stats/Validation Orig place norm': wandb.Histogram(torch.norm(action[:, 1:], dim=-1).detach().cpu().numpy()),
                        'Stats/Validation Trans place norm': wandb.Histogram(torch.norm(place_act, dim=-1).detach().cpu().numpy()),
                        'Stats/Validation Rot det': wandb.Histogram(torch.det(r).detach().cpu().numpy())}

            return val_dict


def run_jobs(process_id, args):
    # Set the beginning of the agent name.
    name = f'{args.task}-{args.exp_name}'
    device = torch.device('cuda')
    mean = torch.tensor([ 0.0159, -0.0017,  0.0003], device='cuda:0')
    std = torch.tensor([0.0626, 0.0094, 0.0609], device='cuda:0')
    mn = torch.tensor([-0.1755, -0.0373, -0.1776], device='cuda:0')
    mx = torch.tensor([0.1765, 0.0579, 0.1769], device='cuda:0')
    print(f"Dataset mean, std: {mean}, {std}")
    gamma = 0.3
    opt_iter = 0

    log_writer = wandb.init(project="GRAVIS_policy",
                            entity='april-lab',
                            job_type=args.task,
                            notes=args.exp_name,
                            group=args.run_group)
    agent = ConvAgent(name,
                      args.task,
                      image_size=IMG_SIZE,
                      scene_size=.8,
                      device=device,
                      step=args.step)
    frozen = ConvAgent(name,
                       args.task,
                       image_size=IMG_SIZE,
                       scene_size=.8,
                       device=device,
                       step=args.step)
    if len(args.load_model) > 0:
        agent.load(args.load_model)
        frozen.load(args.load_model)

    train_ds, val_ds = make_split_ds(0.2, args.data_file)
    epoch_id = 0
    full_covered_area = None
    lr = args.learning_rate

    while (epoch_id < args.num_epoch):
        # train critic with online data
        dataloader = DataLoader(train_ds, batch_size=8192, shuffle=True,
                                num_workers=8)
        optimizer = AdamW(agent.get_params_list(lr))
        print('Start place training')
        for i, data in tqdm(enumerate(dataloader), desc='Training'):
            wandb_dict = {}
            pos = data[0].to(device)
            pos, r, t = center_align(pos)
            # pos = (pos - mean)/std
            next_pos = data[3].to(device)
            next_pos, next_r, next_t = center_align(next_pos)
            # next_pos = (next_pos - mean)/std
            r, t = r.float(), t.float()
            action = data[1].float()
            place_act = torch.bmm(r[:, ::2, ::2], (action[:, 1:].to(device) - t[:, ::2]).unsqueeze(1).permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
            place_act = place_act / 2
            agent_action = torch.cat((data[1][:, 0][:, None].to(device), place_act), dim=-1)
            data = (pos, agent_action, data[2], next_pos, data[4])
            val_pick, val_place, q_pick, q_place = agent.run_batch(*data)

            has_next = data[4].to(val_place.device)
            rewards = data[2].float().to(val_place.device)
            place_tgt = rewards
            with torch.no_grad():
                pick_tgt = frozen.max_place_batch(data[0], data[1], has_next.shape[0])
                pick_tgt[~has_next * (rewards == -1)] = UNATTAINABLE_COST
                qplus = gamma * frozen.max_pick_batch(data[3], has_next.shape[0])
                place_tgt[has_next] += qplus[has_next]
                # What to do with -1 reward ?

            place_loss = F.mse_loss(val_place, place_tgt)
            pick_loss = F.mse_loss(val_pick, pick_tgt)

            loss = 0.5 * pick_loss + 0.5 * place_loss  # +  size_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if opt_iter % 200 == 0:
                val_dict = validate(agent, frozen, val_ds)
                wandb_dict.update(val_dict)
                bs = q_pick.shape[0]
                os.makedirs(osp.join(args.out_dir, 'exp'), exist_ok=True)
                os.makedirs(osp.join(args.out_dir, 'exp/pick'), exist_ok=True)
                os.makedirs(osp.join(args.out_dir, 'exp/place'), exist_ok=True)
                d_nb = torch.randint(bs, (1,))
                val_map = q_pick[d_nb].detach().cpu().numpy()
                pos = data[0].reshape(bs, -1, 3)[d_nb].detach().cpu().numpy()
                np.savez(f'exp/pick/IST_{args.step}_Qpick_{epoch_id}.npz', val_map=val_map, pos=pos)
                place_img = q_place[d_nb].detach().squeeze(0)
                place_img = place_img.permute(1, 2, 0).detach().cpu().numpy()
                plt.imshow(place_img, cmap='seismic', vmin=-1, vmax=72)
                plt.colorbar()
                plt.savefig(f'exp/place/IST_{args.step}_Qplace_{epoch_id}.png')
                plt.clf()

                agent.save()

            wandb_dict.update({f'TRAIN_ONLINE/step {args.step} epoch': epoch_id,
                               f'TRAIN_ONLINE/step {args.step} pick_loss': pick_loss,
                               f'TRAIN_ONLINE/step {args.step} place_loss': place_loss,
                               f'TRAIN_ONLINE/step {args.step} loss': loss,
                               f'Stats/Q Place': wandb.Histogram(q_place.flatten().detach().cpu().numpy()),
                               f'Stats/Q Pick': wandb.Histogram(q_pick.flatten().detach().cpu().numpy()),
                               f'Stats/Orig place norm': wandb.Histogram(torch.norm(action[:, 1:], dim=-1).detach().cpu().numpy()),
                               f'Stats/Trans place norm': wandb.Histogram(torch.norm(place_act, dim=-1).detach().cpu().numpy()),
                               f'Stats/Rot det': wandb.Histogram(torch.det(r).detach().cpu().numpy()),
                               f'TRAIN_ONLINE/opt_iter': opt_iter})
            log_writer.log(wandb_dict)

            opt_iter += 1

        if epoch_id % 10 == 0:
            frozen.clone_model(agent)
        epoch_id += 1


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--num_epoch', type=int, default=1, help='How many test do you need for inferring')
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--run_group', default='')
    parser.add_argument('--data_file', default='', type=str)
    parser.add_argument('--out_dir', default='./', type=str)
    args = parser.parse_args()

    run_jobs(0, args)


if __name__ == '__main__':
    main()
