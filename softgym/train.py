import wandb
import os.path as osp
import os
import argparse
import numpy as np
import cv2
import torch

from softgym.utils.visualization import save_numpy_as_gif
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from dataset import ExpDataset
from agent import IEAgent

import wandb
import pyflex
from matplotlib import pyplot as plt
import tensorflow as tf

import random
import pickle


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--num_iters', type=int, default=1, help='How many iterations do you need for training')
    parser.add_argument('--learning_rate',  default=1e-4, type=float)
    parser.add_argument('--demo_times', default=10, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--max_load',       default=-1, type=int)
    parser.add_argument('--batch',          default=20, type=int)
    parser.add_argument('--model', default='critic', type=str)
    parser.add_argument('--no_perturb', action='store_true')
    parser.add_argument('--run_group', default='')
    args = parser.parse_args()

    device = torch.device('cuda')
    # device = torch.device('cpu')
    dataset = ExpDataset(os.path.join('data', f"{args.task}-step-{args.step}"), device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=8)
    # dataloader = DataLoader(dataset, batch_size=12, shuffle=True, pin_memory=True, num_workers=1)


    # Set up tensorboard logger.
    train_log_dir = os.path.join('logs', args.agent, args.task, args.exp_name, 'train')


    # Set the beginning of the agent name.
    name = f'{args.task}-{args.exp_name}'

    # Initialize agent and limit random dataset sampling to fixed set.
    # tf.random.set_seed(0)

    # Limit random data sampling to fixed set.
    # num_demos = int(args.num_demos)

    # Given `num_demos`, only sample up to that point, and not w/replacement.
    # train_episodes = np.random.choice(range(num_demos), num_demos, False)
    # dataset.set(train_episodes)
    log_writer = wandb.init(project="GRAVIS_policy",
                            entity='april-lab',
                            job_type=args.task,
                            notes=args.exp_name,
                            group=args.run_group)

    agent = IEAgent(name,
                    args.task,
                    image_size=128,
                    scene_size=2.,
                    device=device,
                    step=args.step)

    if len(args.load_model) > 0:
        agent.load(args.load_model)

    # agent.get_mean_and_std(os.path.join('data', f"{args.task}-{args.suffix}"), args.model)
    lr = args.learning_rate
    # odd = False

    while agent.total_iter < args.num_iters:
        if args.model == 'critic':
            # Train critic.
            agent.train_place(dataloader, args.num_iters // 10, lr, log_writer)
        if args.model == 'aff':
            # Train aff.
            agent.train_pick(dataloader, args.num_iters // 10, lr, log_writer)
        # if odd:
        #     lr *= 0.9
        # odd = not odd


if __name__ == '__main__':
    main()
