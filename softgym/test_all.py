import os
import wandb
import torch
import argparse
import numpy as np
import cv2

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

# import agents
from agent import IEAgent
# from models import Critic_MLP

import pyflex
from matplotlib import pyplot as plt
# import tensorflow as tf

import multiprocessing
import random
import pickle


def run_jobs(args, env, agent):
    log_writer = wandb.init(project="GRAVIS_policy",
                            entity='april-lab',
                            job_type=args.task,
                            notes=args.exp_name,
                            group=args.run_group)
    device = torch.device('cuda')
    # from flat configuration
    full_covered_area = env._set_to_flatten()
    pyflex.step()

    step_i = 0

    while step_i < args.step:
        # print("step_i: ", step_i)
        prev_obs, prev_depth = pyflex.render_cloth()
        prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]
        prev_depth = prev_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        mask = np.where(prev_depth[:, :, 0] < 0.348, 255, 0)

        # crumple the cloth by grabbing corner
        # if step_i == 0:
        mask = prev_obs[:, :, 0]
        indexs = np.transpose(np.where(mask != 0))
        corner_id = random.randint(0, 3)
        top, left = indexs.min(axis=0)
        bottom, right = indexs.max(axis=0)

        corners = [[top, left],
                   [top, right],
                   [bottom, right],
                   [bottom, left]]
        u1 = (corners[corner_id][1]) * 2.0 / 720 - 1
        v1 = (corners[corner_id][0]) * 2.0 / 720 - 1

        u2 = random.uniform(-1., 1.)
        v2 = random.uniform(-1., 1.)

        action = np.array([u1, v1, 0, u2, v2, -1])

        _, _, _, info = env.step(action, record_continuous_video=False, img_size=args.img_size)

        if env.action_tool.not_on_cloth:
            # print(f'{step_i} not on cloth')
            # from flat configuration
            full_covered_area = env._set_to_flatten()
            pyflex.step()

            step_i = 0
            continue
        step_i += 1
        now_area = env._get_current_covered_area(pyflex.get_positions())
        now_percent = now_area / full_covered_area
        if now_percent >= (0.65 - args.step * 0.05):
            step_i = 0
            continue

    env.start_record()

    crump_area = env._get_current_covered_area(pyflex.get_positions())
    crump_percent = crump_area / full_covered_area
    print("crump percent: ", crump_percent)

    max_percent = -float("inf")

    vis_img = []
    in_step = 0
    for i in range(args.test_step):
        in_step += 1
        crump_obs, crump_depth = pyflex.render_cloth()

        crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
        crump_depth[crump_depth > 5] = 0
        crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        crump_obs = np.concatenate([crump_obs, crump_depth], 2)
        crump_obs = cv2.resize(crump_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        vis_img.append(cv2.cvtColor(crump_obs[:, :, :3], cv2.COLOR_BGR2RGB).copy())
        pos = pyflex.get_positions()
        pos = torch.tensor(pos).reshape(-1, 4)[:, :3].to(device)
        pick, place = agent.act(pos.clone())

        pick_act = pos[pick][0, :3].detach().cpu().numpy()
        sim_action = np.concatenate((pick_act, place.detach().cpu().numpy(),
                                     np.array([0])), axis=-1)

        _, _, _, info = env.step(sim_action, record_continuous_video=True, img_size=args.img_size)

        curr_area = env._get_current_covered_area(pyflex.get_positions())
        curr_percent = curr_area / full_covered_area
        print("curr percent: ", i, curr_percent)
        if (curr_percent >= 0.85):
            break

    normalize_score = (curr_percent - crump_percent) / (1 - crump_percent)
    if curr_percent >= 0.75:
        result = 'success'
    else:
        result = 'fail'

    log_writer.log({"TEST/Norm score": normalize_score})

    print(normalize_score)
    visual_path = os.path.join('./visual', args.exp_name.split('-')[0])
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    visual_path = os.path.join(visual_path, f'{args.exp_name}-{args.test_id}-{normalize_score}-{result}.jpg')
    print(f'save to {visual_path}')
    vis_img = np.concatenate(vis_img, axis=1)
    cv2.imwrite(visual_path, vis_img)
    if args.save_video_dir is not None:
        path_name = os.path.join(args.save_video_dir, agent.name + args.exp_name)
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        save_name = os.path.join(path_name, f'{args.test_id}-{in_step}-{normalize_score}.gif')
        save_numpy_as_gif(np.array(env.video_frames), save_name)
        print('Video generated and save to {}'.format(save_name))

    env.end_record()
    return 1


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--image_size', type=int, default=160, help='Size of the input')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--step', default=4, type=int)
    parser.add_argument('--test_step', default=10, type=int)
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--test_id', type=int, default=1, help='which test')
    parser.add_argument('--save_video_dir', type=str, default=None, help='Path to the saved video')
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--run_group', default='')
    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['action_mode'] = 'pickandplacepos'

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    # Set the beginning of the agent name.
    name = f'{args.task}-Aff_Critic-{args.num_demos}'

    # Initialize agent and limit random dataset sampling to fixed set.
    # tf.random.set_seed(0)

    # agent = agents.names[args.agent](name,
    #                                  args.task,
    #                                  load_critic_dir=args.load_critic_dir,
    #                                  load_aff_dir=args.load_aff_dir,
    #                                  )
    device = torch.device('cuda')
    agent = IEAgent(name,
                    args.task,
                    image_size=128,
                    scene_size=1.,
                    device=device,
                    step=args.step)
    if len(args.load_model) > 0:
        agent.load(args.load_model)

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    run_jobs(args, env, agent)

if __name__ == '__main__':
    main()
