"""Trained agent evaluation script"""

import os
import argparse

import cv2
import numpy as np
import pyflex

import torch
import skimage.measure

from tqdm import tqdm
from matplotlib import colors
from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize

from agent import EDAgentLin


IMG_SIZE = 64
UNATTAINABLE_COST = -10
BATCH_SIZE = 1024


def test(args, agent, env, agent_steps, test_id, mean, scale, run_name, gamma):
    device = torch.device("cuda")
    # from flat configuration
    env.seed(int(test_id))
    env.reset(config_id=test_id)

    in_step = 0
    picks = []
    picked = []
    q_picks = []
    places = []
    q_places = []
    states = []
    nstates = []
    error_picks = []
    # error_places = []
    entropies = []
    reward = []
    performance = []
    normalized_performance = []

    crump_obs, crump_depth = pyflex.render_cloth()

    crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
    crump_depth[crump_depth > 5] = 0
    crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
    crump_obs = np.concatenate([crump_obs, crump_depth], 2)
    crump_obs = cv2.resize(
        crump_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA
    )
    visual_path = os.path.join("./evaluate", args.exp_name.split("-")[0], run_name)

    for i in range(agent_steps):
        in_step += 1

        pos = pyflex.get_positions()
        pos = torch.tensor(pos).reshape(-1, 4)[:, :3].to(device)
        norm_pos = (pos - mean) / scale
        pick, place, q_pick, q_place = agent.act(norm_pos, args.func_id, ret_q=True)
        obs, _ = pyflex.render_cloth()
        obs = obs.reshape((720, 720, 4))
        obs_path = os.path.join(visual_path, f"{args.exp_name}-{test_id}-{i}.jpg")
        obs = cv2.cvtColor(obs[::-1, :, :3], cv2.COLOR_BGR2RGB).copy()
        cv2.imwrite(obs_path, obs)
        val_pick = q_pick[pick[0]]
        place_pos = ((IMG_SIZE - 1) * (1 + place) / 2).int()
        val_place = q_place[place_pos[1], place_pos[0]]
        img = norm_pos.detach().cpu().numpy().reshape(40, 40, 3)
        entropy_r = skimage.measure.shannon_entropy(img[..., 0])
        entropy_g = skimage.measure.shannon_entropy(img[..., 1])
        entropy_b = skimage.measure.shannon_entropy(img[..., 2])
        entropies.append((entropy_r + entropy_g + entropy_b) / 3)
        picks.append(pick.detach().cpu().numpy())
        places.append(place.detach().cpu().numpy())
        q_picks.append(q_pick.detach().cpu().numpy())
        q_places.append(q_place.detach().cpu().numpy())
        states.append(pos.detach().cpu().numpy())

        place_img = q_place.detach().cpu().numpy()
        norm = colors.Normalize(vmin=-1, vmax=72)
        place_img = norm(place_img)
        place_img = plt.cm.seismic(place_img)[..., :3]
        place_img = (place_img * 255).astype(np.uint8)

        pick_act = pos[pick][0, :3].detach().cpu().numpy()
        sim_action = np.concatenate(
            (pick_act, place.detach().cpu().numpy(), np.array([0])), axis=-1
        )

        _, _, _, info = env.step(
            sim_action, record_continuous_video=False, img_size=args.img_size
        )
        performance.append(info["performance"])
        normalized_performance.append(info["normalized_performance"])
        if args.func_id == 0:
            picked.append(info["picked_id"])
        else:
            picked.append(0)

        reward.append(env.compute_reward())

        error_pick = (val_pick - q_place.max()) ** 2
        error_picks.append(error_pick.detach().cpu().numpy())

        npos = pyflex.get_positions()
        npos = torch.tensor(npos).reshape(-1, 4)[:, :3].to(device)
        npos = (npos - mean) / scale
        _, _, _, _ = agent.act(npos, args.func_id, ret_q=True)
        nstates.append(npos.detach().cpu().numpy())
        if args.func_id == 0 and info["performance"] >= 0.95 * 0.062499997206032304:
            break
        if args.func_id == 1 and info["performance"] >= -0.5:
            break

    pos = pyflex.get_positions()
    pos = torch.tensor(pos).reshape(-1, 4)[:, :3].to(device)
    states.append(pos.detach().cpu().numpy())

    visual_path = os.path.join("./evaluate", args.exp_name.split("-")[0], run_name)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    transition_path = os.path.join(visual_path, f"{args.exp_name}-{test_id}.npz")
    # print(visual_path)
    np.savez(
        transition_path,
        picks=np.stack(picks),
        places=np.stack(places),
        q_picks=np.stack(q_picks),
        q_places=np.stack(q_places),
        states=np.stack(states),
        nstates=np.stack(nstates),
        picked=np.stack(picked),
        entropies=np.stack(entropies),
        error_picks=np.stack(error_picks),
        # error_places=np.stack(error_places),
    )
    reward = np.array(reward)
    performance = np.array(performance)
    normalized_performance = np.array(normalized_performance)

    return np.array(
        [
            reward.mean(),
            performance.mean(),
            normalized_performance.mean(),
            reward[-1],
            performance[-1],
            normalized_performance[-1],
        ]
    )


def agent_eval(args, agent, env, agent_steps, nb_test, mean, scale, run_name, gamma):
    scores = np.zeros((nb_test, 1, 6))
    agent.eval()
    for i in tqdm(range(nb_test)):
        scores[i, :] = test(
            args, agent, env, agent_steps, i, mean, scale, run_name, gamma
        )
    return scores


def eval_metrics(x, agent_steps, prefix):
    """Returns metrix and save score file. X should be nruns x ntasks"""
    score_dict = {"run": x}
    metric_names = [
        "Median",
        "IQM",
        "Mean",
        "Optimality Gap",
        "success rate 75%",
        "success rate 85%",
        "success rate 95%",
    ]
    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
            np.count_nonzero(x > 0.75) / x.size,
            np.count_nonzero(x > 0.85) / x.size,
            np.count_nonzero(x > 0.95) / x.size,
        ]
    )
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=5000
    )
    data_dict = {
        f"{prefix}_score": score_dict,
        f"{prefix}_agent_step": agent_steps,
        f"{prefix}_aggregate_scores": aggregate_scores,
        f"{prefix}_aggregate_score_cis": aggregate_score_cis,
    }
    for k in score_dict:
        print(k)
        for m, s in zip(metric_names, aggregate_scores[k]):
            print(m, s)
    return data_dict


def evaluate_model(args, env_kwargs):
    name = f"{args.task}-{args.exp_name}"
    device = torch.device("cuda")
    mean = torch.tensor([0.0159, -0.0017, 0.0003], device="cuda:0")
    std = torch.tensor([0.0626, 0.0094, 0.0609], device="cuda:0")
    mn = torch.tensor([-0.1755, -0.0373, -0.1776], device="cuda:0")
    mx = torch.tensor([0.1765, 0.0579, 0.1769], device="cuda:0")
    scale = std
    print(f"Dataset mean, std: {mean}, {std}")
    gamma = 0.9
    pick_layers = [64, 16, 8, 32, 32, 16, 8, 8, 8]
    place_layers = [16, 8, 8, 8]
    residual = False
    encoder = "conv"
    agent = EDAgentLin(
        name,
        args.task,
        image_size=IMG_SIZE,
        device=device,
        pick_layers=pick_layers,
        place_layers=place_layers,
        step=args.step,
        batch_norm=False,
        layer_norm=True,
        nfunc=args.nfunc,
        pool="conv",
        residual=residual,
        encoder=encoder,
    )
    if len(args.load_model) > 0:
        agent.load(args.load_model)

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    run_name = os.path.basename(args.load_model.strip("/")).replace(":", "")
    os.makedirs(
        os.path.join("./evaluate", args.exp_name.split("-")[0], run_name),
        exist_ok=True,
    )

    score = agent_eval(
        args, agent, env, 10, args.num_variations, mean, scale, run_name, gamma
    )
    eval_data = {}
    print("Mean reward")
    eval_data.update(eval_metrics(score[..., 0], 10, "mean_reward"))
    print("Mean performance")
    eval_data.update(eval_metrics(score[..., 1], 10, "mean_perf"))
    print("Mean performance normalized")
    eval_data.update(eval_metrics(score[..., 2], 10, "mean_nperf"))
    print("Final reward")
    eval_data.update(eval_metrics(score[..., 3], 10, "fin_reward"))
    print("Final performance")
    eval_data.update(eval_metrics(score[..., 4], 10, "fin_perf"))
    print("Final performance normalized")
    eval_data.update(eval_metrics(score[..., 5], 10, "fin_nperf"))
    np.savez(
        os.path.join(
            "./evaluate", args.exp_name.split("-")[0], run_name, "eval_data.npz"
        ),
        **eval_data,
    )


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument("--env_name", type=str, default="ClothDrop")
    parser.add_argument(
        "--img_size", type=int, default=720, help="Size of the recorded videos"
    )
    parser.add_argument(
        "--image_size", type=int, default=160, help="Size of input observation"
    )
    parser.add_argument(
        "--headless",
        type=int,
        default=0,
        help="Whether to run the environment with headless rendering",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=1,
        help="Number of environment variations to be generated",
    )
    parser.add_argument("--nfunc", default=1, type=int)
    parser.add_argument("--func_id", default=0, type=int)
    parser.add_argument("--task", type=str, default="cloth-flatten")
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--exp_name", type=str, default="0809-01")
    parser.add_argument("--load_model", default="")
    parser.add_argument("--def_aff", default=False, action="store_true")
    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs["use_cached_states"] = False
    env_kwargs["save_cached_states"] = False
    env_kwargs["num_variations"] = args.num_variations
    env_kwargs["render"] = True
    env_kwargs["headless"] = args.headless
    env_kwargs["action_mode"] = "pickandplacepos"
    env_kwargs["num_picker"] = 1

    if not env_kwargs["use_cached_states"]:
        print(
            "Waiting to generate environment variations. May take 1 minute for each variation..."
        )

    evaluate_model(args, env_kwargs)


if __name__ == "__main__":
    main()
