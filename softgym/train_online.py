"""Online training script"""

import os
import argparse

import cv2
import wandb
import numpy as np
import pyflex

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from matplotlib import colors
from matplotlib import pyplot as plt

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize

from agent import MOptAgent
from data_utils import ReplayBuffer


TAU = 0.01
IMG_SIZE = 64


def test(args, agent, env, agent_steps, test_id, mean, scale, run_name, sample_steps):
    device = torch.device("cuda")
    # from flat configuration
    full_covered_area = env._set_to_flatten()
    pyflex.step()
    test_id = np.random.randint(18000, high=20000)
    env.seed(int(test_id))
    env.reset(config_id=test_id)

    vis_img = []
    qpl_img = []
    in_step = 0
    picks = []
    picked = []
    q_picks = []
    places = []
    q_places = []
    states = []
    nstates = []
    reward = []
    performance = []
    normalized_performance = []

    crump_area = env._get_current_covered_area(pyflex.get_positions())
    crump_percent = crump_area / full_covered_area

    crump_obs, crump_depth = pyflex.render_cloth()

    crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
    crump_depth[crump_depth > 5] = 0
    crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
    crump_obs = np.concatenate([crump_obs, crump_depth], 2)
    crump_obs = cv2.resize(
        crump_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA
    )
    vis_img.append(cv2.cvtColor(crump_obs[:, :, :3], cv2.COLOR_BGR2RGB).copy())

    for i in range(agent_steps):
        in_step += 1

        pos = pyflex.get_positions()
        pos = torch.tensor(pos).reshape(-1, 4)[:, :3].to(device)
        norm_pos = (pos - mean) / scale
        pick, place, q_pick, q_place = agent.act(norm_pos, args.func_id, ret_q=True)
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
        qpl_img.append(place_img)

        if args.pix_pick:
            pick_act = torch.cat(
                [
                    pick.div(40, rounding_mode="floor") / 20 - 1,
                    pick.remainder(40) / 20 - 1,
                ],
                dim=-1,
            )
            pick_act = (
                torch.cat([pick_act, torch.zeros(1, device=pick.device)])
                .detach()
                .cpu()
                .numpy()
            )
        else:
            pick_act = pos[pick][0, :3].detach().cpu().numpy()
        sim_action = np.concatenate(
            (pick_act, place.detach().cpu().numpy(), np.array([0])), axis=-1
        )

        _, _, _, info = env.step(
            sim_action, record_continuous_video=False, img_size=args.img_size
        )
        performance.append(info["performance"])
        normalized_performance.append(info["normalized_performance"])
        picked.append(info["picked_id"])
        reward.append(env.compute_reward())

        curr_area = env._get_current_covered_area(pyflex.get_positions())
        curr_percent = curr_area / full_covered_area
        npos = pyflex.get_positions()
        npos = torch.tensor(npos).reshape(-1, 4)[:, :3].to(device)
        npos = (npos - mean) / scale
        nstates.append(npos.detach().cpu().numpy())
        curr_obs, curr_depth = pyflex.render_cloth()
        curr_obs = curr_obs.reshape((720, 720, 4))[::-1, :, :3]
        curr_depth[curr_depth > 5] = 0
        curr_depth = curr_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        curr_obs = np.concatenate([curr_obs, curr_depth], 2)
        curr_obs = cv2.resize(
            curr_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA
        )
        vis_img.append(cv2.cvtColor(curr_obs[:, :, :3], cv2.COLOR_BGR2RGB).copy())
        if args.func_id == 0 and info["performance"] >= 0.97 * 0.062499997206032304:
            break
        if args.func_id == 1 and info["performance"] >= -0.5:
            break

    pos = pyflex.get_positions()
    pos = torch.tensor(pos).reshape(-1, 4)[:, :3].to(device)
    states.append(pos.detach().cpu().numpy())

    normalize_score = (curr_percent - crump_percent) / (1 - crump_percent)

    visual_path = os.path.join("./visual", run_name)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    transition_path = os.path.join(
        visual_path,
        f"{test_id}-{sample_steps}-{normalize_score:.4f}.npz",
    )
    visual_path1 = os.path.join(
        visual_path,
        f"{test_id}-{sample_steps}-{normalize_score:.4f}.jpg",
    )
    visual_path2 = os.path.join(
        visual_path,
        f"{test_id}-{sample_steps}-{normalize_score:.4f}-qfunc.jpg",
    )
    np.savez(
        transition_path,
        picks=np.stack(picks),
        places=np.stack(places),
        q_picks=np.stack(q_picks),
        q_places=np.stack(q_places),
        states=np.stack(states),
        nstates=np.stack(nstates),
        picked=np.stack(picked),
    )
    vis_img = np.concatenate(vis_img, axis=1)
    qpl_img = np.concatenate(qpl_img, axis=1)
    # cv2.imwrite(visual_path, np.concatenate((qpl_img, vis_img), axis=0))
    cv2.imwrite(visual_path2, cv2.cvtColor(qpl_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(visual_path1, vis_img)
    reward = np.array(reward)
    performance = np.array(performance)
    normalized_performance = np.array(normalized_performance)

    return np.array(
        [
            normalize_score,
            reward.mean(),
            performance.mean(),
            normalized_performance.mean(),
            reward[-1],
            performance[-1],
            normalized_performance[-1],
            i,
        ]
    )


def _make_score_dict(scores):
    names = [
        "eval score",
        "success rate 75%",
        "success rate 85%",
        "success rate 95%",
        "mean reward",
        "mean perf",
        "mean norm perf",
        "last reward",
        "last perf",
        "last norm perf",
        "episode length",
    ]
    values = [
        scores[:, 0].mean(),
        np.count_nonzero(scores[:, 0] > 0.75) / scores.shape[0],
        np.count_nonzero(scores[:, 0] > 0.85) / scores.shape[0],
        np.count_nonzero(scores[:, 0] > 0.95) / scores.shape[0],
    ]
    for i in range(1, 8):
        values.append(scores[:, i].mean())

    return dict(zip(names, values))


def agent_eval(
    args, agent, env, agent_steps, nb_test, mean, scale, run_name, sample_steps
):
    scores = []
    for i in range(nb_test):
        print(f"Eval {i}")
        scores.append(
            test(args, agent, env, agent_steps, i, mean, scale, run_name, sample_steps)
        )
    scores = np.stack(scores, axis=0)
    return _make_score_dict(scores)


def _flatten_reward(pos, env, full_covered_area):
    reward = env.compute_reward()
    # final_area = env._get_current_covered_area(pyflex.get_positions())
    final_percent = reward / full_covered_area
    return 50 * final_percent


def _get_fold_rect_groups():
    group_a = []
    group_b = []
    for i in range(40):
        group_a.append([20 + i * 40 - j for j in range(1, 21)])
        group_b.append([20 + i * 40 + j for j in range(0, 20)])
    group_a = np.concatenate(group_a)
    group_b = np.concatenate(group_b)
    return group_a, group_b


def _get_fold_diag_groups():
    group_a = []
    group_b = []
    for i in range(40):
        group_a.append([40 * i + j for j in range(i + 1, 40)])
        group_b.append([40 * j + i for j in range(i + 1, 40)])
    group_a = np.concatenate(group_a)
    group_b = np.concatenate(group_b)
    return group_a.astype(np.int64), group_b.astype(np.int64)


def _fold_rect_reward(pos, env, full_covered_area):
    pos = pos.numpy()
    flat_pos = np.load("flatpos.npy")
    flat_pos = flat_pos[:, :3]
    fold_group_a, fold_group_b = _get_fold_rect_groups()
    pos_group_a = pos[fold_group_a]
    pos_group_b = pos[fold_group_b]
    pos_group_b_init = flat_pos[fold_group_b]
    r1 = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=-1), axis=-1)
    r2 = np.linalg.norm(np.mean(pos_group_b - pos_group_b_init, axis=-1), axis=-1)
    curr_dist = r1 + 1.2 * r2
    reward = 50 - curr_dist / 6.1251
    return reward


def _fold_diag_reward(pos, env, full_covered_area):
    pos = pos.numpy()
    flat_pos = np.load("flatpos.npy")
    flat_pos = flat_pos[:, :3]
    fold_group_a, fold_group_b = _get_fold_diag_groups()
    pos_group_a = pos[fold_group_a]
    pos_group_b = pos[fold_group_b]
    pos_group_b_init = flat_pos[fold_group_b]
    r1 = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=-1), axis=-1)
    r2 = np.linalg.norm(np.mean(pos_group_b - pos_group_b_init, axis=-1), axis=-1)
    curr_dist = r1 + 1.2 * r2
    reward = 50 - curr_dist / 6.1251
    return reward


def compute_rewards(positions, env, full_covered_area):
    r1 = _flatten_reward(positions, env, full_covered_area)
    r2 = _fold_rect_reward(positions, env, full_covered_area)
    r3 = _fold_diag_reward(positions, env, full_covered_area)

    return np.array([r1, r2, r3])


def run_jobs(args, env_kwargs):
    name = f"{args.task}"
    device = torch.device("cuda")
    replay_buffer = ReplayBuffer(args.replay_buffer_size, name, "data/", restrain=True)
    if len(args.replay_buffer_fill) > 1:
        replay_buffer.pre_fill(args.replay_buffer_fill)
    replay_buffer.make_idcs()
    mean = torch.tensor([0.0159, -0.0017, 0.0003], device="cuda:0")
    std = torch.tensor([0.0626, 0.0094, 0.0609], device="cuda:0")
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

    config = {
        "lr": lr,
        "tau": TAU,
        "gamma": gamma,
        "img_size": IMG_SIZE,
        "batch_size": args.batch_size,
        "pick_layers": pick_layers,
        "place_layers": place_layers,
        "batch_norm": batch_norm,
        "layer_norm": layer_norm,
        "encoder": encoder,
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
    n_params = agent.show_param_count()
    config["n_params"] = n_params
    log_writer = wandb.init(
        project="GRAVIS_policy",
        entity="april-lab",
        job_type=args.task,
        group=args.run_group,
        config=config,
    )
    if len(args.load_model) > 0:
        agent.load(args.load_model)
        frozen.load(args.load_model)

    env_kwargs["num_variations"] = args.num_variations
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))

    sample_steps = 0
    gradient_steps = 0

    for train_iter in range(args.total_iter):
        full_covered_area = None

        for online_id in tqdm(range(args.num_online), desc=f"Simulation {train_iter}"):
            train_reward = []
            config_id = np.random.randint(18000)
            env.reset(config_id=config_id)
            full_coverset_to_flatten()
            env.reset(config_id=config_id)

            # Agent
            agent_states = []
            for _ in range(10):
                pos = pyflex.get_positions()
                pos = torch.tensor(pos).reshape(-1, 4)[:, :3]
                with torch.no_grad():
                    npos = pos.clone().to(device)
                    pick, place = agent.act(npos, args.func_id)
                    agent_act = True

                if torch.rand(1).item() < args.eps_pick:
                    pick = torch.randint(pos.shape[0], (1,))
                    agent_act = False
                if torch.rand(1).item() < args.eps_place:
                    place = -1 + 2 * torch.rand(2)
                    agent_act = False

                pick, place = pick.detach().cpu(), place.detach().cpu()
                if args.pix_pick:
                    pick_act = torch.cat(
                        [
                            pick.div(40, rounding_mode="floor") / 20 - 1,
                            pick.remainder(40) / 20 - 1,
                        ],
                        dim=-1,
                    )
                    pick_act = (
                        torch.cat([pick_act, torch.zeros(1, device=pick.device)])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    pick_act = pos[pick][0, :3].detach().cpu().numpy()

                agent_action = torch.cat((pick, place), dim=-1)
                sim_action = np.concatenate(
                    (
                        pick_act,
                        agent_action[-2:].detach().numpy(),
                        np.array([-1 if args.pix_pick else 0]),
                    ),
                    axis=-1,
                )

                _, _, _, info = env.step(
                    sim_action,
                    record_continuous_video=False,
                    img_size=args.img_size,
                )
                sample_steps += 1

                if info["picked_id"] is None:
                    continue

                if not args.pix_pick:
                    agent_action[0] = info["picked_id"]

                next_pos = pyflex.get_positions()
                next_pos = torch.tensor(next_pos).reshape(-1, 4)[:, :3]

                rewards = compute_rewards(next_pos, env, full_covered_area)
                if env.action_tool.not_on_cloth:
                    rewards[0] = -10

                if agent_act:
                    train_reward.append(rewards[args.func_id] / 50)

                state_agent = env.get_state()
                agent_states.append(state_agent)
                replay_buffer.append(
                    (
                        pos.clone(),
                        agent_action.clone(),
                        torch.tensor(rewards[args.func_id], dtype=torch.float),
                        next_pos.clone(),
                        torch.tensor(True),
                    )
                )

        train_reward = np.array(train_reward)

        dataloader = DataLoader(
            replay_buffer,
            batch_size=min(len(replay_buffer), args.batch_size),
            shuffle=True,
            num_workers=8,
        )
        optimizer = AdamW(agent.get_params_list(lr))
        for i, (states, actions, rewards, next_states, _) in tqdm(
            enumerate(dataloader), desc="Training"
        ):
            log_dict = {
                "Stats/train_reward": wandb.Histogram(train_reward),
                "Stats/train_reward avg": train_reward.mean(),
                "Stats/sample steps": sample_steps,
            }
            if opt_iter % int(5 * args.nb_batches) == 0:
                run_name = log_writer.name.replace(" ", "").replace("/", "")
                with torch.no_grad():
                    score_dict = agent_eval(
                        args,
                        agent,
                        env,
                        10,
                        5,
                        mean,
                        scale,
                        run_name,
                        sample_steps,
                    )
                    for key, value in score_dict.items():
                        log_dict.update(
                            {
                                f"Stats/{key}": value,
                            }
                        )
                    log_dict.update(
                        {
                            "TRAIN_ONLINE/opt_iter": opt_iter,
                        }
                    )

            if i > args.nb_batches:
                break

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
                pick_tgt = frozen.max_place_batch(states, actions)[:, args.func_id]
                qplus = gamma * frozen.max_pick_batch(next_states)
                qplus = qplus[:, args.func_id]
                # What to do with -1 reward ?

            place_tgt = (
                (1 - term_mask) * rewards
                + (1 - term_mask) * qplus * gamma
                + term_mask * (rewards / (1 - gamma))
            )

            place_loss = F.mse_loss(val_place[:, args.func_id], place_tgt)
            pick_loss = F.mse_loss(val_pick[:, args.func_id], pick_tgt)
            loss = 0.5 * pick_loss + 0.5 * place_loss

            loss.backward()
            gradient_steps += 1
            optimizer.step()
            optimizer.zero_grad()

            if opt_iter % int(5 * args.nb_batches) == 0:
                with torch.no_grad():
                    # print("Validation and saving")
                    bs = q_pick.shape[0]
                    out_dir = os.path.join("exp", run_name)
                    os.makedirs(out_dir, exist_ok=True)
                    os.makedirs(os.path.join(out_dir, "pick"), exist_ok=True)
                    os.makedirs(os.path.join(out_dir, "place"), exist_ok=True)
                    d_nb = torch.randint(bs, (1,))
                    val_map = q_pick[d_nb, args.func_id].detach().cpu().numpy()
                    pos = states.reshape(bs, -1, 3)[d_nb].detach().cpu().numpy()
                    np.savez(
                        os.path.join(out_dir, f"pick/IST_Qpick_{train_iter}.npz"),
                        val_map=val_map,
                        pos=pos,
                    )
                    # i_nb = torch.randint(q_place.shape[0], (1,))
                    place_img = q_place[d_nb, args.func_id].detach()
                    # divnorm = colors.TwoSlopeNorm(vmin=-1., vcenter=0, vmax=50)
                    place_img = place_img.permute(1, 2, 0).detach().cpu().numpy()
                    plt.imshow(place_img, cmap="seismic", vmin=-1, vmax=72)
                    plt.colorbar()
                    plt.savefig(
                        os.path.join(
                            out_dir,
                            f"place/IST_Qplace_{train_iter}.png",
                        )
                    )
                    plt.clf()

                run_name = log_writer.name.replace(" ", "").replace("/", "")
                agent.save(prefix=run_name)

            log_dict.update(
                {
                    f"TRAIN_ONLINE/pick_loss": pick_loss,
                    f"TRAIN_ONLINE/place_loss": place_loss,
                    f"TRAIN_ONLINE/loss": loss,
                    "Stats/Q Place": wandb.Histogram(
                        q_place[:, args.func_id].flatten().detach().cpu().numpy()
                    ),
                    "Stats/Tgt Place": wandb.Histogram(
                        place_tgt.flatten().detach().cpu().numpy()
                    ),
                    "Stats/Q Pick": wandb.Histogram(
                        q_pick[:, args.func_id].flatten().detach().cpu().numpy()
                    ),
                    "Stats/Tgt Pick": wandb.Histogram(
                        pick_tgt.flatten().detach().cpu().numpy()
                    ),
                    "TRAIN_ONLINE/opt_iter": opt_iter,
                    "TRAIN_ONLINE/train_iter": train_iter,
                    "Stats/gradient steps": gradient_steps,
                }
            )
            wandb.log(log_dict)

            opt_iter += 1
            frozen.update_weights(agent, TAU)

        online_id += 1


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
        default=20000,
        help="Number of environment variations to be generated",
    )
    parser.add_argument("--task", type=str, default="cloth-flatten")
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument(
        "--num_online",
        type=int,
        default=1,
        help="How many test do you need for inferring",
    )
    parser.add_argument("--nfunc", default=3, type=int)
    parser.add_argument("--func_id", default=0, type=int)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--run_group", default="")
    parser.add_argument("--nb_batches", default=15, type=int)
    parser.add_argument("--total_iter", default=100000, type=int)
    parser.add_argument("--eps_pick", default=2e-2, type=float)
    parser.add_argument("--eps_place", default=2e-2, type=float)
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--replay_buffer_size", default=512, type=int)
    parser.add_argument("--replay_buffer_fill", default="", type=str)
    parser.add_argument("--pix_pick", default=False, action="store_true")
    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs["use_cached_states"] = True
    env_kwargs["save_cached_states"] = False
    env_kwargs["num_variations"] = args.num_variations
    env_kwargs["render"] = True
    env_kwargs["headless"] = args.headless
    env_kwargs["action_mode"] = "pickandplacepos"

    if not env_kwargs["use_cached_states"]:
        print(
            "Waiting to generate environment variations. May take 1 minute for each variation..."
        )

    run_jobs(args, env_kwargs)


if __name__ == "__main__":
    main()
