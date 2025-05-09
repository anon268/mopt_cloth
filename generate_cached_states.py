### Courtensy of Xingyu-Lin/softagent
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import numpy as np
import os.path as osp

SAVE_PATH = "./data/videos"


def generate_env_state(env_name):
    env_kwargs = env_arg_dict[env_name]
    env_kwargs["headless"] = True
    env_kwargs["use_cached_states"] = False
    env_kwargs["num_variations"] = 20000
    env_kwargs["save_cached_states"] = True
    # Env wrappter
    env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))
    return env


if __name__ == "__main__":
    env_name = "ClothFlatten"
    generate_env_state(env_name)
