from envs.plane_task_us_env import PlaneTaskUsEnv
from envs.phantom import (
    ScatterersPhantom,
    Ball,
    Teddy
)
from envs.imaging import ImagingSystem, Probe
from envs.generator import ConstPhantomGenerator, RandomProbeGenerator
from envs.logger import TrajectoryLogger

from agents.custom_feature_extractor import CustomFeaturesExtractor

import torch as th
import torch.nn as nn
from gym.spaces import Box, Discrete
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.env_checker import check_env

import gym
import numpy as np
import argparse
import os
import shutil
from rl_zoo3.utils import linear_schedule

N_STEPS_PER_EPISODE = 16
N_STEPS_PER_EPOCH = 64
EPOCHS = 1000 # NO_EPISODES = (NSTEPS_PER_EPOCH/NSTEPS_PER_EPISODE)*EPOCHS
N_WORKERS = 4

def env_fn(trajectory_logger):
    probe = Probe(
        pos=np.array([-20 / 1000, 0]), # only X and Y
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=10 / 1000
    )
    teddy = Teddy(
        belly_pos=np.array([0 / 1000, 0, 50 / 1000]),
        scale=12 / 1000,
        head_offset=.9
    )
    phantom = ScatterersPhantom(
        objects=[teddy],
        x_border=(-40 / 1000, 40 / 1000),
        y_border=(-40 / 1000, 40 / 1000),
        z_border=(0, 90 / 1000),
        n_scatterers=int(1e4),
        n_bck_scatterers=int(1e3),
        seed=None, # default value
    )
    imaging = ImagingSystem(
        c=1540,
        fs=100e6,
        image_width=40 / 1000,
        image_height=90 / 1000,
        image_resolution=(40, 90),  # [pixels]
        median_filter_size=5,
        dr_threshold=-200,
        dec=1,
        no_lines=64
    )
    env = PlaneTaskUsEnv(
        dx_reward_coeff=1,
        angle_reward_coeff=1,
        imaging=imaging,
        phantom_generator=ConstPhantomGenerator(phantom),
        probe_generator=RandomProbeGenerator(
            ref_probe=probe,
            object_to_align=teddy,
            seed=None, # default value
            x_pos= np.arange(-15/1000, 19/1000, step=5/1000),
            focal_pos=[50/1000], # same as for Teddy
            angle=[45, 60, 75, 90]
        ),
        max_steps=N_STEPS_PER_EPISODE,
        no_workers=N_WORKERS,
        use_cache=True,
        trajectory_logger=trajectory_logger,
        step_size=5/1000,
        rot_deg=15
    )
    return env

def load_last_model(fpath):

    if os.path.exists(fpath):

        # Checkpoint are saved as rl_model_XXXXXX_steps.zip
        checkpoints = [int(x[9:-10]) for x in os.listdir(fpath) if x.endswith('.zip')]
        itr = max(checkpoints)
    else:
        itr = 0
        
    return itr

def delete_trajectory__files(fpath, itr):

    if os.path.exists(fpath):

        # trajectories are stored in trajectory_logger file as episode_XXX
        # equal is used because first episode is 0.
        itr_episodes = [int(x[8:]) for x in os.listdir(fpath) if int(x[8:]) >= itr]
        episodes = ['episode_' + str(number) for number in itr_episodes]

        for episode in episodes:
            shutil.rmtree(os.path.join(fpath, episode))
                    

def main():

    #np.random.seed(2442)

    parser = argparse.ArgumentParser(description="Train agent in env: %s" %
                                                    PlaneTaskUsEnv.__name__)
    parser.add_argument("--exp_dir", dest="exp_dir",
                        help="Where to put all information about the experiment",
                        required=True)

    args = parser.parse_args()

    # paths to logs
    TRAJECTORY_LOGS = os.path.join(args.exp_dir, 'trajectory_logger')
    TENSORBOARD_LOGS = os.path.join(args.exp_dir, 'tensorboard_logs')
    CHECKPOINT_LOGS = os.path.join(args.exp_dir, 'checkpoints')

    # Create the enviroment
    trajectory_logger = TrajectoryLogger(
        log_dir=TRAJECTORY_LOGS,
        log_action_csv_freq=1,
        log_state_csv_freq=1,
        log_state_render_freq=10
    )

    env_builder = lambda: env_fn(trajectory_logger)
    env = DummyVecEnv([env_builder])    

    # Save a checkpoint every N_STEPS_PER_EPOCH steps
    checkpoint_callback = CheckpointCallback(
        save_freq=N_STEPS_PER_EPOCH,
        save_path=CHECKPOINT_LOGS,
        name_prefix="rl_model",
        save_vecnormalize=True,
    )

        policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        net_arch=dict(pi=[], vf=[]),
        activation_fn=nn.ReLU(),
    )
    
    existing_model = load_last_model(CHECKPOINT_LOGS)

    if not existing_model:
        model = A2C(
            'CnnPolicy', 
            env,
            n_steps=N_STEPS_PER_EPOCH,
            policy_kwargs=policy_kwargs,
            use_rms_prop=True,
            ent_coef=0.001,
            learning_rate=55*1e-5,
            gae_lambda=0.97,
            normalize_advantage=True,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOGS,
            seed=np.random.seed(42)
        )

        model.learn(total_timesteps=N_STEPS_PER_EPOCH*EPOCHS, 
            callback=checkpoint_callback,
            log_interval=N_STEPS_PER_EPOCH,
            tb_log_name='A2C',
        )

    else:
        itr = int(existing_model/N_STEPS_PER_EPISODE)

        for enviroment in env.envs:
            enviroment.current_episode = itr-1

        delete_trajectory__files(TRAJECTORY_LOGS, itr)        

        model = A2C.load(
            os.path.join(CHECKPOINT_LOGS, 'rl_model_%d_steps.zip' %existing_model),
            env,
            #tensorboard_log=TENSORBOARD_LOGS
        )

        # Check if there is more training steps left.
        remaining_steps = (N_STEPS_PER_EPOCH*EPOCHS) - existing_model
        assert remaining_steps>=0, 'Steps argument is less than 0. There is no training.'

        model.learn(total_timesteps=remaining_steps, 
            callback=checkpoint_callback,
            log_interval=N_STEPS_PER_EPOCH,
            reset_num_timesteps=False
        )

    print('training is done.')

if __name__ == "__main__":
    main()
