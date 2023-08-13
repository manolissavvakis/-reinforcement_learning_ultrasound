#from spinup import vpg_tf1 as vpg
#from spinup.utils.logx import restore_tf_graph

#import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete
from envs.combined_task_us_env import CombinedTaskUsEnv
from envs.phantom import (
    ScatterersPhantom,
    Ball,
    Teddy
)
from envs.imaging import ImagingSystem, Probe
from envs.generator import ConstPhantomGenerator, RandomProbeGenerator
from envs.logger import TrajectoryLogger
import matplotlib
import argparse


N_STEPS_PER_EPISODE = 16
N_STEPS_PER_EPOCH = 64
# 251
EPOCHS = 500 # NO_EPISODES = (NSTEPS_PER_EPOCH/NSTEPS_PER_EPISODE)*EPOCHS
N_WORKERS = 4

def env_fn(trajectory_logger):
    probe = Probe(
        pos=np.array([-20 / 1000, -5 / 1000, 0]),
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
        z_border=(0, 100 / 1000),
        n_scatterers=int(1e4),
        n_bck_scatterers=int(1e3),
        seed=42,
    )
    imaging = ImagingSystem(
        c=1540,
        fs=100e6,
        image_width=40 / 1000,
        image_height=100 / 1000,
        image_resolution=(40, 100),  # [pixels]
        median_filter_size=5,
        dr_threshold=-200,
        dec=1,
        no_lines=64
    )
    env = CombinedTaskUsEnv(
        imaging=imaging,
        phantom_generator=ConstPhantomGenerator(phantom),
        probe_generator=RandomProbeGenerator(
            ref_probe=probe,
            object_to_align=teddy,
            seed=42,
            x_pos= np.arange(-15/1000, 19/1000, step=5/1000),
            y_pos= np.arange(-15/1000, 19/1000, step=5/1000),
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

def main():
    matplotlib.use('agg')
    np.random.seed(2442)


    parser = argparse.ArgumentParser(description="Train agent in env: %s" % CombinedTaskUsEnv.__name__)
    parser.add_argument("--exp_dir", dest="exp_dir",
                        help="Where to put all information about the experiment",
                        required=True)

    args = parser.parse_args()

    trajactory_logger = logger.TrajectoryLogger(
        log_dir=".",
        log_action_csv_freq=1,
        log_state_csv_freq=1,
        # ----- Changing render freq from 2500 to 10.
        log_state_render_freq=10
    )
    spinup_logger_kwargs = dict(output_dir=".", exp_name='log_files')
    env_builder = lambda: env_fn(trajactory_logger)

    vpg(env_fn=env_builder,
        actor_critic=loaded_cnn_actor_critic,
        ac_kwargs=AC_KWARGS,
        steps_per_epoch=N_STEPS_PER_EPOCH,
        epochs=EPOCHS,
        max_ep_len=N_STEPS_PER_EPISODE,
        logger_kwargs=spinup_logger_kwargs,
        save_freq=200,
        lam=0.97,
        pi_lr=1e-4
    )
        

if __name__ == "__main__":
    main()