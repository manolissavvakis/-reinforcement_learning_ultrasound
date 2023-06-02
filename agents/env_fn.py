from envs.imaging import Probe, ImagingSystem
from envs.phantom import Teddy, ScatterersPhantom
from envs.generator import RandomProbeGenerator, ConstPhantomGenerator
from envs.combined_task_us_env import CombinedTaskUsEnv
import numpy as np

N_STEPS_PER_EPISODE = 50
N_STEPS_PER_EPOCH = 100
EPOCHS = 1000 # NO_EPISODES = (NSTEPS_PER_EPOCH/NSTEPS_PER_EPISODE)*EPOCHS
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
        belly_pos=np.array([0 / 1000, 0 / 1000, 50 / 1000]),
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
        seed=42,
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