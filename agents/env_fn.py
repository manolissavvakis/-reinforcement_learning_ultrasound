from envs.imaging import Probe, ImagingSystem
from envs.phantom import Teddy, ScatterersPhantom
from envs.generator import RandomProbeGenerator, ConstPhantomGenerator
from envs.combined_task_us_env import CombinedTaskUsEnv
from envs.utils import Config
import numpy as np

def env_fn(trajectory_logger, config_path: str):

    config = Config(config_path)
    
    probe = Probe(
        pos = np.array(config.get_probe_values('pos')),
        angle = config.get_probe_values('angle'),
        width = config.get_probe_values('width'),
        height = config.get_probe_values('height'),
        focal_depth = config.get_probe_values('focal_depth')
    )
    teddy = Teddy(
        belly_pos = np.array(config.get_teddy_values('belly_pos')),
        scale = config.get_teddy_values('scale'),
        head_offset = config.get_teddy_values('head_offset')
    )
    phantom = ScatterersPhantom(
        objects = [teddy],
        x_border = config.get_scatters_values('x_border'),
        y_border = config.get_scatters_values('y_border'),
        z_border = config.get_scatters_values('z_border'),
        n_scatterers = int(config.get_scatters_values('n_scatterers')),
        n_bck_scatterers = int(config.get_scatters_values('n_bck_scatterers')),
    )
    imaging = ImagingSystem(
        c = config.get_imaging_values('c'),
        fs = config.get_imaging_values('fs'),
        image_width = config.get_imaging_values('image_width'),
        image_height = config.get_imaging_values('image_height'),
        image_resolution = config.get_imaging_values('image_resolution'),  # [pixels]
        median_filter_size = config.get_imaging_values('median_filter_size'),
        dr_threshold = config.get_imaging_values('dr_threshold'),
        dec = config.get_imaging_values('dec'),
        no_lines = config.get_imaging_values('no_lines')
    )
    env = CombinedTaskUsEnv(
        imaging=imaging,
        phantom_generator=ConstPhantomGenerator(phantom),
        probe_generator=RandomProbeGenerator(
            ref_probe=probe,
            object_to_align=teddy,
            x_pos = config.get_generator_values('x_pos'),
            y_pos = config.get_generator_values('y_pos'),
            focal_pos = config.get_generator_values('focal_pos'),
            angle = config.get_generator_values('angle')
        ),
        max_steps = config.get_value('n_steps_per_episode'),
        no_workers = config.get_env_values('no_workers'),
        use_cache = config.get_env_values('use_cache'),
        trajectory_logger = trajectory_logger,
        step_size = config.get_env_values('step_size'),
        rot_deg = config.get_env_values('rot_deg')
    )

    return env