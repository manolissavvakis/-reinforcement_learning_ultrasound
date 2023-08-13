from envs.imaging import Probe, ImagingSystem
from envs.phantom import Teddy, ScatterersPhantom
from envs.generator import RandomProbeGenerator, ConstProbeGenerator, ConstPhantomGenerator
from envs.us_env import PhantomUsEnv
from envs.focal_point_task_us_env import FocalPointTaskUsEnv
from envs.plane_task_us_env import PlaneTaskUsEnv
from envs.utils import Config
import numpy as np

def env_fn(trajectory_logger, config_file):
    """
    env_fn: Function the creates an enviroment based on the
    values given in the 'config.json'.

    :param trajectory_logger: Trajectory Logger object.
    :param config_file: Path of the configurations file.
    """        
    config = Config(config_file)
    
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
        objects=[teddy],
        x_border = config.get_scatters_values('x_border'),
        y_border = config.get_scatters_values('y_border'),
        z_border = config.get_scatters_values('z_border'),
        n_scatterers = int(config.get_scatters_values('n_scatterers')),
        n_bck_scatterers = int(config.get_scatters_values('n_bck_scatterers'))
    )
    imaging = ImagingSystem(
        c = config.get_imaging_values('c'),
        fs = config.get_imaging_values('fs'),
        image_width = config.get_imaging_values('image_width'),
        image_height = config.get_imaging_values('image_height'),
        image_resolution = config.get_imaging_values('image_resolution'),
        median_filter_size = config.get_imaging_values('median_filter_size'),
        dr_threshold = config.get_imaging_values('dr_threshold'),
        dec = config.get_imaging_values('dec'),
        no_lines = config.get_imaging_values('no_lines')
    )
    if config.get_generator_values('random'):
        x_values = config.get_generator_values('x_pos')
        y_values = config.get_generator_values('y_pos')
        focal_values = config.get_generator_values('focal_pos')
        angle_values = config.get_generator_values('angle')

        probe_generator = RandomProbeGenerator(
            ref_probe=probe,
            object_to_align=teddy,
            x_pos = np.arange(x_values[0], x_values[1], x_values[2]),
            y_pos = np.arange(y_values[0], y_values[1], y_values[2]),
            focal_pos = np.arange(focal_values[0], focal_values[1], focal_values[2]),
            angle = np.arange(angle_values[0], angle_values[1], angle_values[2])
        )
    else:
        probe_generator = ConstProbeGenerator(probe)
    
    env_task = {
        'us_env': PhantomUsEnv,
        'focal': FocalPointTaskUsEnv,
        'plane': PlaneTaskUsEnv,
    }
    task = config.get_value('task_type')
    env_init=env_task[task]

    env = env_init(
        imaging=imaging,
        phantom_generator=ConstPhantomGenerator(phantom),
        probe_generator=probe_generator,
        trajectory_logger = trajectory_logger,
        max_steps = config.get_value('n_steps_per_episode'),
        no_workers = config.get_env_values('no_workers'),
        use_cache = config.get_env_values('use_cache'),   
        step_size = config.get_env_values('step_size'),
        focal_step = config.get_env_values('focal_step'),
        rot_deg = config.get_env_values('rot_deg'),
        reward_params = {"a_p": config.get_reward_values('a_p'),
                        "a_r": config.get_reward_values('a_r'),
                        "e_thresh": config.get_reward_values('e_thresh')},
        steps_tolerance = config.get_env_values('steps_tolerance'),
        noise_prob = config.get_env_values('noise_prob'),
        max_probe_dislocation = config.get_env_values('max_probe_dislocation'),
        noise_seed = config.get_env_values('noise_seed')
    )
    return env
