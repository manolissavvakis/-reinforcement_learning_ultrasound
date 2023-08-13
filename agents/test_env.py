import numpy as np
from matplotlib import pyplot as plt
from envs.imaging import Probe, ImagingSystem
from envs.phantom import Teddy, ScatterersPhantom
from envs.generator import RandomProbeGenerator, ConstProbeGenerator, ConstPhantomGenerator
from envs.us_env import PhantomUsEnv
from envs.focal_point_task_us_env import FocalPointTaskUsEnv
from envs.plane_task_us_env import PlaneTaskUsEnv
from envs.utils import Config

config_dir = '/home/spbtu/Manolis_Files/Thesis_Project/rlus'
CONFIG_PATH = os.path.join(config_dir, 'config.json')

# paths to logs
EXP_DIR = os.path.join(config.get_value('log_dir'), 'test_experiment')
TRAJECTORY_LOGS = os.path.join(EXP_DIR, 'trajectory_logger')

N_STEPS_PER_EPISODE=config.get_value('n_steps_per_episode')
N_STEPS_PER_EPOCH=config.get_value('n_steps_per_epoch')
EPOCHS=config.get_value('epochs')

# Create the TrajectoryLogger object.
TRAJECTORY_LOGGER = TrajectoryLogger(
    log_dir = TRAJECTORY_LOGS,
    log_action_csv_freq = config.get_traj_values('log_action_csv_freq'),
    log_state_csv_freq = config.get_traj_values('log_state_csv_freq'),
    log_state_render_freq = config.get_traj_values('log_state_render_freq')
)

def test_env_fn(trajectory_logger, config_file, task, probe_type):
    """
        test_env_fn: Function the creates an enviroment based on the
        values given in the 'config.json'.
        
        :param trajectory_logger: Trajectory Logger object.
        :param config_file: Configurations file directory.
        :param task: Environment task: focal, plane or combined.
        :param probe_type: probe generator type: static or random.
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
    
    if probe_type == 'random':
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
    elif probe_type == 'static':
        probe_generator = ConstProbeGenerator(probe)
    
    env_task = {
        'us_env': PhantomUsEnv,
        'focal': FocalPointTaskUsEnv,
        'plane': PlaneTaskUsEnv,
    }
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
        noise_seed = 0
    )
    return env

def test_reset():
    """
    Test created to check a single observation/env state visualization.
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'combined', 'static')
    env.reset()

def test_moving_probe_works():
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'static')
    env.reset()
    env.step(1) # X_NEG
    env.step(1) # X_NEG
    env.step(2) # X_POS (should come from cache)
    env.step(2) # X_POS (should come from cache)
    env.step(2) # X_POS
    env.step(5) # Z_NEG
    env.step(6) # Z_POS (cached)
    env.step(6) # Z_POS
    env.step(3) # Y_NEG
    env.step(3) # Y_NEG
    env.step(4) # Y_POS (cached)

def test_rewards_1():
    """
    Test reward gained if agent steps out of bounds.
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'plane', 'static')
    env.reset()
    env.step(1) # left - BUMP

def test_rewards_2():
    """
    Test reward gained if probe's angle is out of boundaries.
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'plane', 'static')
    env.reset()
    env.step(5) # 10 deg

def test_rewards_3():
    """
    Test reward gained if probe's angle is in angle range and inside phantom.
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'plane', 'static')
    env.reset()
    env.step(5) # 10 deg
    env.step(2) # X_POS

def test_nop():
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'static')
    env.reset()
    env.step(0) # NOP
    env.step(2) # X_POS
    env.step(0) # NOP

def test_cannot_move_probe_outside_phantom_area():
    """
    Should raise an error if out_of_bounds mode is set.
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'combined', 'static')
    env.reset()
    env.step(1) # X_NEG - BUMP
    env.reset()
    env.step(2) # X_POS 0
    env.step(2) # X_POS 10
    env.step(2) # X_POS 20
    env.step(2) # X_POS - BUMP
    env.reset()
    env.step(3) # Y_NEG - BUMP
    env.reset()
    env.step(4) # Y_POS 0
    env.step(4) # Y_POS 10
    env.step(4) # Y_POS 20
    env.step(4) # Y_POS - BUMP
    env.reset()
    env.step(5) # Z_NEG 0
    env.step(5) # Z_NEG - BUMP
    env.reset()
    env.step(6) # Z_POS 10
    env.step(6) # Z_POS 20
    env.step(6) # Z_POS 30
    env.step(6) # Z_POS 40
    env.step(6) # Z_POS 50
    env.step(6) # Z_POS 60
    env.step(6) # Z_POS 70
    env.step(6) # Z_POS 80
    env.step(6) # Z_POS 90
    env.step(6) # Z_POS - BUMP

def test_caching_works():
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'static')
    env.reset()
    env.step(1) # X_NEG
    env.step(2) # X_POS (should come from cache)

def test_random_probe_generator():
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'random')
    env.reset()
    env.step(1) # X_NEG - BUMP
    env.reset()
    env.step(2) # X_POS
    env.reset()
    env.step(3) # Y_NEG
    env.reset()
    env.step(3) # Y_NEG
    env.reset()
    env.step(6) # Z_POS
    env.reset()
    env.step(5) # Z_NEG

def test_deep_focus():
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'static')
    env.reset()
    env.step(6)  # down - 10
    env.step(6)  # 20
    env.step(6)  # 30
    env.step(6)  # 40
    env.step(6)  # 50
    env.step(6)  # 60
    env.step(6)  # 70
    env.step(6)  # 80
    env.step(6)  # 90

def test_random_dislocation_1():
    """
    Just check if dislocation are drawn for this env.
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'static')
    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)

def test_random_dislocation_2():
    """
    Check if dislocations are drawn, and are properly applicated (
    should not impact the last reward, should be observable in next state).
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'static')
    env.reset()
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)

def test_random_no_dislocation_3():
    """
    Check if dislocations are drawn, and are properly applicated (
    should not impact the last reward, should be observable in next state).
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'focal', 'static')
    env.reset()
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)
    env.step(1)
    env.step(1)
    env.step(2)
    env.step(2)

def test_rotate_1():
    """
    rotate in the center of the object 540 degree,
    in one direction, in the other direction
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'plane', 'static')
    env.reset()
    env.step(5)  # 45
    env.step(5)  # 90
    env.step(5)  # 135
    env.step(5)  # 180
    env.step(5)  # 225
    env.step(5)  # 270
    env.step(5)  # 315
    env.step(5)  # 0
    env.step(5)  # 45
    env.step(6)  # should use cache
    env.step(6)
    env.step(6)
    env.step(6)
    env.step(6)
    env.step(6)

def test_rotate_2():
    """
    X_NEG, X_NEG, ROT_CC, ROT_CC, X_POS, X_POS, X_POS, ROT_C, ROT_C
    """
    env = test_env_fn(TRAJECTORY_LOGGER, CONFIG_PATH, 'plane', 'static')
    env.reset()
    env.step(1)
    env.step(1)
    env.step(6)
    env.step(6)
    env.step(2)
    env.step(2)
    env.step(2)
    env.step(5)
    env.step(5)

if __name__ == "__main__":
    #test_reset()
    #test_moving_probe_works()
    #test_rewards_1()
    #test_rewards_2()
    #test_rewards_3()
    #test_nop()
    #test_cannot_move_probe_outside_phantom_area()
    test_caching_works()
    #test_random_probe_generator()
    #test_deep_focus()
    #test_random_dislocation_1()
    #test_random_dislocation_2()
    #test_random_no_dislocation_2()
    #test_rotate_1()
    #test_rotate_2()
    #test_random_probe_generator_with_angle()