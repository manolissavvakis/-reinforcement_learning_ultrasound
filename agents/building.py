from envs.combined_task_us_env import CombinedTaskUsEnv
from envs.logger import TrajectoryLogger
from envs.env_fn import env_fn, N_STEPS_PER_EPISODE, N_STEPS_PER_EPOCH, EPOCHS
from envs.utils import load_last_model, delete_trajectory__files

from custom_feature_extractor import CustomFeaturesExtractor
from callbacks import ConfigCallback, LoggingCallback, TestConfigCallback


import torch.nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import numpy as np
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description="Train agent in env: %s" %
                                                    CombinedTaskUsEnv.__name__)
    parser.add_argument("--exp_dir", dest="exp_dir", type=str,
                        help="Where to put all information about the experiment",
                        required=True)
    parser.add_argument("--exp_name", dest="exp_name", type=str,
                        help="What's the name of the experiment",
                        required=True)

    args = parser.parse_args()

    # paths to logs
    LOG_DIR = os.path.join(args.exp_dir, args.exp_name)
    TRAJECTORY_LOGS = os.path.join(LOG_DIR, 'trajectory_logger')
    TENSORBOARD_LOGS = os.path.join(LOG_DIR, 'tensorboard_logs')
    CHECKPOINT_LOGS = os.path.join(LOG_DIR, 'checkpoints')
    MONITOR_LOGS = os.path.join(LOG_DIR, 'monitor')

    # Create the experiment directory.
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    # Create the TrajectoryLogger object.
    trajectory_logger = TrajectoryLogger(
        log_dir=TRAJECTORY_LOGS,
        log_action_csv_freq=1,
        log_state_csv_freq=1,
        log_state_render_freq=10
    )    
    env_builder = lambda: env_fn(trajectory_logger)
        
    # Apply a DummyVecEnv Wrapper to the env.
    env = DummyVecEnv([env_builder])
    
    # Apply a Monitor Wrapper to the env.
    #env = Monitor(env, MONITOR_LOGS, info_keywords=("is_success",))

    # Define all the callbacks needed for training.
    checkpoint_callback = CheckpointCallback(
        save_freq=N_STEPS_PER_EPOCH,
        save_path=CHECKPOINT_LOGS,
        name_prefix="rl_model",
        save_vecnormalize=True
    )
        
    config_callback = ConfigCallback(
        log_dir=LOG_DIR
    )
    
    logs_callback = LoggingCallback(
        log_dir=LOG_DIR,
        gif_freq=500 
    )
    
    callback = CallbackList([checkpoint_callback, config_callback, logs_callback])

    # Create a dictionary with policy parameters.
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        net_arch=dict(pi=[], vf=[]),
        activation_fn=nn.ReLU(),
    )

    # Check if there's any model to load.
    existing_model = load_last_model(CHECKPOINT_LOGS)

    # If there is no model, create one and train.
    if not existing_model:
        model = A2C(
            'CnnPolicy',
            env,
            n_steps=N_STEPS_PER_EPOCH,
            policy_kwargs=policy_kwargs,
            gamma=0.95,
            use_rms_prop=True,
            #ent_coef=0.001,
            learning_rate=1e-3,
            normalize_advantage=True,
            verbose=1,
            #stats_window_size=10,
            tensorboard_log=TENSORBOARD_LOGS,
            seed=np.random.seed(42)
        )
        
        model.learn(total_timesteps=N_STEPS_PER_EPOCH*EPOCHS, 
            callback=callback,
            log_interval=N_STEPS_PER_EPOCH,
            tb_log_name='A2C',
        )

    # If there is a model, load it and delete all the logs that occured
    # after the model was last time saved.
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
            callback=callback,
            log_interval=N_STEPS_PER_EPOCH,
            reset_num_timesteps=False
        )

    print('training is done.')

if __name__ == "__main__":
    main()
