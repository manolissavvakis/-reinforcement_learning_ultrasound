from envs.combined_task_us_env import CombinedTaskUsEnv
from envs.logger import TrajectoryLogger
from envs.env_fn import env_fn
from envs.confidence_wrapper import ConfidenceWrapper
from envs.utils import load_last_model, delete_trajectory__files, Config
from custom_feature_extractor import CustomFeaturesExtractor
from callbacks import LoggingCallback
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import shutil
import numpy as np
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description="Train agent in env: %s" %
                                                    CombinedTaskUsEnv.__name__)
    parser.add_argument("--config_path", dest="config_path", type=str,
                        help="Directory of the configurations of the experiment",
                        required=True)
    parser.add_argument("--exp_name", dest="exp_name", type=str,
                        help="What's the name of the experiment",
                        required=True)

    args = parser.parse_args()

    CONFIG_PATH = os.path.join(args.config_path, 'config.json')
    config = Config(CONFIG_PATH)

    # paths to logs
    EXP_DIR = os.path.join(config.get_value('log_dir'), args.exp_name)
    TRAJECTORY_LOGS = os.path.join(EXP_DIR, 'trajectory_logger')
    TENSORBOARD_LOGS = os.path.join(EXP_DIR, 'tensorboard_logs')
    CHECKPOINT_LOGS = os.path.join(EXP_DIR, 'checkpoints')
    MONITOR_LOGS = os.path.join(EXP_DIR, 'monitor')
    
    N_STEPS_PER_EPISODE=config.get_value('n_steps_per_episode')
    N_STEPS_PER_EPOCH=config.get_value('n_steps_per_epoch')
    EPOCHS=config.get_value('epochs')

    # Create the experiment directory.
    if not os.path.exists(EXP_DIR):
        os.mkdir(EXP_DIR)

    # Copy config file in the experiment directory.
    shutil.copy(CONFIG_PATH, EXP_DIR)

    # Create the TrajectoryLogger object.
    trajectory_logger = TrajectoryLogger(
        log_dir = TRAJECTORY_LOGS,
        log_action_csv_freq = config.get_traj_values('log_action_csv_freq'),
        log_state_csv_freq = config.get_traj_values('log_state_csv_freq'),
        log_state_render_freq = config.get_traj_values('log_state_render_freq')
    )    
    #env_builder = lambda: env_fn(trajectory_logger)
        
    # Apply a DummyVecEnv Wrapper to the env.
    #env = DummyVecEnv([env_builder])
    env = env_fn(trajectory_logger, CONFIG_PATH) 

    # Apply Confidence Map Wrapper, if needed.
    if config.get_value('use_confidence'):
        env = ConfidenceWrapper(env, config)
    
    # Apply a Monitor Wrapper to the env.
    env = Monitor(env, MONITOR_LOGS, info_keywords=("is_success",))

    # Define all the callbacks needed for training.
    checkpoint_callback = CheckpointCallback(
        save_freq=N_STEPS_PER_EPOCH,
        save_path=CHECKPOINT_LOGS,
        name_prefix="rl_model",
        save_vecnormalize=True
    )
    
    logs_callback = LoggingCallback(
        exp_dir=EXP_DIR,
        gif_freq=config.get_value('gif_freq')
    )
    
    callback = CallbackList([checkpoint_callback, logs_callback])

    # Create a dictionary with policy parameters.
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        net_arch= config.get_agent_values('net_arch'),
        activation_fn=config.get_agent_values('activation_fn'),
    )
    
    # Check if there's any model to load.
    existing_model = load_last_model(CHECKPOINT_LOGS)

    # If there is no model, create one and train.
    if not existing_model:
        model = A2C(
            'CnnPolicy',
            env,
            n_steps = N_STEPS_PER_EPOCH,
            policy_kwargs = policy_kwargs,
            gamma = config.get_agent_values('gamma'),
            use_rms_prop = config.get_agent_values('use_rms_prop'),
            ent_coef = config.get_agent_values('ent_coef'),
            learning_rate = config.get_agent_values('learning_rate'),
            normalize_advantage = config.get_agent_values('normalize_advantage'),
            verbose = config.get_agent_values('verbose'),
            #stats_window_size=10,
            tensorboard_log=TENSORBOARD_LOGS,
            seed = np.random.seed(config.get_agent_values('seed'))
    )    
        model.learn(total_timesteps=N_STEPS_PER_EPOCH*EPOCHS, 
            callback=callback,
            log_interval=N_STEPS_PER_EPOCH,
            tb_log_name=f'{args.exp_name}_A2C',
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
            reset_num_timesteps=config.get_agent_values('reset_num_timesteps')
        )

    print('Training is completed.')

if __name__ == "__main__":
    main()

