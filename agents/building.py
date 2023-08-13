from envs.logger import TrajectoryLogger
from envs.env_fn import env_fn
from envs.confidence_wrapper import ConfidenceWrapper
from envs.utils import Config, Loader
from custom_feature_extractor import CustomFeaturesExtractor
from callbacks import LoggingCallback, SaveCacheCallback
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
import shutil
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description="Train agent in ultrasound enviroment.")
    parser.add_argument("--config_path", dest="config_path", type=str,
                        help="Directory of the configurations of the experiment",
                        required=True)
    parser.add_argument("--exp_name", dest="exp_name", type=str,
                        help="What's the name of the experiment",
                        required=True)

    args = parser.parse_args()

    CONFIG_PATH = os.path.join(args.config_path, 'config.json')
    config = Config(CONFIG_PATH)

    # Paths to logs
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
    EXP_CONFIG = os.path.join(EXP_DIR, 'config.json')
    if not os.path.exists(EXP_CONFIG):
        shutil.copy(CONFIG_PATH, EXP_DIR)
    config.change_config_file(EXP_CONFIG)

    # Check if there's any model to load.
    loader = Loader(EXP_DIR)

    # Create the TrajectoryLogger object.
    trajectory_logger = TrajectoryLogger(
        log_dir = TRAJECTORY_LOGS,
        log_action_csv_freq = config.get_traj_values('log_action_csv_freq'),
        log_state_csv_freq = config.get_traj_values('log_state_csv_freq'),
        log_state_render_freq = config.get_traj_values('log_state_render_freq')
    )
    
    # Create the environment.
    env = env_fn(trajectory_logger, EXP_CONFIG) 

    # Apply Confidence Map Wrapper, if needed.
    use_confidence=config.get_value('use_confidence')
    if use_confidence:
        a_p = config.get_value('a_p', 'confidence')
        a_r = config.get_value('a_r', 'confidence')
        conf_reward_params = {'a_p': a_p, 'a_r': a_r}
        env = ConfidenceWrapper(env, conf_reward_params)

	# Apply a Monitor Wrapper to the env.
    env = Monitor(env, MONITOR_LOGS, info_keywords=("is_success",))

    # Define all the callbacks needed for training.
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get_callback_values('save_freq')*N_STEPS_PER_EPOCH,
        save_path=CHECKPOINT_LOGS,
        name_prefix="rl_model",
        save_vecnormalize=True
    )
    
    logs_callback = LoggingCallback(
        exp_dir=EXP_DIR,
        gif_freq=config.get_callback_values('gif_freq'),
        moving_window=config.get_callback_values('moving_window'),
        success_window=config.get_callback_values('success_window'),
        use_confidence=use_confidence,
        conf_reward_params = conf_reward_params if use_confidence else None
    )
    
    cache_callback = SaveCacheCallback(
        exp_dir=EXP_DIR
    )
    """
    max_no_improvement_evals = config.get_callback_values('max_no_improvement_evals')
    min_evals = config.get_callback_values('min_evals')
    verbose = config.get_callback_values('verbose')
    n_eval_episodes = config.get_callback_values('n_eval_episodes')
    eval_freq = config.get_callback_values('eval_freq')
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=max_no_improvement_evals,
                                                           min_evals=min_evals,
                                                           verbose=verbose)
    EVAL_DIR = os.path.join(EXP_DIR, 'eval_logs')

    eval_trajectory_logger = TrajectoryLogger(
        log_dir = os.path.join(EVAL_DIR, 'trajectory_logger'),
        log_action_csv_freq = config.get_traj_values('log_action_csv_freq'),
        log_state_csv_freq = config.get_traj_values('log_state_csv_freq'),
        log_state_render_freq = 1
    )

    eval_env = env_fn(eval_trajectory_logger, EXP_CONFIG) 
    eval_callback = EvalCallback(
        eval_env = eval_env,
        n_eval_episodes = n_eval_episodes,
        eval_freq = eval_freq,
        callback_after_eval = stop_train_callback,
        log_path=EVAL_DIR,
        best_model_save_path = os.path.join(EVAL_DIR, 'saves'),
        verbose=1
    )
    """
    #callback = CallbackList([checkpoint_callback, logs_callback, eval_callback])
    callback = CallbackList([checkpoint_callback, logs_callback, cache_callback])
    
    # Create a dictionary with policy parameters.
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        net_arch= config.get_agent_values('net_arch'),
        activation_fn=config.get_agent_values('activation_fn'),
    )

    # If there is no model, create and train one.
    if not loader.load:
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
            seed = np.random.seed(config.get_agent_values('seed')),
    )
        model.learn(total_timesteps=N_STEPS_PER_EPOCH*EPOCHS, 
            callback=callback,
            log_interval=N_STEPS_PER_EPOCH,
            tb_log_name=f'{args.exp_name}_A2C',
        )

    # If there is a model, load it and delete all the logs that occured
    # after the model was last time saved.
    else:
        model = A2C.load(
            os.path.join(CHECKPOINT_LOGS, 'rl_model_%d_steps.zip' %loader.load),
            env,
        )
        # Check if there is more training steps left.
        remaining_steps = (N_STEPS_PER_EPOCH*EPOCHS) - loader.load
        assert remaining_steps>=0, 'Steps argument is less than 0. There is no training.'

        model.learn(total_timesteps=remaining_steps,
            callback=callback,
            log_interval=N_STEPS_PER_EPOCH,
            reset_num_timesteps=config.get_agent_values('reset_num_timesteps')
        )

    print('training is done.')

if __name__ == "__main__":
    main()
