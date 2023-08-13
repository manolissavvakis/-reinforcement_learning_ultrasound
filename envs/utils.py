import copy
import json
import jsonpickle
import shutil
import os
import torch
import pandas as pd
import numpy as np

def copy_and_apply(src, deep=False, **kwargs):
    if deep:
        cpy = copy.deepcopy(src)
    else:
        cpy = copy.copy(src)
    for k, v in kwargs.items():
        setattr(cpy, k, v)
    return cpy

def to_string(obj):
    return jsonpickle.encode(obj)

class Loader:
    """
    Loader: A class used to load a previously trained model.
    
    :param checkpoints_dir: Directory where checkpoints are saved.
    :param trajectory_dir: Directory where trajectory logs are saved.
    :param load: Which checkpoint number to load.
    :param episode: The episode from which the training is loaded.
    """    
    def __init__(self, exp_dir):
        self.checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
        self.trajectory_dir = os.path.join(exp_dir, 'trajectory_logger')
        self.load = self._load_last_model()
        self.episode = self.load_last_episode() if self.load else None
    
    def _load_last_model(self):
        """
        Load the last saved model.
        """
        if not os.path.exists(self.checkpoints_dir):
            load = 0
        else:
            # Checkpoint are saved as rl_model_XXXXXX_steps.zip
            checkpoints = [int(x[9:-10]) for x in os.listdir(self.checkpoints_dir) if x.endswith('.zip')]
            if not len(checkpoints):
                load = 0
            else:
                load = max(checkpoints)
            
        return load

    def load_last_episode(self):
        """
        Returns the last episode that was executed while training.
        """
        step_n = 0
        step_list = []

        while os.path.exists(os.path.join(self.trajectory_dir, f"episode_{len(step_list)}", "action.csv")) and \
            os.path.exists(os.path.join(self.trajectory_dir, f"episode_{len(step_list)}", "state.csv")) and \
            step_n < self.load:
            
            data = pd.read_csv(os.path.join(self.trajectory_dir, f"episode_{len(step_list)}", 'state.csv'), sep='\t')
            step = data['step'].iloc[-1]
            step_list.append(int(step))
            step_n += step_list[-1]
        
        while np.sum(step_list) > self.load:
            step_list.pop()
        
        last_episode = len(step_list) - 1

        return last_episode
    
    def delete_logs(self):
        """
        Delete state and action logs greater than the episode number which was loaded.
        """
        episodes_list = [int(x[8:]) for x in os.listdir(self.trajectory_dir) if int(x[8:]) > self.episode]
        episodes = ['episode_' + str(number) for number in episodes_list]

        for del_episode in episodes:
            shutil.rmtree(os.path.join(self.trajectory_dir, del_episode))

class Config:
    """
    Config: A class used to load the configuration of the environment.
    
    :param config_file: Path of the configuration file.
    """
    def __init__(self, config_filepath):
        config_file = open(config_filepath)
        self.config_dict = json.load(config_file)

    def change_config_file(self, f_path):
        config_file = open(f_path)
        self.config_dict = json.load(config_file)

    def get_value(self, key):
        try:
            value = self.config_dict[key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )    
	
    def get_agent_values(self, key):
        try:
            value = self.config_dict['agent'][key]
            if key == 'activation_fn':
                act_fn_dict={"torch.nn.ReLU": torch.nn.ReLU(), "torch.nn.Tanh": torch.nn.Tanh()}
                value = act_fn_dict[self.config_dict['agent'][key]]
            return value
        except KeyError:
            raise KeyError(
            f'Key "{key}" does not exist in config file!!!'
        )

    def get_traj_values(self, key):
        try:
            value = self.config_dict['env']['trajectory_logger'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )

    def get_probe_values(self, key):
        try:
            value = self.config_dict['env']['probe'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )

    def get_teddy_values(self, key):
        try:
            value = self.config_dict['env']['teddy'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )
        
    def get_scatters_values(self, key):
        try:
            value = self.config_dict['env']['phantom'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )
        
    def get_imaging_values(self, key):
        try:
            value = self.config_dict['env']['imaging'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )
        
    def get_generator_values(self, key):
        try:
            value = self.config_dict['env']['probe_generator'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )
        
    def get_env_values(self, key):
        try:
            value = self.config_dict['env'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )
    
    def get_callback_values(self, key):
        try:
            value = self.config_dict['callbacks'][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )
    
    def get_reward_values(self, key, reward_type=None):
        try:
            if reward_type is None:
                reward_type = 'classic'
            value = self.config_dict['reward'][reward_type][key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'      
            )  
            
    def get_evaluation_values(self, key):
        try:
            value = self.config_dict['evaluation'][key]
            return value
        except KeyError:
            raise KeyError(
            f'Key "{key}" does not exist in config file!!!'
            )
