import copy
import json
import jsonpickle
import shutil
import os
import torch

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

def load_last_model(fpath):
    if not os.path.exists(fpath):
        itr = 0
    else:
        # Checkpoint are saved as rl_model_XXXXXX_steps.zip
        checkpoints = [int(x[9:-10]) for x in os.listdir(fpath) if x.endswith('.zip')]
        if not len(checkpoints):
            itr = 0
            traj_dir = os.path.join(os.path.dirname(fpath), 'trajectory_logger')
            delete_trajectory__files(traj_dir, itr)
        else:
            itr = max(checkpoints)
        
    return itr

def delete_trajectory__files(fpath, itr):
    if os.path.exists(fpath):

        # trajectories are stored in trajectory_logger file as episode_XXX
        # equal is used because first episode is 0.
        itr_episodes = [int(x[8:]) for x in os.listdir(fpath) if int(x[8:]) >= itr]
        episodes = ['episode_' + str(number) for number in itr_episodes]

        for episode in episodes:
            shutil.rmtree(os.path.join(fpath, episode))

class Config:
    def __init__(self, config_filepath):
        config_file = open(config_filepath)
        self.config_dict = json.load(config_file)

    def get_value(self, key):
        try:
            value = self.config_dict[key]
            return value
        except KeyError:
            raise KeyError(
                f'Key "{key}" does not exist in config file!!!'
            )    

    def get_list(self, keyword='slide_ids'):
        try:
            slide_ids = self.config_dict[keyword].split(',')
            return slide_ids
        except KeyError:
            raise KeyError(
                'Requested list does not exist in config file!!!'
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