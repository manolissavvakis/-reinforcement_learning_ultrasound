from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from envs.utils import convert_json
from building import env_fn
import json
import imageio
import os
import numpy as np
from mpi4py import MPI

class TestConfigCallback(BaseCallback):
    """
    ConfigCallback:
    _on_training_start: save the configurations of the
    experiment in ``save_path`` directory. Parameters saved
    are what's included in ``self.locals``.

    :param save_path: Path to the folder where the model will be saved.
    :param exp_name: Experiment name.
    """
    def __init__(self, save_path: str, exp_name: str):
        super().__init__()
        self.save_path = save_path
        self.exp_name = exp_name
        
    def _proc_id():
    """Get rank of calling process."""
        return MPI.COMM_WORLD.Get_rank()
        
    def _on_training_start(self) -> None:
        config_json = convert_json(self.locals)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if _proc_id()==0:
            output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
            print(output)
            with open(os.path.join(self.save_path, "config.json"), 'w') as out:
                out.write(output)

class ConfigCallback(BaseCallback):
    """
    ConfigCallback:
    _on_training_start: save the configurations of the
    experiment in ``save_path`` directory. Parameters saved
    are what's included in ``self.locals``.

    :param save_path: Path to the folder where the model will be saved.
    :param exp_name: Experiment name.
    """
    def __init__(self, save_path: str, exp_name: str):
        super().__init__()
        json_writer = JSONOutputFormat(os.path.join(save_path, exp_name, 'config.json'))
        
    def _on_training_start(self) -> None:
        json_writer.write(self.locals)
        json_writer.close()
        
class LoggingCallback(BaseCallback):
    """
    LoggingCallback:
    _on_step: log the error after each step. Also, log a gif 
    with agent's behaviour every "gif_freq" episodes.
    
    :param save_path: Path to the folder where the model will be saved.
    :param exp_name: Experiment name.
    :param gif_freq: The frequency of create behaviour gif.
    """
    def __init__(self, save_path: str, exp_name: str, gif_freq: int):
        super().__init__()
        self.save_path = save_path
        self.exp_name = exp_name      
        
    def _on_step(self) -> None:
        # Log error value.
        error = self.training_env.get_error()
        self.logger.record("error", error)
        
        # Log a gif with agent's behaviour every gif_freq episodes.
        if self.training_env.current_episode %  == gif_freq:
            images = []
            eval_agent = self.model
            eval_agent.set_env(env_fn(None))
            
            obs = eval_agent.env.reset()
            img = eval_agent.env.render(mode="rgb_array", views=["env"])
            images.append(img)
            
            steps = 0
            episode_over = False
            while steps < eval_agent.env.max_steps and not episode_over:
                action, _ = eval_agent.predict(obs)
                obs, _, episode_over, _ = eval_agent.env.step(action)
                img = eval_agent.env.render(mode="rgb_array", views=["env"])
            
            gif_file = os.path.join(self.save_path, self.exp_name, 'gifs', 
                                    f"gif_step_{self.training_env.current_episode}")
            imageio.mimsave(f"{gif_file}.gif", np.array(images) , fps=5)
            
            del eval_agent
    
    
