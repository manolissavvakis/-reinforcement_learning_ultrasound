from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import JSONOutputFormat
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.env_fn import env_fn
from envs.utils import convert_json
import json
import imageio
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpi4py import MPI
from PIL import Image, ImageDraw, ImageFont

class TestConfigCallback(BaseCallback):
    """
    ConfigCallback:
    _on_training_start: save the configurations of the
    experiment in ``save_path`` directory. Parameters saved
    are what's included in ``self.locals``.

    :param save_path: Path to the folder where the model will be saved.
    :param exp_name: Experiment name.
    """
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        
    def _proc_id():
    # Get rank of calling process.
        return MPI.COMM_WORLD.Get_rank()
    
    def _on_step(self) -> bool:
        return super()._on_step()
 
    def _on_training_start(self) -> None:
        config_json = convert_json(self.locals)

        from os.path import basename
        config_json['exp_name'] = basename(self.log_dir)

        if self._proc_id()==0:
            output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
            print(output)
            with open(os.path.join(self.log_dir, "config.json"), 'w') as out:
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
    def __init__(self, log_dir: str):
        super().__init__()
        self.json_writer = JSONOutputFormat(os.path.join(log_dir, 'config.json'))

    def _on_step(self) -> bool:
        return super()._on_step()
        
    def _on_training_start(self) -> None:
        #def toJSON(model):
        #    return json.dumps(model, default=lambda o: o.__dict__, 
        #        sort_keys=True, indent=4)
        #self.json_writer.write(toJSON(self.model))
        def to_json(obj):
            return str(obj)
        include = {'model.policy': to_json(self.model.policy), 'env': to_json(self.model.env), 'policy_kwargs': to_json(self.model.policy_kwargs),
                   'n_envs': self.model.n_envs, 'entropy_coef': self.model.ent_coef, 'vf_coef': self.model.vf_coef, 
                   'gae_lambda': self.model.gae_lambda, 'gamma': self.model.gamma, 'learning_rate': self.model.learning_rate, 
                   'max_grad_norm': self.model.max_grad_norm, 'normalize_advantage': self.model.normalize_advantage, 'seed': self.model.seed, 
                    'use_sde': self.model.use_sde, 'verbose': self.model.verbose, 'total_training_steps': self.model._total_timesteps, 'epoch_steps': self.model.n_steps}
        self.json_writer.write(include, {})
        self.json_writer.close()
        
class LoggingCallback(BaseCallback):
    """
    LoggingCallback:
    _on_step: log the error after each step. Also, log a gif 
    with agent's behaviour every "gif_freq" episodes.
    
    :param save_path: Path to the folder where the model will be saved.
    :param exp_name: Experiment name.
    :param gif_freq: The frequency of create behaviour gif.
    """
    def __init__(self, log_dir: str, gif_freq: int):
        super().__init__()
        self.traj_dir = os.path.join(log_dir, 'trajectory_logger')
        self.plots_dir = os.path.join(log_dir, 'plots')
        self.gif_freq = gif_freq
        
    def _on_step(self):
        # Log error value.
        error = self.training_env.envs[0].get_error()
        self.logger.record("error", error)
        
        # Log a gif with agent's behaviour every gif_freq episodes.
        if self.training_env.envs[0].current_episode % self.gif_freq == 0:
            images = []
            env_builder = lambda: env_fn(None)
            eval_env = DummyVecEnv([env_builder])
            eval_agent = self.model
            eval_agent.set_env(eval_env)
            
            obs = eval_agent.env.reset()
            img = eval_agent.env.envs[0].render(mode="rgb_array", views=["env"])

            # Draw on the top left corner the episode, step number and reward.
            def draw_image_info(img, episode, step=0, reward=None):
                image = Image.fromarray(img)
                draw_img = ImageDraw.Draw(image)
                font = ImageFont.truetype('LiberationSans-Regular.ttf', size=20)
                text = f"Episode: {episode}\n"\
                            f"Step: {step}\n"\
                            f"Reward: {reward}"
                draw_img.text((5, 5), text, font=font, fill='black')
                return np.array(image)

            images.append(draw_image_info(img, eval_agent.env.envs[0].current_episode))
            
            steps = 0
            episode_over = False
            while steps < eval_agent.env.envs[0].max_steps and not episode_over:
                action, _ = eval_agent.predict(obs)
                obs, reward, episode_over, _ = eval_agent.env.step(action)
                img = eval_agent.env.envs[0].render(mode="rgb_array", views=["env"])
                images.append(draw_image_info(img, eval_agent.env.envs[0].current_episode, steps, round(reward.item(), 4)))
                steps+= 1
            
            if not os.path.exists(self.plots_dir):
                os.mkdir(self.plots_dir)
            gif_file = os.path.join(self.plots_dir, f"gif_episode_{eval_agent.env.envs[0].current_episode}")
            imageio.mimsave(f"{gif_file}.gif", np.array(images) , fps=5)
            
            del eval_agent, eval_env
                                    
    def _on_training_end(self):
        # Create and save mean cumulative reward and mean error plots. Also save
        # training's success rate.  
        episode_n = 0
        collected_sum_rewards = []
        collected_mean_rewards = []
        collected_errors = []
        collected_success = []
        
        while os.path.exists(os.path.join(self.traj_dir, f"episode_{episode_n}", "action.csv")) and \
        os.path.exists(os.path.join(self.traj_dir, f"episode_{episode_n}", "state.csv")) and \
        episode_n <= self.training_env.envs[0].current_episode:
        
            def _get_cumulative_reward(episode_n):
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'))
                rewards = data['reward'].to_numpy()
                return np.sum(rewards)
            
            def _get_mean_reward(episode_n):
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'))
                rewards = data['reward'].to_numpy()
                return np.mean(rewards)

            def _get_mean_error(episode_n):
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'))
                errors = data['error'].to_numpy()
                return np.mean(errors)
        
            def _get_success(episode_n):
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'))
                is_success = data['is_success'].iloc[-1]
                return is_success
        
            collected_sum_rewards.append(_get_cumulative_reward(episode_n))
            collected_mean_rewards.append(_get_mean_reward(episode_n))
            collected_errors.append(_get_mean_error(episode_n))
            collected_success.append(_get_success(episode_n))
            episode_n += 1
            
        window = 40; success_window = 10
        cumul_reward = pd.Series(collected_sum_rewards).rolling(window, min_periods=1, center=True).mean()
        mean_reward = pd.Series(collected_mean_rewards).rolling(window, min_periods=1, center=True).mean()
        mean_error = pd.Series(collected_errors).rolling(window, min_periods=1, center=True).mean()
        
        ind = np.arange(len(collected_success), step=success_window)
        success_rate = []
        
        # Warning, this works only if max_episodes % 100 == 0.
        for i in ind:
            success_rate.append(np.mean(collected_success[i:i+success_window]))
        
        suffix = ['eps', 'png']
        plt.rc('font', family='serif')
        plt.rc('lines', linewidth=2)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('axes', linewidth=0.6, labelsize=10, titlesize=14)
        plt.rc('figure', figsize=(4, 3))
        plt.rcParams['axes.spines.right'] = False
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(cumul_reward)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Cumulative Reward")
        ax.set_title("Mean Cumulative Reward over Episodes")
        [fig.savefig(os.path.join(self.plots_dir, f'cumul_reward.{suf}'), bbox_inches='tight', dpi=1000) for suf in suffix]
        plt.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(mean_reward)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward over Episodes")
        [fig.savefig(os.path.join(self.plots_dir, f'mean_reward.{suf}'), format=suf, bbox_inches='tight', dpi=1000) for suf in suffix]
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(mean_error)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Error")
        ax.set_title("Mean Error over Episodes")
        [fig.savefig(os.path.join(self.plots_dir, f'error.{suf}'), format=suf, bbox_inches='tight', dpi=1000) for suf in suffix]
        plt.close()  
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(success_rate)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Mean Success Rate over {success_window} Episodes")
        [fig.savefig(os.path.join(self.plots_dir, f'success_rate.{suf}'), format=suf, bbox_inches='tight', dpi=1000) for suf in suffix]
        plt.close()  

            
            
