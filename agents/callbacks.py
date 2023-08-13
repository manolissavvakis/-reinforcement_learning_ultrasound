from stable_baselines3.common.callbacks import BaseCallback
from envs.env_fn import env_fn
from envs.confidence_wrapper import ConfidenceWrapper
from envs.logger import TrajectoryLogger
import imageio
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class LoggingCallback(BaseCallback):
    """
    LoggingCallback:
    _on_step: log the error after each step. Also, log a gif 
        with agent's behaviour every "gif_freq" episodes.
    
    _on_training_end: plot and save training metrics such as
        mean reward, cumulative mean reward, mean error and
        success rate. These metrics are calculated with moving mean.
    
    :param exp_dir: Experiment's directory.
    :param gif_freq: Frequency of gif's creation.
    :param moving_window: Window used to calculate the moving average.
    :param success_window: Window of episodes used to calculate the success rate.
    :param exp_dir: Whether to use confidence map wrapper.
    :param conf_reward_params: Parameters used in confidence based reward function.
    """
    def __init__(self,
                exp_dir: str,
                gif_freq: int,
                moving_window: int,
                success_window: int,
                use_confidence: bool,
                conf_reward_params = None
                ):
        super().__init__()
        self.config_file = os.path.join(exp_dir, 'config.json')
        self.traj_dir = os.path.join(exp_dir, 'trajectory_logger')
        self.plots_dir = os.path.join(exp_dir, 'plots')
        self.gif_freq = gif_freq
        self.window = moving_window
        self.success_window = success_window
        self.use_confidence = use_confidence
        self.conf_reward_params = conf_reward_params
        self.trigger = True

    def _on_step(self):
        # Log error value.
        error = self.training_env.envs[0].get_error()
        self.logger.record("error", error)
        
        # Log a gif with agent's behaviour every gif_freq episodes.
        episode = self.training_env.envs[0].current_episode
        if episode % self.gif_freq != 0:
            self.trigger = True
        else:
            if self.trigger:
                self.trigger = False
                images = []
                
                eval_trajectory_logger = TrajectoryLogger(
                    log_dir = os.path.join(self.plots_dir, 'evaluation_logs'),
                    log_action_csv_freq = 1,
                    log_state_csv_freq = 1,
                    log_state_render_freq = 0
                )
                
                # Initialize an env for the gif creation.
                eval_env = env_fn(eval_trajectory_logger, self.config_file)
                if self.use_confidence:
                    eval_env = ConfidenceWrapper(eval_env, self.conf_reward_params)
                
                obs = eval_env.reset()
                img = eval_env.render(mode="rgb_array", views=["env"])

                def draw_image_info(img, episode, step=0, reward=None):
                    """
                    Draw on the top left corner the episode, step number and reward.
                    
                    :param img: env observation. Can be (np.array, str) type.
                    :param episode: training's episode number.
                    :param step: evaluation's step number.
                    :param reward: reward gained from action in evaluation env.
                    :return: drawn image of (np.array, PIL.Image) type.
                    """
                    if isinstance(img, np.ndarray):
                        image = Image.fromarray(img)
                    elif isinstance(img, str):
                        image = Image.open(img)
                    else:
                        raise Exception('img input not a np.array or an image path.')
                    draw_img = ImageDraw.Draw(image)
                    font = ImageFont.truetype('LiberationSans-Regular.ttf', size=20)
                    text = f"Episode: {episode}\n"\
                                f"Step: {step}\n"\
                                f"Reward: {reward}"
                    draw_img.text((5, 5), text, font=font, fill='black')
                    return np.array(image) if isinstance(img, np.ndarray) else image

                images.append(draw_image_info(img, episode))
                
                steps = 1
                episode_over = False
                
                # Collect env observations until episode is over.
                while steps <= eval_env.max_steps and not episode_over:
                    action, _ = self.model.predict(obs)
                    obs, reward, episode_over, _ = eval_env.step(action.item())
                    img = eval_env.render(mode="rgb_array", views=["env"])
                    images.append(draw_image_info(img, episode, steps, round(reward, 4)))
                    steps+= 1
                
                if not os.path.exists(self.plots_dir):
                    os.mkdir(self.plots_dir)
                gif_file = os.path.join(self.plots_dir, f"gif_episode_{episode}")
                
                # Save gif.
                imageio.mimsave(f"{gif_file}.gif", np.array(images) , duration=200)
                
                eval_env.field_session.close()
                del eval_env
                                    
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
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'), sep='\t')
                rewards = data['reward'].to_numpy()
                return np.sum(rewards)
            
            def _get_mean_reward(episode_n):
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'), sep='\t')
                rewards = data['reward'].to_numpy()
                return np.mean(rewards)

            def _get_mean_error(episode_n):
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'), sep='\t')
                errors = data['error'].to_numpy()
                return np.mean(errors)
        
            def _get_success(episode_n):
                data = pd.read_csv(os.path.join(self.traj_dir, f"episode_{episode_n}", 'action.csv'), sep='\t')
                is_success = data['is_success'].iloc[-1]
                return is_success
        
            collected_sum_rewards.append(_get_cumulative_reward(episode_n))
            collected_mean_rewards.append(_get_mean_reward(episode_n))
            collected_errors.append(_get_mean_error(episode_n))
            collected_success.append(_get_success(episode_n))
            episode_n += 1
            
        cumul_reward = pd.Series(collected_sum_rewards).rolling(self.window, min_periods=1, center=True).mean()
        mean_reward = pd.Series(collected_mean_rewards).rolling(self.window, min_periods=1, center=True).mean()
        mean_error = pd.Series(collected_errors).rolling(self.window, min_periods=1, center=True).mean()
        
        success_rate = []

        ind = np.arange(len(collected_success), step=self.success_window)
        for i in ind:
            if not i+self.success_window > len(collected_success)-1:
                success_rate.append(np.mean(collected_success[i:i+self.success_window]))
                if i+self.success_window == len(collected_success)-1:
                    break
            else:
                if i+self.success_window - (len(collected_success)-1):
                    success_rate.append(np.mean(collected_success[i-self.success_window:]))
        
        suffix = ['eps', 'png']
        plt.rc('font', family='serif')
        plt.rc('lines', linewidth=1)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('axes', linewidth=0.6, labelsize=10, titlesize=14)
        plt.rc('figure', figsize=(4, 3))
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        
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
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(f"{self.success_window} episodes batch")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Mean Success Rate over {self.success_window} Episodes")
        [fig.savefig(os.path.join(self.plots_dir, f'success_rate.{suf}'), format=suf, bbox_inches='tight', dpi=1000) for suf in suffix]
        plt.close()
        
class SaveCacheCallback(BaseCallback):
    """
    SaveCacheCallback:
    _on_training_end: Save cache memory to a file when training is over.
    
    :param exp_dir: experiment's directory.
    """
    def __init__(self, exp_dir: str):
        super().__init__()
        self.exp_dir = exp_dir
        self.trigger = True
    
    def _on_step(self):
        episode = self.training_env.envs[0].current_episode
        if episode % 200 != 0:
            self.trigger = True
        else:
            if self.trigger:
                self.trigger = False
                cache = self.training_env.envs[0]._cache
                np.savez(os.path.join(self.exp_dir, 'cache_memory.npz'), **cache)
                
        
    def _on_training_end(self):
        cache = self.training_env.envs[0]._cache
        np.savez(os.path.join(self.exp_dir, 'cache_memory.npz'), **cache)
        

