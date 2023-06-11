from stable_baselines3.common.callbacks import BaseCallback
from envs.env_fn import env_fn
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
    
    :param save_path: Path to the folder where the model will be saved.
    :param exp_name: Experiment name.
    :param gif_freq: The frequency of create behaviour gif.
    """
    def __init__(self, exp_dir: str, gif_freq: int):
        super().__init__()
        self.config_file = os.path.join(exp_dir, 'config.json')
        self.traj_dir = os.path.join(exp_dir, 'trajectory_logger')
        self.plots_dir = os.path.join(exp_dir, 'plots')
        self.gif_freq = gif_freq
        self.last_episode_trigger = 0
        
    def _on_step(self):
        # Log error value.
        error = self.training_env.envs[0].get_error()
        self.logger.record("error", error)
        
        # Log a gif with agent's behaviour every gif_freq episodes.
        episode = self.training_env.envs[0].current_episode
        if episode % self.gif_freq == 0 and episode == self.last_episode_trigger:
            images = []
            eval_env = env_fn(None, self.config_file)
            
            obs = eval_env.reset()
            img = eval_env.render(mode="rgb_array", views=["env"])

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

            images.append(draw_image_info(img, episode))
            
            steps = 0
            episode_over = False
            while steps < eval_env.max_steps and not episode_over:
                action, _ = self.model.predict(obs)
                obs, reward, episode_over, _ = eval_env.step(action.item())
                img = eval_env.render(mode="rgb_array", views=["env"])
                images.append(draw_image_info(img, episode, steps, round(reward, 4)))
                steps+= 1
            
            if not os.path.exists(self.plots_dir):
                os.mkdir(self.plots_dir)
            gif_file = os.path.join(self.plots_dir, f"gif_episode_{episode}")
            imageio.mimsave(f"{gif_file}.gif", np.array(images) , fps=5)
            
            self.last_episode_trigger += self.gif_freq
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
            
        window = 40; success_window = 10
        cumul_reward = pd.Series(collected_sum_rewards).rolling(window, min_periods=1, center=True).mean()
        mean_reward = pd.Series(collected_mean_rewards).rolling(window, min_periods=1, center=True).mean()
        mean_error = pd.Series(collected_errors).rolling(window, min_periods=1, center=True).mean()
        
        ind = np.arange(len(collected_success), step=success_window)

        success_rate = []
        for i in ind:
            success_rate.append(np.mean(collected_success[i:i+success_window]))
        
        # This means some episodes towards the end are left out.
        if not len(collected_success) % success_window:
            success_rate[-1] = np.mean(collected_success[ind[-1]:])
        
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
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(f"{success_window} episodes batch")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Mean Success Rate over {success_window} Episodes")
        [fig.savefig(os.path.join(self.plots_dir, f'success_rate.{suf}'), format=suf, bbox_inches='tight', dpi=1000) for suf in suffix]
        plt.close()  