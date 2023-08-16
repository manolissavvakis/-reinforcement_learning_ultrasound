import os
import csv
from PIL import Image
import pandas as pd


class TrajectoryLogger:
    """
    Logger object that records information about the actions of the agent.
    
    :param log_dir: where to store logging information
    :param log_action_csv_freq: how often performed actions should be logged.
        When False, do not log actions to CSV files
    :param log_state_csv_freq:  how often obtained states should be logged.
        When False, do not log the data to CSV files
    :param log_state_render_freq: how often env.renders should be logged to file,
        When False, do not log the data
    :param state_views: type of rendering.
    """
    def __init__(self,
                log_dir=None,
                log_action_csv_freq=1,
                log_state_csv_freq=1,
                log_state_render_freq=10,  # 0 or None means do not log info
                state_views=('env', 'observation')
     ):

        self.log_action_csv_freq = log_action_csv_freq
        self.log_state_csv_freq = log_state_csv_freq
        self.log_state_render_freq = log_state_render_freq
        self.state_views = state_views
        self.log_dir = log_dir
        
        # CSVLogger instances for different logs.
        self.action_logger, self.state_logger = None, None
        
        # UsPhantomEnvRenderLogger instance.
        self.state_render_loggers = []

    def restart(self, episode_nr):
        """
        Restarts trajectory recorder, creates all necessary resources
        required to save logs (directory tree structure, etc.).

        Should be called at the start of the new episode.

        param: episode_nr: the number of the next episode.
        """

        # Create a directory for each episode.
        episode_dir = os.path.join(self.log_dir, "episode_%d" % episode_nr)
        os.makedirs(episode_dir, exist_ok=True)

        if self.log_action_csv_freq:
            # Create a csv logger which records actions.
            log_file = os.path.join(episode_dir, "action.csv")
            self.action_logger = CSVLogger(
                log_file, fieldnames=[
                    "step",
                    "action_code",
                    "action_name",
                    "reward",
                    "error",
                    "is_success"
                ])

        if self.log_state_csv_freq:
            # Create a csv logger which records states (probe and object).
            log_file = os.path.join(episode_dir, "state.csv")
            self.state_logger = CSVLogger(
                log_file, fieldnames=[
                    "step",
                    "probe_x",
                    "probe_y",
                    "probe_z",
                    "probe_angle",
                    "obj_x",
                    "obj_y",
                    "obj_z",
                    "obj_angle",
                    "out_of_bounds"
                ])
        if self.log_state_render_freq and self.state_views and self.log_state_render_freq:
            self.state_render_loggers = []

            # For each view option (env, observation), create a render logger for that episode.
            for v in self.state_views:
                self.state_render_loggers.append(UsPhantomEnvRenderLogger(episode_dir, v))

    def log_action(self, episode, step, action_code,
                   action_name, reward, error, is_success=None):
        if episode % self.log_action_csv_freq == 0:
            self.action_logger.log(
                step=step,
                action_code=action_code,
                action_name=action_name,
                reward=reward,
                error=error,
                is_success=is_success
            )

    def log_state(self, episode, step, env):
        if episode % self.log_state_csv_freq == 0:

            state = {}
            # Add step number
            state['step'] = step

            # Get probe's and Teddy's (belly) state.
            for key, value in env.get_state_desc().items():
                state[key] = value

            # Add whether probe's out of constraints.
            state['out_of_bounds'] = None if not env.out_of_bounds else True

            self.state_logger.log(**state)

        # For each render logger,
        if self.log_state_render_freq:
            if episode % self.log_state_render_freq == 0:
                for logger in self.state_render_loggers:
                    logger.log(step, env)
                
    def save_trajectory(self, done):
        """
        When episode's over, save all the data collected to a csv file.
        """
        if done:
            for logger in (self.action_logger, self.state_logger):
                df = pd.DataFrame(logger.content, columns=logger.fieldnames)
                df.to_csv(logger.output_file, index=False, sep='\t')

class CSVLogger:
    """
    Logger which logs data to csv.
    
    :param fieldnames: column names of the csv file
    :param output_file: path to the csv file
    :param content: data logged.
    """
    def __init__(self, output_file, fieldnames):
        self.fieldnames = fieldnames
        self.output_file = output_file
        self.content = []
            
    def log(self, **kwargs):
        row = list(kwargs.values())
        self.content.append(row)
        
class UsPhantomEnvRenderLogger:
    """
    Logger which logs images from the experiment.
    
    :param output_dir: produced images' directory
    :param view: type of rendering
    """
    def __init__(self, output_dir, view):
        self.output_dir = output_dir
        self.view = view

    def log(self, step, env):
        # Create a screenshot of the enviroment (render function closes the figure).
        r = env.render(mode='rgb_array', views=[self.view])

        # Create the image from the array given and save it.
        img = Image.fromarray(r, mode="RGB")
        img.save(os.path.join(self.output_dir, "%s_step_%03d.png" % (self.view, step)),
                 dpi=(300,300), optimize=False, compress_level=1)