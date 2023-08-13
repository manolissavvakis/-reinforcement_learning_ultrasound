import math
import numpy as np
import gym
import json
from gym import spaces
from envs.generator import ConstPhantomGenerator
from envs.fieldii import Field2
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
import logging

_LOGGER = logging.getLogger(__name__)

class PhantomUsEnv(gym.Env):
    """
    Ultrasound environment of the Phantom.
    
    :param imaging: imaging instance used.
    :param phantom_generator: phantom generator instance used.
    :param probe_generator: probe generator instance used (constant or random).
    :param max_steps: max number of steps executed per episode.
    :param no_workers: number of workers used in the Field2 session.
    :param step_size: step size to move in x,y axes.
    :param focal_step: step size 
    :param rot_deg: rotation step size.
    :param steps_tolerance: steps tolerance to consider a final position
        as successful.
    :param use_cache: whether to use cache memory. If parameter is string
        type, cache is loaded from 'cache_memory.npz' file. Else if parameter
        is boolean type, if cache is used it will be initiliazed as empty dict.
    :param reward_params: reward singal parameters value.
    :param noise_prob: propability to apply noise (value in [0, 1]).
    :param max_probe_dislocation: max number of steps to apply as noise.
    :param noise_seed: seed given to noise application.
    :param trajectory_logger: trajectory logger instance.
    """

    def __init__(
        self,
        imaging,
        phantom_generator,
        probe_generator,
        max_steps,
        no_workers,
        step_size,
        focal_step,
        rot_deg,
        steps_tolerance,
        use_cache,
        reward_params,
        noise_prob=None,
        max_probe_dislocation=None,
        noise_seed=None,
        trajectory_logger=None,
    ):
        # Cache is used only with ConstPhantomGenerator.
        if use_cache and not isinstance(phantom_generator, ConstPhantomGenerator):
            raise ValueError("Cache can be used with %s instances only." %
                             ConstPhantomGenerator.__name__)

        self.phantom, self.probe = None, None
        self.phantom_generator = phantom_generator
        self.probe_generator = probe_generator
        self.imaging = imaging
        self.max_steps = max_steps
        self.step_size = step_size
        self.focal_step = focal_step
        self.rot_deg = rot_deg
        self.steps_tolerance = steps_tolerance
        self.noise_prob = noise_prob,
        self.max_probe_dislocation = max_probe_dislocation,
        self.noise_seed = noise_seed,
        self.current_step = 0
        self.current_episode = -1
        self.out_of_bounds = None
        self.current_observation = None
        self.last_error = None
        self.field_session = Field2(no_workers=no_workers)
        self.use_cache = use_cache
        self.reward_params = reward_params
        if self.use_cache:
            if isinstance(self.use_cache, bool):
                self._cache = {}
            elif isinstance(self.use_cache, str):
                try:
                    self._cache = np.load('cache_memory.npz')
                except: 
                    raise Exception('The cache file specified does not exist.')

        self.action_space = spaces.Discrete(len(self._get_action_map()))
        observation_shape = (
            1,
            self.imaging.image_resolution[1],
            self.imaging.image_resolution[0]
            )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=observation_shape,
            dtype=np.float64)
        self.metadata = {
            'render.modes': ['rgb_array']
        }
        if noise_seed:
            self.noise_rng = random.Random(noise_seed)
        else:
            self.noise_rng = None
        
        self.trajectory_logger = trajectory_logger
        _LOGGER.debug("Created environment: %s" % repr(self))
        _LOGGER.debug("Action space: %s" % repr(self.action_space))
        _LOGGER.debug("Observations space: %s" % repr(self.observation_space))

    def _get_action_map(self):
        # (x, y, z, theta)
        return {
            0: (0, 0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0, 0),  # x axis movement to the left
            2: (self.step_size,  0, 0, 0),  # x axis movement to the right
            3: (0, -self.step_size,  0, 0), # y axis movement to the left
            4: (0, self.step_size,  0, 0),  # y axis movement to the right
            5: (0, 0, -self.focal_step, 0),  # z axis move upwards
            6: (0, 0, self.focal_step, 0),   # z axis move downwards
            7: (0, 0, 0, -self.rot_deg),    # clockwise rotation
            8: (0, 0, 0, self.rot_deg)      # counter-clockwise rotation
        }

    def _get_action(self, action_number):
        return self._get_action_map()[action_number]

    def get_action_name(self, action_number):
        """
        :return: string representation for given action number
        (e.g. when logging trajectory to file)
        """
        return {
            0: "NOP",
            1: "X_NEG",
            2: "X_POS",
            3: "Y_NEG",
            4: "Y_POS",
            5: "Z_NEG",
            6: "Z_POS",      
            7: "ROT_C",
            8: "ROT_CC"
        }.get(action_number, None)

    def reset(self):
        """
        Resets the environment and increments current episode counter.

        :return: first observation after the reset
        """
        _LOGGER.debug("Restarting environment.")
        self.phantom = next(self.phantom_generator)
        self.probe = next(self.probe_generator)
        self.out_of_bounds = False
        self.last_error = self.get_error()
        self.current_step = 0
        self.current_episode += 1
        o = self._get_observation()
        self.current_observation = o

        if self.trajectory_logger is not None:
            self.trajectory_logger.restart(episode_nr=self.current_episode)
            self.trajectory_logger.log_state(
                episode=self.current_episode,
                step=self.current_step,
                env=self)
        return o

    def step(self, action):
        """
        Perform action and move environment state to the next timestep.
        The sequence followed is:
        perform action -> get_observation -> compute reward -> update_state -> log state

        :param action: action to perform.
        :return: observation, reward, episode over?, diagnostic info
            (whether episode was successful)
        """
        if self._check_termination_conditions()[0]:
            raise RuntimeError("This episode is over, reset the environment.")
        self.current_step += 1
        self._perform_action(action)
        o = self._get_observation()
        
        reward = self._get_reward()
        # Apply noise to the current state.
        self._update_state(action)

        self.current_observation = o
        info = dict(is_success = None)
        episode_over, info["is_success"] = self._check_termination_conditions()

        if self.trajectory_logger is not None:
            self.trajectory_logger.log_action(
                episode=self.current_episode,
                step=self.current_step,
                action_code=action,
                action_name=self.get_action_name(action),
                reward=reward,
                error=self.last_error,
                is_success=info["is_success"]
            )
            self.trajectory_logger.log_state(
                episode=self.current_episode,
                step=self.current_step,
                env=self
            )
            self.trajectory_logger.save_trajectory(episode_over)
        return o, reward, episode_over, info

    def render(self, mode='rgb_array', views=None):
        """
        Renders current state of the environment.

        :param mode: rendering mode (see self.metdata['render.modes'] for supported modes.
        :param views: imaging views, any combination of values: {'env', 'observation'},
            if views is None, all ['env', 'observation'] modes are used.
        :return: an output of the renderer
        """
        if views is None:
            views = ["env", "observation"]
        if mode == 'rgb_array':
            return self._render_to_array(views)
        else:
            super(PhantomUsEnv).render(mode=mode)

    def close(self):
        """
        Terminate a Field2 session.
        """
        self.field_session.close()

    def get_state_desc(self):
        return {
            "probe_x": self.probe.pos[0],
            "probe_y": self.probe.pos[1],
            "probe_z": self.probe.focal_depth,
            "probe_angle": self.probe.angle,
            "obj_x": self.phantom.get_main_object().belly.pos[0],
            "obj_y": self.phantom.get_main_object().belly.pos[1],
            "obj_z": self.phantom.get_main_object().belly.pos[2],
            "obj_angle": self.phantom.get_main_object().angle
        }

    def _get_reward(self):
        """
        Calculate the reward signal gained.
        """
        if self.out_of_bounds:
            return -1
        else:
            e = self.get_error()
            a_p = self.reward_params['a_p']
            a_r = self.reward_params['a_r']
            e_thresh = self.reward_params['e_thresh']
            
            reward_clipped = 1 - (e/e_thresh) if e<=e_thresh else 0
            
            if not math.isclose(e, self.last_error, rel_tol = 1e-3):
                if e > self.last_error:
                    reward_hint = -a_p
                else:
                    reward_hint = a_r
            else:
                reward_hint = 0

            self.last_error = e

            reward = reward_clipped + reward_hint
            
            return reward
    
    def get_error(self):
        """
        Calculate error signal between probe and the goal.
        """
        dx, dy, dz = self._get_pos_diff()  
        dtheta = self._get_angle_diff()
        
        # Normalize x,y,z,theta values.
        x = dx/(self.phantom.x_border[1]/2)
        y = dy/(self.phantom.y_border[1]/2)
        z = dz/(self.phantom.get_main_object().belly.pos[2])
        theta = np.sin(np.radians(dtheta/2))
        
        error = 1/4 * np.sum(np.power([x, y, z, theta], 2))
        return error

    def _perform_action(self, action):
        """
        Move and rotate and probe.
        """
        x_t, y_t, z_t, theta_t = self._get_action(action)
        _LOGGER.debug("Executing action: %s" % str((x_t, y_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, y_t, z_t)
        self.probe = self.probe.rotate(theta_t)
    
    def _get_pos_diff(self):
        """
        Get difference between probe and goal in x,y,z axis.
        """
        # Position in the current timestep.
        x_t, y_t, z_t = self.probe.get_focal_point_pos()
        # Position of the goal.
        x_g, y_g, z_g = self.phantom.get_main_object().belly.pos
        return x_t - x_g, y_t - y_g, z_t - z_g
        
    def _get_angle_diff(self):
        """
        Get angle difference between probe and goal.
        """
        # Angle in the current timestep.
        theta_t = self.probe.angle
        # Angle of the goal.
        theta_g = self.phantom.get_main_object().angle
        return theta_t - theta_g
    
    def _get_distance(self):
        """
        Get Euclidean distance between probe and goal.
        """
        delta_xyz = np.array(self._get_pos_diff())
        distance = np.sqrt(np.sum(np.power(delta_xyz, 2)))
        return distance

    def _move_focal_point_if_possible(self, x_t, y_t, z_t):
        """
        Move probe in the phantom. If probe moves outside the phantom
        limits, the move is cancelled.
        
        :return: new probe position and focal depth.
        """
        pr_pos_x_l = (self.probe.pos[0] - self.probe.width/2) + x_t
        pr_pos_x_r = (self.probe.pos[0] + self.probe.width/2) + x_t
        pr_pos_y_l = (self.probe.pos[1] - self.probe.height/2) + y_t
        pr_pos_y_r = (self.probe.pos[1] + self.probe.height/2) + y_t
        pr_pos_z = self.probe.focal_depth + z_t
        x_border_l, x_border_r = self.phantom.x_border
        y_border_l, y_border_r = self.phantom.y_border
        z_border_l, z_border_r = self.phantom.z_border

        rel_tol = 10/1000

        def le(a, b):
            return a < b or math.isclose(a, b, rel_tol=rel_tol)

        def ge(a, b):
            return a > b or math.isclose(a, b, rel_tol=rel_tol)

        if le(x_border_l, pr_pos_x_l) and ge(x_border_r, pr_pos_x_r):
            self.probe = self.probe.translate(np.array([x_t, 0, 0]))
        else:
            self.out_of_bounds = True
        if le(y_border_l, pr_pos_y_l) and ge(y_border_r, pr_pos_y_r):
            self.probe = self.probe.translate(np.array([0, y_t, 0]))
        else:
            self.out_of_bounds = True 
        if le(z_border_l, pr_pos_z) and ge(z_border_r, pr_pos_z):
            self.probe = self.probe.change_focal_depth(z_t)
        else:
            self.out_of_bounds = True
        self.probe.pos = np.round(self.probe.pos, decimals=3)
        self.probe.focal_depth = round(self.probe.focal_depth, ndigits=3)

    def _update_state(self, action):
        """
        Apply noise depending on which plane the action is taken.
        """
        if self.noise_rng and self.noise_rng.random() < self.noise_prob:
            
            # Number of steps to add as noise.
            dislocation = self.noise_rng.choice(list(range(1, self.max_probe_dislocation+1)))
            x_t, y_t, z_t, theta_t = self._get_action(action)
            x_t, y_t, z_t, theta_t = (coord * dislocation for coord in (x_t, y_t, z_t, theta_t))
            self._move_focal_point_if_possible(x_t, y_t, z_t)
            self.probe.rotate(theta_t)

    def _get_observation(self):
        if self.use_cache:
            return self._get_cached_observation()
        else:
            return self._get_image() 

    def _get_cached_observation(self):
        """
        .. warning:
            Assumes that objects in the phantom does not move (are 'static').
        
        :return: a previously observed state from cache.
        """
        state = (
            int(round(self.probe.pos[0], 3)*1e3),
            int(round(self.probe.pos[1], 3)*1e3),
            int(round(self.probe.focal_depth, 3)*1e3),
            int(round(self.probe.angle))
        )
        if state in self._cache:
            _LOGGER.info("Using cached value for probe state (x, y, z, theta)=%s"
                          % str(state))
        else:
            bmode = self._get_image()
            self._cache[state] = bmode
        return self._cache[state]
            
    def _check_termination_conditions(self):
        """
        Conditions to terminate an episode:
        1) Probe is outside the phantom.
        2) Reached max number of steps.
        
        :return: whether episode's over and goal is reached.
        """
        episode_over = False
        if (self.current_step >= self.max_steps or self.out_of_bounds):
            episode_over = True
        if not episode_over:
            success = None
        else:
            success = self.is_episode_successful()
        return episode_over, success

    def _get_image(self):
        """
        Feed probe's field of view to Field2 simulation.
        
        :return: bmode image
        """
        points, amps, _ = self.probe.get_fov(self.phantom)
        rf_array, _ = self.field_session.simulate_linear_array(
            points, amps,
            sampling_frequency=self.imaging.fs,
            no_lines=self.imaging.no_lines,
            z_focus=self.probe.focal_depth,
            image_width=self.imaging.image_width)
        bmode = self.imaging.image(rf_array)
        bmode = bmode.reshape((1,)+bmode.shape)
        _LOGGER.debug("B-mode image shape: %s" % str(bmode.shape))
        return bmode
    
    def is_episode_successful(self):
        """
        Success of an episode. To consider probe's final position
        as success, there is a number of steps tolerance.
        
        :return: If the function returns true, the 
        position and angle of the goal are reached.
        """
        if self.out_of_bounds:
            return False
        else:
            steps_tol = self.steps_tolerance
            final_pos = self.probe.get_focal_point_pos()
            final_angle = self.probe.angle
            if 180 < final_angle <= 360:
                final_angle = -(360 - final_angle)    
            goal_pos = self.phantom.get_main_object().belly.pos
            goal_angle = self.phantom.get_main_object().angle

            pos_success = np.allclose(final_pos[0:2], goal_pos[0:2], atol=float(steps_tol) * self.step_size)
            foc_success = math.isclose(final_pos[2], goal_pos[2], abs_tol=float(steps_tol) * self.focal_step)
            rot_success = math.isclose(final_angle, goal_angle, abs_tol=steps_tol * self.rot_deg)
            
            return pos_success and foc_success and rot_success

    def _render_to_array(self, views):
        fig = plt.figure(figsize=(4, 4), dpi=200)
        title_elements = [
             ("Episode: %d", self.current_episode),
             ("step: %d", self.current_step),
             ("reward: %.2f", self._get_reward())
        ]
        title_elements = [el[0] % el[1] for el in title_elements if el[1] is not None]
        
        view_handlers = {
            'env': self._plot_env,
            'observation': self._plot_bmode
        }
        projections = {
            'env': '3d',
            'observation': None # default
        }
        for i, v in enumerate(views):
            projection = projections[v]
            view_handler = view_handlers[v]
            ax = fig.add_subplot(len(views), 1, i+1, projection=projection)
            view_handler(ax)
        fig.canvas.draw()
        plt.tight_layout()
        b = fig.canvas.tostring_rgb()
        width, height = fig.canvas.get_width_height()
        result = np.fromstring(b, dtype=np.uint8).copy().reshape(height, width, 3)
        plt.close(fig)
        return result

    def _plot_bmode(self, ax):
        """
        :return: axes of bmode plot
        """
        if self.current_observation is None:
            raise RuntimeError("Please call 'restart' method first.")
        ax.imshow(self.current_observation.squeeze(), cmap='gray')
        ax.set_xlabel("width $(px)$")
        ax.set_ylabel("depth $(px)$")
        return ax

    def _plot_env(self, ax):
        """
        :return: axes of environment observation plot
        """
        ax.set_xlabel("$X (mm)$")
        ax.set_ylabel("$Y (mm)$")
        ax.set_zlabel("$Z (mm)$")
        ax.set_xlim(self.phantom.x_border)
        ax.set_ylim(self.phantom.y_border)
        ax.set_zlim(self.phantom.z_border)
        
        def mm_formatter_fn(x, pos):
            return "%.0f" % (x * 1000)
        mm_formatter = matplotlib.ticker.FuncFormatter(mm_formatter_fn)
        ax.xaxis.set_major_formatter(mm_formatter)
        ax.yaxis.set_major_formatter(mm_formatter)
        ax.zaxis.set_major_formatter(mm_formatter)
        # ax.view_init(0, azim=(-self.probe.angle))
        ax.invert_zaxis()
        # Plot phantom objects.
        for obj in self.phantom.objects:
            obj.plot_mesh(ax)
        # Plot focal point position (THE GOAL).
        focal_point_x = self.probe.pos[0]
        focal_point_y = self.probe.pos[1]
        focal_point_z = self.probe.focal_depth
        ax.scatter(
            focal_point_x, focal_point_y, focal_point_z,
            s=400, c='yellow', marker='X')
        # Plot probe line.
        probe_x = .5*self.probe.width*math.cos(math.radians(self.probe.angle))
        probe_y = .5*self.probe.width*math.sin(math.radians(self.probe.angle))
        probe_pt_1 = np.array([probe_x, probe_y, 0])+self.probe.pos
        probe_pt_2 = -np.array([probe_x, probe_y, 0])+self.probe.pos
        ax.plot(
            xs=[probe_pt_1[0], probe_pt_2[0]],
            ys=[probe_pt_1[1], probe_pt_2[1]],
            zs=[0,0],
            c='yellow',
            linewidth=5
        )
        return ax
