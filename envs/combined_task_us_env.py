from us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np
import random
import math

_LOGGER = logging.getLogger(__name__)

class CombinedTaskUsEnv(PhantomUsEnv):
    def __init__(
            self,
            angle_range=None,
            dx_reward_coeff=1,
            dz_reward_coeff=1,
            angle_reward_coeff=1,            
            probe_dislocation_prob=None,
            max_probe_dislocation=None,
            max_probe_disrotation=None,
            dislocation_seed=None,
            **kwargs):

        super().__init__(**kwargs)
        self.angle_range = angle_range
        self.angle_reward_coeff = angle_reward_coeff
        self.dz_reward_coeff = dz_reward_coeff
        self.dx_reward_coeff = dx_reward_coeff
        self.max_probe_disrot = max_probe_disrotation
        self.max_probe_dislocation = max_probe_dislocation
        self.probe_dislocation_prob = probe_dislocation_prob
        self.action_space = spaces.Discrete(len(self._get_action_map()))
        if dislocation_seed:
            self.dislocation_rng = random.Random(dislocation_seed)
        else:
            self.dislocation_rng = None

    """    
    _get_action_map and get_action_name are inherited from us_env
    
    """

    def _perform_action(self, action):
        x_t, z_t, theta_t = self._get_action(action)

        _LOGGER.debug("Executing action: %s" % str((x_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, z_t)
        p = self.probe.rotate(theta_t)
        if self.angle_range is None or self._is_in_angle_range(p.angle):
            self.probe = p
        
    def _is_in_angle_range(self, angle):
        left, right = self.angle_range
        return left <= angle <= right

    def _get_l1_distance(self):
        # \in [0,1], 0 is better
        tracked_pos = self.phantom.get_main_object().belly.pos
        current_pos = self.probe.get_focal_point_pos()

        dx = np.abs(tracked_pos[0]-current_pos[0])
        dz = np.abs(tracked_pos[2]-current_pos[2])

        av_x_l, av_x_r = self._get_available_x_pos()
        av_z_l, av_z_r = self._get_available_z_pos()
        # WARN: assuming that the tracked object is a static object (does
        # not move)
        max_dx = max(abs(tracked_pos[0]-av_x_l), abs(tracked_pos[0]-av_x_r))
        max_dz = max(abs(tracked_pos[2]-av_z_l), abs(tracked_pos[2]-av_z_r))

        return dx/max_dx, dz/max_dz
    
    def _get_angle_distance(self):
        tracked_angle = self.phantom.get_main_object().angle
        probe_angle = self.probe.angle

        # \in [0,1], 0 is better
        # why not just use difference between angles?
        # "Convert" to reward.
        angle_diff_sin = math.sin(math.radians( (probe_angle - tracked_angle) % 360 ))

        return abs(angle_diff_sin)
        

    def _get_reward(self):
        d_x, d_z = self._get_l1_distance()
        d_a = self._get_angle_distance()
        reward = -(self.dx_reward_coeff * d_x + self.dz_reward_coeff * d_z + self.angle_reward_coeff * d_a)

        return reward

    def _update_state(self):
        if self.dislocation_rng and self.dislocation_rng.random() < self.probe_dislocation_prob:
            # Add noise depending on which plane the action is taken.
            if self._get_action() == 1 or 2:
                # Dislocate probe on along OX axis.
                x_dislocation = self.dislocation_rng.choice(list(range(1, self.max_probe_dislocation+1)))
                x_dislocation *= self.step_size
                self._move_focal_point_if_possible(x_t=x_dislocation, z_t=0)

            elif self._get_action() == 3 or 4:
                z_dislocation = self.dislocation_rng.choice(list(range(1, self.max_probe_dislocation+1)))
                z_dislocation *= self.step_size
                self._move_focal_point_if_possible(x_t=0, z_t=z_dislocation)

            elif self._get_action() == 5 or 6:
                disrotation = self.dislocation_rng.choice(list(range(1, self.max_probe_disrotation+1)))
                disrotation *= self.rot_deg
                self.probe.rotate(disrotation)
 