from envs.us_env import PhantomUsEnv
from gym import spaces
from gym.utils import EzPickle
import logging
import numpy as np
import random
import math

_LOGGER = logging.getLogger(__name__)


class PlaneTaskUsEnv(PhantomUsEnv, EzPickle):
    def __init__(
            self,
            angle_range=[-30, 30],
            dx_reward_coeff=1,
            angle_reward_coeff=1,
            probe_dislocation_prob=None,
            max_probe_disloc=None,
            max_probe_disrot=None,
            dislocation_seed=None,
            reward_fn=None,
            **kwargs):
        """
        Args:
            dx_reward_coeff: L1 dx multiplier
            dz_reward_coeff: L1 dz multiplier
            probe_dislocation_prob: the probability, that probe will be randomly
            dislocated in given timestep
            max_probe_dislocation: maximum random probe dislocation, that can
            be performed, in the number of self.step_sizes
        """
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_fn = reward_fn
        self.angle_range = angle_range
        self.angle_reward_coeff = angle_reward_coeff
        self.dx_reward_coeff = dx_reward_coeff
        self.max_probe_disrot = max_probe_disrot
        self.max_probe_disloc = max_probe_disloc
        self.probe_dislocation_prob = probe_dislocation_prob
        self.action_space = spaces.Discrete(len(self._get_action_map()))
        if dislocation_seed:
            self.dislocation_rng = random.Random(dislocation_seed)
        else:
            self.dislocation_rng = None
            

    def _get_action_map(self):
        return {
            # x, y, theta
            0: (0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0),
            2: (self.step_size,  0, 0),
            3: (0, 0, -self.rot_deg),
            4: (0, 0, self.rot_deg),
        }

    def get_action_name(self, action_number):
        """
        Returns string representation for given action number
        (e.g. when logging trajectory to file)
        """
        return {
            0: "NOP",
            1: "LEFT",
            2: "RIGHT",
            3: "ROT_C",
            4: "ROT_CC",
        }.get(action_number, None)

    def _perform_action(self, action):
        x_t, _, theta_t = self._get_action(action)
        z_t = 0
        _LOGGER.debug("Executing action: %s" % str((x_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, z_t)
        p = self.probe.rotate(theta_t)
        self.probe = p

    def _get_x_improvement(self):
        # Implementation: (d_t - d_t+1)/d_step
        d_t1 = self._get_distance()
        delta_dt = (self.current_distance - d_t1)/self.step_size # self.current_distance is d_t
        self.current_distance = d_t1 # Update current_distance
        return delta_dt
        
    def _get_angle_improvement(self):
        # Implementation: (theta_t - theta_t+1)/theta_step
        theta_t1 = self.probe.angle
        delta_thetat = (self.current_angle - theta_t1)/self.rot_deg # self.current_angle is theta_t
        self.current_angle = theta_t1
        return delta_thetat
    
    def _check_distance_list(self):
        if len(self.distance_list) < 2:
            self.distance_list.append(self.current_distance)        
        elif len(self.distance_list) == 2:
            if not math.isclose(self.distance_list[0], self.distance_list[1], abs_tol=0.01):
                self.distance_list = self.distance_list[1]
            if math.isclose(self.distance_list[1], self.current_distance, abs_tol=0.01):
                self.step_size = self.step_size - self.step_size/5
                self.distance_list.clear()
            else:
                self.distance_list = self.current_distance
                
    def _get_reward(self):
        if self.out_of_bounds:
            return -1
        if self._is_in_angle_range(self.probe.angle):
            return -0.5
        else:
            # use only OX distance
            delta_distance = self._get_x_improvement()
            delta_angle = self._get_angle_improvement()

            return self.dx_reward_coeff*delta_distance + self.angle_reward_coeff*delta_angle

    def _update_state(self):
        if self.dislocation_rng and \
           self.dislocation_rng.random() < self.probe_dislocation_prob:
            # draw, whether we rotate or accidentally move the probe
            if self.dislocation_rng.random() < 0.5:
                # Dislocate probe on along OX axis.
                x_disloc = self.dislocation_rng.choice(
                    list(range(-self.max_probe_disloc, 0)) +
                    list(range(1, self.max_probe_disloc+1)))
                x_disloc *= self.step_size
                self._move_focal_point_if_possible(x_t=x_disloc, z_t=0)
            else:
                disrot = self.dislocation_rng.choice(
                    list(range(-self.max_probe_disrot, 0)) +
                    list(range(1, self.max_probe_disrot+1)))
                disrot *= self.rot_deg
                self.probe.rotate(disrot)

    def _is_in_angle_range(self, angle):
        left, right = self.angle_range
        return left <= angle <= right

