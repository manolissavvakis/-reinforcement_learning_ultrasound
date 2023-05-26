from envs.us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np
import math
import random

_LOGGER = logging.getLogger(__name__)


class FocalPointTaskUsEnv(PhantomUsEnv):
    def __init__(
            self,
            probe_dislocation_prob=None,
            max_probe_dislocation=None,
            dislocation_seed=None,
            **kwargs):
        """
        Args:
            probe_dislocation_prob: the probability, that probe will be randomly
            dislocated in given timestep
            max_probe_dislocation: maximum random probe dislocation, that can
            be performed, in the number of self.step_sizes
        """
        super().__init__(**kwargs)
        self.max_probe_dislocation = max_probe_dislocation
        self.probe_dislocation_prob = probe_dislocation_prob
        self.action_space = spaces.Discrete(len(self._get_action_map()))
        if dislocation_seed:
            self.dislocation_rng = random.Random(dislocation_seed)
        else:
            self.dislocation_rng = None

    def _get_action_map(self):
        return {
            0: (0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0),  # move to the left
            2: (self.step_size,  0, 0),  # move to the right
            3: (0, -self.step_size, 0),  # move upwards
            4: (0,  self.step_size, 0),  # move downwards
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
            3: "UP",
            4: "DOWN",
        }.get(action_number, None)

    def _perform_action(self, action):
        x_t, z_t, theta_t = self._get_action(action)
        _LOGGER.debug("Moving the probe: %s" % str((x_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, z_t)

    def get_error(self):
        dx, dz = self._get_pos_diff()  
        x = dx/self.phantom.x_border[1]
        z = dz/self.phantom.z_border[1]/2
        error = 1/2 * np.sum(np.power([x, z], 2))
        return error

    def _update_state(self):
        if self.dislocation_rng and \
           self.dislocation_rng.random() < self.probe_dislocation_prob:
            # draw a random dislocation
            x_disloc = self.dislocation_rng.choice(list(range(1, self.max_probe_dislocation+1)))
            x_disloc *= self.step_size
            self._move_focal_point_if_possible(x_t=x_disloc, z_t=0)
