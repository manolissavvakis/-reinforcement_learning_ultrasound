from envs.us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np
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
            0: (0, 0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0, 0),  # x axis movement to the left
            2: (self.step_size,  0, 0, 0),  # x axis movement to the right
            3: (0, -self.step_size,  0, 0), # y axis movement to the left
            4: (0, self.step_size,  0, 0),  # y axis movement to the right
            5: (0, 0, -self.step_size, 0),  # z axis move upwards
            6: (0, 0, self.step_size, 0),   # z axis move downwards
        }

    def get_action_name(self, action_number):
        """
        Returns string representation for given action number
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
        }.get(action_number, None)

    def _perform_action(self, action):
        x_t, y_t, z_t, theta_t = self._get_action(action)
        _LOGGER.debug("Moving the probe: %s" % str((x_t, y_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, y_t, z_t)

    def get_error(self):
        dx, dy, dz = self._get_pos_diff()  
        x = dx/(self.phantom.x_border[1]/2)
        y = dy/(self.phantom.y_border[1]/2)
        z = dz/(self.phantom.get_main_object().belly.pos[2])
        error = 1/3 * np.sum(np.power([x, y, z], 2))
        return error
    
    def _check_step_reduction(self, action):
        step_reduction = None
        if action in range(1, 7):
            step_reduction = self._check_distance_list()
        return step_reduction, None

    def _update_state(self):
        if self.dislocation_rng and \
           self.dislocation_rng.random() < self.probe_dislocation_prob:
            # draw a random dislocation
            x_disloc = self.dislocation_rng.choice(list(range(1, self.max_probe_dislocation+1)))
            x_disloc *= self.step_size
            self._move_focal_point_if_possible(x_t=x_disloc, z_t=0)
