from envs.us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np
import math
import random

_LOGGER = logging.getLogger(__name__)


class FocalPointTaskUsEnv(PhantomUsEnv):
    """
    Wrapper for the PhantomUsEnv class.
    Action space is restricted to x, y and focal depth alterations.
    """
    
    def __init__(
            self,
            **kwargs):

        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(len(self._get_action_map())) 

    def _get_action_map(self):
        return {
            0: (0, 0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0, 0),  # x axis movement to the left
            2: (self.step_size,  0, 0, 0),  # x axis movement to the right
            3: (0, -self.step_size,  0, 0), # y axis movement to the left
            4: (0, self.step_size,  0, 0),  # y axis movement to the right
            5: (0, 0, -self.focal_step, 0),  # z axis move upwards
            6: (0, 0, self.focal_step, 0),   # z axis move downwards
        }

    def get_action_name(self, action_number):
        """
        Returns string representation for given action number
        (e.g. when logging trajectory to file).
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

    def get_error(self):
        """
        Error signal expression for this type of environment.
        """
        dx, dy, dz = self._get_pos_diff()  
        x = dx/(self.phantom.x_border[1]/2)
        y = dy/(self.phantom.y_border[1]/2)
        z = dz/(self.phantom.get_main_object().belly.pos[2])
        error = 1/3 * np.sum(np.power([x, y, z], 2))
        return error
