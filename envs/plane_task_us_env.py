from envs.us_env import PhantomUsEnv
from gym import spaces
import logging
import numpy as np
import math
import random

_LOGGER = logging.getLogger(__name__)


class PlaneTaskUsEnv(PhantomUsEnv):
    """
    Wrapper for the PhantomUsEnv class.
    Action space is restricted to x, y and angle alterations.
    
    :param angle_range: range of angles (e.g. to set restrictions)
    """   
    def __init__(
            self,
            angle_range=None,
            **kwargs):

        super().__init__(**kwargs)
        self.angle_range = angle_range
        self.action_space = spaces.Discrete(len(self._get_action_map()))        

    def _get_action_map(self):
        return {
            0: (0, 0, 0, 0),  # NOP
            1: (-self.step_size, 0, 0, 0),  # x axis movement to the left
            2: (self.step_size,  0, 0, 0),  # x axis movement to the right
            3: (0, -self.step_size,  0, 0), # y axis movement to the left
            4: (0, self.step_size,  0, 0),  # y axis movement to the right
            5: (0, 0, 0, -self.rot_deg),    # clockwise rotation
            6: (0, 0, 0, self.rot_deg)      # counter-clockwise rotation
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
            5: "ROT_C",
            6: "ROT_CC",
        }.get(action_number, None)
        
    def get_error(self):
        """
        Error signal expression for this type of environment.
        """
        dx, dy, _ = self._get_pos_diff()  
        dtheta = self._get_angle_diff()
        x = dx/(self.phantom.x_border[1]/2)
        y = dy/(self.phantom.y_border[1]/2)
        theta = np.sin(np.radians(dtheta/2))
        error = 1/3 * np.sum(np.power([x, y, theta], 2))
        return error
        
    def _is_in_angle_range(self, angle):
        left, right = self.angle_range
        return left <= angle <= right
            
            
        