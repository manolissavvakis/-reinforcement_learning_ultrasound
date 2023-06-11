from envs.us_env import PhantomUsEnv
from gym import spaces
from gym.utils import EzPickle
import logging
import numpy as np
import random

_LOGGER = logging.getLogger(__name__)


class PlaneTaskUsEnv(PhantomUsEnv, EzPickle):
    def __init__(
            self,
            angle_range=None,
            probe_dislocation_prob=None,
            max_probe_dislocation=None,
            max_probe_disrotation=None,
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
        EzPickle.__init__(self, **kwargs)
        self.angle_range = angle_range
        self.max_probe_disrotation = max_probe_disrotation
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

    def _perform_action(self, action):
        x_t, y_t, z_t, theta_t = self._get_action(action)
        _LOGGER.debug("Executing action: %s" % str((x_t, y_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, y_t, z_t)
        p = self.probe.rotate(theta_t)
        self.probe = p
        
    def get_error(self):
        dx, dy, _ = self._get_pos_diff()  
        dtheta = self._get_angle_diff()
        x = dx/(self.phantom.x_border[1]/2)
        y = dy/(self.phantom.y_border[1]/2)
        theta = np.sin(np.radians(dtheta/2))
        error = 1/3 * np.sum(np.power([x, y, theta], 2))
        return error
    
    def _check_step_reduction(self, action):
        step_reduction, rot_reduction = None, None
        if action in range(1, 5):
            step_reduction = self._check_distance_list()
        elif action in range(5, 7):
            rot_reduction = self._check_rotation_list()
        return step_reduction, rot_reduction
    

    def _update_state(self):
        if self.dislocation_rng and \
           self.dislocation_rng.random() < self.probe_dislocation_prob:
            # draw, whether we rotate or accidentally move the probe
            if self.dislocation_rng.random() < 0.5:
                # Dislocate probe on along OX axis.
                x_disloc = self.dislocation_rng.choice(list(range(1, self.max_probe_dislocation+1)))
                x_disloc *= self.step_size
                self._move_focal_point_if_possible(x_t=x_disloc, z_t=0)
            else:
                disrot = self.dislocation_rng.choice(list(range(1, self.max_probe_disrotation+1)))
                disrot *= self.rot_deg
                self.probe.rotate(disrot)

    def _is_in_angle_range(self, angle):
        left, right = self.angle_range
        return left <= angle <= right
            
            
        