from envs.us_env import PhantomUsEnv
import logging
import numpy as np
import random

_LOGGER = logging.getLogger(__name__)

class CombinedTaskUsEnv(PhantomUsEnv):
    def __init__(
            self,
            angle_range=None,          
            probe_dislocation_prob=None,
            max_probe_dislocation=None,
            max_probe_disrotation=None,
            dislocation_seed=None,
            **kwargs):

        super().__init__(**kwargs)
        self.angle_range = angle_range
        self.max_probe_disrot = max_probe_disrotation
        self.max_probe_dislocation = max_probe_dislocation
        self.probe_dislocation_prob = probe_dislocation_prob
        if dislocation_seed:
            self.dislocation_rng = random.Random(dislocation_seed)
        else:
            self.dislocation_rng = None

    def _perform_action(self, action):
        x_t, y_t, z_t, theta_t = self._get_action(action)
        _LOGGER.debug("Executing action: %s" % str((x_t, y_t, z_t, theta_t)))
        self._move_focal_point_if_possible(x_t, y_t, z_t)
        p = self.probe.rotate(theta_t)
        if self.angle_range is None or self._is_in_angle_range(p.angle):
            self.probe = p
    
    def get_error(self):
        dx, dy, dz = self._get_pos_diff()  
        dtheta = self._get_angle_diff()
        x = dx/(self.phantom.x_border[1]/2)
        y = dy/(self.phantom.y_border[1]/2)
        z = dz/(self.phantom.get_main_object().belly.pos[2])
        theta = np.sin(np.radians(dtheta/2))
        error = 1/4 * np.sum(np.power([x, y, z, theta], 2))
        return error
    
    def _check_step_reduction(self, action):
        step_reduction, rot_reduction = None, None
        if action in range(1, 8):
            step_reduction = self._check_distance_list()
        elif action in range(8, 10):
            rot_reduction = self._check_rotation_list()
        return step_reduction, rot_reduction    
        
    def _is_in_angle_range(self, angle):
        left, right = self.angle_range
        return left <= angle <= right        

    def _update_state(self):
        if self.dislocation_rng and self.dislocation_rng.random() < self.probe_dislocation_prob:
            # Add noise depending on which plane the action is taken.
            if self._get_action() == 1 or 2:
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
 