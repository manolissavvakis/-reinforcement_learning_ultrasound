import gym
import numpy as np
import os
import time
import atexit
import glob
import subprocess
import math
import logging
import tempfile
from envs.utils import Config

_LOGGER = logging.getLogger(__name__)

class ConfidenceWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        config,
        working_dir=None,
        ):
        super().__init__(env)
        if working_dir is None:
            self.working_dir = tempfile.TemporaryDirectory(suffix='_confidence')
        if env.use_cache:
            self._map_cache = {} 
        self.confidence_maps = np.ndarray(shape=(2,
                                                 config.get_imaging_values('image_resolution')[1],
                                                 config.get_imaging_values('image_resolution')[0]))
        atexit.register(self._cleanup)
        self._start_sessions()
            
    def reset(self):
        o = self.env.reset()
        self.confidence_maps[0] = self._get_map(o)
        return o
    
    def step(self, action):
        if self.env._check_termination_conditions()[0]:
            raise RuntimeError("This episode is over, reset the environment.")
        self.env.current_step += 1
        self.env._perform_action(action)
        o = self.env._get_observation()
        self.confidence_maps[1] = self._get_map(o)
        
        reward = self._get_reward()
        # Update current state independently to the action
        # (for example apply shaking noise to the probe position).
        self.env._update_state()
        self.env.current_observation = o
        self.confidence_maps[0] = self.confidence_maps[1]
        info = dict(is_success = None)
        episode_over, info["is_success"] = self.env._check_termination_conditions()

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
        return o, reward, episode_over, info
        
    def _get_reward(self):
        classic_reward = self.env._get_reward()
        if self.env.out_of_bounds:
            return classic_reward
        else:
            quality_reward = np.mean(self.confidence_maps[1])
            delta_q = self._get_quality_improvement()
            a_p, a_r = 0.5, 0.2
            if not math.isclose(delta_q, 0., abs_tol = 1e-3):
                if delta_q > 0.:
                    reward_hint = -a_p
                else:
                    reward_hint = a_r
            else:
                reward_hint = 0
                
        return classic_reward + quality_reward + reward_hint
            
    def _get_quality_improvement(self):
        # Implementation: c_t+1 - c_t, where c_t is the average confidence.
        c_t, c_t1 = np.mean(self.confidence_maps, axis=(1, 2))
        delta_q = c_t1 - c_t
        return delta_q

    def _get_map(self, bmode):
        if self.use_cache:
            return self._get_cached_confidence_map(bmode)
        else:
            return self._get_confidence_map(bmode)
      
    def _get_cached_confidence_map(self, bmode):
        # Assumes, that objects in the phantom does not move (are 'static').
        state = (
            int(round(self.env.probe.pos[0], 3)*1e3),
            int(round(self.env.probe.pos[1], 3)*1e3),
            int(round(self.env.probe.focal_depth, 3)*1e3),
            int(round(self.env.probe.angle))
        )
        if state in self._map_cache:
            _LOGGER.info("Using cached confidence map value for probe state (x, y, z, theta)=%s"
                          % str(state))
        else:
            conf_map = self._get_confidence_map(bmode)
            self._map_cache[state] = conf_map
        return self._map_cache[state]
        
    def _get_confidence_map(self, bmode):
        self._assert_workers_exists()
        np.savetxt(os.path.join(self.working_dir.name, 'bmode.csv'), bmode.squeeze(), delimiter=",") 
        # Create "go file"
        open(os.path.join(self.working_dir.name, ('go_conf')), 'a').close()
        # Wait till all matlab processes finish the job.
        ready_sign_file = os.path.join(self.working_dir.name, 'ready_conf')
        i = 0
        while not os.path.isfile(ready_sign_file):
            time.sleep(1)
            if i % 10 == 0:
                self._assert_workers_exists()
            i = i+1
        
        # Confidence Map is ready
        conf_file = os.path.join(self.working_dir.name, 'confidence_map.csv')
        conf_map = np.genfromtxt(conf_file, delimiter=',')

        # Cleanup.
        os.remove(conf_file)
        os.remove(os.path.join(self.working_dir.name, 'bmode.csv'))
        go_file = os.path.join(self.working_dir.name, "go_conf")
        ready_file = os.path.join(self.working_dir.name, "ready_conf")
        if os.path.isfile(go_file):
            os.remove(go_file)
        if os.path.isfile(ready_file):
            os.remove(ready_file)

        _LOGGER.debug("Confidence Map created from bmode data")

        return conf_map

    def _start_sessions(self):
        self._pipes = self._start_session()
        timeout = 120
        print("Waiting max. %d [s] till all Confidence Map worker will be available..." % timeout)
        started_sign_pattern = os.path.join(self.working_dir.name, 'conf_started')
        while not os.path.isfile(started_sign_pattern) and timeout > 0:
            time.sleep(1)
            timeout -= 1
        if timeout <= 0:
             raise RuntimeError("Timeout waiting for MATLAB processes, stopping.")
        print("Checking state of confidence map worker...")
        self._assert_workers_exists()
        print("...OK!")
        
    def _start_session(self):
        """
        Call confidence estimation for B-mode.
        Save as bmode as a csv file. Call Matlab and estimate the Confidence
        Map, which is saved as "confidence_map.csv".
        Load the confidence map back to Python and delete the csv file.
        """
        fn_call = (
            "addpath('/home/spbtu/Manolis_Files/Thesis_Project/rlus/ConfidenceMap'), " +
            "try, " +
            ("confMap(\'%s\', \'%s\'), " %(self.working_dir.name, 'bmode.csv')) +
            "exit(0),"
            "catch ex, " +
            "fprintf('%s, %s \\n', ex.identifier, ex.message)," +
            "exit(1), " +
            "end ")

        matlab_call = ["/usr/local/MATLAB/R2018a/bin/matlab", "-nosplash", "-nodesktop", "-r", fn_call]
        pipe = subprocess.Popen(matlab_call)
        return pipe
        
    def _assert_workers_exists(self):
        if self._pipes.poll() is not None:
            raise RuntimeError("Confidence Map worker is dead! Check logs, why he has been stopped.")
        
    def _cleanup(self):
        open(os.path.join(self.working_dir.name, 'die_conf'), 'a').close()
        print("Waiting till all child processes die (conf_map)...")
        for pipe in self._pipes:
            while pipe.poll() is None:
                time.sleep(2)
        print("All subprocesses are dead now, session is closed (conf_map).")