import random
from envs.utils import copy_and_apply
import numpy as np

class PhantomGenerator:
    """
    Base phantom generator constructor.
    """
    def __init__(self):
        pass

    def __next__(self):
        return self.get_next_phantom()

    def get_next_phantom(self):
        raise NotImplementedError


class ProbeGenerator:
    """
    Base probe generator constructor.
    """
    def __init__(self):
        pass

    def __next__(self):
        return self.get_next_probe()

    def get_next_probe(self):
        raise NotImplementedError


class ConstPhantomGenerator(PhantomGenerator):
    """
    This generator initializes a static phantom at
        the beginning of each episode.
    """
    def __init__(self, phantom):
        super().__init__()
        self.phantom = phantom

    def get_next_phantom(self):
        return self.phantom


class ConstProbeGenerator(ProbeGenerator):
    """
    This generator initializes the probe at a static point
        at the beginning of each episode.
    """
    def __init__(self, probe):
        super().__init__()
        self.probe = probe

    def get_next_probe(self):
        return self.probe

class RandomProbeGenerator(ProbeGenerator):
    """
    This generator initializes the probe at a random point
        at the beginning of each episode.
        
    :param ref_probe: reference probe
    :param object_to_align: object to align, in case angles list is None.
    :param x_pos: list of x positions to initialize the probe
    :param y_pos: list of y positions to initialize the probe
    :param focal_pos: list of focal positions to initialize the probe
    :param angle: list of angles to initialize the probe
    :param seed: seed    
    """
    def __init__(self,
                ref_probe,
                object_to_align,
                x_pos=None,
                y_pos=None,
                focal_pos=None,
                angle=None,
                seed=None
                ):
        super().__init__()
        self.ref_probe = ref_probe
        self.object_to_align = object_to_align
        if x_pos is None:
            self.x_pos = [i/1000 for i in range(-15, 16, 1)]
        else:
            self.x_pos = x_pos
        if y_pos is None:
            self.y_pos = [i/1000 for i in range(-15, 16, 1)]
        else:
            self.y_pos = y_pos
        if focal_pos is None:
            self.focal_pos = [i/1000 for i in range(10, 90, 5)]
        else:
            self.focal_pos = focal_pos
        if angle is None:
            self.angle = [self.object_to_align.angle]
        else:
            self.angle = angle
        self.rng = random.Random(seed)

    def get_next_probe(self):
    """
    Get a new random starting positions for the probe.
    
    :return: a probe
    """
        x = self.rng.choice(self.x_pos)
        y = self.rng.choice(self.y_pos)
        fd = self.rng.choice(self.focal_pos)
        a = self.rng.choice(self.angle)
        probe_pos = np.array([x, y, 0])
        return copy_and_apply(
            self.ref_probe, deep=True,
            pos=probe_pos,
            angle=a,
            focal_depth=fd
        )
