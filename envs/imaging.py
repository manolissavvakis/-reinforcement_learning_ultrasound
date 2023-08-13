import numpy as np
import random
from scipy import signal, interpolate
from envs.utils import to_string, copy_and_apply


class Probe:
    """
    Probe used to scan the environment.
    
    :param pos: 3-D position of the probe.
    :param angle: angle of the probe.
    :param width: width of the probe (x axis)
    :param height: height of the probe (z axis)
    :param focal_depth: focal depth of the probe.
    """
    def __init__(self, pos, angle, width, height, focal_depth):
        self.pos = np.array(pos)
        self.angle = angle
        self.width = width  # OX
        self.height = height  # OZ
        self.focal_depth = focal_depth

    def translate(self, t):
        """
        Moves the position of the probe.

        :param t: translation vector
        :return: displaced probe (a copy)
        """
        return copy_and_apply(
            self, deep=True,
            pos=self.pos+t
        )

    def rotate(self, angle):
        """
        Rotates scanning plane of the probe.

        :param angle: rotation angle (in degrees between [-180, 180]).
        :return: rotated probe (a copy)
        """
        return copy_and_apply(
            self, deep=True,
            angle=(self.angle+angle)%360)

    def change_focal_depth(self, delta_z):
        """
        Moves upwards/downwards a focal depth of the imaging system.

        :param: delta_z: displacement of the focal point
        :return: a probe with new position of the focal point
        """
        return copy_and_apply(
            self, deep=True,
            focal_depth=self.focal_depth+delta_z)

    def get_focal_point_pos(self):
        """
        :return: 3-D array with the position of the focal point.
        """
        return np.array([self.pos[0], self.pos[1], self.focal_depth])

    def get_fov(self, phantom):
        """
        Returns the Field of View from given position and angle of
        the probe.

        :return: points, amplitudes, phantom in new FOV
        """
        ph_cpy = phantom.translate(-self.pos)
        ph_cpy = ph_cpy.rotate_xy(-self.angle)
        points, amps = ph_cpy.get_points(
            window=(self.width, self.height))
        return points, amps, ph_cpy

    def __str__(self):
        return to_string(self)

class ImagingSystem:

    """
    ImagingSystem: A class used to create images of the env.

    :param c: speed of sound
    :param fs: sampling frequency
    :param image_width: width of the output image, in [m]
    :param image_height: height of the output image, in [m]
    :param image_resolution: image resolution, (width, height) [pixels]
    :param median_filter_size: the size of median filter
    :param dr_threshold: dynamic range threshold
    :param dec: RF data decimation factor
    :param no_lines: number of lines of RF data. 
    """
    def __init__(
        self,
        c,
        fs,
        image_width,
        image_height,
        image_resolution,
        median_filter_size,
        dr_threshold,
        no_lines,
        dec=1
    ):

        self.c = c
        self.fs = fs
        self.image_width = image_width
        self.image_height = image_height
        self.image_resolution = image_resolution
        self.median_filter_size = median_filter_size
        self.dr_threshold = dr_threshold
        self.dec = dec
        self.no_lines = no_lines

    def _interp(self, data):
        input_xs = np.arange(0, data.shape[1])*(self.image_width/data.shape[1])
        input_zs = np.arange(0, data.shape[0])*(self.c/(2*self.fs))
        output_xs = np.arange(
            self.image_width,
            step=self.image_width/self.image_resolution[0])
        output_zs = np.arange(
            self.image_height,
            step=self.image_height/self.image_resolution[1])
        return interpolate.interp2d(input_xs, input_zs, data, kind="cubic")\
            (output_xs, output_zs)

    def _detect_envelope(self, data):
        return np.abs(signal.hilbert(data, axis=0))

    def _adjust_dynamic_range(self, data, dr=-60):
        nonzero_idx = data != 0
        data = 20*np.log10(np.abs(data)/np.max((np.abs(data[nonzero_idx]))))
        return np.clip(data, dr, 0)

    def image(self, rf):
        """
        Computes new B-mode image from given RF data.

        :param rf: recorded ultrasound signal to image
        :return: B-mode image with values in [0, 1]
        """
        data = rf[::self.dec, :]
        data = self._detect_envelope(data)
        data = self._adjust_dynamic_range(data, dr=self.dr_threshold)
        data = self._interp(data)
        data = signal.medfilt(data, kernel_size=self.median_filter_size)
        data = data-data.min()
        data = data/data.max()
        return data

