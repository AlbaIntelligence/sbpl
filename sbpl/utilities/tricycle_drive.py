from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from sbpl.utilities.path_tools import normalize_angle, diff_angles


class IndustrialTricycleV1Dimensions(object):
    @staticmethod
    def front_wheel_from_axis():
        # Front wheel is 964mm in front of the origin (center of rear-axle)
        return 0.964

    @staticmethod
    def max_front_wheel_angle():
        return 0.5*170*np.pi/180.

    @staticmethod
    def max_front_wheel_speed():
        return 60.*np.pi/180.  # deg per second to radians

    @staticmethod
    def max_linear_acceleration():
        return 1./2.5  # m/s per second. It needs few seconds to achieve 1 m/s speed

    @staticmethod
    def max_angular_acceleration():
        return 1./2.  # rad/s per second. It needs 2 seconds to achieve 1 rad/s rotation speed

    @staticmethod
    def front_column_model_p_gain():
        return 0.16  # P-gain value based on the fitting the RW data to this model



def industrial_tricycle_footprint(footprint_scaler):
    footprint = np.array([
        [1348.35, 0.],
        [1338.56, 139.75],
        [1306.71, 280.12],
        [1224.36, 338.62],
        [1093.81, 374.64],
        [-214.37, 374.64],
        [-313.62, 308.56],
        [-366.36, 117.44],
        [-374.01, -135.75],
        [-227.96, -459.13],
        [-156.72, -458.78],
        [759.8, -442.96],
        [849.69, -426.4],
        [1171.05, -353.74],
        [1303.15, -286.54],
        [1341.34, -118.37]
    ]) / 1000.
    assert (footprint[0, 1] == 0)  # bumper front-center has to be the first one (just so that everything is correct)
    footprint[:, 1] *= 0.95

    return footprint*footprint_scaler

