from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle


def angle_cont_to_discrete(angle, num_angles):
    '''
    Python port of ContTheta2Disc from sbpl utils (from continuous angle to one of uniform ones)
    :param angle float: float angle
    :param num_angles int: number of angles in 2*pi range
    :return: discrete angle
    '''
    theta_bin_size = 2.0 * np.pi / num_angles
    return (int)((normalize_angle(angle + theta_bin_size / 2.0) + 2*np.pi) / (2.0 * np.pi) * num_angles)


def angle_discrete_to_cont(angle_cell, num_angles):
    '''
    Python port of DiscTheta2Cont from sbpl utils (from discrete angle to continuous)
    :param angle_cell int: discrete angle
    :param num_angles int: number of angles in 2*pi range
    :return: discrete angle
    '''
    bin_size = 2*np.pi/num_angles
    return normalize_angle(angle_cell*bin_size)
