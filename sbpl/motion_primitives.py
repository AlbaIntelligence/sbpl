from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import _sbpl_module

import numpy as np
import os


def mprim_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../matlab/mprim'))


def load_motion_pritimives(mprim_filename):
    """Load motion primitives from file (uses proxy environment to use SBPL code)"""
    with open(mprim_filename) as f:
        l0 = f.readline().strip()
        if not l0.startswith('resolution_m: '):
            raise AssertionError("Invalid resolution entry")
        resolution = float(l0[len('resolution_m: '):])

        l1 = f.readline().strip()
        if not l1.startswith('numberofangles: '):
            raise AssertionError("Invalid number of angles entry")
        number_of_angles = int(l1[len('numberofangles: '):])

    params = _sbpl_module.EnvNAVXYTHETALAT_InitParms()
    params.size_x = 1
    params.size_y = 1
    params.numThetas = number_of_angles
    params.cellsize_m = resolution
    params.startx = 0
    params.starty = 0
    params.starttheta = 0

    params.goalx = 0
    params.goaly = 0
    params.goaltheta = 0
    empty_map = np.zeros((params.size_y, params.size_x), dtype=np.uint8)
    env = _sbpl_module.EnvironmentNAVXYTHETALAT(np.array([[0., 0.]]), mprim_filename, empty_map, params)
    return env.get_motion_primitives()


def display_motion_primitives(motion_primitives):
    pass


if __name__ == '__main__':
    mprimtives = load_motion_pritimives(os.path.join(mprim_folder(), 'pr2.mprim'))

    for p in mprimtives:
        print(p.get_intermediate_states())
