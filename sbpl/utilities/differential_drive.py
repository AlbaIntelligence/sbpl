from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


def industrial_diffdrive_footprint(footprint_scaler):
    # Note: This is NOT the real footprint, just a mock for the simulator in order to develop a strategy
    footprint = np.array([
        [644.5, 0],
        [634.86, 61],
        [571.935, 130.54],
        [553.38, 161],
        [360.36, 186],
        # Right attachement
        [250, 186],
        [250, 186],
        [100, 186],
        [100, 186],
        # End of right attachement
        [0, 196],
        [-119.21, 190.5],
        [-173.4, 146],
        [-193, 0],
        [-173.4, -143],
        [-111.65, -246],
        [-71.57, -246],
        # Left attachement
        [100, -246],
        [100, -246],
        [250, -246],
        [250, -246],
        # End of left attachement
        [413.085, -223],
        [491.5, -204.5],
        [553, -161],
        [634.86, -62]
    ]) / 1000.

    assert (footprint[0, 1] == 0)  # bumper front-center has to be the first one (just so that everything is correct)
    return footprint*footprint_scaler
