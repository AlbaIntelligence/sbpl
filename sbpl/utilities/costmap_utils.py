from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from bc_gym_planning_env.utilities.path_tools import get_pixel_footprint
from sbpl.utilities.costmap_2d_python import CostMap2D
from sbpl.utilities.path_tools import world_to_pixel_sbpl


def pose_collides(pose, footprint, costmap_data, origin, resolution):
    '''
    Check if robot footprint at x, y (world coordinates) and
        oriented as yaw collides with lethal obstacles.
    '''
    kernel_image = get_pixel_footprint(pose[2],
                                       footprint,
                                       resolution)
    # Get the coordinates of where the footprint is inside the kernel_image (on pixel coordinates)
    kernel = np.where(kernel_image)
    # Move footprint to (x,y), all in pixel coordinates
    x, y = world_to_pixel_sbpl(pose[:2], origin, resolution)
    collisions = y + kernel[0] - kernel_image.shape[0] // 2, x + kernel[1] - kernel_image.shape[1] // 2

    # Check if the footprint pixel coordinates are valid, this is, if they are not negative and are inside the map
    good = np.logical_and(np.logical_and(collisions[0] >= 0, collisions[0] < costmap_data.shape[0]),
                          np.logical_and(collisions[1] >= 0, collisions[1] < costmap_data.shape[1]))

    # Just from the footprint coordinates that are good, check if they collide
    # with obstacles inside the map
    return bool(np.any(costmap_data[collisions[0][good],
                                    collisions[1][good]] == CostMap2D.LETHAL_OBSTACLE))
