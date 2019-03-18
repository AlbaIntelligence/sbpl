from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.map_drawing_utils import get_pixel_footprint_for_drawing, \
    get_drawing_coordinates_from_physical
from bc_gym_planning_env.utilities.path_tools import blit
from bc_gym_planning_env.utilities.costmap_inflation import INSCRIBED_INFLATED_OBSTACLE

"""
NOTE: All draw functions in here assument that the image is already flipped for drawing... i.e. The lowet y value 
corresponds to the last row of the image array.
"""


def draw_world_map_inflation(img, costmap_data):
    '''
    Draws obstacles and unknowns
    :param img array(W, H, 3)[uint8]: canvas to draw on
    :param costmap_data(W, H)[uint8]: costmap data
    '''
    # flip image to show it in physical orientation like rviz
    costmap = np.flipud(costmap_data)
    img[costmap > 0] = (200, 0, 0)
    img[costmap == CostMap2D.LETHAL_OBSTACLE] = (0, 255, 255)
    img[costmap == INSCRIBED_INFLATED_OBSTACLE] = (200, 200, 0)
    img[costmap == CostMap2D.NO_INFORMATION] = (70, 70, 70)


def draw_robot(image_to_draw, footprint, pose, resolution, origin, color=(30, 150, 30), color_axis=None, fill=True):
    px, py = get_drawing_coordinates_from_physical(image_to_draw.shape,
                                                   resolution,
                                                   origin,
                                                   pose[0:2])
    kernel = get_pixel_footprint_for_drawing(pose[2], footprint, resolution, fill=fill)
    blit(kernel, image_to_draw, px, py, color, axis=color_axis)
    return px, py
