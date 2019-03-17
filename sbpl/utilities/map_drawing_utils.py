from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.map_drawing_utils import get_pixel_footprint_for_drawing
from bc_gym_planning_env.utilities.path_tools import blit
from bc_gym_planning_env.utilities.costmap_inflation import INSCRIBED_INFLATED_OBSTACLE

from sbpl.utilities.path_tools import world_to_pixel_floor


"""
NOTE: All draw functions in here assument that the image is already flipped for drawing... i.e. The lowet y value 
corresponds to the last row of the image array.
"""


def get_drawing_coordinates_from_physical_floor(map_shape, resolution, origin, physical_coords, enforce_bounds=False):
    '''
    :param physical_coords: either (x, y)  or n x 2 array of (x, y), in physical units
    :param enforce_bounds: Can be:
        False: Allow points to be outside range of costmap
        True: Raise an error if points fall out of costmap
        'filter': Filter out points which fall out of costmap.
    :return: same in coordinates suitable for drawing (y axis is flipped)
    '''
    assert enforce_bounds in (True, False, 'filter')
    physical_coords = np.array(physical_coords)
    assert physical_coords.ndim <= 2
    assert physical_coords.shape[physical_coords.ndim - 1] == 2
    assert np.array(map_shape).ndim == 1

    pixel_coords = world_to_pixel_floor(physical_coords, origin, resolution)
    # flip the y because we flip image for display
    pixel_coords[..., 1] = map_shape[0] - 1 - pixel_coords[..., 1]

    if enforce_bounds and (not (pixel_coords < map_shape[1::-1]).all() or (np.amin(pixel_coords) < 0)):
        raise IndexError("Point %s, in pixels (%s) is outside the map (shape %s)." % (physical_coords, pixel_coords, map_shape))
    return pixel_coords


def draw_trajectory(array_to_draw, resolution, origin, trajectory, color=(0, 255, 0),
                    enforce_bounds=False, thickness=1):
    if len(trajectory) == 0:
        return
    drawing_coords = get_drawing_coordinates_from_physical_floor(
        array_to_draw.shape,
        resolution,
        origin,
        trajectory[:, :2],
        enforce_bounds=enforce_bounds)

    cv2.polylines(array_to_draw, [drawing_coords], False, color, thickness=thickness)


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


def draw_wide_path(img, path, robot_width, origin, resolution, color=(220, 220, 220)):
    """
    Draw a path as a tube to follow
    :param img array(N, M, 3)[uint8]: BGR image on which to draw (mutates image)
    :param path array(K, 3)[float]: array of (x, y, angle) of the path
    :param robot_width float: robot's width in meters
    :param origin array(2)[float]: x, y origin of the image
    :param resolution float: resolution of the costmap in meters
    :param color tuple[int]: BGR color tuple
    """
    drawing_coords = get_drawing_coordinates_from_physical_floor(
        img.shape,
        resolution,
        origin,
        path[:, :2],
        enforce_bounds=False)

    cv2.polylines(img, [drawing_coords], False, color, thickness=int(robot_width / resolution))


def draw_robot(image_to_draw, footprint, pose, resolution, origin, color=(30, 150, 30), color_axis=None, fill=True):
    px, py = get_drawing_coordinates_from_physical_floor(image_to_draw.shape,
                                                         resolution,
                                                         origin,
                                                   pose[0:2])
    kernel = get_pixel_footprint_for_drawing(pose[2], footprint, resolution, fill=fill)
    blit(kernel, image_to_draw, px, py, color, axis=color_axis)
    return px, py
