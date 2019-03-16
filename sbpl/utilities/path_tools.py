from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

import cv2

from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle


def get_pixel_footprint(angle, robot_footprint, map_resolution, fill=True):
    '''
    Return a binary image of a given robot footprint, in pixel coordinates,
    rotated over the appropriate angle range.
    Point (0, 0) in world coordinates is in the center of the image.
    angle_range: if a 2-tuple, the robot footprint will be rotated over this range;
        the returned footprint results from superimposing the footprint at each angle.
        If a single number, a single footprint at that angle will be returned
    robot_footprint: n x 2 numpy array with ROS-style footprint (x, y coordinates),
        in metric units, oriented at 0 angle
    map_resolution:
    :param angle Float: orientation of the robot
    :param robot_footprint array(N, 2)[float64]: n x 2 numpy array with ROS-style footprint (x, y coordinates),
        in metric units, oriented at 0 angle
    :param map_resolution Float: length in metric units of the side of a pixel
    :param fill bool: if True, the footprint will be solid; if False, only the contour will be traced
    :return array(K, M)[uint8]: image of the footprint drawn on the image in white
    '''
    assert not isinstance(angle, tuple)
    angles = [angle]
    m = np.empty((2, 2, len(angles)))  # stack of 2 x 2 matrices to rotate the footprint across all desired angles
    c, s = np.cos(angles), np.sin(angles)
    m[0, 0, :], m[0, 1, :], m[1, 0, :], m[1, 1, :] = (c, -s, s, c)
    rot_pix_footprints = np.rollaxis(np.dot(robot_footprint / map_resolution, m), -1)  # n_angles x n_footprint_corners x 2
    # From all the possible footprints, get the outer corner
    footprint_corner = np.maximum(np.amax(rot_pix_footprints.reshape(-1, 2), axis=0),
                                  -np.amin(rot_pix_footprints.reshape(-1, 2), axis=0))
    pic_half_size = np.ceil(footprint_corner).astype(np.int32)
    int_footprints = np.round(rot_pix_footprints).astype(np.int32)

    # int_footprints = np.floor(rot_pix_footprints).astype(np.int32)
    # get unique int footprints to save time; using http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    flat_int_footprints = int_footprints.reshape(len(angles), -1)
    row_view = np.ascontiguousarray(flat_int_footprints).view(np.dtype((np.void, flat_int_footprints.dtype.itemsize * flat_int_footprints.shape[1])))
    _, idx = np.unique(row_view, return_index=True)
    unique_int_footprints = int_footprints[idx]
    kernel = np.zeros(2 * pic_half_size[::-1] + 1, dtype=np.uint8)
    for f in unique_int_footprints:
        if fill:
            cv2.fillPoly(kernel, [f + pic_half_size], (255, 255, 255))
        else:
            cv2.polylines(kernel, [f + pic_half_size], 1, (255, 255, 255))
    return kernel


def world_to_pixel_floor(world_coords, origin, resolution):
    """
    Convert a numpy set of world coordinates (... x 2 numpy array)
    to pixel coordinates, given origin ((x, y) in world coordinates)
    and resolution (in world units per pixel)
    Instead of rounding, this uses floor.
    Python implementation of SBPL CONTXY2DISC
    #define CONTXY2DISC(X, CELLSIZE) (((X)>=0)?((int)((X)/(CELLSIZE))):((int)((X)/(CELLSIZE))-1))

    The returned array is of type np.int, same shape as world_coords

    :param world_coords: An Array(..., 2)[float] array of (x, y) world coordinates in meters.
    :param origin: A (x, y) point representing the location of the origin in meters.
    :param resolution: Resolution in meters/pixel.
    :returns: An Array(..., 2)[int] of (x, y) pixel coordinates
    """
    assert len(origin) == 2

    if not isinstance(world_coords, np.ndarray):
        world_coords = np.asarray(world_coords)
    if not isinstance(origin, np.ndarray):
        origin = np.asarray(origin)
    assert world_coords.shape[world_coords.ndim - 1] == 2
    # (((X)>=0)?((int)((X)/(CELLSIZE))):((int)((X)/(CELLSIZE))-1))
    return np.floor((world_coords - origin) / resolution).astype(np.int)


def world_to_pixel_sbpl(world_coords, origin, resolution):
    """
    Convert a numpy set of world coordinates (... x 2 numpy array)
    to pixel coordinates, given origin ((x, y) in world coordinates)
    and resolution (in world units per pixel)
    Instead of rounding, this uses floor.
    Python implementation of SBPL CONTXY2DISC
    #define CONTXY2DISC(X, CELLSIZE) (((X)>=0)?((int)((X)/(CELLSIZE))):((int)((X)/(CELLSIZE))-1))

    The returned array is of type np.int, same shape as world_coords

    :param world_coords: An Array(..., 2)[float] array of (x, y) world coordinates in meters.
    :param origin: A (x, y) point representing the location of the origin in meters.
    :param resolution: Resolution in meters/pixel.
    :returns: An Array(..., 2)[int] of (x, y) pixel coordinates
    """
    assert len(origin) == 2

    if not isinstance(world_coords, np.ndarray):
        world_coords = np.asarray(world_coords)
    if not isinstance(origin, np.ndarray):
        origin = np.asarray(origin)
    assert world_coords.shape[world_coords.ndim - 1] == 2

    # (((X)>=0)?((int)((X)/(CELLSIZE))):((int)((X)/(CELLSIZE))-1))

    result = ((world_coords - origin) / np.float(resolution)).astype(np.int)
    result[world_coords < 0] -= 1
    return result


def pixel_to_world_centered(pixel_coords, origin, resolution):
    '''
    Convert a numpy set of pixel coordinates (... x 2 numpy array)
    to world coordinates, given origin ((x, y) in world coordinates) and
    resolution (in world units per pixel)
    Gives center of the pixel like SBPL
    #define DISCXY2CONT(X, CELLSIZE) ((X)*(CELLSIZE) + (CELLSIZE)/2.0)

    The returned array is of type np.float32, same shape as pixel_coords
    '''
    pixel_coords = np.asarray(pixel_coords)
    assert pixel_coords.shape[pixel_coords.ndim - 1] == 2
    return pixel_coords.astype(np.float64) * resolution + np.array(origin, dtype=np.float64) + resolution*0.5


def normalize_angle_0_2pi(angle):
    # get to the range from -2PI, 2PI
    if np.abs(angle) > 2 * np.pi:
        angle = angle - ((int)(angle / (2 * np.pi))) * 2 * np.pi

    # get to the range 0, 2PI
    if angle < 0:
        angle += 2 * np.pi

    return angle


def angle_cont_to_discrete(angle, num_angles):
    '''
    Python port of ContTheta2Disc from sbpl utils (from continuous angle to one of uniform ones)
    :param angle float: float angle
    :param num_angles int: number of angles in 2*pi range
    :return: discrete angle
    '''
    theta_bin_size = 2.0 * np.pi / num_angles
    return (int)(normalize_angle_0_2pi(angle + theta_bin_size / 2.0) / (2.0 * np.pi) * num_angles)


def angle_discrete_to_cont(angle_cell, num_angles):
    '''
    Python port of DiscTheta2Cont from sbpl utils (from discrete angle to continuous)
    :param angle_cell int: discrete angle
    :param num_angles int: number of angles in 2*pi range
    :return: discrete angle
    '''
    bin_size = 2*np.pi/num_angles
    return normalize_angle(angle_cell*bin_size)


def compute_robot_area(resolution, robot_footprint):
    '''
    Computes robot footprint area in pixels
    '''
    return float(np.count_nonzero(get_pixel_footprint(0., robot_footprint, resolution)))
