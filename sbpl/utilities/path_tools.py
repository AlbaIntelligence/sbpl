from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle, world_to_pixel


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
