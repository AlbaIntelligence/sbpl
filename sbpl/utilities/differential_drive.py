from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from sbpl.utilities.path_tools import normalize_angle, diff_angles


def kinematic_body_pose_motion_step(pose, linear_velocity, angular_velocity, dt):
    """
    Compute a new pose of the robot based on the previous pose and velocity.

    :param pose: A (n_poses, 3) array of poses.
        The second dimension corresponds to (x_position, y_position, angle)
    :param linear_velocity: An (n_poses, ) Array indicating forward velocity
    :param angular_velocity: An (n_poses, ) Array indicating angular velocity
    :param dt: A float time-step (e.g. 0.1)
    :return: A (n_poses, 3) array of poses after 1 step.
    """
    pose_result = np.array(pose, dtype=float)
    angle = pose[..., 2]
    half_wdt = 0.5*angular_velocity*dt
    v_factor = linear_velocity * dt * np.sinc(half_wdt / np.pi)
    pose_result[..., 0] += v_factor * np.cos(angle + half_wdt)
    pose_result[..., 1] += v_factor * np.sin(angle + half_wdt)
    pose_result[..., 2] = normalize_angle(angle + angular_velocity*dt)

    return pose_result


def diff_drive_control_to_tricycle(linear_vel, angular_vel, front_wheel_angle,
                                   max_front_wheel_angle, front_wheel_from_axis_distance):
    '''
    Based on kinematic model:
    linear_vel = front_wheel_linear_velocity*cos(front_wheel_angle)
    angular_vel = front_wheel_linear_velocity*sin(front_wheel_angle)/front_wheel_from_axis_distance
    '''

    # compute desired angle of the front wheel
    if np.abs(linear_vel) < 1e-6:
        desired_angle = np.sign(angular_vel)*max_front_wheel_angle
    else:
        desired_angle = np.arctan(front_wheel_from_axis_distance*angular_vel/linear_vel)

    desired_angle = np.clip(desired_angle, -max_front_wheel_angle, max_front_wheel_angle)

    # invert kinematic model to compute lin velocity of the front wheel.
    # There two estimates will be consistent when the wheel turns to the desired angle
    # however while its turning, we need to pick some value
    # front_wheel_linear_velocity = linear_vel/cos(front_wheel_angle)
    # front_wheel_linear_velocity = angular_vel*front_wheel_from_axis_distance/sin(front_wheel_angle)
    wheel_sin = np.sin(front_wheel_angle)
    wheel_cos = np.cos(front_wheel_angle)

    # we want the center of mass to move with v and rotate with w.
    # It means we move front wheel center move with v + [w, d].
    # Wheel is nonholonomic, so to get wheel linear velocity we project (v + [w, d]) on wheel direction:
    front_wheel_linear_velocity = linear_vel*wheel_cos + angular_vel*front_wheel_from_axis_distance*wheel_sin
    front_wheel_linear_velocity = max(front_wheel_linear_velocity, 0.)

    # however if we want to rotate in place, we do not apply linear velocity if the angle is not close to the desired
    if np.abs(linear_vel) < 1e-6 and np.abs(diff_angles(front_wheel_angle, desired_angle)) > np.pi/8:
        front_wheel_linear_velocity = 0.

    return front_wheel_linear_velocity, desired_angle
