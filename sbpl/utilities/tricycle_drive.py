from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from bc_gym_planning_env.robot_models.differential_drive import kinematic_body_pose_motion_step


def tricycle_kinematic_step(pose, current_wheel_angle, dt, control_signals, max_front_wheel_angle, front_wheel_from_axis,
                            max_front_wheel_speed, front_column_p_gain, model_front_column_pid=True):
    '''
    :param pose: ... x 3  (x, y, angle)
    :param pose: ... x 1  (wheel_angle)
    :param dt: integration timestep
    :param control_signals: ... x 2 (wheel_v, wheel_angle) controls
    :return: ... x 3 pose and ... x 1 state (wheel_angle) after one timestep
    '''
    # rotate the front wheel first
    if model_front_column_pid:
        new_wheel_angle = tricycle_front_wheel_column_step(
            current_wheel_angle, control_signals[:, 1],
            max_front_wheel_angle, max_front_wheel_speed, front_column_p_gain,
            dt
        )
    else:
        new_wheel_angle = np.clip(control_signals[:, 1], -max_front_wheel_angle, max_front_wheel_angle)

    desired_wheel_v = control_signals[:, 0]
    linear_velocity = desired_wheel_v * np.cos(new_wheel_angle)
    angular_velocity = desired_wheel_v * np.sin(new_wheel_angle) / front_wheel_from_axis

    pose_result = kinematic_body_pose_motion_step(pose, linear_velocity, angular_velocity, dt)
    return pose_result, new_wheel_angle


def tricycle_front_wheel_column_step(current_front_wheel_angle, desired_front_wheel_angle,
                                     max_front_wheel_angle, max_front_wheel_speed, front_column_p_gain,
                                     dt):
    '''
    The model of the front wheel column which includes PID and takes into account the constraints
    on the angular velocity of the wheel and maximum angle
    '''
    # rotate the front wheel first emulating a pid controller on the front wheel with a finite rotation speed
    max_front_wheel_delta = max_front_wheel_speed*dt
    clip_first = False
    if clip_first:
        desired_wheel_delta = desired_front_wheel_angle - current_front_wheel_angle
        desired_wheel_delta = np.clip(desired_wheel_delta, -max_front_wheel_delta, max_front_wheel_delta)
        new_front_wheel_angle = current_front_wheel_angle + front_column_p_gain*desired_wheel_delta
    else:
        desired_wheel_delta = front_column_p_gain*(desired_front_wheel_angle - current_front_wheel_angle)
        desired_wheel_delta = np.clip(desired_wheel_delta, -max_front_wheel_delta, max_front_wheel_delta)
        new_front_wheel_angle = current_front_wheel_angle + desired_wheel_delta
    new_front_wheel_angle = np.clip(new_front_wheel_angle, -max_front_wheel_angle, max_front_wheel_angle)
    return new_front_wheel_angle



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

