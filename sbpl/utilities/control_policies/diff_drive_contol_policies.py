from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle
from builtins import range
import numpy as np
import itertools
import logging

from sbpl.utilities.control_policies.common_control_policies import statefull_branching, copy_control_policy


def gen_vw_step_choices(min_v, max_v, max_w, v_samples, w_samples_in_each_direction, enable_turn_in_place):
    '''
    A certain algorithm to get an array of reasonable combinations of v and w applicable to
    the robot.
    :param min_v, max_v, v_samples: range of linear velocities
        where v_samples number of linear velocity choices not counting the max_v:
        Example:
            min_v, max_v, v_samples = 0, 1, 4
            resulting v choices: [0, 0.25, 0.5, 0.75, 1]
    :param max_w, w_samples_in_each_direction: choices of angular velocity.
        Example:
            max_w, w_samples_in_each_direction = 1, 2
            resulting w choices: [-1, -0.5, 0, 0.5, 1]
    Note, that exhaustive choices are pruned to exclude slower than max_w rotations if linear velocity is not max.
    '''
    v_choices = np.linspace(min_v, max_v, v_samples+1)
    # make sure we include zero into the angular velocities
    w_choices = np.hstack((
        np.linspace(-max_w, 0, w_samples_in_each_direction+1),
        np.linspace(0, max_w, w_samples_in_each_direction+1)[1:]
    ))
    step_choices_candidates = list(itertools.product(v_choices, w_choices))

    step_choices = []
    for v, w in step_choices_candidates:
        if v != max_v:
            if abs(w) != max_w:
                continue
        step_choices.append((v, w))

    if enable_turn_in_place:
        # add turn in place policy here
        # to avoid stuck, temporarily only add anti-clock turn in place
        # TODO: figure out a way to add turn in place in two directions
        step_choices.append((0, 1.0))
        # step_choices.append((0, -1.0))

    step_choices = np.array(step_choices)
    return step_choices


def control_choices_diff_drive_exhaustive(max_v, max_w, forward_model, initial_state, exhausitve_dt=0.4,
                                          refine_dt=0.1, n_steps=2,
                                          w_samples_in_each_direction=15,
                                          enable_turn_in_place=True,
                                          v_samples=4):
    """
    Exhaustive set of diff drive control policies given max_v and max_w for few steps
    We apply exhaustive search every exhausitve_dt and then copy the state for the rest of the steps (with refine_dt)
    :param max_v: e.g. 1
    :param max_w: e.g. 1
    :param forward_model: A function of the form:
        next_pose, next_state = forward_model(pose, initial_states, dt, control_signals)
    :param initial_state: vector of the initial state for forward model
    :param v_samples: number of velocity samples
    :return:
    """
    exhaustive_policy = lambda pose, state, control: gen_vw_step_choices(
        min_v = 0.1*max_v,
        max_v = max_v,
        max_w = max_w,
        v_samples=v_samples,
        w_samples_in_each_direction=w_samples_in_each_direction,
        enable_turn_in_place=enable_turn_in_place)

    pose_evolution, state_evolution, control_evolution = statefull_branching(
        initial_pose = [0., 0., 0.],  # (x, y, theta)
        initial_state= initial_state,
        initial_control = [0., 0],  # (v, w)
        list_of_policies=[(exhaustive_policy, 1), (copy_control_policy, (int(exhausitve_dt/refine_dt)-1))]*n_steps,
        forward_model=forward_model,
        dt = refine_dt
    )

    logging.info('DWA diff drive exhaustive control generated %d trajectories (%d steps)' %
                 (state_evolution.shape[0], n_steps))

    control_costs = np.zeros(control_evolution.shape[0], dtype=float)
    return pose_evolution, state_evolution, control_evolution, refine_dt, control_costs


def control_choices_diff_drive_constant_distance(
        max_v, max_w, forward_model, initial_state, exhausitve_dt=0.4,
        refine_dt=0.1, n_steps=2,
        w_samples_in_each_direction=15,
        enable_turn_in_place=True):
    """
    Exhaustive set of diff drive control policies given max_v and max_w for few steps
    We apply exhaustive search every exhausitve_dt and then copy the state for the rest of the steps (with refine_dt)
    :param max_v: e.g. 1
    :param max_w: e.g. 1
    :param forward_model: A function of the form:
        next_pose, next_state = forward_model(pose, initial_states, dt, control_signals)
    :param initial_state: vector of the initial state for forward model
    :return:
    """
    exhaustive_policy = lambda pose, state, control: gen_vw_step_choices(
        min_v = 0.1*max_v,
        max_v = max_v,
        max_w = max_w,
        v_samples=3,
        w_samples_in_each_direction=w_samples_in_each_direction,
        enable_turn_in_place=enable_turn_in_place)

    pose_evolution, state_evolution, control_evolution = statefull_branching(
        initial_pose = [0., 0., 0.],  # (x, y, theta)
        initial_state= initial_state,
        initial_control = [0., 0],  # (v, w)
        list_of_policies=[(exhaustive_policy, 1), (copy_control_policy, (int(exhausitve_dt/refine_dt)-1))]*n_steps,
        forward_model=forward_model,
        dt = refine_dt
    )

    logging.info('DWA diff drive exhaustive control generated %d trajectories (%d steps)' %
                 (state_evolution.shape[0], n_steps))

    control_costs = np.zeros(control_evolution.shape[0], dtype=float)
    return pose_evolution, state_evolution, control_evolution, refine_dt, control_costs


def recovery_choices_diff_drive(forward_model, start_state, max_v, max_w):
    """
    This function use the robot's forward model and a few constraints to generate multiple turn in place policies
    :param forward_model: a function getting robot pose and state, dt and controls and returning the new pose and state
    :param start_state: the initial state of the robot
    :param max_v: linear velocity limit
    :param max_w: angular velocity limit
    :return: Generate multiple turns in place policies
    """
    refine_dt = 0.5
    rotation_angular_velocity = max_w
    # compute how many iterations is needed to make a full circle
    turn_full_circle_steps = int((2*np.pi/(rotation_angular_velocity))/refine_dt + 0.5)

    def _turn_in_place_policy(pose, state, last_controls, v, direction, angle_to_turn_to):
        last_robot_angle = pose[-1, 2]

        # In order to make trajectories the same length, we need to fill the last poses
        # with 'stay in place' for those trajectories that finished early. So here we
        # return stationary position if the robot pose and wheel are good enough.
        # The actual turn in place is happening below.
        do_nothing_command = np.array([[0., 0.]])

        if reach_end_of_turn(direction, last_robot_angle, angle_to_turn_to):
            return do_nothing_command

        # Here is the actual turn in place
        return np.array([[v, direction*rotation_angular_velocity]])

    # TODO: this is confusing, need refactoring
    def _generate_turn_in_place_trajectory(v, angle_to_turn_to, direction):
        """
        This function do planning trajectory rollout given a set of parameters (v, angle_to_turn_to, direction)
        :param v: commanding linear velocity for the in place turn
        :param angle_to_turn_to: end of turn angle
        :param direction: turning direction, i.e. clockwise or counterclockwise
        :return: robot's pose, state, control input in time series with turn_full_circle_steps steps
        """
        initial_pose = [0., 0., 0.]     # (x, y, theta)
        initial_state = [None, max_v, 0.]    # (wheel_angle, linear_velocity, ang_velocity)
        initial_control = [0., 0]  # (linear_velocity, angular_velocity)
        # control policy function handler: (control_generator, n_steps_to_apply_this_generator)
        # in this case, list below contains only 1 control policy with turn_full_circle_steps steps
        list_of_policies = [(lambda pose, state, controls: _turn_in_place_policy(
            pose, state, controls, v=v, direction=direction, angle_to_turn_to=angle_to_turn_to), turn_full_circle_steps)]
        return statefull_branching(initial_pose, initial_state, initial_control, list_of_policies, forward_model, refine_dt)

    pose_evolution = np.empty((0, turn_full_circle_steps, 3), dtype=float)
    state_evolution = np.empty((0, turn_full_circle_steps, 3), dtype=float)
    control_evolution = np.empty((0, turn_full_circle_steps, 2), dtype=float)

    num_of_end_of_turn_positions = 5    # 5 used here is an arbitrary parameter
    control_costs = []
    # clockwise and anticlockwise
    for direction in [1, -1]:
        for angle_i in range(1, num_of_end_of_turn_positions):
            angle_to_turn_to = direction*normalize_angle((float(angle_i)/num_of_end_of_turn_positions)*2*np.pi)
            pose_rotation, state_rotation, control_rotation = _generate_turn_in_place_trajectory(
                v=0.0, angle_to_turn_to=angle_to_turn_to, direction=direction)
            pose_evolution = np.vstack((pose_evolution, pose_rotation))
            state_evolution = np.vstack((state_evolution, state_rotation))
            control_evolution = np.vstack((control_evolution, control_rotation))

            # compute the cost of this turn in place policy
            part_of_angle = (float(angle_i)/num_of_end_of_turn_positions)
            current_control_costs = 1.0 * part_of_angle
            control_costs.append(current_control_costs)

    control_costs = np.array(control_costs)
    return pose_evolution, state_evolution, control_evolution, refine_dt, control_costs


def reach_end_of_turn(direction, last_robot_angle, angle_to_turn_to):
    """
    Check if robot's last angle has reached the end of the in place turn
    :param direction: turning direction clockwise is 1, counterclockwise is -1
    :param last_robot_angle: the current/last angle the robot reached
    :param angle_to_turn_to: the target angle we want the robot to turn to
    :return: True if the robot reached the target angle, otherwise False
    """
    if direction > 0:
        if last_robot_angle < 0:
            last_robot_angle += 2 * np.pi
        if angle_to_turn_to < 0:
            angle_to_turn_to += 2 * np.pi
        if last_robot_angle >= angle_to_turn_to:
            return True
        return False
    elif direction < 0:
        if last_robot_angle > 0:
            last_robot_angle -= 2 * np.pi
        if angle_to_turn_to > 0:
            angle_to_turn_to -= 2 * np.pi
        if last_robot_angle <= angle_to_turn_to:
            return True
        return False
    else:
        raise NotImplementedError("Direction should not be 0")


def control_choices_diffdrive_classic_dwa(
        max_v,
        max_w,
        forward_model,
        initial_state):
    return control_choices_diff_drive_exhaustive(
        max_v, max_w, forward_model, initial_state, exhausitve_dt=0.7,
        refine_dt=0.1, n_steps=1,
        w_samples_in_each_direction=60,
        enable_turn_in_place=True,
        v_samples=3
    )


def control_choices_diffdrive(
        policy_name,
        max_v,
        max_w,
        forward_model,
        initial_state):
    """
    A particular choice of diff drive control policies determined empirically

    :param policy_name: A string identifying the control policy (see below)
    :param max_v: e.g. 1
    :param max_w: e.g. 1
    :param forward_model: A function of the form:
        next_pose, next_state = forward_model(pose, initial_states, dt, control_signals)
    :param initial_state: vector of the initial state for forward model
    """
    policies = {'diffdrive_exhaustive': control_choices_diff_drive_exhaustive,
                'classic_dwa': control_choices_diffdrive_classic_dwa}

    return policies[policy_name](
        max_v=max_v,
        max_w=max_w,
        forward_model=forward_model,
        initial_state=initial_state)
