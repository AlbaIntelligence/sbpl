from __future__ import print_function
from __future__ import absolute_import

from builtins import range
from builtins import zip

import numpy as np

"""
Terminology used here:
                                                Tricycle
pose: The current position of the robot:        (x, y, theta)
state: The state of the robot:                  (wheel_angle, linear_velocity, ang_velocity)
control: The control signal to the robot:       (speed, target_wheel_angle)
"""


def statefull_branching(initial_pose, initial_state, initial_control, list_of_policies, forward_model, dt):
    """
    Given a matrix of initial_pose, inital_state and initial_control propagate it forward in time for n_steps using
    a function that given a state history, generates possible controls in this state

    :param initial_pose: vectory of (x, y, angle)
    :param initial_state: vector of states
    :param list_of_policies - list of elements, where every element is a pair of
        (control_generator, n_steps_to_apply_this_generator).
        control_generator: function that takes pose, state and control history returns vector of possible
            controls in this state
    :param forward_model - function that given poses, states and controls propagates the poses and states dt forward
    :param dt - timestep for forward model propagation
    :return: matrices of poses, states and controls
        (e.g. for states maxtrix shape (num_of_trajectories, num_of_timesteps, length_of_state_vector)

    Example:
        1) Tricycle drive. Pose (x, y, angle), state is [wheel_angle, linear_velocity, ang_velocity], control (v, target angle)
         initial_pose - N possible initial poses at the t=0 - matrix of (N, 1, 3) shape
         initial_state - N possible initial states at the t=0 - matrix of (N, 1, 3) shape
         control_generator - function that given current wheel angle and velocity returns possible front wheel
          angles and velocities after dt.
          Lets assume that given a particular state, control_generator generates only
          2 possible velocities and 2 possible new wheel target angles. This result to 4 possible branches given a state.
          Assume that initial state dimenstions is (1, 1, M) and n_steps is 3, then result would be
          of (4*4*4, 3, M) dimensions meaning that there are 56 possible states that system can reach in 3 steps.
    """
    new_pose, new_state, new_control = _prepare_initial_matrices(initial_pose, initial_state, initial_control)
    for control_generator, n_steps in list_of_policies:
        for step in range(n_steps):
            new_pose, new_state, new_control = _propagate_state_matrix_once(
                new_pose, new_state, new_control, control_generator, forward_model, dt)

    # inital state and poses was trivially expanded, so there is no need to return it
    return new_pose[:, 1:, :], new_state[:, 1:, :], new_control[:, 1:, :]


def _propagate_state_matrix_once(pose_matrix, state_matrix, control_matrix,
                                 control_generator, forward_model, dt):
    '''
    Helper for state evolution functions. See statefull_branching doc.
    :param state_matrix: 3d (policy count, time, state vector)
    '''
    assert(pose_matrix.shape[0:2] == state_matrix.shape[0:2])
    assert(state_matrix.shape[0:2] == control_matrix.shape[0:2])
    new_controls = []
    for pose_evolution, state_evolution, control_evolution in zip(pose_matrix, state_matrix, control_matrix):
        possible_next_controls = control_generator(pose_evolution, state_evolution, control_evolution)
        new_controls.append(possible_next_controls)

    # how many branches came out of the particular state
    branching_factors = [s.shape[0] for s in new_controls]

    pose_matrix = np.repeat(pose_matrix, branching_factors, axis=0)
    state_matrix = np.repeat(state_matrix, branching_factors, axis=0)
    control_matrix = np.repeat(control_matrix, branching_factors, axis=0)

    new_controls = np.vstack(new_controls)
    new_poses, new_state = forward_model(pose_matrix[:, -1, :], state_matrix[:, -1, :], dt, new_controls)

    def _extend_evolution_matrix(matrix, new_row):
        new_row = new_row[:, None, :]
        assert(matrix.shape[0] == new_row.shape[0])
        return np.hstack((matrix, new_row))

    new_pose_evolution = _extend_evolution_matrix(pose_matrix, new_poses)
    new_state_evolution = _extend_evolution_matrix(state_matrix, new_state)
    new_control_evolution = _extend_evolution_matrix(control_matrix, new_controls)

    return new_pose_evolution, new_state_evolution, new_control_evolution


def _prepare_initial_matrices(initial_pose, initial_state, initial_control):
    '''
    Prepare single initial pose, state and control vector to be fed to statefull_branching function.
    :param initial_pose: 1d vector of robot initial pose (e.g. [x, y, angle])
    :param initial_state: 1d vector of robot initial state (e.g. [velocity, omega])
    :param initial_control: 1d vector of robot initial control (e.g. [wheel_vel, target_angle])
    '''
    def _prepare_variable(variable):
        variable = np.asarray(variable)
        assert(variable.ndim == 1)
        return variable.reshape(1, 1, variable.shape[0])

    return _prepare_variable(initial_pose), _prepare_variable(initial_state), _prepare_variable(initial_control)


def copy_control_policy(pose_evolution, state_evolution, control_evolution):
    '''
    This policy just copies the last control forward:
    control0, -> control0,
    '''
    last_control = control_evolution[-1, :]
    result = last_control[None, :]
    return result
