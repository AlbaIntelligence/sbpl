from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from builtins import range
from builtins import zip
import itertools
import numpy as np

from bc_gym_planning_env.robot_models.tricycle_model import diff_drive_control_to_tricycle
from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle
from sbpl.control_policies.common_control_policies import copy_control_policy, statefull_branching


def tricycle_branching_policy(pose_evolution, state_evolution, control_evolution,
                              dt, theta_samples, v_choices, max_front_wheel_angle,
                              front_wheel_speed, front_wheel_speed_coefficients=None):
    """
    Function for statefull_branching method that generates possible front wheel angles given vehicle constraints.
    state_evolution: matrix of (N, (vel and angle)) where N is the history of the state evolution
    returns: (choices)
    """
    wheel_angle = state_evolution[-1, 0]
    # We better have odd theta_samples to generate straight movements always
    assert(theta_samples >= 2 and theta_samples % 2 == 1)
    if front_wheel_speed_coefficients is None:
        angle_choices = np.linspace(wheel_angle - dt * front_wheel_speed, wheel_angle + dt * front_wheel_speed, theta_samples)
        angle_choices = np.clip(angle_choices, -max_front_wheel_angle, +max_front_wheel_angle)
        choices = list(itertools.product(v_choices, angle_choices))
    else:
        assert len(front_wheel_speed_coefficients)==len(v_choices)
        choices = []
        for v, front_wheel_speed_coeff in zip(v_choices, front_wheel_speed_coefficients):
            angle_choices = np.linspace(wheel_angle - dt * front_wheel_speed * front_wheel_speed_coeff,
                                        wheel_angle + dt * front_wheel_speed * front_wheel_speed_coeff,
                                        theta_samples)
            angle_choices = np.clip(angle_choices, -max_front_wheel_angle, +max_front_wheel_angle)
            choices += list(itertools.product([v], angle_choices))
    result = np.array(choices)
    return result


def control_choices_tricycle_exhaustive(
        forward_model,
        wheel_angle, max_v,
        exhausitve_dt,
        refine_dt,
        n_steps,
        theta_samples,
        v_samples,
        extra_copy_n_steps,
        max_front_wheel_angle,
        max_front_wheel_speed):
    """
    Generates all possible control policies for tricycle drive given initial wheel_angle that
    can be reached in n_steps of dt length.
    """

    v_choices = np.linspace(max_v*0.5, max_v, v_samples + 1)[1:]

    exhaustive_policy = lambda pose, state, controls: tricycle_branching_policy(
        pose,
        state,
        controls,
        # this is not strictly correct because it is legally supposed to be refine_dt.
        # However it does work well in practice with real tricycles.
        # Effectively it makes tricycle think that it can turn the front wheel 7 times faster than it can.
        # Probably in real world tricycle has so much inertia that makes it move slowly but front wheel doesn't
        # have much inertia so effectively you move slower but turn wheel the same.
        exhausitve_dt,
        theta_samples=theta_samples,
        v_choices=v_choices,
        max_front_wheel_angle=max_front_wheel_angle,
        front_wheel_speed=max_front_wheel_speed
    )

    # policies = [(exhaustive_policy, 1), (copy_control_policy, (int(exhausitve_dt/refine_dt)-1))]*n_steps
    # if extra_copy_n_steps > 0:
    #     # add extra continuation to the policy to extend the horizon without switching point
    #     policies += [(copy_control_policy, extra_copy_n_steps)]
    policies = [(exhaustive_policy, 1), (copy_control_policy, extra_copy_n_steps)]

    pose_evolution, state_evolution, control_evolution = statefull_branching(
        initial_pose = [0., 0., 0.],
        initial_state= [wheel_angle, max_v, 0.],
        initial_control = [0., 0],
        list_of_policies=policies,
        forward_model=forward_model,
        dt = refine_dt
    )

    return pose_evolution, state_evolution, control_evolution, refine_dt


def control_choices_tricycle_classic_dwa(forward_model, wheel_angle, max_v,
                                         max_front_wheel_angle, max_front_wheel_speed):
    """
    This analogous to control_choices_tricycle_exhaustive, but parameters are to mimic original dwa
    that doesn't have switch points and was used for demo
    """
    return control_choices_tricycle_exhaustive(
        forward_model,
        wheel_angle, max_v,
        exhausitve_dt=0.35,
        refine_dt=0.1,
        n_steps=1,
        theta_samples=21,
        v_samples=3,
        extra_copy_n_steps=8,
        max_front_wheel_angle=max_front_wheel_angle,
        max_front_wheel_speed=max_front_wheel_speed
    )


def control_choices_tricycle_classic_dwa_agressive(forward_model, wheel_angle, max_v,
                                                   max_front_wheel_angle, max_front_wheel_speed):
    """
    forward_model: a function getting robot pose and state, dt and controls and returning
        the new pose and state
    The same as control_choices_tricycle_exhaustive, but more agressive on the front wheel column
    to avoid introducing extra control delay that comes from the sensory delay.
    """
    return control_choices_tricycle_exhaustive(
        forward_model,
        wheel_angle, max_v,
        exhausitve_dt=0.75,
        refine_dt=0.1,
        n_steps=1,
        theta_samples=21,
        v_samples=4,
        extra_copy_n_steps=8,
        max_front_wheel_angle=max_front_wheel_angle,
        max_front_wheel_speed=max_front_wheel_speed
    )


def control_choices_tricycle_constant_distance(
        forward_model, wheel_angle, max_v,
        max_front_wheel_angle, max_front_wheel_speed,
        scale_steering_speed=False):
    """
    forward_model: a function getting robot pose and state, dt and controls and returning
        the new pose and state
    This analogous to control_choices_tricycle_exhaustive, but parameters are to mimic original dwa
    that doesn't have switch points and was used for demo
    """

    # coefficient that scales the range of choices of front wheel angle at the beginning of  the first arc
    exhausitve_dt = 0.3
    # coefficient that scales the range of choices of front wheel angle for the second arc
    match_distance_exhaustive_dt = 0.75

    refine_dt = 0.1
    # resolution of choices of front wheel angle for the first arc
    theta_samples = 21
    # resolution of choices of front wheel angle for the second arc
    match_distance_theta_samples = 11

    v_samples = 3

    first_stage_steps = 8
    match_distance_steps = 8
    extra_copy_n_steps = 0

    def _tricycle_branching_policy_constant_distance(pose_evolution, state_evolution, control_evolution,
                                                     exhaustive_dt, refine_dt, theta_samples,
                                                     max_v, steps_to_evolve, front_wheel_speed):
        previous_linear_v = state_evolution[-1, 1]
        desired_distance = max_v*refine_dt*pose_evolution.shape[0]
        # TODO: compute actual distance now
        distance_traveled = previous_linear_v*refine_dt*pose_evolution.shape[0]

        new_v = (desired_distance-distance_traveled)/(steps_to_evolve*refine_dt)

        if new_v < max_v*0.1:
            # If you need a small velocity to reach required distance,
            # then you have a little space to maneuver at the need,
            # so we optimize on number of trajectories to save on computation later.
            return copy_control_policy(pose_evolution, state_evolution, control_evolution)
        else:
            # Here we need to cover decent amount of distance. We can cover it with just new_v.
            # But since we are in the beginning of the fan, we want more possibilities to finish the fan.
            # That's why we branch out not only on angles but also on velocities.
            # For example, we can choose new_v and few other velocities (e.g. new_v and max_v)
            # But here we just simply use [0.5*max_v, max_v].
            # 0.5 max_v to have a decent length even if new_v is small.
            matching_v_choices = [0.5*max_v, max_v]

            return tricycle_branching_policy(
                pose_evolution,
                state_evolution,
                control_evolution,
                dt=match_distance_exhaustive_dt,
                theta_samples=match_distance_theta_samples,
                v_choices=matching_v_choices,
                max_front_wheel_angle=max_front_wheel_angle,
                front_wheel_speed=max_front_wheel_speed,
            )

    distance_matching_policy = lambda pose, state, control: _tricycle_branching_policy_constant_distance(
        pose,
        state,
        control,
        exhausitve_dt,
        refine_dt,
        theta_samples=theta_samples,
        max_v=max_v,
        steps_to_evolve=match_distance_steps,
        front_wheel_speed=max_front_wheel_speed)

    v_choices = np.linspace(0., max_v, v_samples + 1)[1:]
    if scale_steering_speed:
        front_wheel_speed_coefficients = np.linspace(0., 1., v_samples + 1)[1:][::-1]
    else:
        front_wheel_speed_coefficients = None

    exhaustive_policy = lambda pose, state, control: tricycle_branching_policy(
        pose,
        state,
        control,
        # this is not strictly correct because it is legally supposed to be refine_dt.
        # However it does work well in practice with real tricycles.
        # Effectively it makes tricycle think that it can turn the front wheel 7 times faster than it can.
        # Probably in real world tricycle has so much inertia that makes it move slowly but front wheel doesn't
        # have much inertia so effectively you move slower but turn wheel the same.
        exhausitve_dt,
        theta_samples=theta_samples,
        v_choices=v_choices,
        max_front_wheel_angle=max_front_wheel_angle,
        front_wheel_speed=max_front_wheel_speed,
        front_wheel_speed_coefficients=front_wheel_speed_coefficients)

    policies = [(exhaustive_policy, 1), (copy_control_policy, first_stage_steps-1),
                (distance_matching_policy, 1), (copy_control_policy, match_distance_steps-1)]

    if extra_copy_n_steps > 0:
        policies += [(copy_control_policy, extra_copy_n_steps)]

    pose_evolution, state_evolution, control_evolution = statefull_branching(
        [0., 0., 0.], [wheel_angle, max_v, 0.], [0., 0.], policies, forward_model, refine_dt)

    return pose_evolution, state_evolution, control_evolution, refine_dt


def control_choices_tricycle_constant_distance_smooth(
        forward_model, wheel_angle, max_v,
        max_front_wheel_angle, max_front_wheel_speed):
    return control_choices_tricycle_constant_distance(
        forward_model, wheel_angle, max_v,
        max_front_wheel_angle, max_front_wheel_speed,
        scale_steering_speed=True
    )


def control_choices_tricycle_constant_distance_2_smooth(
        forward_model, wheel_angle, max_v,
        max_front_wheel_angle, max_front_wheel_speed):
    return control_choices_tricycle_constant_distance_2(
        forward_model, wheel_angle, max_v,
        max_front_wheel_angle, max_front_wheel_speed,
        scale_steering_speed=True
    )


def control_choices_tricycle(forward_model, policy_name, wheel_angle, max_v, max_front_wheel_angle, max_front_wheel_speed):
    """
    A particular choice of tricycle control policies determined empirically

    :param forward_model: A function of the form:
        next_pose, next_state = forward_model(pose, initial_states, dt, control_signals)
    :param policy_name: A string identifying the control policy (see below)
    :param wheel_angle: A float identifying wheel angle (in radians?)
    :param max_v: A float indicating max velocity
    :param max_front_wheel_angle: A float
    :param max_front_wheel_speed: A float
    :return: pose_evolution, state_evolution, control_evolution, refine_dt, control_costs
        pose_evolution: A (n_policies, n_steps, n_state_vars=3) array of n_polcies different poses as they evolve
        state_evolution: A (n_policies, n_steps, n_state_vars=3) array
        control_evolution: A (n_policies, n_steps, n_control_vars=2) array
        refine_dt: A float, indicating ???
        control_costs: A (n_policies, ) array of costs (appears to be unused now, but hard to tell)
    """
    policies = {'constant_distance': control_choices_tricycle_constant_distance,
                'constant_distance_2': control_choices_tricycle_constant_distance_2,
                'constant_distance_smooth': control_choices_tricycle_constant_distance_smooth,
                'constant_distance_smooth_2': control_choices_tricycle_constant_distance_2_smooth,
                'classic_dwa_agressive': control_choices_tricycle_classic_dwa_agressive,
                'classic_dwa': control_choices_tricycle_classic_dwa,
                'recovery_aggressive': control_choices_tricycle_recovery_aggressive
                }
    (pose_evolution,
     state_evolution,
     control_evolution,
     refine_dt) = policies[policy_name](forward_model,
                                        wheel_angle,
                                        max_v,
                                        max_front_wheel_angle=max_front_wheel_angle,
                                        max_front_wheel_speed=max_front_wheel_speed)

    control_costs = np.zeros(control_evolution.shape[0], dtype=float)
    return pose_evolution, state_evolution, control_evolution, refine_dt, control_costs


def recovery_choices_tricycle(forward_model, wheel_angle, max_v, max_w,
                              max_front_wheel_angle, max_front_wheel_speed, front_wheel_from_axis):
    """
    forward_model: a function getting robot pose and state, dt and controls and returning
        the new pose and state
    Generate multiple turns in place
    """
    refine_dt = 0.5
    rotation_angular_velocity = max_w*0.33
    turn_the_wheel_steps = int(1.5*(2*max_front_wheel_angle/max_front_wheel_speed)/refine_dt + 0.5)
    turn_around_steps = int((2*np.pi/rotation_angular_velocity)/refine_dt + 0.5) + turn_the_wheel_steps

    def _turn_in_place_policy(pose, state, direction, angle_to_turn_to):
        last_wheel_angle = state[-1, 0]
        last_robot_angle = pose[-1, 2]

        # In order to make trajectories the same length, we need to fill the last poses
        # with 'stay in place' for those trajectories that finished early. So here we
        # return stationary position if the robot pose and wheel are good enough.
        # The actual turn in place is happening below.
        do_nothing_command = np.array([[0., last_wheel_angle]])
        if direction > 0:
            if last_robot_angle < 0:
                last_robot_angle += 2*np.pi
            if angle_to_turn_to < 0:
                angle_to_turn_to += 2*np.pi
            if last_robot_angle >= angle_to_turn_to:
                return do_nothing_command
        elif direction < 0:
            if last_robot_angle > 0:
                last_robot_angle -= 2*np.pi
            if angle_to_turn_to > 0:
                angle_to_turn_to -= 2*np.pi
            if last_robot_angle <= angle_to_turn_to:
                return do_nothing_command
        else:
            assert False

        # Here is the actual turn in place
        v, desired_angle = diff_drive_control_to_tricycle(0.0, direction*rotation_angular_velocity,
                                                          front_wheel_angle=last_wheel_angle,
                                                          max_front_wheel_angle=max_front_wheel_angle,
                                                          front_wheel_from_axis_distance=front_wheel_from_axis)
        return np.array([[v, desired_angle]])

    def _prepare_rotation(angle_to_turn_to, direction):
        return statefull_branching(
            [0., 0., 0.], [wheel_angle, max_v, 0.], [0., 0],
            [(lambda pose, state, _: _turn_in_place_policy(
                pose, state, direction=direction, angle_to_turn_to=angle_to_turn_to), turn_around_steps)],
            forward_model, refine_dt)

    pose_evolution = np.empty((0, turn_around_steps, 3), dtype=float)
    state_evolution = np.empty((0, turn_around_steps, 3), dtype=float)
    control_evolution = np.empty((0, turn_around_steps, 2), dtype=float)

    angle_resolution = np.pi/5
    discretization = int(np.pi*2/angle_resolution)
    control_costs = []
    # clockwise and anticlockwise
    for direction in [1, -1]:
        for angle_i in range(1, discretization):
            angle_to_turn_to = direction*normalize_angle((float(angle_i)/discretization)*2*np.pi)
            pose_rotation, state_rotation, control_rotation = _prepare_rotation(angle_to_turn_to=angle_to_turn_to,
                                                                                direction=direction)
            pose_evolution = np.vstack((pose_evolution, pose_rotation))
            state_evolution = np.vstack((state_evolution, state_rotation))
            control_evolution = np.vstack((control_evolution, control_rotation))

            # penalize turning the wheel if its against the rotation
            if wheel_angle*direction < 0:
                costs_to_turn_the_wheel = 1.
            else:
                costs_to_turn_the_wheel = 0.

            # how much costs are given to turning the wheel vs turning the robot.
            # 0.5 - means equal costs
            # smaller number means that robot will prefer to turn the wheel vs turning the whole body.
            turn_wheel_weight = 0.3
            part_of_angle = (float(angle_i)/discretization)
            current_control_costs = (1-turn_wheel_weight)*part_of_angle + \
                turn_wheel_weight*costs_to_turn_the_wheel
            control_costs.append(current_control_costs)

    control_costs = np.array(control_costs)
    return pose_evolution, state_evolution, control_evolution, refine_dt, control_costs


def control_choices_tricycle_constant_distance_2(
        forward_model, wheel_angle, max_v,
        max_front_wheel_angle, max_front_wheel_speed,
        scale_steering_speed=False):
    '''
    forward_model: a function getting robot pose and state, dt and controls and returning
        the new pose and state
    This analogous to control_choices_gtx_exhaustive, but parameters are to mimic original dwa
    that doesn't have switch points and was used for demo
    '''

    # coefficient that scales the range of choices of front wheel angle at the beginning of  the first arc
    exhausitve_dt = 0.3
    # coefficient that scales the range of choices of front wheel angle for the second arc
    match_distance_exhaustive_dt = 0.75

    refine_dt = 0.1
    # resolution of choices of front wheel angle for the first arc
    theta_samples = 41
    theta_samples_2 = 3
    # resolution of choices of front wheel angle for the second arc
    match_distance_theta_samples = 3

    v_samples = 2
    first_stage_steps = 8
    match_distance_steps = 4
    extra_copy_n_steps = 0

    def _gtx_branching_policy_constant_distance(pose_evolution, state_evolution, control_evolution,
                                                exhaustive_dt, refine_dt, theta_samples,
                                                max_v, steps_to_evolve, front_wheel_speed):
        previous_linear_v = state_evolution[-1, 1]
        desired_distance = max_v*refine_dt*pose_evolution.shape[0]
        # TODO: compute actual distance now
        distance_traveled = previous_linear_v*refine_dt*pose_evolution.shape[0]

        new_v = (desired_distance-distance_traveled)/(steps_to_evolve*refine_dt)

        if new_v < max_v*0.1:
            return copy_control_policy(pose_evolution, state_evolution, control_evolution)
        else:
            matching_v_choices = [0.5*max_v, max_v]

            return tricycle_branching_policy(
                pose_evolution,
                state_evolution,
                control_evolution,
                dt=match_distance_exhaustive_dt,
                theta_samples=match_distance_theta_samples,
                v_choices=matching_v_choices,
                max_front_wheel_angle=max_front_wheel_angle,
                front_wheel_speed=max_front_wheel_speed,
            )

    distance_matching_policy = lambda pose, state, control: _gtx_branching_policy_constant_distance(
        pose,
        state,
        control,
        exhausitve_dt,
        refine_dt,
        theta_samples=theta_samples,
        max_v=max_v,
        steps_to_evolve=match_distance_steps,
        front_wheel_speed=max_front_wheel_speed)

    v_choices = np.linspace(0., max_v, v_samples + 1)[1:]
    if scale_steering_speed:
        front_wheel_speed_coefficients = np.linspace(0., 1., v_samples + 1)[1:][::-1]
    else:
        front_wheel_speed_coefficients = None

    exhaustive_policy = lambda pose, state, control: tricycle_branching_policy(
        pose,
        state,
        control,
        # this is not strictly correct because it is legally supposed to be refine_dt.
        # However it does work well in practice with real GTX.
        # Effectivelly it makes gtx think that it can turn the front wheel 7 times faster than it can.
        # Probably in real world gtx has so much inertia that makes it move slowly but front wheel doesn't
        # have much inertia so effectively you move slower but turn wheel the same.
        exhausitve_dt,
        theta_samples=theta_samples,
        v_choices=v_choices,
        max_front_wheel_angle=max_front_wheel_angle,
        front_wheel_speed=max_front_wheel_speed,
        front_wheel_speed_coefficients=front_wheel_speed_coefficients)

    exhaustive_policy_2 = lambda pose, state, control: tricycle_branching_policy(
        pose,
        state,
        control,
        # this is not strictly correct because it is legally supposed to be refine_dt.
        # However it does work well in practice with real GTX.
        # Effectivelly it makes gtx think that it can turn the front wheel 7 times faster than it can.
        # Probably in real world gtx has so much inertia that makes it move slowly but front wheel doesn't
        # have much inertia so effectively you move slower but turn wheel the same.
        exhausitve_dt,
        theta_samples=theta_samples_2,
        v_choices=v_choices,
        max_front_wheel_angle=max_front_wheel_angle,
        front_wheel_speed=max_front_wheel_speed,
        front_wheel_speed_coefficients=front_wheel_speed_coefficients)

    policies = [(exhaustive_policy, 1), (copy_control_policy, first_stage_steps-1),
                (exhaustive_policy_2, 1), (copy_control_policy, first_stage_steps - 1),
                (distance_matching_policy, 1), (copy_control_policy, match_distance_steps-1)]

    if extra_copy_n_steps > 0:
        policies += [(copy_control_policy, extra_copy_n_steps)]

    pose_evolution, state_evolution, control_evolution = statefull_branching(
        [0., 0., 0.], [wheel_angle, max_v, 0.], [0., 0.], policies, forward_model, refine_dt)

    return pose_evolution, state_evolution, control_evolution, refine_dt


def control_choices_tricycle_recovery_aggressive(
        forward_model, wheel_angle, max_v,
        max_front_wheel_angle, max_front_wheel_speed,
        scale_steering_speed=False):
    '''
    forward_model: a function getting robot pose and state, dt and controls and returning
        the new pose and state
    This analogous to control_choices_gtx_exhaustive, but parameters are to mimic original dwa
    that doesn't have switch points and was used for demo
    '''
    max_v = 0.33*max_v
    # coefficient that scales the range of choices of front wheel angle at the beginning of  the first arc
    exhausitve_dt = 0.3
    # coefficient that scales the range of choices of front wheel angle for the second arc
    exhausitve_dt_2 = 0.75

    refine_dt = 0.1
    # resolution of choices of front wheel angle for the first arc
    theta_samples = 181
    # resolution of choices of front wheel angle for the second arc
    theta_samples_2 = 11

    v_samples = 2
    first_stage_steps = 8
    match_distance_steps = 8
    extra_copy_n_steps = 0

    v_choices = np.linspace(0., max_v, v_samples + 1)[1:]
    if scale_steering_speed:
        front_wheel_speed_coefficients = np.linspace(0., 1., v_samples + 1)[1:][::-1]
    else:
        front_wheel_speed_coefficients = None

    exhaustive_policy = lambda pose, state, control: tricycle_branching_policy(
        pose,
        state,
        control,
        exhausitve_dt,
        theta_samples=theta_samples,
        v_choices=v_choices,
        max_front_wheel_angle=max_front_wheel_angle,
        front_wheel_speed=max_front_wheel_speed,
        front_wheel_speed_coefficients=front_wheel_speed_coefficients)

    exhaustive_policy_2 = lambda pose, state, control: tricycle_branching_policy(
        pose,
        state,
        control,
        exhausitve_dt_2,
        theta_samples=theta_samples_2,
        v_choices=v_choices,
        max_front_wheel_angle=max_front_wheel_angle,
        front_wheel_speed=max_front_wheel_speed,
        front_wheel_speed_coefficients=front_wheel_speed_coefficients)

    policies = [(exhaustive_policy, 1), (copy_control_policy, first_stage_steps-1),
                (exhaustive_policy_2, 1), (copy_control_policy, match_distance_steps-1)
                ]

    if extra_copy_n_steps > 0:
        policies += [(copy_control_policy, extra_copy_n_steps)]

    pose_evolution, state_evolution, control_evolution = statefull_branching(
        [0., 0., 0.], [wheel_angle, max_v, 0.], [0., 0.], policies, forward_model, refine_dt)

    return pose_evolution, state_evolution, control_evolution, refine_dt
