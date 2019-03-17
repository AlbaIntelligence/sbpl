from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import sbpl._sbpl_module

import numpy as np
import os
import cv2

from bc_gym_planning_env.robot_models.differential_drive import kinematic_body_pose_motion_step
from bc_gym_planning_env.utilities.coordinate_transformations import diff_angles, normalize_angle, \
    from_egocentric_to_global
from bc_gym_planning_env.robot_models.tricycle_model import tricycle_kinematic_step
from bc_gym_planning_env.utilities.frozenarray import freeze_array

from sbpl.utilities.control_policies.diff_drive_contol_policies import control_choices_diff_drive_exhaustive
from sbpl.utilities.control_policies.tricycle_control_policies import control_choices_tricycle_exhaustive
from sbpl.utilities.map_drawing_utils import draw_trajectory, draw_arrow
from sbpl.utilities.path_tools import pixel_to_world_centered, angle_discrete_to_cont, \
    world_to_pixel_sbpl, angle_cont_to_discrete
from sbpl.utilities.tricycle_drive import IndustrialTricycleV1Dimensions


def mprim_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../matlab/mprim'))


class MotionPrimitives(object):
    def __init__(self, resolution, number_of_angles, mprim_list):
        self._primitives = mprim_list
        self._resolution = resolution
        self._number_of_angles = number_of_angles

        self._angle_to_primitive = {}
        for p in self._primitives:
            try:
                angle_primitives = self._angle_to_primitive[p.starttheta_c]
            except KeyError:
                angle_primitives = {}
                self._angle_to_primitive[p.starttheta_c] = angle_primitives
            angle_primitives[p.motprimID] = p

    def get_primitives(self):
        return self._primitives

    def get_resolution(self):
        return self._resolution

    def get_number_of_angles(self):
        return self._number_of_angles

    def find_primitive(self, angle_id, primitive_id):
        return self._angle_to_primitive[angle_id][primitive_id]


class MotionPrimitive(object):
    """Python implementation of motion primitives.
    SBPL-generated primitives are wrapped with cpp SBPL_xytheta_mprimitiveWrapper that has the same interface"""
    def __init__(self, primitive_id, start_theta_discrete, action_cost_multiplier,
                 end_cell, intermediate_states, control_signals=None):
        self._id = primitive_id
        self._start_theta_discrete = start_theta_discrete
        self._action_cost_multiplier = action_cost_multiplier
        self._end_cell = freeze_array(np.array(end_cell))
        self._intermediate_states = freeze_array(np.array(intermediate_states))
        if control_signals is not None:
            control_signals = freeze_array(control_signals.copy())
        self._control_signals = control_signals

    @property
    def motprimID(self):
        return self._id

    @property
    def starttheta_c(self):
        return self._start_theta_discrete

    @property
    def additionalactioncostmult(self):
        return self._action_cost_multiplier

    @property
    def endcell(self):
        return self._end_cell

    @property
    def turning_radius(self):
        raise NotImplementedError("Turning radius is for non-uniform angles which are not implemented")

    def get_intermediate_states(self):
        return self._intermediate_states

    def get_control_signals(self):
        return self._control_signals


def load_motion_pritimives(mprim_filename):
    """Load motion primitives from file (uses proxy environment to use SBPL code)"""
    with open(mprim_filename) as f:
        l0 = f.readline().strip()
        if not l0.startswith('resolution_m: '):
            raise AssertionError("Invalid resolution entry")
        resolution = float(l0[len('resolution_m: '):])

        l1 = f.readline().strip()
        if not l1.startswith('numberofangles: '):
            raise AssertionError("Invalid number of angles entry")
        number_of_angles = int(l1[len('numberofangles: '):])

    params = sbpl._sbpl_module.EnvNAVXYTHETALAT_InitParms()
    params.size_x = 1
    params.size_y = 1
    params.numThetas = number_of_angles
    params.cellsize_m = resolution
    params.startx = 0
    params.starty = 0
    params.starttheta = 0

    params.goalx = 0
    params.goaly = 0
    params.goaltheta = 0
    empty_map = np.zeros((params.size_y, params.size_x), dtype=np.uint8)
    env = sbpl._sbpl_module.EnvironmentNAVXYTHETALAT(np.array([[0., 0.]]), mprim_filename, empty_map, params,
                                                     False)
    return MotionPrimitives(resolution, number_of_angles, env.get_motion_primitives_list())


def check_motion_primitives(motion_primitives):

    # check that intermediate states start at 0 with proper orientation
    for p in motion_primitives.get_primitives():
        first_state = p.get_intermediate_states()[0]
        assert first_state[0] == 0
        assert first_state[1] == 0
        d_angles = diff_angles(angle_discrete_to_cont(p.starttheta_c, motion_primitives.get_number_of_angles()), first_state[2])
        assert abs(d_angles) < 1e-4

    angle_to_primitive = {}
    for p in motion_primitives.get_primitives():
        try:
            angle_to_primitive[p.starttheta_c].append(p)
        except KeyError:
            angle_to_primitive[p.starttheta_c] = [p]

    # every angle has the same number of primitives
    motion_primitive_lenghts = np.array([len(v) for v in angle_to_primitive.values()])
    assert np.all(len(angle_to_primitive.values()[0]) == motion_primitive_lenghts)
    assert set(angle_to_primitive.keys()) == set(range(len(angle_to_primitive)))

    # angles are ordered
    angles = np.array([p.starttheta_c for p in motion_primitives.get_primitives()])
    assert np.all(np.diff(angles) >= 0)

    for angle_primitives in angle_to_primitive.values():
        # ids are ordered inside the angle
        primitive_ids = np.array([p.motprimID for p in angle_primitives])
        assert np.all(np.arange(len(angle_primitives)) == primitive_ids)

    return angle_to_primitive


def debug_motion_primitives(motion_primitives, only_zero_angle=False):
    angle_to_primitive = check_motion_primitives(motion_primitives)

    all_angles = normalize_angle(
        np.arange(motion_primitives.get_number_of_angles())*np.pi*2/motion_primitives.get_number_of_angles())
    print('All angles: %s' % all_angles)
    print('(in degrees: %s)'% np.degrees(all_angles))

    for angle_c, primitives in angle_to_primitive.items():
        print('------', angle_c)
        for p in primitives:

            if np.all(p.get_intermediate_states()[0, :2] == p.get_intermediate_states()[:, :2]):
                turn_in_place = 'turn in place'
            else:
                turn_in_place = ''
            print(turn_in_place, p.endcell, p.get_intermediate_states()[0], p.get_intermediate_states()[-1])
            final_float_state = pixel_to_world_centered(p.endcell[:2], np.zeros((2,)), motion_primitives.get_resolution())
            final_float_angle = angle_discrete_to_cont(p.endcell[2], motion_primitives.get_number_of_angles())
            try:
                np.testing.assert_array_almost_equal(final_float_state, p.get_intermediate_states()[-1][:2])
            except AssertionError:
                print("POSE DOESN'T LAND TO A CELL", final_float_state, p.get_intermediate_states()[-1])

            try:
                np.testing.assert_array_almost_equal(final_float_angle, p.get_intermediate_states()[-1][2], decimal=3)
            except AssertionError:
                print("ANGLE DOESN'T LAND TO AN ANGLE CELL", np.degrees(final_float_angle), np.degrees(p.get_intermediate_states()[-1][2]))

        endcells = np.array([p.endcell for p in primitives])
        image_half_width = np.amax(np.amax(np.abs(endcells), 0)[:2]) + 1
        zoom = 10
        img = np.full((zoom*image_half_width*2, zoom*image_half_width*2, 3), 255, dtype=np.uint8)
        resolution = motion_primitives.get_resolution()/zoom
        origin = np.array((-image_half_width*motion_primitives.get_resolution(),
                           -image_half_width*motion_primitives.get_resolution()))

        for p in primitives:
            draw_arrow(img, p.get_intermediate_states()[0], 0.7*motion_primitives.get_resolution(),
                       origin, resolution, color=(0, 0, 0))

            draw_trajectory(img, resolution, origin, p.get_intermediate_states()[:, :2], color=(0, 200, 0))
            draw_arrow(img, p.get_intermediate_states()[-1], 0.5*motion_primitives.get_resolution(),
                       origin, resolution, color=(0, 0, 200))
        cv2.imshow('a', img)
        cv2.waitKey(-1)
        if only_zero_angle:
            return


def dump_motion_primitives(motion_primitives, filename):
    check_motion_primitives(motion_primitives)

    with open(filename, 'w') as f:
        f.write('resolution_m: %.6f\n' % motion_primitives.get_resolution())
        f.write('numberofangles: %d\n' % motion_primitives.get_number_of_angles())
        f.write('totalnumberofprimitives: %d\n' % len(motion_primitives.get_primitives()))

        for p in motion_primitives.get_primitives():
            f.write('primID: %d\n' % p.motprimID)
            f.write('startangle_c: %d\n' % p.starttheta_c)
            f.write('endpose_c: %d %d %d\n' % (p.endcell[0], p.endcell[1], p.endcell[2]))
            f.write('additionalactioncostmult: %d\n' % p.additionalactioncostmult)
            states = p.get_intermediate_states()
            f.write('intermediateposes: %d\n' % len(states))
            for s in states:
                f.write('%.4f %.4f %.4f\n' % (s[0], s[1], s[2]))


def assert_motion_primitives_equal(motion_primitives_0, motion_primitives_1):
    assert motion_primitives_0.get_resolution() == motion_primitives_1.get_resolution()
    assert motion_primitives_0.get_number_of_angles() == motion_primitives_1.get_number_of_angles()
    assert len(motion_primitives_0.get_primitives()) == len(motion_primitives_1.get_primitives())

    for p0, p1 in zip(motion_primitives_0.get_primitives(), motion_primitives_1.get_primitives()):
        assert p0.motprimID == p1.motprimID
        assert p0.starttheta_c == p1.starttheta_c
        np.testing.assert_array_equal(p0.endcell, p1.endcell)
        assert p0.additionalactioncostmult == p1.additionalactioncostmult
        np.testing.assert_almost_equal(p0.get_intermediate_states(), p1.get_intermediate_states(), decimal=4)


def linear_intermediate_states(
        start_theta_discrete, endcell, number_of_intermediate_states, resolution, number_of_angles):
    assert number_of_intermediate_states >= 2, "There has to be at least start and final state"
    angle_bin_size = 2 * np.pi / number_of_angles
    interpolation = np.arange(number_of_intermediate_states)/(number_of_intermediate_states-1)
    start_angle = normalize_angle(start_theta_discrete*angle_bin_size)
    end_angle = normalize_angle(endcell[2]*angle_bin_size)
    angle_diff = normalize_angle(end_angle - start_angle)

    states = np.array([
        interpolation * endcell[0] * resolution,
        interpolation * endcell[1] * resolution,
        normalize_angle(start_angle + interpolation*angle_diff)
    ]).T
    return np.ascontiguousarray(states)


def create_linear_primitive(
        primitive_id,
        start_theta_discrete,
        action_cost_multiplier,
        end_cell,
        number_of_intermediate_states,
        resolution,
        number_of_angles):

    return MotionPrimitive(
        primitive_id=primitive_id,
        start_theta_discrete=start_theta_discrete,
        action_cost_multiplier=action_cost_multiplier,
        end_cell=end_cell,
        intermediate_states=linear_intermediate_states(
            start_theta_discrete, end_cell, number_of_intermediate_states, resolution, number_of_angles
        ))


def exhaustive_geometric_primitives(resolution, number_of_intermediate_states, number_of_angles):
    batch = []
    action_cost_multiplier = 1

    def normalize_theta_cell(theta_c):
        if theta_c > number_of_angles:
            return normalize_theta_cell(theta_c - number_of_angles)
        elif theta_c < 0:
            return normalize_theta_cell(theta_c + number_of_angles)
        return theta_c

    for start_theta_discrete in range(number_of_angles):
        for i, end_cell in enumerate([[1, 0, start_theta_discrete],
                                      [0, 1, start_theta_discrete],
                                      [-1, 0, start_theta_discrete],
                                      [0, -1, start_theta_discrete],
                                      [0, 0, normalize_theta_cell(start_theta_discrete+1)],
                                      [0, 0, normalize_theta_cell(start_theta_discrete-1)]]):
            batch.append(create_linear_primitive(
                primitive_id=i,
                start_theta_discrete=start_theta_discrete,
                action_cost_multiplier=action_cost_multiplier,
                end_cell=end_cell,
                number_of_intermediate_states=number_of_intermediate_states,
                resolution=resolution,
                number_of_angles=number_of_angles))

    return MotionPrimitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        mprim_list=batch
    )



def forward_model_diffdrive_motion_primitives(
        resolution, number_of_angles, target_v, target_w, w_samples_in_each_direction,
        primitives_duration, refine_dt=0.05):

    def forward_model(pose, state, dt, control_signals):
        new_pose = kinematic_body_pose_motion_step(
            pose=pose,
            linear_velocity=control_signals[:, 0],
            angular_velocity=control_signals[:, 1],
            dt=dt)
        next_state = state.copy()
        next_state[:, 0] = control_signals[:, 0]
        next_state[:, 1] = control_signals[:, 1]
        return new_pose, next_state

    pose_evolution, state_evolution, control_evolution, refine_dt, control_costs = control_choices_diff_drive_exhaustive(
        max_v=target_v,
        max_w=target_w,
        forward_model=forward_model,
        initial_state=(0., 0.),
        refine_dt=refine_dt,
        exhausitve_dt=refine_dt*primitives_duration,
        n_steps=1,
        w_samples_in_each_direction=w_samples_in_each_direction,
        enable_turn_in_place=True,
        v_samples=1
    )

    primitives = []
    for start_theta_discrete in range(number_of_angles):
        current_primitive_cells = []
        for primitive_id, (ego_poses, controls) in enumerate(zip(pose_evolution, control_evolution)):
            ego_poses = np.vstack(([[0., 0., 0.]], ego_poses))
            start_angle = angle_discrete_to_cont(start_theta_discrete, number_of_angles)
            poses = from_egocentric_to_global(
                ego_poses,
                ego_pose_in_global_coordinates=np.array([0., 0., start_angle]))

            # to model precision loss while converting to .mprim file, we round it here
            poses = np.around(poses, decimals=4)
            last_pose = poses[-1]
            end_cell = np.zeros((3,), dtype=int)

            center_cell_shift = pixel_to_world_centered(np.zeros((2,)), np.zeros((2,)), resolution)
            end_cell[:2] = world_to_pixel_sbpl(center_cell_shift + last_pose[:2], np.zeros((2,)), resolution)
            perfect_last_pose = np.zeros((3,), dtype=float)
            end_cell[2] = angle_cont_to_discrete(last_pose[2], number_of_angles)

            current_primitive_cells.append(tuple(end_cell))

            perfect_last_pose[:2] = pixel_to_world_centered(end_cell[:2], np.zeros((2,)), resolution)
            perfect_last_pose[2] = angle_discrete_to_cont(end_cell[2], number_of_angles)

            # penalize slow movement forward and sudden jerns
            if controls[0, 0] < target_v*0.5 or abs(controls[0, 1]) > 0.5*target_w:
                action_cost_multiplier = 100
            else:
                action_cost_multiplier = 1

            primitive = MotionPrimitive(
                primitive_id=primitive_id,
                start_theta_discrete=start_theta_discrete,
                action_cost_multiplier=action_cost_multiplier,
                end_cell=end_cell,
                intermediate_states=poses,
                control_signals=controls
            )
            primitives.append(primitive)

        # print('There are %d unique primitives from %d' % (len(set(current_primitive_cells)),
        #                                                   len(current_primitive_cells)))

    return MotionPrimitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        mprim_list=primitives
    )



def forward_model_tricycle_motion_primitives(
        resolution, number_of_angles, target_v, tricycle_angle_samples,
        primitives_duration, front_wheel_rotation_speedup, v_samples, refine_dt=0.1):

    max_front_wheel_angle=IndustrialTricycleV1Dimensions.max_front_wheel_angle()
    front_wheel_from_axis=IndustrialTricycleV1Dimensions.front_wheel_from_axis()
    max_front_wheel_speed=IndustrialTricycleV1Dimensions.max_front_wheel_speed()
    front_column_model_p_gain=IndustrialTricycleV1Dimensions.front_column_model_p_gain()

    def forward_model(pose, initial_states, dt, control_signals):
        current_wheel_angles = initial_states[:, 0]
        next_poses, next_angles = tricycle_kinematic_step(
            pose, current_wheel_angles, dt, control_signals,
            max_front_wheel_angle,
            front_wheel_from_axis,
            max_front_wheel_speed,
            front_column_model_p_gain,
            model_front_column_pid=False
        )
        next_state = initial_states.copy()
        next_state[:, 0] = next_angles[:]
        next_state[:, 1] = control_signals[:, 0]
        next_state[:, 2] = 0.  # angular velocity is ignored
        return next_poses, next_state


    pose_evolution, state_evolution, control_evolution, refine_dt = control_choices_tricycle_exhaustive(
        forward_model,
        wheel_angle=0.,
        max_v=target_v,
        exhausitve_dt=refine_dt*front_wheel_rotation_speedup,
        refine_dt=refine_dt,
        n_steps=1,
        theta_samples=tricycle_angle_samples,
        v_samples=v_samples,
        extra_copy_n_steps=primitives_duration-1,
        max_front_wheel_angle=max_front_wheel_angle,
        max_front_wheel_speed=max_front_wheel_speed
    )

    primitives = []
    for start_theta_discrete in range(number_of_angles):
        current_primitive_cells = []
        for primitive_id, (ego_poses, controls) in enumerate(zip(pose_evolution, control_evolution)):
            ego_poses = np.vstack(([[0., 0., 0.]], ego_poses))
            start_angle = angle_discrete_to_cont(start_theta_discrete, number_of_angles)
            poses = from_egocentric_to_global(
                ego_poses,
                ego_pose_in_global_coordinates=np.array([0., 0., start_angle]))

            # to model precision loss while converting to .mprim file, we round it here
            poses = np.around(poses, decimals=4)
            last_pose = poses[-1]
            end_cell = np.zeros((3,), dtype=int)

            center_cell_shift = pixel_to_world_centered(np.zeros((2,)), np.zeros((2,)), resolution)
            end_cell[:2] = world_to_pixel_sbpl(center_cell_shift + last_pose[:2], np.zeros((2,)), resolution)
            perfect_last_pose = np.zeros((3,), dtype=float)
            end_cell[2] = angle_cont_to_discrete(last_pose[2], number_of_angles)

            current_primitive_cells.append(tuple(end_cell))

            perfect_last_pose[:2] = pixel_to_world_centered(end_cell[:2], np.zeros((2,)), resolution)
            perfect_last_pose[2] = angle_discrete_to_cont(end_cell[2], number_of_angles)

            # # penalize slow movement forward and sudden jerns
            # if controls[0, 0] < target_v*0.5 or abs(controls[0, 1]) > 0.5*target_w:
            #     action_cost_multiplier = 100
            # else:
            #     action_cost_multiplier = 1

            action_cost_multiplier = 1

            primitive = MotionPrimitive(
                primitive_id=primitive_id,
                start_theta_discrete=start_theta_discrete,
                action_cost_multiplier=action_cost_multiplier,
                end_cell=end_cell,
                intermediate_states=poses,
                control_signals=controls
            )
            primitives.append(primitive)

        # print('There are %d unique primitives from %d' % (len(set(current_primitive_cells)),
        #                                                   len(current_primitive_cells)))

    return MotionPrimitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        mprim_list=primitives
    )


if __name__ == '__main__':
    # mprimtives = load_motion_pritimives(os.path.join(mprim_folder(), 'custom/gtx_32_10.mprim'))

    start_theta_discrete = 0
    number_of_intermediate_states = 3
    resolution = 0.1
    number_of_angles = 1
    batch = []
    action_cost_multiplier = 1
    for i, end_cell in enumerate([[1, 0, 0],
                                  [0, 1, 0],
                                  [-1, 0, 0],
                                  [0, -1, 0]]):
        batch.append(create_linear_primitive(
            primitive_id=i,
            start_theta_discrete=start_theta_discrete,
            action_cost_multiplier=action_cost_multiplier,
            end_cell=end_cell,
            number_of_intermediate_states=number_of_intermediate_states,
            resolution=resolution,
            number_of_angles=number_of_angles))

    motion_primitives = MotionPrimitives(
        resolution=0.1,
        number_of_angles=1,
        mprim_list=batch
    )

    debug_motion_primitives(motion_primitives)
