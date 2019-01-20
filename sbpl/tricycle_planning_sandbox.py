from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import cv2

from bc_gym_planning_env.robot_models.tricycle_model import tricycle_kinematic_step
from sbpl.motion_primitives import MotionPrimitives, MotionPrimitive, debug_motion_primitives
from sbpl.planners import perform_single_planning
from sbpl.utilities.control_policies.tricycle_control_policies import control_choices_tricycle_exhaustive
from sbpl.utilities.coordinate_transformations import from_egocentric_to_global
from sbpl.utilities.map_drawing_utils import add_wall_to_static_map, draw_robot, prepare_canvas, draw_world_map

from bc_gym_planning_env.envs.rw_corridors.tdwa_test_environments import \
    get_random_maps_squeeze_between_obstacle_in_corridor_on_path
from sbpl.utilities.path_tools import pixel_to_world_centered, normalize_angle, angle_discrete_to_cont, \
    world_to_pixel_sbpl, angle_cont_to_discrete
from sbpl.utilities.tricycle_drive import industrial_tricycle_footprint, IndustrialTricycleV1Dimensions


def forward_model_tricycle_motion_primitives(
        resolution, number_of_angles, target_v, tricycle_angle_samples,
        primitives_duration, refine_dt=0.1):

    front_wheel_rotation_speedup = 10

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
        v_samples=1,
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

        print('There are %d unique primitives from %d' % (len(set(current_primitive_cells)),
                                                          len(current_primitive_cells)))

    return MotionPrimitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        mprim_list=primitives
    )


def run_sbpl_tricycle_motion_primitive_planning(
        number_of_angles,
        target_v, target_w,
        tricycle_angle_samples,
        primitives_duration,
        footprint_scale):
    original_costmap, static_path, test_maps = get_random_maps_squeeze_between_obstacle_in_corridor_on_path()

    test_map = test_maps[1]
    resolution = test_map.get_resolution()

    motion_primitives = forward_model_tricycle_motion_primitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        target_v=target_v,
        tricycle_angle_samples=tricycle_angle_samples,
        primitives_duration=primitives_duration
    )

    add_wall_to_static_map(test_map, (1, -4.6), (1.5, -4.6))
    footprint = industrial_tricycle_footprint(footprint_scaler=footprint_scale)

    plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps, environment = perform_single_planning(
        planner_name='arastar',
        footprint=footprint,
        motion_primitives=motion_primitives,
        forward_search=True,
        costmap=test_map,
        start_pose=static_path[0],
        goal_pose=static_path[-10],
        target_v=target_v,
        target_w=target_w,
        allocated_time=np.inf,
        cost_scaling_factor=4.,
        debug=False)

    params = environment.get_params()
    costmap = environment.get_costmap()

    img = prepare_canvas(costmap.shape)
    draw_world_map(img, costmap)
    start_pose = static_path[0]
    start_pose[:2] -= test_map.get_origin()
    goal_pose = static_path[-10]
    goal_pose[:2] -= test_map.get_origin()

    trajectory_through_primitives = np.array([start_pose])

    plan_xytheta_cell = np.vstack(([environment.xytheta_real_to_cell(start_pose)], plan_xytheta_cell))
    for i in range(len(actions)):
        angle_c, motor_prim_id = actions[i]
        collisions = environment.get_primitive_collision_pixels(angle_c, motor_prim_id)
        pose_cell = plan_xytheta_cell[i]
        assert pose_cell[2] == angle_c
        collisions[:, 0] += pose_cell[0]
        collisions[:, 1] += pose_cell[1]

        primitive_start = pixel_to_world_centered(pose_cell[:2], np.zeros((2,)), test_map.get_resolution())
        primitive = motion_primitives.find_primitive(angle_c, motor_prim_id)
        primitive_states = primitive.get_intermediate_states().copy()
        primitive_states[:, :2] += primitive_start

        trajectory_through_primitives = np.vstack((trajectory_through_primitives, primitive_states))

        # img = np.flipud(img)
        # img[collisions[:, 1], collisions[:, 0], 1] = 70
        # img[pose_cell[1], pose_cell[0], :] = 255
        # img = np.flipud(img)
        #
        # magnify = 2
        # cv2.imshow("planning result",
        #            cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(-1)
        #
        # for pose in primitive_states:
        #     draw_robot(img, footprint, pose, params.cellsize_m, np.zeros((2,)),
        #                color=70, color_axis=(0, 1))
        #
        # cv2.imshow("planning result",
        #            cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(-1)

    for pose in trajectory_through_primitives:
        draw_robot(img, footprint, pose, params.cellsize_m, np.zeros((2,)),
                   color=70, color_axis=(1, 2))

    # draw_trajectory(img, params.cellsize_m, np.zeros((2,)), plan_xytheta)
    draw_robot(img, footprint, start_pose, params.cellsize_m, np.zeros((2,)))
    draw_robot(img, footprint, goal_pose, params.cellsize_m, np.zeros((2,)))
    magnify = 2
    img = cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("planning result", img)
    cv2.waitKey(-1)


if __name__ == '__main__':
    run_sbpl_tricycle_motion_primitive_planning(
        target_v=0.65,
        target_w=1.0,
        number_of_angles=32,
        tricycle_angle_samples=5,
        primitives_duration=8,
        footprint_scale=0.95
    )
