from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

from bc_gym_planning_env.envs.rw_corridors.tdwa_test_environments import \
    get_random_maps_squeeze_between_obstacle_in_corridor_on_path
from sbpl.motion_primitives import forward_model_tricycle_motion_primitives, debug_motion_primitives
from sbpl.planners import perform_single_planning
from sbpl.utilities.map_drawing_utils import add_wall_to_static_map, draw_robot, prepare_canvas, draw_world_map
from sbpl.utilities.path_tools import pixel_to_world_centered
from sbpl.utilities.tricycle_drive import industrial_tricycle_footprint


def run_sbpl_tricycle_motion_primitive_planning(
        number_of_angles,
        target_v, target_w,
        tricycle_angle_samples,
        primitives_duration,
        footprint_scale,
        do_debug_motion_primitives
    ):
    original_costmap, static_path, test_maps = get_random_maps_squeeze_between_obstacle_in_corridor_on_path()

    test_map = test_maps[0]
    resolution = test_map.get_resolution()

    motion_primitives = forward_model_tricycle_motion_primitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        target_v=target_v,
        tricycle_angle_samples=tricycle_angle_samples,
        primitives_duration=primitives_duration
    )

    if do_debug_motion_primitives:
        debug_motion_primitives(motion_primitives)

    add_wall_to_static_map(test_map, (1, -4.6), (1.5, -4.6))
    footprint = industrial_tricycle_footprint(footprint_scaler=footprint_scale)

    plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps, environment = perform_single_planning(
        planner_name='arastar',
        footprint=footprint,
        motion_primitives=motion_primitives,
        forward_search=False,
        costmap=test_map,
        start_pose=static_path[0],
        goal_pose=static_path[-10],
        target_v=target_v,
        target_w=target_w,
        allocated_time=np.inf,
        cost_scaling_factor=4.,
        debug=False,
        use_full_kernels=True
    )

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
        number_of_angles=180,
        tricycle_angle_samples=15,
        primitives_duration=2,
        footprint_scale=0.95,
        do_debug_motion_primitives=True
    )
