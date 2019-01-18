from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import cv2

from sbpl.motion_primitives import MotionPrimitives, create_linear_primitive, exhaustive_geometric_primitives
from sbpl.planners import perform_single_planning
from sbpl.utilities.costmap_2d_python import CostMap2D
from sbpl.utilities.map_drawing_utils import add_wall_to_static_map, draw_trajectory, draw_robot, prepare_canvas, \
    draw_world_map


def box_2d_planning(debug):
    costmap = CostMap2D.create_empty((4, 4), 0.2, np.zeros((2,)))

    gap = 1.2
    add_wall_to_static_map(costmap, (0, 2), (2, 2), width=0.0)
    add_wall_to_static_map(costmap, (2+gap, 2), (4, 2), width=0.0)

    footprint_width = 0.79
    footprint = np.array(
        [[0.5*footprint_width, 0.5*footprint_width],
         [-0.5 * footprint_width+1e-6, 0.5 * footprint_width],
         [-0.5 * footprint_width+1e-6, -0.5 * footprint_width+1e-6],
         [0.5 * footprint_width, -0.5 * footprint_width+1e-6]]
    )

    start_theta_discrete = 0
    number_of_intermediate_states = 2
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
            resolution=costmap.get_resolution(),
            number_of_angles=number_of_angles))

    motion_primitives = MotionPrimitives(
        resolution=costmap.get_resolution(),
        number_of_angles=1,
        mprim_list=batch
    )

    start_pose = np.array([2.3, 1.3, 0.])
    goal_pose = np.array([2.6, 2.8, 0.])
    plan_xytheta, plan_xytheta_cell, controls, plan_time, solution_eps, environment = perform_single_planning(
        planner_name='arastar',
        footprint=footprint,
        motion_primitives=motion_primitives,
        forward_search=True,
        costmap=costmap,
        start_pose=start_pose,
        goal_pose=goal_pose,
        target_v=0.65,
        target_w=1.0,
        allocated_time=np.inf,
        cost_scaling_factor=40.,
        debug=False)

    if debug:

        print(environment.xytheta_real_to_cell(start_pose))
        print(environment.xytheta_real_to_cell(goal_pose))

        print(plan_xytheta_cell)
        print("done with the solution of size=%d and sol. eps=%f" % (len(plan_xytheta_cell), solution_eps))
        print("actual path (with intermediate poses) size=%d" % len(plan_xytheta))

        params = environment.get_params()
        costmap = environment.get_costmap()
        img = prepare_canvas(costmap.shape)
        draw_world_map(img, costmap)
        for pose in plan_xytheta:
            draw_robot(img, footprint, pose, params.cellsize_m, np.zeros((2,)),
                       color=70, color_axis=(1, 2))
        draw_trajectory(img, params.cellsize_m, np.zeros((2,)), plan_xytheta)
        draw_robot(img, footprint, start_pose, params.cellsize_m, np.zeros((2,)))
        draw_robot(img, footprint, goal_pose, params.cellsize_m, np.zeros((2,)))
        magnify = 16
        img = cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("planning result", img)
        cv2.waitKey(-1)


def box_3d_planning(debug):
    costmap = CostMap2D.create_empty((4, 4), 0.05, np.zeros((2,)))

    gap = 1.4
    add_wall_to_static_map(costmap, (0, 2), (2, 2), width=0.5)
    add_wall_to_static_map(costmap, (2+gap, 2), (4, 2), width=0.5)

    footprint_width = 0.79
    footprint = np.array(
        [[0.5*footprint_width, 0.5*footprint_width],
         [-0.5 * footprint_width+1e-6, 0.5 * footprint_width],
         [-0.5 * footprint_width+1e-6, -0.5 * footprint_width+1e-6],
         [0.5 * footprint_width, -0.5 * footprint_width+1e-6]]
    )

    motion_primitives = exhaustive_geometric_primitives(
        costmap.get_resolution(), 10, 32
    )

    start_pose = np.array([2.3, 1.0, np.pi/4])
    goal_pose = np.array([2.6, 3.0, np.pi/4])
    plan_xytheta, plan_xytheta_cell, controls, plan_time, solution_eps, environment = perform_single_planning(
        planner_name='arastar',
        footprint=footprint,
        motion_primitives=motion_primitives,
        forward_search=True,
        costmap=costmap,
        start_pose=start_pose,
        goal_pose=goal_pose,
        target_v=0.65,
        target_w=1.0,
        allocated_time=np.inf,
        cost_scaling_factor=40.,
        debug=False)

    if debug:

        print(environment.xytheta_real_to_cell(start_pose))
        print(environment.xytheta_real_to_cell(goal_pose))

        print(plan_xytheta_cell)
        print("done with the solution of size=%d and sol. eps=%f" % (len(plan_xytheta_cell), solution_eps))
        print("actual path (with intermediate poses) size=%d" % len(plan_xytheta))

        params = environment.get_params()
        costmap = environment.get_costmap()
        img = prepare_canvas(costmap.shape)
        draw_world_map(img, costmap)
        for pose in plan_xytheta:
            draw_robot(img, footprint, pose, params.cellsize_m, np.zeros((2,)),
                       color=70, color_axis=(1, 2))
        draw_trajectory(img, params.cellsize_m, np.zeros((2,)), plan_xytheta)
        draw_robot(img, footprint, start_pose, params.cellsize_m, np.zeros((2,)))
        draw_robot(img, footprint, goal_pose, params.cellsize_m, np.zeros((2,)))
        magnify = 4
        img = cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("planning result", img)
        cv2.waitKey(-1)



if __name__ == '__main__':
    box_3d_planning(debug=True)
