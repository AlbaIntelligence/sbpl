from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sbpl._sbpl_module
import numpy as np
import cv2

from bc_gym_planning_env.utilities.map_drawing_utils import prepare_canvas, draw_trajectory, draw_robot
from sbpl.environments import EnvNAVXYTHETALAT_InitParms, EnvironmentNAVXYTHETALAT
from bc_gym_planning_env.utilities.costmap_inflation import compute_cost_possibly_circumscribed_thresh, inflate_costmap
from sbpl.utilities.map_drawing_utils import draw_world_map_inflation


def create_planner(planner_name, environment, forward_search):
    return {
        'arastar': sbpl._sbpl_module.ARAPlanner,
        'adstar': sbpl._sbpl_module.ADPlanner,
        'anastar': sbpl._sbpl_module.anaPlanner,
        'lazy_arastar': sbpl._sbpl_module.LazyARAPlanner,
        'rstar': sbpl._sbpl_module.RSTARPlanner
    }[planner_name](environment, forward_search)


def perform_single_planning(
        planner_name,
        footprint,
        motion_primitives,
        forward_search,
        costmap,
        start_pose,
        goal_pose,
        target_v=0.65,
        target_w = 1.0,
        allocated_time=np.inf,
        cost_scaling_factor = 4.,
        debug=False,
        use_full_kernels=False):

    assert costmap.get_resolution() == motion_primitives.get_resolution()

    cost_possibly_circumscribed_thresh = compute_cost_possibly_circumscribed_thresh(
        footprint, costmap.get_resolution(),
        cost_scaling_factor=cost_scaling_factor
    )
    inflated_costmap = inflate_costmap(
        costmap, cost_scaling_factor, footprint
    )

    params = EnvNAVXYTHETALAT_InitParms()
    params.size_x = costmap.get_data().shape[1]
    params.size_y = costmap.get_data().shape[0]
    params.numThetas = motion_primitives.get_number_of_angles()
    params.cellsize_m = costmap.get_resolution()
    params.nominalvel_mpersecs = target_v
    params.timetoturn45degsinplace_secs = 1./target_w/8.
    params.obsthresh = 254
    params.costinscribed_thresh = 253
    params.costcircum_thresh = cost_possibly_circumscribed_thresh
    params.startx = 0
    params.starty = 0
    params.starttheta = 0
    params.goalx = 0
    params.goaly = 0
    params.goaltheta = 0

    environment = EnvironmentNAVXYTHETALAT(
        footprint,
        motion_primitives,
        inflated_costmap.get_data(),
        params,
        use_full_kernels=use_full_kernels
    )

    planner = create_planner(planner_name, environment, forward_search)

    start_pose = start_pose.copy()
    start_pose[:2] -= costmap.get_origin()

    goal_pose = goal_pose.copy()
    goal_pose[:2] -= costmap.get_origin()

    planner.set_start(start_pose, environment, True)
    planner.set_goal(goal_pose, environment, True)

    plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps = planner.replan(
        environment, allocated_time=allocated_time, final_epsilon=1)

    if debug:
        print("done with the solution of size=%d and sol. eps=%f in %ss" % (len(plan_xytheta_cell), solution_eps, plan_time))
        print("actual path (with intermediate poses) size=%d" % len(plan_xytheta))

        # angle_to_primitive = {}
        # for p in motion_primitives.get_primitives():
        #     try:
        #         angle_primitives = angle_to_primitive[p.starttheta_c]
        #     except KeyError:
        #         angle_primitives = {}
        #         angle_to_primitive[p.starttheta_c] = angle_primitives
        #
        #     angle_primitives[p.motprimID] = p
        #
        #
        # trajectory_through_primitives = np.array([start_pose])
        # for angle_id, motor_prim_id in actions:
        #     primitive = angle_to_primitive[angle_id][motor_prim_id]
        #
        #     # angle = primitive.get_intermediate_states()[0, 2]
        #     angle = trajectory_through_primitives[-1, 2]
        #
        #     print(np.degrees(trajectory_through_primitives[-1, 2]),
        #           np.degrees(primitive.get_intermediate_states()[0, 2]))
        #     pose = np.array([trajectory_through_primitives[-1, 0],
        #                      trajectory_through_primitives[-1, 1],
        #                      angle])
        #     dt = 0.1
        #     states_through_control = np.array([pose])
        #     for c in primitive.get_control_signals():
        #         pose = kinematic_body_pose_motion_step(
        #             pose=pose,
        #             linear_velocity=c[0],
        #             angular_velocity=c[1],
        #             dt=dt)
        #         states_through_control = np.vstack((states_through_control, [pose]))
        #
        #     trajectory_through_primitives = np.vstack((trajectory_through_primitives, states_through_control))
        #
        # plan_xytheta = trajectory_through_primitives

        params = environment.get_params()
        costmap = environment.get_costmap()

        img = prepare_canvas(costmap.shape)
        draw_world_map_inflation(img, costmap)
        for pose in plan_xytheta:
            draw_robot(img, footprint, pose, params.cellsize_m, np.zeros((2,)),
                       color=70, color_axis=(1, 2))
        draw_trajectory(img, params.cellsize_m, np.zeros((2,)), plan_xytheta)
        draw_robot(img, footprint, start_pose, params.cellsize_m, np.zeros((2,)))
        draw_robot(img, footprint, goal_pose, params.cellsize_m, np.zeros((2,)))
        magnify = 2
        img = cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("planning result", img)
        cv2.waitKey(-1)

    return plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps, environment
