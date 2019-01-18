from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sbpl._sbpl_module
import numpy as np
import cv2
import tempfile
import os
import shutil

from bc_gym_planning_env.robot_models.differential_drive import kinematic_body_pose_motion_step

from bc_gym_planning_env.utilities.coordinate_transformations import from_egocentric_to_global

from sbpl.environments import EnvNAVXYTHETALAT_InitParms, EnvironmentNAVXYTHETALAT
from sbpl.motion_primitives import dump_motion_primitives
from sbpl.utilities.costmap_inflation import compute_cost_possibly_circumscribed_thresh, inflate_costmap
from sbpl.utilities.map_drawing_utils import prepare_canvas, draw_world_map, draw_robot, draw_trajectory


def create_planner(planner_name, environment, forward_search):
    return {
        'arastar': sbpl._sbpl_module.ARAPlanner,
        'adstar': sbpl._sbpl_module.ADPlanner,
        'anastar': sbpl._sbpl_module.anaPlanner,
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
        debug=False):

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
        params)

    planner = create_planner(planner_name, environment, forward_search)

    start_pose = start_pose.copy()
    start_pose[:2] -= costmap.get_origin()

    goal_pose = goal_pose.copy()
    goal_pose[:2] -= costmap.get_origin()

    planner.set_start(start_pose, environment)
    planner.set_goal(goal_pose, environment)

    plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps = planner.replan(
        environment, allocated_time=allocated_time)

    if debug:
        print("done with the solution of size=%d and sol. eps=%f" % (len(plan_xytheta_cell), solution_eps))
        print("actual path (with intermediate poses) size=%d" % len(plan_xytheta))

        angle_to_primitive = {}
        for p in motion_primitives.get_primitives():
            try:
                angle_primitives = angle_to_primitive[p.starttheta_c]
            except KeyError:
                angle_primitives = {}
                angle_to_primitive[p.starttheta_c] = angle_primitives

            angle_primitives[p.motprimID] = p


        trajectory_through_primitives = np.array([start_pose])
        for angle_id, motor_prim_id in actions:
            primitive = angle_to_primitive[angle_id][motor_prim_id]
            states = primitive.get_intermediate_states().copy()

            pose = np.array([0., 0., states[0, 2]])
            dt = 0.1
            states_through_control = np.array([pose])
            for c in primitive.get_control_signals():
                pose = kinematic_body_pose_motion_step(
                    pose=pose,
                    linear_velocity=c[0],
                    angular_velocity=c[1],
                    dt=dt)
                states_through_control = np.vstack((states_through_control, [pose]))

            states[:, :2] += trajectory_through_primitives[-1, :2]
            trajectory_through_primitives = np.vstack((trajectory_through_primitives, states))

        # dt = 0.1
        # pose = start_pose
        # for angle_id, motor_prim_id in actions:
        #     primitive = angle_to_primitive[angle_id][motor_prim_id]
        #     for c in primitive.get_control_signals():
        #         pose = kinematic_body_pose_motion_step(
        #             pose=pose,
        #             linear_velocity=c[0],
        #             angular_velocity=c[1],
        #             dt=dt)
        #         trajectory_through_primitives = np.vstack((trajectory_through_primitives, [pose]))

        plan_xytheta = trajectory_through_primitives

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
        magnify = 2
        img = cv2.resize(img, dsize=(0, 0), fx=magnify, fy=magnify, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("planning result", img)
        cv2.waitKey(-1)

    return plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps, environment
