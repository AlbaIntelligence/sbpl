from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import _sbpl_module
import os
import numpy as np
import cv2

from sbpl.environments import EnvironmentNAVXYTHETALAT
from sbpl.motion_primitives import mprim_folder
from sbpl.planners import create_planner
from sbpl.utilities.costmap_2d_python import CostMap2D

from sbpl.utilities.map_drawing_utils import prepare_canvas, draw_robot, draw_trajectory
from sbpl.utilities.map_drawing_utils import draw_world_map


def env_examples_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../env_examples'))


def planandnavigatexythetalat(environment_config, motion_primitives, planner_name):
    """
    Python port of planandnavigatexythetalat from sbpl test/main.cpp
    """
    true_env = EnvironmentNAVXYTHETALAT(environment_config)
    params = true_env.get_params()
    cost_obstacle, cost_inscribed, cost_possibly_circum = true_env.get_cost_thresholds()
    true_costmap = true_env.get_costmap()

    assert cost_obstacle == CostMap2D.LETHAL_OBSTACLE
    assert cost_inscribed == CostMap2D.INSCRIBED_INFLATED_OBSTACLE

    # check the start and goal obtained from the true environment
    print("start: %f %f %f, goal: %f %f %f\n" % (
        params.startx, params.starty, params.starttheta,
        params.goalx, params.goaly, params.goaltheta))

    footprint = np.zeros((4, 2))
    halfwidth = 0.01
    halflength = 0.01
    footprint[0, :] = (-halflength, -halfwidth)
    footprint[1, :] = (halflength, -halfwidth)
    footprint[2, :] = (halflength, halfwidth)
    footprint[3, :] = (-halflength, halfwidth)

    empty_map = np.zeros((params.size_y, params.size_x), dtype=np.uint8)
    env = EnvironmentNAVXYTHETALAT(footprint, motion_primitives, empty_map, params)

    # compute sensing as a square surrounding the robot with length twice that of the
    # longest motion primitive

    max_mot_prim_length_squared = 0
    for p in env.get_motion_primitives().get_primitives():
        dx = p.endcell[0]
        dy = p.endcell[1]
        primitive_length = dx * dx + dy * dy
        if primitive_length > max_mot_prim_length_squared:
            print("Found a longer motion primitive with dx = %s and dy = %s from starttheta = %s" %
                  (dx, dy, p.starttheta_c))
            max_mot_prim_length_squared = primitive_length
    max_motor_primitive_length = np.sqrt(max_mot_prim_length_squared)
    print("Maximum motion primitive length: %s" % max_motor_primitive_length)

    incremental_sensing = _sbpl_module.IncrementalSensing(10*int(max_motor_primitive_length + 0.5))

    planner = create_planner(planner_name, env, False)

    planner.set_start_goal_from_env(env)
    planner.set_planning_params(
        initial_epsilon=3.0,
        search_until_first_solution=False
    )

    params = env.get_params()
    goal_pose = np.array([params.goalx, params.goaly, params.goaltheta])
    print("goal cell: %s" % env.xytheta_real_to_cell(goal_pose))

    goaltol_x = 0.001
    goaltol_y = 0.001
    goaltol_theta = 0.001
    steps_along_the_path = 1

    start_pose = np.array((params.startx, params.starty, params.starttheta))

    # now comes the main loop
    while (abs(start_pose[0] - params.goalx) > goaltol_x or
           abs(start_pose[1] - params.goaly) > goaltol_y or
           abs(start_pose[2] - params.goaltheta) > goaltol_theta):

        changed_cells = incremental_sensing.sense_environment(start_pose, true_env, env)
        planner.apply_environment_changes(changed_cells, env)

        print("new planning...")
        plan_xytheta, plan_xytheta_cell, plan_time, solution_eps = planner.replan(env, allocated_time=10.)
        print("done with the solution of size=%d and sol. eps=%f", len(plan_xytheta_cell), solution_eps)
        print("actual path (with intermediate poses) size=%d", len(plan_xytheta))

        if len(plan_xytheta_cell):
            # move until we move into the end of motion primitive
            cell_to_move = plan_xytheta_cell[min(len(plan_xytheta_cell) - 1, steps_along_the_path)]
            print("Moving %s -> %s" % (env.xytheta_real_to_cell(start_pose), cell_to_move))
            # this check is weak since true configuration does not know the actual perimeter of the robot
            if not true_env.is_valid_configuration(cell_to_move):
                raise Exception("ERROR: robot is commanded to move into an invalid configuration according to true environment")

            new_start_pose = env.xytheta_cell_to_real(cell_to_move)
            planner.set_start(new_start_pose, env)

        else:
            new_start_pose = start_pose
            print("No move is made")

        img = prepare_canvas(true_costmap.shape)
        draw_world_map(img, true_costmap)
        draw_trajectory(img, params.cellsize_m, np.zeros((2,)), plan_xytheta)
        draw_robot(img, footprint, start_pose, params.cellsize_m, np.zeros((2,)))
        draw_robot(img, footprint, goal_pose, params.cellsize_m, np.zeros((2,)))
        cv2.imshow("current map", img)
        cv2.waitKey(-1)

        start_pose = new_start_pose

    print('Goal reached')


if __name__ == '__main__':
    planandnavigatexythetalat(
        environment_config=os.path.join(env_examples_folder(), 'nav3d/env1.cfg'),
        # environment_config=os.path.join(env_examples_folder(), 'nav3d/willow-25mm-inflated-env.cfg'),
        motion_primitives=os.path.join(mprim_folder(), 'pr2.mprim'),
        planner_name='adstar'
    )
