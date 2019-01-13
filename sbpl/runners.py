from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import _sbpl_module
import os
import numpy as np
import matplotlib.pyplot as plt



def create_planner(planner_name, environment, forward_search):
    return {
        'arastar': _sbpl_module.ARAPlanner,
        'adstar': _sbpl_module.ADPlanner,
        'anastar': _sbpl_module.anaPlanner,
    }[planner_name](environment, forward_search)



def mprim_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../matlab/mprim'))


def env_examples_folder():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(current_dir, '../env_examples'))


if __name__ == '__main__':
    true_env = _sbpl_module.EnvironmentNAVXYTHETALAT(
        os.path.join(env_examples_folder(), 'nav3d/env1.cfg')
        # os.path.join(env_examples_folder(), 'nav3d/willow-25mm-inflated-env.cfg')
    )
    params = true_env.get_params()

    # check the start and goal obtained from the true environment
    print("start: %f %f %f, goal: %f %f %f\n" % (
        params.startx, params.starty, params.starttheta,
        params.goalx, params.goaly, params.goaltheta))

    # costmap = true_env.get_costmap()
    # plt.imshow(costmap, vmin=0, vmax=254)
    # plt.ylim([0, costmap.shape[0]])
    # plt.xlim([0, costmap.shape[1]])
    # plt.show()

    footprint = np.zeros((4, 2))
    halfwidth = 0.01
    halflength = 0.01
    footprint[0, :] = (-halflength, -halfwidth)
    footprint[1, :] = (halflength, -halfwidth)
    footprint[2, :] = (halflength, halfwidth)
    footprint[3, :] = (-halflength, halfwidth)

    motion_primitives = os.path.join(mprim_folder(), 'pr2.mprim')
    # motion_primitives = os.path.join(mprim_folder(), 'pr2_10cm.mprim')

    empty_map = np.zeros((params.size_y, params.size_x), dtype=np.uint8)
    env = _sbpl_module.EnvironmentNAVXYTHETALAT(footprint, motion_primitives, empty_map, params)

    # compute sensing as a square surrounding the robot with length twice that of the
    # longest motion primitive

    primitives = env.get_motion_primitives()
    max_mot_prim_length_squared = 0
    for p in env.get_motion_primitives():
        dx = p.endcell[0]
        dy = p.endcell[1]
        primitive_length = dx * dx + dy * dy
        if primitive_length > max_mot_prim_length_squared:
            print("Found a longer motion primitive with dx = %s and dy = %s from starttheta = %s" %
                  (dx, dy, p.starttheta_c))
            max_mot_prim_length_squared = primitive_length
    max_motor_primitive_length = np.sqrt(max_mot_prim_length_squared)
    print("Maximum motion primitive length: %s" % max_motor_primitive_length)

    sensing_range = int(max_motor_primitive_length + 0.5)

    planner = create_planner("arastar", env, False)
    planner.set_start_goal_from_env(env)
    planner.set_planning_params(
        initial_epsilon=3.0,
        search_until_first_solution=False
    )
    # planner = create_planner("adstar", env, False)
    # planner = create_planner("anastar", env, False)

    _sbpl_module.planandnavigatexythetalat(
        # os.path.join(env_examples_folder(), 'nav3d/willow-25mm-inflated-env.cfg'),
        true_env,
        env,
        planner,
        sensing_range
    )
