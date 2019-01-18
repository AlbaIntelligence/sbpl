from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


from bc_gym_planning_env.envs.rw_corridors.tdwa_test_environments import \
    get_random_maps_squeeze_between_obstacle_in_corridor_on_path
from sbpl.motion_primitives import forward_model_diffdrive_motion_primitives
from sbpl.planners import perform_single_planning
from sbpl.utilities.differential_drive import industrial_diffdrive_footprint
from sbpl.utilities.map_drawing_utils import add_wall_to_static_map


def run_sbpl_motiont_primitive_planning_benchmark(
        number_of_angles,
        target_v, target_w,
        w_samples_in_each_direction,
        primitives_duration,
        footprint_scale):
    original_costmap, static_path, test_maps = get_random_maps_squeeze_between_obstacle_in_corridor_on_path()

    test_map = test_maps[0]
    resolution = test_map.get_resolution()

    motion_primitives = forward_model_diffdrive_motion_primitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        target_v=target_v,
        target_w=target_w,
        w_samples_in_each_direction=w_samples_in_each_direction,
        primitives_duration=primitives_duration
    )

    add_wall_to_static_map(test_map, (1, -4.6), (1.5, -4.6))

    plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps, environment = perform_single_planning(
        planner_name='arastar',
        footprint=industrial_diffdrive_footprint(footprint_scaler=footprint_scale),
        motion_primitives=motion_primitives,
        forward_search=True,
        costmap=test_map,
        start_pose=static_path[0],
        goal_pose=static_path[-10],
        target_v=target_v,
        target_w=target_w,
        allocated_time=np.inf,
        cost_scaling_factor=4.,
        debug=True)


if __name__ == '__main__':
    run_sbpl_motiont_primitive_planning_benchmark(
        target_v=0.65,
        target_w=1.0,
        number_of_angles=32,
        w_samples_in_each_direction=4,
        primitives_duration=5,
        footprint_scale=1.8
    )
