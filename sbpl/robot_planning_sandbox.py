from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


from sbpl.motion_primitives import forward_model_diffdrive_motion_primitives
from sbpl.planners import perform_single_planning
from sbpl.utilities.costmap_2d_python import CostMap2D
from sbpl.utilities.differential_drive import industrial_diffdrive_footprint
from sbpl.utilities.map_drawing_utils import add_wall_to_static_map


def run_sbpl_motiont_primitive_planning_benchmark(
        number_of_angles,
        target_v, target_w,
        w_samples_in_each_direction,
        primitives_duration,
        footprint_scale):

    test_map = CostMap2D(
        data=np.zeros((140, 70), dtype=np.uint8),
        origin=np.array([0., 0.]),
        resolution=0.03
    )
    test_map.get_data()[:] = 0
    print(test_map.get_origin(), test_map.get_data().shape)
    resolution = test_map.get_resolution()

    motion_primitives = forward_model_diffdrive_motion_primitives(
        resolution=resolution,
        number_of_angles=number_of_angles,
        target_v=target_v,
        target_w=target_w,
        w_samples_in_each_direction=w_samples_in_each_direction,
        primitives_duration=primitives_duration
    )

    add_wall_to_static_map(test_map, (0.07828677, 2.15250846), (0.07828677, 3.95250846))
    add_wall_to_static_map(test_map, (0.97828677, 2.15250846), (1.47828677, 2.15250846))

    start_pose = np.array([0.9818883, 3.52881533, -np.pi/2])
    goal_pose = np.array([0.97828677, 1.56319919, -np.pi/2])
    print(start_pose)
    print(goal_pose)

    plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps, environment = perform_single_planning(
        planner_name='arastar',
        footprint=industrial_diffdrive_footprint(footprint_scaler=footprint_scale),
        motion_primitives=motion_primitives,
        forward_search=True,
        costmap=test_map,
        start_pose=start_pose,
        goal_pose=goal_pose,
        target_v=target_v,
        target_w=target_w,
        allocated_time=np.inf,
        cost_scaling_factor=1.,
        debug=True)


if __name__ == '__main__':
    run_sbpl_motiont_primitive_planning_benchmark(
        target_v=0.65,
        target_w=1.0,
        number_of_angles=32,
        w_samples_in_each_direction=2,
        primitives_duration=5,
        footprint_scale=1.8
    )
