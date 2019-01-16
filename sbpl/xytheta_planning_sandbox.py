from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np

from sbpl.motion_primitives import MotionPrimitives, create_linear_primitive
from sbpl.planners import perform_single_planning
from sbpl.utilities.costmap_2d_python import CostMap2D
from sbpl.utilities.map_drawing_utils import add_wall_to_static_map


def box_planning():
    costmap = CostMap2D.create_empty((10, 10), 0.2, np.zeros((2,)))

    gap = 1.3
    add_wall_to_static_map(costmap, (0, 5), (4, 5), width=0.0)
    add_wall_to_static_map(costmap, (4+gap, 5), (10, 5), width=0.0)

    footprint_width = 1.
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

    perform_single_planning(
        planner_name='arastar',
        footprint=footprint,
        motion_primitives=motion_primitives,
        forward_search=True,
        costmap=costmap,
        start_pose=np.array([5., 4.37, 0.]),
        goal_pose=np.array([5., 5.7, 0.]),
        target_v=0.65,
        target_w=1.0,
        allocated_time=np.inf,
        cost_scaling_factor=40.,
        debug=True)


if __name__ == '__main__':
    box_planning()
