

def run_sbpl_motiont_primitive_planning_benchmark(config):
    original_costmap, static_path, test_maps = get_random_maps_squeeze_between_obstacle_in_corridor_on_path()
    robot = create_robot(config['robot_name'], footprint_scale=config['footprint_scale'])

    resolution = test_maps[0].get_resolution()

    motion_primitives = forward_model_diffdrive_motion_primitives(
        resolution=resolution,
        number_of_angles=config['number_of_angles'],
        target_v=config['target_v'],
        target_w=config['target_w'],
        w_samples_in_each_direction=config['w_samples_in_each_direction'],
        primitives_duration=config['primitives_duration']
    )

    def _run_planning(test_map):
        add_wall_to_static_map(test_map, (1, -4.6), (1.5, -4.6))

        plan_xytheta, plan_xytheta_cell, actions, plan_time, solution_eps, environment = perform_single_planning(
            planner_name='arastar',
            footprint=robot.get_footprint(),
            motion_primitives=motion_primitives,
            forward_search=True,
            costmap=test_map,
            start_pose=static_path[0],
            goal_pose=static_path[-10],
            target_v=config['target_v'],
            target_w=config['target_w'],
            allocated_time=np.inf,
            cost_scaling_factor=4.,
            debug=True)

        return len(plan_xytheta) > 0, plan_time

    def _progress_callback(done, total):
        print("Done %d out of %d" % (done, total))

    start_time = time.time()
    results = multiprocessing_map(_run_planning, test_maps, progress_callback=_progress_callback, n_workers=1)
    successes = [r[0] for r in results]
    planning_times = [r[1] for r in results]

    print(time.time() - start_time, np.sum(successes), np.mean(planning_times))
    '''
    run_sbpl_motiont_primitive_planning_benchmark(
        dict(label='sbpl_diffdrive_forward_model',
             planner_name='sbpl_path_forward',
             robot_name=RobotNames.INDUSTRIAL_DIFFDRIVE_V1,
             footprint_scale=1.8,
             target_v=0.65,
             target_w=1.0,
             number_of_angles=32)
    )
    '''
    # 124.759207964 377 0.7519494995
    # 540.658827782 480 1.82676428372 (w_samples_in_each_direction=16,
    # 176.419268847 415 0.61856025974 (w_samples_in_each_direction=4